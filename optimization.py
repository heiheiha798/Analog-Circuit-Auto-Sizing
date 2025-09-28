import argparse
import pyAether as ae
import copy
import random

from src.utils import read_parameters, parse_symmetry_constraints, apply_symmetry_constraints
from src.optimizer import SimulatePlatform
from src.graph_builder import build_graph_from_eda
from src.circuit_analyzer import analyze_circuit_constraints

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

def parse_arguments():
    """Parse command-line arguments for the optimization tool"""
    parser = argparse.ArgumentParser(
        description='Circuit Optimization Platform',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ae_lib', required=True, help='Aether library name')
    parser.add_argument('--ae_cell', required=True, help='Aether cell name')
    parser.add_argument('--ae_view', required=True, help='Aether view name')
    parser.add_argument('--mde_cell', required=True, help='MDE cell name')
    parser.add_argument('--mde_view', required=True, help='MDE view name')
    parser.add_argument('--param_file', required=True, help='Parameter file path')
    parser.add_argument('--output_path', required=True, help='Output directory path')
    parser.add_argument('--output_file', required=True, help='Output file name')
    parser.add_argument('--symmetry_devices', default='', 
                        help='Symmetry devices in format "(inst1,inst2,inst3),(inst4,inst5)"')
    parser.add_argument('--dummy_devices', default='', 
                        help='Dummy devices in format "inst1,inst2,inst3"')

    # Optimization parameters
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--pop_size', type=int, default=300,
                        help='Population size for differential evolution')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output for debugging')
    parser.add_argument('--set_params', action='store_true',
                        help='Set parameters using specific param_file')
    parser.add_argument('--evaluate', action='store_true',
                        help='Simulate using specific parameters')
    parser.add_argument('--fw_scan', action='store_true',
                    help='Run a sensitivity scan on all MOS fw parameters.')
    parser.add_argument('--run_bo', action='store_true',
                            help='Run Bayesian Optimization.')    
    parser.add_argument('--run_gd', action='store_true',
                        help='Run adaptive Gradient Descent optimization.')
    parser.add_argument('--greedy_alpha', type=float, default=5.0,
                        help='Greedy jump threshold in percent (e.g., 5 for 5%%). Set to 0 for absolute greedy.')

    return parser.parse_args()

# ====================================================================
# =========== 核心算法: 两阶段层级优化 (V4) =======================
# ====================================================================
def run_gradient_descent(platform: SimulatePlatform, initial_parameters: dict, 
                         max_iterations: int = 10, 
                         line_search_depth: int = 3, # No longer used
                         perturb_ratio: float = 0.05, # Now 5% for fw/l
                         greedy_threshold_pct: float = 5.0):
    """
    执行一个两阶段的层级优化算法 (V4)。
    阶段一：只优化整数参数 'm' (粗调)。
    阶段二：固定 'm'，只优化连续参数 'fw' 和 'l' (精调)。
    """
    import random

    print("\n=======================================================")
    print("===   HIERARCHICAL GRADIENT DESCENT MODE (V4)       ===")
    print("=======================================================")

    # 目标函数 get_score (保持不变)
    platform.only_set_params(initial_parameters)
    baseline_scores = platform.evaluate()
    if not baseline_scores:
        print("FATAL: Baseline simulation failed."); return
    baseline_ugb = baseline_scores.get('UGB', 1.0)
    baseline_area = baseline_scores.get('Total_Area', 1.0)

    def get_score(scores: dict):
        if not scores: return 1e12
        pm = scores.get('Phase_Margin', -180.0); gain = scores.get('Gain_db', -200.0); gm = scores.get('Gain_Margin', 100.0)
        if pm <= -180.0 or gain <= -200.0 or gm >= 100.0: return 1e12
        iopa_ma = scores.get('I_OPA', 1.0) * 1000.0
        
        # --- 【核心修改】: 引入对低电流的惩罚 ---
        LOW_CURRENT_THRESHOLD = 2.0  # 定义一个低电流阈值，例如 2.0mA

        pm_viol = max(0, (50.0 - pm) / 50.0)
        gain_viol = max(0, (80.0 - gain) / 80.0)
        gm_viol = max(0, (gm - (-10.0)) / abs(-10.0))
        
        # 新的 iopa_viol 计算逻辑
        if iopa_ma < LOW_CURRENT_THRESHOLD:
            # 如果低于阈值，惩罚与距离成正比
            iopa_viol = (LOW_CURRENT_THRESHOLD - iopa_ma) / LOW_CURRENT_THRESHOLD
        elif iopa_ma > 3.0: # 硬上限保持不变
            iopa_viol = (iopa_ma - 3.0) / 3.0
        else:
            iopa_viol = 0 # 在 [2.0mA, 3.0mA] 区间内无惩罚

        # ---------------------------------------------
        
        total_violation = pm_viol + gain_viol + gm_viol + iopa_viol
        
        if total_violation > 0: 
            return 1e9 + total_violation * 1e6
        else:
            # 只有当所有指标都达标时，才计算质量分
            ugb_norm = scores.get('UGB', 0) / baseline_ugb
            area_norm = scores.get('Total_Area', 0) / baseline_area
            return 0.5 * area_norm - 0.5 * ugb_norm

    # ----------------------------------------------------------------
    # 辅助函数: 封装单次优化循环的逻辑，将在两个阶段中被复用
    # ----------------------------------------------------------------
    def run_optimization_stage(stage_name: str, start_params: dict, optimizable_param_names: list, stage_max_iter: int):
        print(f"\n{'#'*25} Starting {stage_name} {'#'*25}")
        print(f"Optimizing {len(optimizable_param_names)} parameters in this stage.")
        
        X_current_params = copy.deepcopy(start_params)

        for i in range(stage_max_iter):
            print(f"\n--- {stage_name} Iteration {i+1}/{stage_max_iter} ---")
            platform.only_set_params(X_current_params)
            score_current = get_score(platform.evaluate())
            print(f"  - Current score: {score_current:.4f}")

            found_immediate_jump = False
            best_neighbor_so_far = {'params': None, 'score': score_current}
            shuffled_param_names = random.sample(optimizable_param_names, len(optimizable_param_names))

            for name in shuffled_param_names:
                original_value = X_current_params[name].value
                param_type = X_current_params[name].type
                
                delta = 1 if param_type == 'integer' else original_value * perturb_ratio

                for sign in [1, -1]:
                    # 跳过 m=1 时的 -1 操作
                    if param_type == 'integer' and original_value <= 1 and sign == -1: continue

                    params_probe = copy.deepcopy(X_current_params)
                    params_probe[name].value += sign * delta
                    
                    platform.only_set_params(params_probe)
                    score_probe = get_score(platform.evaluate())
                    print(f"    - Probing {name} ({'+' if sign > 0 else '-'}{'1' if param_type == 'integer' else f'{perturb_ratio*100}%'})... Score: {score_probe:.4f}")
                    
                    if score_probe < best_neighbor_so_far['score']:
                        best_neighbor_so_far = {'params': params_probe, 'score': score_probe}

                    improvement = score_current - score_probe
                    threshold = 0
                    if score_current >= 1e9:
                        penalty_part = score_current - 1e9
                        threshold = penalty_part * (greedy_threshold_pct / 100.0)

                    if improvement > threshold:
                        print(f"  >>> GREEDY JUMP! Found significant improvement. Moving immediately.")
                        X_current_params = params_probe
                        found_immediate_jump = True
                        break
                if found_immediate_jump:
                    break

            if found_immediate_jump:
                continue

            if best_neighbor_so_far['score'] < score_current:
                print(f"  - Found a modest improvement. Moving to best neighbor.")
                X_current_params = best_neighbor_so_far['params']
            else:
                print(f"\n--- CONVERGENCE in {stage_name}: No further improvement found. ---")
                break
        
        return X_current_params

    # ======================== 主流程 ========================
    
    # --- 阶段一: 只优化 'm' 参数 (粗调) ---
    m_param_names = [name for name, param in initial_parameters.items() if name.endswith('_m') and not param.is_dummy]
    # 将总迭代次数分配给两个阶段
    stage1_max_iter = int(max_iterations * 0.5) # 50%的预算给m
    params_after_stage1 = run_optimization_stage(
        stage_name="Stage 1: Integer Space (m)",
        start_params=initial_parameters,
        optimizable_param_names=m_param_names,
        stage_max_iter=stage1_max_iter
    )

    # --- 阶段二: 固定 'm'，优化 'fw' 和 'l' (精调) ---
    fw_l_param_names = [name for name, param in initial_parameters.items() if (name.endswith('_fw') or name.endswith('_l')) and not param.is_dummy]
    stage2_max_iter = max_iterations - stage1_max_iter # 剩下的预算给fw/l
    final_params = run_optimization_stage(
        stage_name="Stage 2: Continuous Space (fw & l)",
        start_params=params_after_stage1,
        optimizable_param_names=fw_l_param_names,
        stage_max_iter=stage2_max_iter
    )

    # --- 结束和保存 ---
    print("\n=======================================================")
    print("===      HIERARCHICAL OPTIMIZATION COMPLETED        ===")
    print("=======================================================")
    print("Final best parameters found:")
    platform.only_set_params(final_params)
    final_score = get_score(platform.evaluate())
    
    output_dir = f"{platform.output_path}"
    os.makedirs(output_dir, exist_ok=True)
    final_result_file = f"{output_dir}/gd_v4_final_solution.txt"
    with open(final_result_file, 'w') as f:
        f.write(f"Optimization Status: Finished Hierarchical Optimization.\n")
        f.write(f"Final Score: {final_score:.4f}\n\n")
        f.write("Optimized Parameters:\n")
        all_optimizable_names = m_param_names + fw_l_param_names
        for name, param_obj in final_params.items():
            if name in all_optimizable_names:
                formatted_value = param_obj.format_value(param_obj.value)
                f.write(f"  {name}: {formatted_value}\n")

def run_bayesian_optimization(platform: SimulatePlatform, initial_parameters: dict, n_calls: int = 100):
    """
    使用scikit-optimize执行贝叶斯优化。

    :param platform: SimulatePlatform的实例，用于执行仿真。
    :param initial_parameters: 从文件中读取的初始参数字典，用于定义搜索空间和基准。
    :param n_calls: 贝叶斯优化器总的评估（仿真）次数。
    """
    print("\n=======================================================")
    print("===      STARTING BAYESIAN OPTIMIZATION MODE        ===")
    print("=======================================================")

    # 1. 定义搜索空间 (Search Space)
    #    告诉优化器每个参数的名字、类型和范围
    search_space = []
    param_names = []

    initial_values = {name: param.value for name, param in initial_parameters.items()}

    for name, param in initial_parameters.items():
        if param.is_dummy:
            continue
        
        param_names.append(name)
        
        # 定义一个更小的、围绕初始值的搜索范围
        # 例如，初始值的50%到200%
        low_bound = max(param.min, initial_values[name] * 0.5)
        high_bound = min(param.max, initial_values[name] * 2.0)

        # 确保下界不大于上界
        if low_bound > high_bound:
            low_bound, high_bound = high_bound, low_bound
            
        if param.type == 'integer':
            # 对于整数类型，确保范围至少为1
            high_bound = max(high_bound, low_bound + 1)
            space_dim = Integer(low=int(round(low_bound)), high=int(round(high_bound)), name=name)
        else: # continuous
            space_dim = Real(low=low_bound, high=high_bound, name=name)

        search_space.append(space_dim)
        # print(f"  - Defining space for {name}: range [{space_dim.low}, {space_dim.high}]")

    # 2. 获取基准性能，用于后续目标归一化
    print("\n--- Running simulation for baseline performance... ---")
    platform.only_set_params(initial_parameters)
    baseline_scores = platform.evaluate()
    if not baseline_scores:
        print("FATAL: Baseline simulation failed. Cannot proceed with optimization.")
        return
    
    baseline_ugb = baseline_scores.get('UGB', 1.0) # 避免除以0
    baseline_area = baseline_scores.get('Total_Area', 1.0)

    # 3. 定义内部目标函数 (Objective Function for BO)
    #    这个函数是黑盒的核心，它接收参数，返回一个单一的“成本”值
    @use_named_args(search_space)
    def objective_for_bo(**params_kwargs):
        current_params = copy.deepcopy(initial_parameters)
        for name, value in params_kwargs.items():
            current_params[name].value = value

        platform.only_set_params(current_params)
        scores = platform.evaluate()

        if not scores:
            return 1e12

        # --- 新的、更精细的成本计算 ---
        
        pm = scores.get('Phase_Margin', -180.0) # 无效时给最差的物理值
        gain = scores.get('Gain_db', -200.0)    # 无效时给最差的物理值
        gm = scores.get('Gain_Margin', 100.0)   # 无效时给最差的物理值
        iopa = scores.get('I_OPA', 1.0) * 1000.0 # 无效时给一个很大的电流值

        # a. 检查仿真是否收敛 (AC部分)
        if pm <= -180.0 or gain <= -200.0 or gm >= 100.0:
            print("  [Objective] Sim result indicates non-convergence. Returning max penalty.")
            return 1e12

        # b. 计算归一化的约束偏离度 (0表示满足，>0表示不满足)
        pm_viol = max(0, (50.0 - pm) / 50.0)           # 偏离50度的百分比
        gain_viol = max(0, (80.0 - gain) / 80.0)         # 偏离80dB的百分比
        gm_viol = max(0, (gm - (-10.0)) / abs(-10.0)) # 偏离-10dB的百分比
        iopa_viol = max(0, (iopa - 3.0) / 3.0)           # 超出3mA的百分比

        total_violation = pm_viol + gain_viol + gm_viol + iopa_viol

        # c. 根据约束偏离度决定成本
        if total_violation > 0:
            # 即使不可行，也返回一个能反映“差多远”的有限值
            cost = 1e9 + total_violation * 1e6 # 基础惩罚+偏离度惩罚
            print(f"  [Objective] Infeasible solution. Violation = {total_violation:.3f}, Cost = {cost:.2f}")
            return cost
        else:
            # d. 只有完全满足约束，才计算质量分
            ugb_norm = scores['UGB'] / baseline_ugb
            area_norm = scores['Total_Area'] / baseline_area
            
            w_ugb = 0.5
            w_area = 0.5
            quality_score = w_area * area_norm - w_ugb * ugb_norm
            
            print(f"  [Objective] Feasible solution. Quality Score = {quality_score:.4f}")
            return quality_score

    # 4. 运行贝叶斯优化
    print(f"\n--- Starting Bayesian Optimization with {n_calls} calls... ---")
    result = gp_minimize(
        func=objective_for_bo,
        dimensions=search_space,
        n_calls=n_calls,          # 总仿真次数
        n_initial_points=10,      # 初始随机探索的点数
        acq_func="EI",            # 使用经典的Expected Improvement采集函数
        random_state=123          # 保证结果可复现
    )

    # 5. 打印最终结果
    print("\n=======================================================")
    print("===         BAYESIAN OPTIMIZATION COMPLETED         ===")
    print("=======================================================")
    print(f"Best score found: {result.fun:.4f}")
    print("\nBest parameters found:")
    
    best_params_dict = {}
    for dim, value in zip(search_space, result.x):
        # 将结果格式化后打印
        param_obj = initial_parameters[dim.name]
        formatted_value = param_obj.format_value(value)
        print(f"  - {dim.name}: {formatted_value}")
        best_params_dict[dim.name] = value

    # (可选) 用找到的最佳参数再跑一次仿真，以展示最终性能
    print("\n--- Evaluating the final best parameters... ---")
    final_params = copy.deepcopy(initial_parameters)
    for name, value in best_params_dict.items():
        final_params[name].value = value
    platform.only_set_params(final_params)
    platform.evaluate()

if __name__ == "__main__":
    args = parse_arguments()

    # 2. 初始化EDA环境
    print("Initializing Aether environment...")
    ae.emyInitAether('-adv')
    
    # 3. 创建仿真平台实例
    # 注意: 我们在这里创建实例，因为它包含了所有模式都需要的配置信息
    platform = SimulatePlatform(
        ae_lib=args.ae_lib,
        ae_cell=args.ae_cell,
        ae_view=args.ae_view,
        mde_cell=args.mde_cell,
        mde_view=args.mde_view,
        output_path=args.output_path,
        output_file=args.output_file,
    )

    # --- 核心逻辑分支 ---

    if args.fw_scan:
        print("\n=======================================================")
        print("===         STARTING FW SENSITIVITY SCAN MODE         ===")
        print("=======================================================")

        # a. 打开电路设计视图，用于分析
        print(f"\nOpening design for analysis: {args.ae_lib}/{args.ae_cell}/{args.ae_view}")
        cv = ae.dbOpenCV(args.ae_lib, args.ae_cell, args.ae_view)
        if cv is None:
            print("Error: Failed to open design view for analysis.")
            exit(1)
        
        try:
            # b. 构建电路图并分析对称约束
            print("\n--- Building Circuit Graph and Analyzing Constraints ---")
            circuit_graph = build_graph_from_eda(cv, args.param_file)
            analyze_circuit_constraints(circuit_graph)
        except Exception as e:
            print(f"\nAn error occurred during graph construction or analysis: {e}")
            exit(1)
        finally:
            print("\nClosing design view after analysis.")
            ae.dbCloseCV(cv)

        # c. 加载初始参数文件，作为所有修改的“黄金副本”
        print(f"\n--- Loading initial parameters from: {args.param_file} ---")
        initial_parameters = read_parameters(args.param_file)
        if not initial_parameters:
            print("Error: Failed to read initial parameters. Exiting.")
            exit(1)

        # d. 准备一个字典来存储所有仿真结果
        scan_results = {}

        # e. 运行基准仿真 (Baseline)
        print("\n--- [Step 1/3] Running Baseline Simulation ---")
        platform.only_set_params(initial_parameters)
        scan_results['baseline'] = platform.evaluate()

        # f. 根据分析结果构建扫描任务列表
        print("\n--- [Step 2/3] Preparing Scan Tasks Based on Symmetry ---")
        scan_tasks = []
        processed_devices = set()
        
        # 从对称组创建任务
        for group in circuit_graph.constraint_groups:
            # 确保组内是MOS管
            if "MOS" in group.devices[0].device_type:
                device_names = [d.name for d in group.devices]
                scan_tasks.append({'name': group.group_type, 'devices': device_names})
                processed_devices.update(device_names)
        
        # 从独立的MOS管创建任务
        for name, device in circuit_graph.devices.items():
            if ("MOS" in device.device_type) and (name not in processed_devices):
                scan_tasks.append({'name': name, 'devices': [name]})
        
        print(f"Found {len(scan_tasks)} unique MOS devices/groups to scan.")

        # g. 执行扫描循环
        print("\n--- [Step 3/3] Executing Parameter Scan Loop ---")
        for i, task in enumerate(scan_tasks):
            task_name = task['name']
            device_list = task['devices']
            print(f"\n[{i+1}/{len(scan_tasks)}] Scanning Target: {task_name}")

            # 获取此任务中器件的原始fw值 (从任意一个器件即可)
            representative_device_name = device_list[0]
            param_key_fw = f"{representative_device_name}_fw"
            if param_key_fw not in initial_parameters:
                print(f"Warning: Cannot find fw parameter for {representative_device_name}. Skipping task.")
                continue
            original_fw_value = initial_parameters[param_key_fw].value

            # --- 扫描 -10% (0.9倍) ---
            print(f"  -> Scanning fw at -10% ({original_fw_value * 0.9:.4e})")
            params_minus_10 = copy.deepcopy(initial_parameters)
            for device_name in device_list:
                params_minus_10[f"{device_name}_fw"].value = original_fw_value * 0.9
            platform.only_set_params(params_minus_10)
            scan_results[f"{task_name}_fw_-10%"] = platform.evaluate()

            # --- 扫描 +10% (1.1倍) ---
            print(f"  -> Scanning fw at +10% ({original_fw_value * 1.1:.4e})")
            params_plus_10 = copy.deepcopy(initial_parameters)
            for device_name in device_list:
                params_plus_10[f"{device_name}_fw"].value = original_fw_value * 1.1
            platform.only_set_params(params_plus_10)
            scan_results[f"{task_name}_fw_+10%"] = platform.evaluate()

        # h. 汇总并打印所有结果
        print("\n\n=======================================================")
        print("===          FW SENSITIVITY SCAN FINAL RESULTS        ===")
        print("=======================================================")
        for name, result in scan_results.items():
            if not result: # 检查仿真失败的情况
                print(f"\n[{name}]:")
                print("  SIMULATION FAILED")
                continue

            ugb = result.get('UGB', 'N/A')
            pm = result.get('Phase_Margin', 'N/A')
            gain = result.get('Gain_db', 'N/A')
            gm = result.get('Gain_Margin', 'N/A')
            iopa = result.get('I_OPA', 'N/A')
            area = result.get('Total_Area', 'N/A')
            
            # 格式化输出
            ugb_str = f"{ugb/1e6:.3f} MHz" if isinstance(ugb, (int, float)) else ugb
            pm_str = f"{pm:.2f} deg" if isinstance(pm, (int, float)) else pm
            gain_str = f"{gain:.2f} dB" if isinstance(gain, (int, float)) else gain
            gm_str = f"{gm:.2f} dB" if isinstance(gm, (int, float)) else gm
            iopa_str = f"{iopa*1000:.4f} mA" if isinstance(iopa, (int, float)) else iopa
            area_str = f"{area:.2f} um^2" if isinstance(area, (int, float)) else area

            print(f"\n[{name}]:")
            print(f"  UGB: {ugb_str}, PM: {pm_str}, Gain: {gain_str}, GM: {gm_str}, I_OPA: {iopa_str}, Area: {area_str}") # <--- 修改这一行

        # 扫描完成后，将电路恢复到初始状态
        print("\n--- Restoring circuit to initial parameters... ---")
        platform.only_set_params(initial_parameters)

    elif args.set_params and not args.evaluate:
        print("\n--- SET PARAMETERS MODE ---")
        parameters = read_parameters(args.param_file)
        platform.only_set_params(parameters)
        print("Parameters have been set on the schematic.")

    elif args.evaluate:
        print("\n--- EVALUATE MODE ---")
        if args.set_params:
            print("Setting parameters before evaluation...")
            parameters = read_parameters(args.param_file)
            platform.only_set_params(parameters)
        print("Running a single simulation...")
        platform.evaluate()
    
    elif args.run_gd:
        initial_parameters = read_parameters(args.param_file)
        if not initial_parameters:
            print("Error: Failed to read initial parameters for GD. Exiting.")
            exit(1)
        
        # 运行我们新的梯度下降主函数
        run_gradient_descent(platform, initial_parameters, 
                             max_iterations=20, # 建议从一个稍大的迭代次数开始
                             perturb_ratio=0.05,
                             greedy_threshold_pct=args.greedy_alpha) # 从命令行传入 alpha
    elif args.run_bo:
        # a. 加载初始参数
        initial_parameters = read_parameters(args.param_file)
        if not initial_parameters:
            print("Error: Failed to read initial parameters for BO. Exiting.")
            exit(1)
        
        # b. 运行我们新的贝叶斯优化主函数
        #    为了快速测试，可以先设置一个较小的 n_calls，例如 50
        run_bayesian_optimization(platform, initial_parameters, n_calls=50)

    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --fw_scan.")
        print("Example: python optimization.py --fw_scan [other_args...]")

    print("\nScript finished.")

