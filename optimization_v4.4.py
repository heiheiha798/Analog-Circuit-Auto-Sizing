# V4.4 步长调整 + 优化停滞终止

import argparse
import pyAether as ae
import copy
import random
import os

from src.utils import read_parameters
from src.optimizer import SimulatePlatform
from src.data_models import CircuitGraph
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
    
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output for debugging')
    parser.add_argument('--set_params', action='store_true',
                        help='Set parameters using specific param_file')
    parser.add_argument('--evaluate', action='store_true',
                        help='Simulate using specific parameters')
    parser.add_argument('--run_gd', action='store_true',
                        help='Run adaptive Gradient Descent optimization.')
    parser.add_argument('--greedy_alpha', type=float, default=5.0,
                        help='Greedy jump threshold in percent (e.g., 5 for 5%%). Set to 0 for absolute greedy.')

    return parser.parse_args()

def run_hybrid_optimization_v4_3(platform: SimulatePlatform, initial_parameters: dict,
                                   circuit_graph: CircuitGraph,
                                   max_iterations: int = 40,
                                   perturb_ratio: float = 0.05,
                                   initial_greedy_alpha_pct: float = 5.0, # MODIFIED: 重命名
                                   min_greedy_alpha_pct: float = 1.0):   # NEW: 最小alpha阈值
    """
    执行一个混合优化算法 (V4.4)，该版本采用用户定义的自适应Alpha阈值逻辑。
    - 当移动到一个“最佳邻居”（非贪婪跳跃）时，将alpha动态调整为实际的改进百分比。
    - 步长保持不变，以隔离alpha调整的效果。
    - 当最佳改进低于min_alpha时，判定为收敛。
    """
    print("\n===================================================================")
    print("===   HYBRID OPTIMIZATION (V4.4) WITH ADAPTIVE ALPHA      ===")
    print("===================================================================")

    # --- 目标函数 get_score (保持不变) ---
    platform.only_set_params(initial_parameters)
    baseline_scores = platform.evaluate()
    # ... (此处代码与之前版本完全相同) ...
    def get_score(scores: dict):
        if not scores: return 1e12
        pm = scores.get('Phase_Margin', -180.0); gain = scores.get('Gain_db', -200.0); gm = scores.get('Gain_Margin', 100.0)
        if pm <= -180.0 or gain <= -200.0 or gm >= 100.0: return 1e12
        iopa_ma = scores.get('I_OPA', 1.0) * 1000.0
        pm_viol = max(0, (50.0 - pm) / 50.0)
        gain_viol = max(0, (80.0 - gain) / 80.0)
        gm_viol = max(0, (gm - (-10.0)) / abs(-10.0))
        iopa_viol = max((iopa_ma - 3.0) / 3.0, 0)
        total_violation = pm_viol + gain_viol + gm_viol + iopa_viol
        if total_violation > 0:
            return 1e9 + total_violation * 1e6
        else:
            baseline_ugb = 1.0
            baseline_area = 1.0
            ugb_norm = scores.get('UGB', 0) / baseline_ugb
            area_norm = scores.get('Total_Area', 0) / baseline_area
            return 0.5 * area_norm - 0.5 * ugb_norm

    # --- 构建对称参数映射 (逻辑不变) ---
    param_mapping = {}
    # ... (此处代码与之前版本完全相同) ...
    devices_in_groups = set()
    for group in circuit_graph.constraint_groups:
        representative_device = group.devices[0]
        for param_name_full, param_obj in initial_parameters.items():
            if param_name_full.startswith(representative_device.name + '_'):
                param_suffix = param_name_full[len(representative_device.name):]
                key = f"{representative_device.name}{param_suffix}"
                param_mapping[key] = [f"{dev.name}{param_suffix}" for dev in group.devices]
        for dev in group.devices:
            devices_in_groups.add(dev.name)
    for name, param in initial_parameters.items():
        device_name = name.split('_')[0]
        if device_name not in devices_in_groups and not param.is_dummy:
            if name not in param_mapping:
                 param_mapping[name] = [name]

    # --- 参数分类和状态初始化 ---
    m_param_names = {name for name in param_mapping.keys() if name.endswith('_m')}
    other_param_names = {name for name in param_mapping.keys() if not name.endswith('_m')}
    frozen_params = set()
    optimization_phase = "coarse_tune_m"
    current_greedy_alpha_pct = initial_greedy_alpha_pct # NEW: 初始化动态alpha

    # --- 全局最优解追踪器 (逻辑不变) ---
    tracking_info = {
        'best_params': copy.deepcopy(initial_parameters),
        'best_score': get_score(baseline_scores),
        'best_iter_num': 0,
        'global_iter_count': 0,
        'best_metrics': baseline_scores
    }

    # ======================== 主优化循环 ========================
    print(f"\n{'#'*25} Starting Main Optimization Loop {'#'*25}")
    X_current_params = copy.deepcopy(initial_parameters)
    last_successful_move = None

    for i in range(max_iterations):
        tracking_info['global_iter_count'] += 1
        current_iter_num = tracking_info['global_iter_count']

        # ... (阶段判断逻辑不变) ...
        if optimization_phase == "coarse_tune_m":
            active_param_names = list(m_param_names - frozen_params)
            if not active_param_names:
                print(f"\n{'!'*15} All 'm' parameters frozen. Switching to fine-tuning phase. {'!'*15}")
                optimization_phase = "fine_tune_all"
                active_param_names = list(other_param_names)
        else: # fine_tune_all
            active_param_names = list(other_param_names)

        if not active_param_names:
            print("\n--- CONVERGENCE: No active parameters left to optimize. ---")
            break

        print(f"\n--- Iteration {i+1}/{max_iterations} (Phase: {optimization_phase}, Alpha: {current_greedy_alpha_pct:.2f}%) ---")
        platform.only_set_params(X_current_params)
        eval_results = platform.evaluate()
        score_current = get_score(eval_results)
        print(f"  - Current score: {score_current:.4f}")

        # ... (全局最优解追踪和惯性搜索逻辑不变) ...
        
        found_immediate_jump = False
        best_neighbor_so_far = {'params': None, 'score': score_current, 'name': None, 'sign': None}
        shuffled_param_names = random.sample(active_param_names, len(active_param_names))

        for name in shuffled_param_names:
            # ... (内部探测逻辑，包括计算delta等，与V4.2相同) ...
            original_value = X_current_params[name].value
            param_type = X_current_params[name].type
            delta = 1 if param_type == 'integer' else original_value * perturb_ratio

            found_improvement_for_this_param = False
            for sign in [1, -1]:
                # ...
                params_probe = copy.deepcopy(X_current_params)
                # ...
                
                platform.only_set_params(params_probe)
                probe_eval_results = platform.evaluate()
                score_probe = get_score(probe_eval_results)

                # ... (全局最优更新) ...

                if score_probe < best_neighbor_so_far['score']:
                    best_neighbor_so_far = {'params': params_probe, 'score': score_probe, 'name': name, 'sign': sign}
                
                if score_probe < score_current:
                    found_improvement_for_this_param = True
                
                # MODIFIED: 使用动态alpha计算阈值
                improvement = score_current - score_probe
                threshold = 0
                if score_current >= 1e9:
                    penalty_part = score_current - 1e9
                    threshold = penalty_part * (current_greedy_alpha_pct / 100.0)

                if improvement > threshold:
                    print(f"  >>> GREEDY JUMP! Found improvement > {current_greedy_alpha_pct:.2f}%. Moving immediately.")
                    X_current_params = params_probe
                    found_immediate_jump = True
                    last_successful_move = {'name': name, 'sign': sign}
                    # NEW: 贪婪跳跃后，将alpha重置为初始值，鼓励继续大步前进
                    current_greedy_alpha_pct = initial_greedy_alpha_pct
                    break
            
            if found_immediate_jump:
                break
            
            if not found_improvement_for_this_param and name in m_param_names:
                print(f"  - Parameter '{name}' hit a local optimum. Freezing.")
                frozen_params.add(name)

        if found_immediate_jump:
            continue

        # --- NEW: Adaptive Alpha and Convergence Logic ---
        if best_neighbor_so_far['score'] < score_current:
            # 计算实际改进百分比
            improvement = score_current - best_neighbor_so_far['score']
            improvement_pct = 0
            if score_current >= 1e9:
                penalty_part = score_current - 1e9
                if penalty_part > 0:
                    improvement_pct = (improvement / penalty_part) * 100.0
            # 注意: 这里可以补充在非惩罚区的百分比计算逻辑，但通常惩罚区是主要场景
            
            if improvement_pct < min_greedy_alpha_pct:
                print(f"\n--- CONVERGENCE: Best improvement ({improvement_pct:.2f}%) is below minimum alpha ({min_greedy_alpha_pct:.2f}%). ---")
                # 移动到最后这个微小改进点
                X_current_params = best_neighbor_so_far['params']
                break

            print(f"  - Found a modest improvement ({improvement_pct:.2f}%). Moving to best neighbor.")
            X_current_params = best_neighbor_so_far['params']
            last_successful_move = {'name': best_neighbor_so_far['name'], 'sign': best_neighbor_so_far['sign']}
            
            # 动态调整alpha
            new_alpha = max(min_greedy_alpha_pct, improvement_pct)
            print(f"  >>> Adjusting alpha from {current_greedy_alpha_pct:.2f}% to {new_alpha:.2f}%")
            current_greedy_alpha_pct = new_alpha
        else:
            print(f"\n--- CONVERGENCE: No further improvement found in this iteration. ---")
            last_successful_move = None
            break
            
    # --- 结束和保存 (逻辑不变) ---
    print("\n=======================================================")
    print("===      HYBRID OPTIMIZATION (V4.4) COMPLETED     ===")
    print("=======================================================")
    print("Final best parameters found (from overall optimization):")
    
    print("\n========== Best Evaluation Result (from Iter {}) ==========".format(tracking_info['best_iter_num']))
    for key, value in tracking_info['best_metrics'].items():
        print(f"====== {key:<12} : {value}")

    output_dir = f"{platform.output_path}"
    os.makedirs(output_dir, exist_ok=True)
    final_result_file = f"{output_dir}/hybrid_v4.2_final_solution.txt"
    with open(final_result_file, 'w') as f:
        f.write(f"Optimization Status: Finished Hybrid Optimization.\n")
        f.write(f"Found at Iteration: {tracking_info['best_iter_num']}\n")
        f.write(f"Final Score: {tracking_info['best_score']:.4f}\n")
        f.write(f"Frozen 'm' parameters ({len(frozen_params)}): {sorted(list(frozen_params))}\n\n")
        f.write("Optimized Parameters:\n")
        
        final_best_params = tracking_info['best_params']
        all_optimizable_names = list(m_param_names) + list(other_param_names)
        for name in sorted(all_optimizable_names):
            if name in final_best_params:
                param_obj = final_best_params[name]
                formatted_value = param_obj.format_value(param_obj.value)
                status = " (Frozen)" if name in frozen_params else ""
                f.write(f"  {name}: {formatted_value}{status}\n")

if __name__ == "__main__":
    args = parse_arguments()
    ae.emyInitAether('-adv')
    
    platform = SimulatePlatform(
        ae_lib=args.ae_lib,
        ae_cell=args.ae_cell,
        ae_view=args.ae_view,
        mde_cell=args.mde_cell,
        mde_view=args.mde_view,
        output_path=args.output_path,
        output_file=args.output_file,
    )

    if args.set_params and not args.evaluate:
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
        print("\n--- Building Circuit Graph for Symmetry Analysis ---")
        cv = None # 确保在 try 外可以访问
        try:
            cv = ae.dbOpenCV(args.ae_lib, args.ae_cell, args.ae_view)
            if cv is None:
                print("Error: Failed to open design view for analysis.")
                exit(1)
            
            circuit_graph = build_graph_from_eda(cv, args.param_file)
            analyze_circuit_constraints(circuit_graph)
            
            # 打印找到的对称组，便于调试
            if circuit_graph.constraint_groups:
                print(f"\nSymmetry analysis found {len(circuit_graph.constraint_groups)} groups for optimization.")
                for i, group in enumerate(circuit_graph.constraint_groups):
                    device_names = ', '.join([d.name for d in group.devices])
                    print(f"  - Group {i+1}: [{device_names}]")
            else:
                print("\nNo symmetric groups found. Proceeding with standard optimization.")

        except Exception as e:
            print(f"\nAn error occurred during graph construction or analysis: {e}")
            exit(1)
        finally:
            if cv:
                # print("Closing design view after analysis.")
                ae.dbCloseCV(cv)

        initial_parameters = read_parameters(args.param_file)
        if not initial_parameters:
            print("Error: Failed to read initial parameters for GD. Exiting.")
            exit(1)
        
        run_hybrid_optimization_v4_3(platform, initial_parameters, circuit_graph,
                                    max_iterations=20, # 你可以调整这些值
                                    perturb_ratio=0.05,
                                    initial_greedy_alpha_pct=args.greedy_alpha,
                                    min_greedy_alpha_pct = 1.0) # 新增不敏感性阈值
        
    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --fw_scan.")
        print("Example: python optimization.py --fw_scan [other_args...]")

    print("\nScript finished.")

