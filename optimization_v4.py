# V4 先m后fw、l、r、c，固定分层

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

# ====================================================================
# =========== \u6838\u5fc3\u7b97\u6cd5: \u4e24\u9636\u6bb5\u5c42\u7ea7\u4f18\u5316 (V4) =======================
# ====================================================================
def run_gradient_descent(platform: SimulatePlatform, initial_parameters: dict, 
                         circuit_graph: CircuitGraph,
                         max_iterations: int = 20, 
                         perturb_ratio: float = 0.05,
                         greedy_threshold_pct: float = 5.0):
    """
    执行一个两阶段的层级优化算法 (V4)，并利用对称性约束减少优化变量。
    此版本会追踪并记录全局最优解是在哪一次迭代中发现的。
    """
    print("\n=======================================================")
    print("===   HIERARCHICAL GRADIENT DESCENT (V4) WITH SYMMETRY  ===")
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
        pm_viol = max(0, (50.0 - pm) / 50.0)
        gain_viol = max(0, (80.0 - gain) / 80.0)
        gm_viol = max(0, (gm - (-10.0)) / abs(-10.0))
        iopa_viol = max((iopa_ma - 3.0) / 3.0, 0)
        total_violation = pm_viol + gain_viol + gm_viol + iopa_viol
        if total_violation > 0: 
            return 1e9 + total_violation * 1e6
        else:
            ugb_norm = scores.get('UGB', 0) / baseline_ugb
            area_norm = scores.get('Total_Area', 0) / baseline_area
            return 0.5 * area_norm - 0.5 * ugb_norm

    # ----------------------------------------------------------------
    # 辅助函数: 封装单次优化循环的逻辑，将在两个阶段中被复用
    # ----------------------------------------------------------------
    # MODIFICATION 1: 修改 run_optimization_stage 的函数签名，增加 tracking_info 参数
    def run_optimization_stage(stage_name: str, start_params: dict, optimizable_param_names: list, stage_max_iter: int, param_mapping: dict, tracking_info: dict):
        print(f"\n{'#'*25} Starting {stage_name} {'#'*25}")
        print(f"Optimizing {len(optimizable_param_names)} independent parameters/groups in this stage.")
        
        X_current_params = copy.deepcopy(start_params)

        for i in range(stage_max_iter):
            # MODIFICATION 2: 增加全局迭代计数器
            tracking_info['global_iter_count'] += 1
            current_iter_num = tracking_info['global_iter_count']

            print(f"\n--- {stage_name} Iteration {i+1}/{stage_max_iter} (Global Step: {current_iter_num}) ---")
            platform.only_set_params(X_current_params)
            eval_results = platform.evaluate()
            score_current = get_score(eval_results)
            print(f"  - Current score: {score_current:.4f}")

            # 在每次迭代开始时，也检查一下当前解是否是历史最优
            if score_current < tracking_info['best_score']:
                print(f"  *** New overall best found at the start of an iteration! Score: {score_current:.4f}, Iter: {current_iter_num} ***")
                tracking_info['best_score'] = score_current
                tracking_info['best_params'] = copy.deepcopy(X_current_params)
                tracking_info['best_iter_num'] = current_iter_num
                tracking_info['best_metrics'] = eval_results

            found_immediate_jump = False
            best_neighbor_so_far = {'params': None, 'score': score_current}
            shuffled_param_names = random.sample(optimizable_param_names, len(optimizable_param_names))

            for name in shuffled_param_names:
                original_value = X_current_params[name].value
                param_type = X_current_params[name].type
                
                delta = 1 if param_type == 'integer' else original_value * perturb_ratio

                for sign in [1, -1]:
                    if param_type == 'integer' and original_value <= 1 and sign == -1: continue

                    params_probe = copy.deepcopy(X_current_params)
                    new_value = original_value + sign * delta
                    
                    for actual_param_to_update in param_mapping[name]:
                        params_probe[actual_param_to_update].value = new_value

                    platform.only_set_params(params_probe)
                    probe_eval_results = platform.evaluate()
                    score_probe = get_score(probe_eval_results)
                    print(f"      - Probing {name} ({'+' if sign > 0 else '-'}{'1' if param_type == 'integer' else f'{perturb_ratio*100}%'})... Score: {score_probe:.4f}")
                    
                    # MODIFICATION 3: 每次探测到一个新解时，都与全局最优解比较
                    if score_probe < tracking_info['best_score']:
                        print(f"  *** New overall best found! Score: {score_probe:.4f}, Iter: {current_iter_num} ***")
                        tracking_info['best_score'] = score_probe
                        tracking_info['best_params'] = copy.deepcopy(params_probe)
                        tracking_info['best_iter_num'] = current_iter_num
                        tracking_info['best_metrics'] = probe_eval_results

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

    # ======================== 新增：构建对称参数映射 (代码不变) ========================
    param_mapping = {}
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

    # ======================== 主流程 ========================
    
    # MODIFICATION 4: 初始化全局追踪器
    tracking_info = {
        'best_params': copy.deepcopy(initial_parameters),
        'best_score': get_score(baseline_scores),
        'best_iter_num': 0,
        'global_iter_count': 0,
        'best_metrics': baseline_scores
    }
    
    # --- 阶段一: 只优化 'm' 参数 (粗调) ---
    m_param_names = [name for name in param_mapping.keys() if name.endswith('_m')]
    
    stage1_max_iter = int(max_iterations * 0.5)
    params_after_stage1 = run_optimization_stage(
        stage_name="Stage 1: Integer Space (m)",
        start_params=initial_parameters,
        optimizable_param_names=m_param_names,
        stage_max_iter=stage1_max_iter,
        param_mapping=param_mapping,
        tracking_info=tracking_info # 传递追踪器
    )

    # --- 阶段二: 固定 'm'，优化 'fw' 和 'l' (精调) ---
    fw_l_param_names = [name for name in param_mapping.keys() if (name.endswith('_fw') or name.endswith('_l') or name.endswith('_segW') or name.endswith('_segL'))]
    
    stage2_max_iter = max_iterations - stage1_max_iter
    run_optimization_stage(
        stage_name="Stage 2: Continuous Space (fw, l, etc.)",
        start_params=params_after_stage1,
        optimizable_param_names=fw_l_param_names,
        stage_max_iter=stage2_max_iter,
        param_mapping=param_mapping,
        tracking_info=tracking_info # 传递追踪器
    )

    # --- 结束和保存 ---
    # MODIFICATION 5: 使用 tracking_info 中的信息来保存最终结果
    print("\n=======================================================")
    print("===      HIERARCHICAL OPTIMIZATION COMPLETED      ===")
    print("=======================================================")
    print("Final best parameters found (from overall optimization):")
    
    print("\n========== Best Evaluation Result (from Iter {}) ==========".format(tracking_info['best_iter_num']))
    for key, value in tracking_info['best_metrics'].items():
        print(f"====== {key:<12} : {value}")

    output_dir = f"{platform.output_path}"
    os.makedirs(output_dir, exist_ok=True)
    final_result_file = f"{output_dir}/gd_v4_final_solution.txt"
    with open(final_result_file, 'w') as f:
        f.write(f"Optimization Status: Finished Hierarchical Optimization.\n")
        f.write(f"Found at Iteration: {tracking_info['best_iter_num']}\n")
        f.write(f"Final Score: {tracking_info['best_score']:.4f}\n\n")
        f.write("Optimized Parameters:\n")
        
        final_best_params = tracking_info['best_params']
        all_optimizable_names = m_param_names + fw_l_param_names
        for name in sorted(all_optimizable_names):
            if name in final_best_params:
                param_obj = final_best_params[name]
                formatted_value = param_obj.format_value(param_obj.value)
                f.write(f"  {name}: {formatted_value}\n")

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
        cv = None # \u786e\u4fdd\u5728 try \u5916\u53ef\u4ee5\u8bbf\u95ee
        try:
            cv = ae.dbOpenCV(args.ae_lib, args.ae_cell, args.ae_view)
            if cv is None:
                print("Error: Failed to open design view for analysis.")
                exit(1)
            
            circuit_graph = build_graph_from_eda(cv, args.param_file)
            analyze_circuit_constraints(circuit_graph)
            
            # \u6253\u5370\u627e\u5230\u7684\u5bf9\u79f0\u7ec4\uff0c\u4fbf\u4e8e\u8c03\u8bd5
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
        # --- \u5206\u6790\u7ed3\u675f ---

        initial_parameters = read_parameters(args.param_file)
        if not initial_parameters:
            print("Error: Failed to read initial parameters for GD. Exiting.")
            exit(1)
        
        # \u5c06\u5206\u6790\u7ed3\u679c (circuit_graph) \u4f20\u9012\u7ed9\u4f18\u5316\u51fd\u6570
        run_gradient_descent(platform, initial_parameters, circuit_graph, # <--- \u65b0\u589e circuit_graph \u53c2\u6570
                            max_iterations=20,
                            perturb_ratio=0.05,
                            greedy_threshold_pct=args.greedy_alpha)

    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --fw_scan.")
        print("Example: python optimization.py --fw_scan [other_args...]")

    print("\nScript finished.")
