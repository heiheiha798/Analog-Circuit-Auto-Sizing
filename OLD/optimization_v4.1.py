# V4.1 全参数空间搜索，参数冻结+惯性搜索

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
# =========== 新增辅助函数: 检查硬性指标不敏感性 =================
# ====================================================================
def _check_metric_insensitivity(baseline_scores: dict, probed_scores: dict, threshold_pct: float = 1.0) -> bool:
    """
    比较两次仿真结果的硬性指标，判断参数是否不敏感。

    参数:
        baseline_scores (dict): 基准仿真结果。
        probed_scores (dict): 微扰后的仿真结果。
        threshold_pct (float): 判断为不敏感的相对变化阈值 (百分比)。

    返回:
        bool: 如果所有硬性指标变化都小于阈值，则返回 True (不敏感)。
    """
    # 如果微扰后仿真失败，则认为其影响是巨大的 (敏感的)
    if not probed_scores:
        return False

    hard_metrics = ['Phase_Margin', 'Gain_db', 'Gain_Margin', 'I_OPA']
    threshold = threshold_pct / 100.0

    for metric in hard_metrics:
        base_val = baseline_scores.get(metric)
        probe_val = probed_scores.get(metric)
        
        if base_val is None or probe_val is None:
            if base_val is not probe_val:
                return False
            continue
        
        if abs(base_val) < 1e-9: # 避免除以零
            if abs(probe_val) > 1e-9:
                return False
        else:
            relative_change = abs((probe_val - base_val) / base_val)
            if relative_change > threshold:
                return True

    return False

# ====================================================================
# =========== 核心算法: 带变量衰减和惯性的统一优化 V2 ============
# ====================================================================
def run_optimization_with_decay(platform: SimulatePlatform, initial_parameters: dict,
                                circuit_graph: CircuitGraph,
                                max_iterations: int = 20,
                                perturb_ratio: float = 0.05,
                                greedy_threshold_pct: float = 5.0,
                                insensitivity_threshold_pct: float = 1.0):
    """
    执行一个带参数变量衰减和惯性机制的统一优化算法。
    """
    print("\n===================================================================")
    print("===   UNIFIED OPTIMIZATION WITH VARIABLE DECAY & INERTIA   ===")
    print("===================================================================")

    # 目标函数 get_score (保持不变)
    platform.only_set_params(initial_parameters)
    baseline_scores = platform.evaluate()
    if not baseline_scores:
        print("FATAL: Baseline simulation failed."); return
    baseline_ugb = baseline_scores.get('UGB', 1.0)
    baseline_area = baseline_scores.get('Total_Area', 1.0)

    def get_score(scores: dict):
        # ... (这个内部函数保持不变)
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

    # --- 构建对称参数映射 (逻辑不变) ---
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
    
    all_optimizable_names = list(param_mapping.keys())
    frozen_params = set() 
    
    # ======================== 阶段一: 初始不敏感性扫描 (逻辑不变) ========================
    print(f"\n{'#'*25} Starting Initial Insensitivity Scan {'#'*25}")
    current_params = copy.deepcopy(initial_parameters)
    platform.only_set_params(current_params)
    scan_baseline_scores = platform.evaluate()
    if scan_baseline_scores:
        for name in all_optimizable_names:
            params_probe = copy.deepcopy(current_params)
            original_value = params_probe[name].value
            delta = 1 if params_probe[name].type == 'integer' else original_value * perturb_ratio
            new_value = original_value + delta
            for actual_param_to_update in param_mapping[name]:
                params_probe[actual_param_to_update].value = new_value
            platform.only_set_params(params_probe)
            probed_scores = platform.evaluate()
            is_sensitive = _check_metric_insensitivity(scan_baseline_scores, probed_scores, insensitivity_threshold_pct)
            if not is_sensitive:
                print(f"  - Parameter '{name}' is insensitive. Freezing.")
                frozen_params.add(name)
    print(f"Scan complete. {len(frozen_params)} parameters frozen.")


    # ======================== 阶段二: 主优化循环 (带衰减和惯性) ========================
    print(f"\n{'#'*25} Starting Main Optimization Loop {'#'*25}")
    X_current_params = copy.deepcopy(initial_parameters)
    last_successful_move = None  # <<< NEW: Initialize inertia tracker

    for i in range(max_iterations):
        active_param_names = [p for p in all_optimizable_names if p not in frozen_params]
        if not active_param_names:
            print("\n--- CONVERGENCE: All parameters have been frozen. ---")
            break
        
        print(f"\n--- Main Loop Iteration {i+1}/{max_iterations} (Active Params: {len(active_param_names)}) ---")
        platform.only_set_params(X_current_params)
        score_current = get_score(platform.evaluate())
        print(f"  - Current score: {score_current:.4f}")

        # <<< NEW: Inertia Probe Block >>>
        if last_successful_move and last_successful_move['name'] not in frozen_params:
            print(f"  - Attempting inertia move: {last_successful_move['name']} ({'+' if last_successful_move['sign'] > 0 else '-'})")
            params_probe = copy.deepcopy(X_current_params)
            name, sign = last_successful_move['name'], last_successful_move['sign']
            original_value = params_probe[name].value
            delta = 1 if params_probe[name].type == 'integer' else original_value * perturb_ratio
            new_value = original_value + sign * delta
            
            for actual_param_to_update in param_mapping[name]:
                params_probe[actual_param_to_update].value = new_value
            
            platform.only_set_params(params_probe)
            score_probe = get_score(platform.evaluate())

            if score_probe < score_current:
                print(f"  >>> INERTIA SUCCESS! Score improved to {score_probe:.4f}. Continuing with momentum.")
                X_current_params = params_probe
                # We keep last_successful_move as is, to try it again next iteration
                continue # Skip the random search for this iteration
            else:
                print(f"  - Inertia move failed. Score: {score_probe:.4f}. Resetting momentum.")
                last_successful_move = None # Reset inertia if it fails

        found_immediate_jump = False
        # <<< MODIFIED: Expanded best_neighbor_so_far to track the move >>>
        best_neighbor_so_far = {'params': None, 'score': score_current, 'name': None, 'sign': None}
        shuffled_param_names = random.sample(active_param_names, len(active_param_names))

        for name in shuffled_param_names:
            original_value = X_current_params[name].value
            param_type = X_current_params[name].type
            delta = 1 if param_type == 'integer' else original_value * perturb_ratio

            found_improvement_for_this_param = False
            
            for sign in [1, -1]:
                if param_type == 'integer' and original_value <= 1 and sign == -1: continue

                params_probe = copy.deepcopy(X_current_params)
                new_value = original_value + sign * delta
                
                for actual_param_to_update in param_mapping[name]:
                    params_probe[actual_param_to_update].value = new_value

                platform.only_set_params(params_probe)
                score_probe = get_score(platform.evaluate())
                print(f"      - Probing {name} ({'+' if sign > 0 else '-'}{'1' if param_type == 'integer' else f'{perturb_ratio*100}%'})... Score: {score_probe:.4f}")
                
                if score_probe < best_neighbor_so_far['score']:
                    # <<< MODIFIED: Store the move that led to the best score >>>
                    best_neighbor_so_far = {'params': params_probe, 'score': score_probe, 'name': name, 'sign': sign}
                
                if score_probe < score_current:
                    found_improvement_for_this_param = True

                improvement = score_current - score_probe
                threshold = 0
                if score_current >= 1e9:
                    penalty_part = score_current - 1e9
                    threshold = penalty_part * (greedy_threshold_pct / 100.0)

                if improvement > threshold:
                    print(f"  >>> GREEDY JUMP! Found significant improvement. Moving immediately.")
                    X_current_params = params_probe
                    found_immediate_jump = True
                    last_successful_move = {'name': name, 'sign': sign} # <<< NEW: Record move for inertia
                    break
            
            if found_immediate_jump:
                break

            if not found_improvement_for_this_param:
                print(f"  - Parameter '{name}' hit a local optimum. Freezing.")
                frozen_params.add(name)

        if found_immediate_jump:
            continue

        if best_neighbor_so_far['score'] < score_current:
            print(f"  - Found a modest improvement. Moving to best neighbor.")
            X_current_params = best_neighbor_so_far['params']
            # <<< NEW: Record the best move of the iteration for inertia >>>
            last_successful_move = {'name': best_neighbor_so_far['name'], 'sign': best_neighbor_so_far['sign']}
        else:
            print(f"\n--- CONVERGENCE in Main Loop: No further improvement found. ---")
            last_successful_move = None # <<< NEW: Reset inertia on convergence
            break
            
    print("\n=======================================================")
    print("===      UNIFIED OPTIMIZATION COMPLETED           ===")
    print("=======================================================")
    final_params = X_current_params
    print("Final best parameters found:")
    platform.only_set_params(final_params)
    final_score = get_score(platform.evaluate())
    
    output_dir = f"{platform.output_path}"
    os.makedirs(output_dir, exist_ok=True)
    final_result_file = f"{output_dir}/unified_decay_inertia_final_solution.txt"
    with open(final_result_file, 'w') as f:
        f.write(f"Optimization Status: Finished Unified Optimization with Decay & Inertia.\n")
        f.write(f"Final Score: {final_score:.4f}\n")
        f.write(f"Frozen parameters ({len(frozen_params)}): {sorted(list(frozen_params))}\n\n")
        f.write("Optimized Parameters:\n")
        for name in sorted(all_optimizable_names):
            param_obj = final_params[name]
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

        run_optimization_with_decay(platform, initial_parameters, circuit_graph,
                                    max_iterations=20, # 你可以调整这些值
                                    perturb_ratio=0.05,
                                    greedy_threshold_pct=args.greedy_alpha,
                                    insensitivity_threshold_pct=1.0) # 新增不敏感性阈值

    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --fw_scan.")
        print("Example: python optimization.py --fw_scan [other_args...]")

    print("\nScript finished.")

