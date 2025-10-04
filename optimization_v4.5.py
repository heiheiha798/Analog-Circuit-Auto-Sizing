# V4.5 参数分层搜索 + 惯性 + 分阶段冻结 + 步长调整 + 阈值终止

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

def run_hybrid_optimization_v4_5(platform: SimulatePlatform, initial_parameters: dict,
                                  circuit_graph: CircuitGraph,
                                  max_sim_count: int = 500,
                                  perturb_ratio: float = 0.1,
                                  initial_greedy_alpha_pct: float = 5.0,
                                  min_greedy_alpha_pct: float = 1.0):
    """
    执行一个混合两阶段优化算法 (V4.5)，融合了分层、惯性、冻结、自适应Alpha和阈值终止。
    
    - 核心特性:
      - 分层搜索: 先优化 'm' 参数 (阶段一)，再优化其他连续参数 (阶段二)。
      - 惯性机制: 记录并优先尝试每个参数上次成功的探索方向。
      - 分阶段冻结: 在每个阶段，如果一个参数在所有方向上都无法带来任何改进，则将其冻结。
      - 自适应Alpha: 接受阈值(alpha)会根据上一轮的最佳改进动态调整，变得更灵敏。
      - 阈值终止: 如果一轮迭代中找到的最佳改进低于最小阈值，则认为该阶段收敛。
    - 终止条件:
      - 达到最大仿真次数 (max_sim_count)。
      - 两个阶段都因收敛或参数全部冻结而自然结束。
    """
    print("\n===================================================================")
    print("===   HYBRID TWO-STAGE OPTIMIZATION (V4.5)                  ===")
    print("===   (Hierarchical + Inertia + Freezing + Adaptive Alpha)  ===")
    print("===================================================================")

    # --- 目标函数 get_score (与V4.3版本相同) ---
    platform.only_set_params(initial_parameters)
    baseline_scores = platform.evaluate()
    if not baseline_scores:
        print("FATAL: Baseline simulation failed."); return

    initial_ugb_val = baseline_scores.get('UGB', 1.0)
    initial_area_val = baseline_scores.get('Total_Area', 1.0)

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
            ugb_norm = scores.get('UGB', 0) / initial_ugb_val
            area_norm = scores.get('Total_Area', 0) / initial_area_val
            return 0.5 * area_norm - 0.5 * ugb_norm

    # --- 构建对称参数映射 (与V4.3版本相同) ---
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

    # --- 参数分类和状态初始化 ---
    m_param_names = {name for name in param_mapping.keys() if name.endswith('_m')}
    other_param_names = {name for name in param_mapping.keys() if not name.endswith('_m')}
    
    # 状态变量
    frozen_params_stage1 = set()
    frozen_params_stage2 = set()
    last_success_direction = {name: None for name in param_mapping.keys()}
    current_greedy_alpha_pct = initial_greedy_alpha_pct
    simulation_count = 1

    # --- 全局最优解追踪器 ---
    tracking_info = {
        'best_params': copy.deepcopy(initial_parameters),
        'best_score': get_score(baseline_scores),
        'best_sim_num': 1,
        'best_metrics': baseline_scores
    }
    X_current_params = copy.deepcopy(initial_parameters)

    # ======================== 阶段一: 'm' 参数优化 (带冻结、惯性、自适应Alpha) ========================
    print(f"\n{'#'*25} Starting Stage 1: 'm' Parameter Tuning {'#'*25}")
    stage1_iter_count = 0
    while simulation_count < max_sim_count:
        stage1_iter_count += 1
        active_m_params = list(m_param_names - frozen_params_stage1)
        if not active_m_params:
            print(f"\n--- Stage 1 CONVERGENCE: All 'm' parameters have been frozen. ---")
            break

        print(f"\n--- Stage 1 Iteration {stage1_iter_count} (Alpha: {current_greedy_alpha_pct:.2f}%) ---")
        
        platform.only_set_params(X_current_params)
        eval_results = platform.evaluate(); simulation_count += 1
        score_current = get_score(eval_results)
        print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")

        found_immediate_jump = False
        best_neighbor_so_far = {'params': None, 'score': score_current, 'name': None, 'sign': None}
        shuffled_param_names = random.sample(active_m_params, len(active_m_params))

        for name in shuffled_param_names:
            found_improvement_for_this_param = False
            # 惯性机制：优先尝试上次成功的方向
            directions = [1, -1]
            if last_success_direction[name] is not None:
                directions.insert(0, last_success_direction[name])
                directions = list(dict.fromkeys(directions)) # 移除重复项

            for sign in directions:
                original_value = X_current_params[name].value
                # m参数是整数，步长固定为1
                if X_current_params[name].type == 'integer' and original_value <= 1 and sign == -1: continue

                params_probe = copy.deepcopy(X_current_params)
                new_value = original_value + sign * 1
                for actual_param in param_mapping[name]:
                    params_probe[actual_param].value = new_value
                
                platform.only_set_params(params_probe)
                probe_eval_results = platform.evaluate(); simulation_count += 1
                score_probe = get_score(probe_eval_results)
                
                # 更新全局最优解
                if score_probe < tracking_info['best_score']:
                    print(f"  *** New overall best found! Score: {score_probe:.4f}, Sim: {simulation_count} ***")
                    tracking_info.update({
                        'best_score': score_probe, 'best_params': copy.deepcopy(params_probe),
                        'best_sim_num': simulation_count, 'best_metrics': probe_eval_results
                    })

                # 记录本轮迭代的最佳邻居
                if score_probe < best_neighbor_so_far['score']:
                    best_neighbor_so_far = {'params': params_probe, 'score': score_probe, 'name': name, 'sign': sign}
                
                if score_probe < score_current:
                    found_improvement_for_this_param = True
                else:
                    last_success_direction[name] = None # 失败则重置惯性方向
                
                # 检查是否满足贪婪跳转条件 (使用动态Alpha)
                improvement = score_current - score_probe
                threshold = 0
                if score_current >= 1e9: # 只在惩罚区计算百分比
                    penalty_part = score_current - 1e9
                    threshold = penalty_part * (current_greedy_alpha_pct / 100.0)
                
                if improvement > threshold:
                    print(f"  >>> GREEDY JUMP on '{name}'! Improvement > {current_greedy_alpha_pct:.2f}%.")
                    X_current_params = params_probe
                    last_success_direction[name] = sign
                    found_immediate_jump = True
                    current_greedy_alpha_pct = initial_greedy_alpha_pct # 重置Alpha
                    break
            
            if found_immediate_jump: break
            
            # 分阶段冻结逻辑
            if not found_improvement_for_this_param:
                print(f"  - Parameter '{name}' hit a local optimum. Freezing for Stage 1.")
                frozen_params_stage1.add(name)

        if found_immediate_jump: continue

        # 如果没有贪婪跳转，则检查最佳邻居
        if best_neighbor_so_far['params'] is not None:
            improvement = score_current - best_neighbor_so_far['score']
            improvement_pct = 0
            if score_current >= 1e9 and (score_current - 1e9) > 0:
                improvement_pct = (improvement / (score_current - 1e9)) * 100.0
            
            # 阈值终止逻辑
            if improvement_pct < min_greedy_alpha_pct:
                print(f"\n--- Stage 1 CONVERGENCE: Best improvement ({improvement_pct:.2f}%) is below min alpha ({min_greedy_alpha_pct:.2f}%). ---")
                X_current_params = best_neighbor_so_far['params'] # 做最后一次移动
                break

            print(f"  - No greedy jump. Moving to best neighbor (Improvement: {improvement_pct:.2f}%).")
            X_current_params = best_neighbor_so_far['params']
            last_success_direction[best_neighbor_so_far['name']] = best_neighbor_so_far['sign']
            
            # 自适应Alpha调整
            new_alpha = max(min_greedy_alpha_pct, improvement_pct)
            print(f"  >>> Adjusting alpha from {current_greedy_alpha_pct:.2f}% to {new_alpha:.2f}%")
            current_greedy_alpha_pct = new_alpha
        else:
            print(f"\n--- Stage 1 CONVERGENCE: No improvement found in this iteration. ---")
            break
            
        if simulation_count >= max_sim_count:
            print(f"\n--- HALTING: Max simulations reached during Stage 1. ---")
            break

    # ======================== 阶段二: 'fw/l/r/c' 参数优化 (逻辑与阶段一类似) ========================
    print(f"\n{'#'*25} Starting Stage 2: Continuous Parameter Tuning {'#'*25}")
    stage2_iter_count = 0
    current_greedy_alpha_pct = initial_greedy_alpha_pct # 重置Alpha进入新阶段

    while simulation_count < max_sim_count:
        stage2_iter_count += 1
        active_other_params = list(other_param_names - frozen_params_stage2)
        if not active_other_params:
            print(f"\n--- Stage 2 CONVERGENCE: All continuous parameters have been frozen. ---")
            break

        print(f"\n--- Stage 2 Iteration {stage2_iter_count} (Alpha: {current_greedy_alpha_pct:.2f}%) ---")
        
        platform.only_set_params(X_current_params)
        eval_results = platform.evaluate(); simulation_count += 1
        score_current = get_score(eval_results)
        print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")

        found_immediate_jump = False
        best_neighbor_so_far = {'params': None, 'score': score_current, 'name': None, 'sign': None}
        shuffled_param_names = random.sample(active_other_params, len(active_other_params))

        for name in shuffled_param_names:
            found_improvement_for_this_param = False
            # 惯性机制
            directions = [1, -1]
            if last_success_direction[name] is not None:
                directions.insert(0, last_success_direction[name])
                directions = list(dict.fromkeys(directions))

            for sign in directions:
                params_probe = copy.deepcopy(X_current_params)
                original_value = params_probe[name].value
                step = original_value * perturb_ratio
                new_value = original_value + sign * step
                if new_value <= 0: continue # 确保参数值为正
                
                for actual_param in param_mapping[name]:
                    params_probe[actual_param].value = new_value
                
                platform.only_set_params(params_probe)
                probe_eval_results = platform.evaluate(); simulation_count += 1
                score_probe = get_score(probe_eval_results)
                
                if score_probe < tracking_info['best_score']:
                    print(f"  *** New overall best found! Score: {score_probe:.4f}, Sim: {simulation_count} ***")
                    tracking_info.update({
                        'best_score': score_probe, 'best_params': copy.deepcopy(params_probe),
                        'best_sim_num': simulation_count, 'best_metrics': probe_eval_results
                    })

                if score_probe < best_neighbor_so_far['score']:
                    best_neighbor_so_far = {'params': params_probe, 'score': score_probe, 'name': name, 'sign': sign}
                
                if score_probe < score_current:
                    found_improvement_for_this_param = True
                else:
                    last_success_direction[name] = None
                
                # 检查贪婪跳转
                improvement = score_current - score_probe
                threshold = 0
                if score_current >= 1e9:
                    penalty_part = score_current - 1e9
                    threshold = penalty_part * (current_greedy_alpha_pct / 100.0)
                
                if improvement > threshold:
                    print(f"  >>> GREEDY JUMP on '{name}'! Improvement > {current_greedy_alpha_pct:.2f}%.")
                    X_current_params = params_probe
                    last_success_direction[name] = sign
                    found_immediate_jump = True
                    current_greedy_alpha_pct = initial_greedy_alpha_pct
                    break
            
            if found_immediate_jump: break

            # 分阶段冻结
            if score_current < 1e9 and not found_improvement_for_this_param:
                print(f"  - Parameter '{name}' hit a local optimum (in feasible region). Freezing for Stage 2.")
                frozen_params_stage2.add(name)

        if found_immediate_jump: continue

        # 检查最佳邻居
        if best_neighbor_so_far['params'] is not None:
            improvement = score_current - best_neighbor_so_far['score']
            improvement_pct = 0
            # 在惩罚区计算改进百分比
            if score_current >= 1e9 and (score_current - 1e9) > 0:
                improvement_pct = (improvement / (score_current - 1e9)) * 100.0
            # 在性能区也可以定义百分比，例如基于score的相对变化
            elif score_current > 0:
                 improvement_pct = (improvement / abs(score_current)) * 100.0

            # 阈值终止
            if improvement_pct < min_greedy_alpha_pct and score_current < 1e9: # 通常在性能区更关注阈值终止
                print(f"\n--- Stage 2 CONVERGENCE: Best improvement ({improvement_pct:.2f}%) is below min alpha ({min_greedy_alpha_pct:.2f}%). ---")
                X_current_params = best_neighbor_so_far['params']
                break

            print(f"  - No greedy jump. Moving to best neighbor (Improvement: {improvement_pct:.2f}%).")
            X_current_params = best_neighbor_so_far['params']
            last_success_direction[best_neighbor_so_far['name']] = best_neighbor_so_far['sign']
            
            # 自适应Alpha
            new_alpha = max(min_greedy_alpha_pct, improvement_pct)
            print(f"  >>> Adjusting alpha from {current_greedy_alpha_pct:.2f}% to {new_alpha:.2f}%")
            current_greedy_alpha_pct = new_alpha
            
            perturb_ratio = perturb_ratio * new_alpha / initial_greedy_alpha_pct
            print(f"  >>> Adjusting perturb_ratio to {perturb_ratio:.4f}")
            
        else:
            print(f"\n--- Stage 2 CONVERGENCE: No improvement found in this iteration. ---")
            break
            
        if simulation_count >= max_sim_count:
            print(f"\n--- HALTING: Max simulations reached during Stage 2. ---")
            break

    # ======================== 结束和保存 ========================
    print("\n=======================================================")
    print("===      HYBRID OPTIMIZATION (V4.5) COMPLETED       ===")
    print("=======================================================")
    print(f"Total simulations performed: {simulation_count}")
    print("Final best parameters found (from overall optimization):")
    
    print("\n========== Best Evaluation Result (from Sim #{}) ==========".format(tracking_info['best_sim_num']))
    for key, value in tracking_info['best_metrics'].items():
        print(f"====== {key:<12} : {value}")

    output_dir = f"{platform.output_path}"
    os.makedirs(output_dir, exist_ok=True)
    final_result_file = f"{output_dir}/hybrid_v4.5_final_solution.txt"
    with open(final_result_file, 'w') as f:
        f.write(f"Optimization Status: Finished Hybrid Two-Stage Optimization (V4.5).\n")
        f.write(f"Total Simulations: {simulation_count}\n")
        f.write(f"Best Solution Found at Simulation: {tracking_info['best_sim_num']}\n")
        f.write(f"Final Score: {tracking_info['best_score']:.4f}\n\n")
        f.write(f"Frozen 'm' parameters ({len(frozen_params_stage1)}): {sorted(list(frozen_params_stage1))}\n")
        f.write(f"Frozen other parameters ({len(frozen_params_stage2)}): {sorted(list(frozen_params_stage2))}\n\n")
        
        final_best_params = tracking_info['best_params']
        all_optimizable_names = list(m_param_names) + list(other_param_names)
        for name in sorted(all_optimizable_names):
            if name in final_best_params:
                param_obj = final_best_params[name]
                formatted_value = param_obj.format_value(param_obj.value)
                status = ""
                if name in frozen_params_stage1:
                    status = " (Frozen Stage 1)"
                elif name in frozen_params_stage2:
                    status = " (Frozen Stage 2)"
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
        
        run_hybrid_optimization_v4_5(platform, initial_parameters, circuit_graph,
                                    max_sim_count=500,
                                    perturb_ratio=0.1,
                                    initial_greedy_alpha_pct=args.greedy_alpha,
                                    min_greedy_alpha_pct=1.0)
        
    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --fw_scan.")
        print("Example: python optimization.py --fw_scan [other_args...]")

    print("\nScript finished.")

