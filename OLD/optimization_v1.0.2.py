# 基于V1.0.0 继续开发 V1.0.2

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

# Note: skopt (scikit-optimize) imports removed — not used in current file
# If you later add Bayesian optimization (gp_minimize / Real / Integer / use_named_args), re-enable these imports
#from skopt import gp_minimize
#from skopt.space import Real, Integer
#from skopt.utils import use_named_args

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

def get_score(scores, initial_ugb_val, initial_area_val):
	"""模块级评分函数，等价于原内部 get_score。"""
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

def evaluate_and_count(platform, params, simulation_count):
	"""设置参数并运行一次仿真，返回 (results, new_simulation_count)。"""
	platform.only_set_params(params)
	results = platform.evaluate()
	return results, simulation_count + 1

def update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count):
	"""如果 probe 优于全局最优则更新 tracking_info 并打印提示。"""
	if score_probe < tracking_info['best_score']:
		print(f"  *** New overall best found! Score: {score_probe:.4f}, Sim: {simulation_count} ***")
		tracking_info.update({
			'best_score': score_probe,
			'best_params': copy.deepcopy(params_probe),
			'best_sim_num': simulation_count,
			'best_metrics': probe_eval_results
		})

def finalize_and_save_results(tracking_info, platform, simulation_count, frozen_params, m_param_names, other_param_names, filename="dynamic_v1.0.2_final_solution.txt"):
	"""
	独立的结束与保存逻辑。接受追踪信息、平台和统计变量，打印最终信息并写文件。
	保持与原脚本等价的输出格式。
	"""
	print("\n=======================================================")
	print("===      DYNAMIC OPTIMIZATION COMPLETED             ===")
	print("=======================================================")
	print(f"Total simulations performed: {simulation_count}")
	print("Final best parameters found (from overall optimization):")
	
	print("\n========== Best Evaluation Result (from Sim #{}) ==========".format(tracking_info['best_sim_num']))
	for key, value in tracking_info['best_metrics'].items():
		print(f"====== {key:<12} : {value}")

	output_dir = f"{platform.output_path}"
	os.makedirs(output_dir, exist_ok=True)
	final_result_file = f"{output_dir}/{filename}"
	with open(final_result_file, 'w') as f:
		f.write(f"Optimization Status: Finished Dynamic Two-Stage Optimization (V1.0.2).\n")
		f.write(f"Total Simulations: {simulation_count}\n")
		f.write(f"Best Solution Found at Simulation: {tracking_info['best_sim_num']}\n")
		f.write(f"Final Score: {tracking_info['best_score']:.4f}\n\n")
		f.write(f"Frozen 'm' parameters ({len(frozen_params)}): {sorted(list(frozen_params))}\n\n")
		f.write("Optimized Parameters (Best Found):\n")
		
		final_best_params = tracking_info['best_params']
		all_optimizable_names = list(m_param_names) + list(other_param_names)
		for name in sorted(all_optimizable_names):
			if name in final_best_params:
				param_obj = final_best_params[name]
				formatted_value = param_obj.format_value(param_obj.value)
				status = " (Frozen)" if name in frozen_params else ""
				f.write(f"  {name}: {formatted_value}{status}\n")

def run_dynamic_optimization_v1_0_2(platform: SimulatePlatform, initial_parameters: dict,
                                  circuit_graph: CircuitGraph,
                                  perturb_ratio: float = 0.10,
                                  greedy_threshold_pct: float = 5.0,
                                  beta_threshold_pct: float = 1.0,  # 新增: Stage1 退出阈值
                                  max_sim_count: int = 400): 
    """
    执行一个三阶段的动态收敛优化算法 (V1.0.2)。
    - Stage 1: 优化 'm' 参数，使用冻结和新增的beta阈值提前退出机制。
    - Stage 2: 优化连续参数以满足硬性约束。
    - Stage 3: 在满足约束后，专门优化UGB/Area，使用领域知识启发式搜索。
    """
    print("\n===================================================================")
    print("===   DYNAMIC THREE-STAGE OPTIMIZATION (V1.0.2)             ===")
    print("===================================================================")

    # --- 初始化 (与之前版本相同) ---
    platform.only_set_params(initial_parameters)
    baseline_scores = platform.evaluate()
    if not baseline_scores:
        print("FATAL: Baseline simulation failed."); return

    initial_ugb_val = baseline_scores.get('UGB', 1.0)
    initial_area_val = baseline_scores.get('Total_Area', 1.0)
    if initial_ugb_val == 0: print("WARNING: Initial UGB is 0")
    if initial_area_val == 0: print("WARNING: Initial Total_Area is 0")

    param_mapping = {}
    devices_in_groups = set()
    for group in circuit_graph.constraint_groups:
        representative_device = group.devices[0]
        for param_name_full, param_obj in initial_parameters.items():
            if param_name_full.startswith(representative_device.name + '_'):
                param_suffix = param_name_full[len(representative_device.name):]
                key = f"{representative_device.name}{param_suffix}"
                param_mapping[key] = [f"{dev.name}{param_suffix}" for dev in group.devices]
        for dev in group.devices: devices_in_groups.add(dev.name)
    for name, param in initial_parameters.items():
        device_name = name.split('_')[0]
        if device_name not in devices_in_groups and not param.is_dummy:
            if name not in param_mapping: param_mapping[name] = [name]

    # --- 新增: 为Stage 3定义特定参数 ---
    m_param_names = {name for name in param_mapping.keys() if name.endswith('_m')}
    fine_tuning_suffixes = {'_l', '_segW', '_segL'} # 注意这里用后缀匹配
    fine_tuning_params = {name for name in param_mapping.keys() if any(name.endswith(s) for s in fine_tuning_suffixes)}
    other_param_names = {name for name in param_mapping.keys() if name not in m_param_names and name not in fine_tuning_params}

    print("\n--- DEBUG: Parameter Categorization ---")
    print(f"M-Params (Stage 1, {len(m_param_names)}): {sorted(list(m_param_names))}")
    print(f"Other-Params (Stage 2, {len(other_param_names)}): {sorted(list(other_param_names))}")
    print(f"Fine-Tune-Params (Stage 3, {len(fine_tuning_params)}): {sorted(list(fine_tuning_params))}")

    frozen_params_s1 = set()
    frozen_params_s2 = set()
    frozen_params_s3 = set()
    last_success_direction = {name: None for name in param_mapping.keys()}

    simulation_count = 1
    tracking_info = {
        'best_params': copy.deepcopy(initial_parameters),
        'best_score': get_score(baseline_scores, initial_ugb_val, initial_area_val),
        'best_sim_num': 1,
        'best_metrics': baseline_scores
    }
    X_current_params = copy.deepcopy(initial_parameters)

    # --- 主循环状态机 ---
    current_stage = 1
    while simulation_count < max_sim_count:
        
        # ======================== STAGE 1: 'm' Parameter Tuning ========================
        if current_stage == 1:
            print(f"\n{'#'*25} Entering Stage 1: 'm' Parameter Tuning {'#'*25}")
            stage1_iter_count = 0
            while simulation_count < max_sim_count:
                stage1_iter_count += 1
                active_m_params = list(m_param_names - frozen_params_s1)
                if not active_m_params:
                    print(f"\n--- Stage 1 CONVERGENCE: All 'm' parameters frozen. ---")
                    break

                print(f"\n--- Stage 1 Iteration {stage1_iter_count} (Active 'm' params: {len(active_m_params)}) ---")
                
                eval_results, simulation_count = evaluate_and_count(platform, X_current_params, simulation_count)
                score_current = get_score(eval_results, initial_ugb_val, initial_area_val)
                print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")

                found_immediate_jump = False
                max_improvement_in_iteration = 0.0
                shuffled_param_names = random.sample(active_m_params, len(active_m_params))

                for name in shuffled_param_names:
                    found_improvement_for_this_param = False
                    directions = [1, -1]
                    if last_success_direction[name] is not None:
                        directions = [last_success_direction[name]] + [d for d in directions if d != last_success_direction[name]]
                    
                    for sign in directions:
                        original_value = X_current_params[name].value
                        if X_current_params[name].type == 'integer' and original_value <= 1 and sign == -1: continue
                        
                        params_probe = copy.deepcopy(X_current_params)
                        new_value = params_probe[name].clamp(original_value + sign * 1)
                        if new_value == original_value: continue

                        for p_update in param_mapping[name]: params_probe[p_update].value = new_value
                        
                        probe_eval_results, simulation_count = evaluate_and_count(platform, params_probe, simulation_count)
                        score_probe = get_score(probe_eval_results, initial_ugb_val, initial_area_val)
                        update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count)

                        improvement = score_current - score_probe
                        max_improvement_in_iteration = max(max_improvement_in_iteration, improvement)

                        if score_probe < score_current: found_improvement_for_this_param = True
                        
                        threshold = (score_current - 1e9) * (greedy_threshold_pct / 100.0) if score_current >= 1e9 else 0
                        if improvement > threshold:
                            print(f"  >>> GREEDY JUMP on '{name}'! Moving immediately.")
                            X_current_params = params_probe; last_success_direction[name] = sign
                            found_immediate_jump = True; break
                    
                    if not found_improvement_for_this_param:
                        print(f"  - Parameter '{name}' hit a local optimum. Freezing.")
                        frozen_params_s1.add(name); last_success_direction[name] = None
                    if found_immediate_jump: break
                
                if not found_immediate_jump:
                    print(f"  - No greedy jump in this iteration.")
                    # --- 新增: Beta 退出机制 ---
                    relative_improvement = (max_improvement_in_iteration / abs(score_current)) * 100.0 if score_current != 0 else 0
                    print(f"  - Max improvement this iteration: {max_improvement_in_iteration:.4f} ({relative_improvement:.2f}%)")
                    if relative_improvement < beta_threshold_pct:
                        print(f"  - Improvement potential exhausted ({relative_improvement:.2f}% < {beta_threshold_pct}%)")
                        print(f"--- Stage 1 HALTED early due to low improvement potential. ---")
                        break
            
            # --- 阶段转换逻辑 ---
            final_eval_results, _ = evaluate_and_count(platform, X_current_params, simulation_count-1)
            final_score = get_score(final_eval_results, initial_ugb_val, initial_area_val)
            if final_score < 1e9:
                print("\n--- Constraints met. Proceeding to Stage 3 for fine-tuning. ---")
                current_stage = 3
            else:
                print("\n--- Constraints not met. Proceeding to Stage 2 for constraint solving. ---")
                current_stage = 2
            continue # 进入下一次主循环以切换到新阶段

        # ======================== STAGE 2: Constraint Solving ========================
        elif current_stage == 2:
            print(f"\n{'#'*25} Entering Stage 2: Constraint Solving {'#'*25}")
            stage2_iter_count = 0
            while simulation_count < max_sim_count:
                stage2_iter_count += 1
                
                # --- 阶段转换检查 ---
                eval_results, sim_count_eval = evaluate_and_count(platform, X_current_params, simulation_count)
                score_current_check = get_score(eval_results, initial_ugb_val, initial_area_val)
                if score_current_check < 1e9:
                    print(f"\n--- Constraints met in Stage 2 (Iter {stage2_iter_count}). Switching to Stage 3. ---")
                    current_stage = 3; break

                simulation_count = sim_count_eval
                score_current = score_current_check

                active_other_params = list((other_param_names | fine_tuning_params) - frozen_params_s2)
                if not active_other_params:
                    print(f"\n--- Stage 2 CONVERGENCE: All continuous parameters frozen. ---")
                    break
                
                print(f"\n--- Stage 2 Iteration {stage2_iter_count} (Active params: {len(active_other_params)}) ---")
                print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")

                best_neighbor_so_far = {'params': None, 'score': score_current}
                shuffled_param_names = random.sample(active_other_params, len(active_other_params))
                any_improvement_in_iteration = False

                for name in shuffled_param_names:
                    # ... (Stage 2 的爬山法逻辑与V1.0.0完全相同) ...
                    found_improvement_for_this_param = False
                    original_value = X_current_params[name].value; step = original_value * perturb_ratio
                    directions = [1, -1]
                    if last_success_direction[name] is not None:
                        directions = [last_success_direction[name]] + [d for d in directions if d != last_success_direction[name]]
                    
                    for sign in directions:
                        params_probe = copy.deepcopy(X_current_params)
                        new_value = params_probe[name].clamp(original_value + sign * step)
                        if abs(new_value - original_value) < 1e-12: continue
                        for p_update in param_mapping[name]: params_probe[p_update].value = new_value

                        probe_eval_results, simulation_count = evaluate_and_count(platform, params_probe, simulation_count)
                        score_probe = get_score(probe_eval_results, initial_ugb_val, initial_area_val)
                        update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count)

                        if score_probe < best_neighbor_so_far['score']:
                            best_neighbor_so_far = {'params': params_probe, 'score': score_probe}
                            last_success_direction[name] = sign; found_improvement_for_this_param = True
                    
                    if found_improvement_for_this_param: any_improvement_in_iteration = True
                    else:
                        print(f"  - Parameter '{name}' hit a local optimum. Freezing for Stage 2.")
                        frozen_params_s2.add(name)

                if best_neighbor_so_far['score'] < score_current:
                    print(f"  - Found improvement in iteration. Moving to best neighbor with score {best_neighbor_so_far['score']:.4f}")
                    X_current_params = best_neighbor_so_far['params']
                
                if not any_improvement_in_iteration:
                    print(f"\n--- Stage 2 CONVERGENCE: No improvement found in a full iteration. ---")
                    break
            
            if current_stage == 2: # 如果不是因为切换到Stage 3而退出，则优化结束
                print("\n--- Optimization halted after Stage 2. ---")
                break # 退出主 while 循环
            else:
                continue # 进入下一次主循环以切换到 Stage 3
        
        # ======================== STAGE 3: UGB/Area Fine-Tuning ========================
        elif current_stage == 3:
            print(f"\n{'#'*25} Entering Stage 3: UGB/Area Fine-Tuning {'#'*25}")
            stage3_iter_count = 0
            while simulation_count < max_sim_count:
                stage3_iter_count += 1
                
                # --- 阶段转换检查 ---
                eval_results, sim_count_eval = evaluate_and_count(platform, X_current_params, simulation_count)
                score_current_check = get_score(eval_results, initial_ugb_val, initial_area_val)
                if score_current_check >= 1e9:
                    print(f"\n--- Constraints VIOLATED in Stage 3 (Iter {stage3_iter_count}). Switching back to Stage 2. ---")
                    current_stage = 2; break

                simulation_count = sim_count_eval
                score_current = score_current_check

                active_fine_params = list(fine_tuning_params - frozen_params_s3)
                if not active_fine_params:
                    print(f"\n--- Stage 3 CONVERGENCE: All fine-tuning parameters frozen. ---")
                    break
                
                print(f"\n--- Stage 3 Iteration {stage3_iter_count} (Active params: {len(active_fine_params)}) ---")
                print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")

                best_neighbor_so_far = {'params': None, 'score': score_current}
                shuffled_param_names = random.sample(active_fine_params, len(active_fine_params))
                any_improvement_in_iteration = False

                for name in shuffled_param_names:
                    found_improvement_for_this_param = False
                    original_value = X_current_params[name].value; step = original_value * perturb_ratio
                    directions = [1, -1]
                    
                    # --- 新增: 启发式搜索方向 ---
                    # 对于补偿电容的'l'参数，优先尝试减小，用PM换UGB
                    if name.startswith('C') and name.endswith('_l'):
                        print(f"  - Heuristic applied to '{name}': Prioritizing reduction.")
                        directions = [-1, 1]
                    elif last_success_direction[name] is not None:
                        directions = [last_success_direction[name]] + [d for d in directions if d != last_success_direction[name]]

                    for sign in directions:
                        params_probe = copy.deepcopy(X_current_params)
                        new_value = params_probe[name].clamp(original_value + sign * step)
                        if abs(new_value - original_value) < 1e-12: continue
                        for p_update in param_mapping[name]: params_probe[p_update].value = new_value

                        probe_eval_results, simulation_count = evaluate_and_count(platform, params_probe, simulation_count)
                        score_probe = get_score(probe_eval_results, initial_ugb_val, initial_area_val)
                        update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count)
                        
                        # 在Stage 3，即使分数变差（比如PM掉出约束），也要进行比较，以便best_neighbor能记录下来
                        if score_probe < best_neighbor_so_far['score']:
                            best_neighbor_so_far = {'params': params_probe, 'score': score_probe}
                            last_success_direction[name] = sign; found_improvement_for_this_param = True
                    
                    if found_improvement_for_this_param: any_improvement_in_iteration = True
                    else:
                        print(f"  - Parameter '{name}' hit a local optimum. Freezing for Stage 3.")
                        frozen_params_s3.add(name)

                if best_neighbor_so_far['score'] < score_current:
                    print(f"  - Found improvement in iteration. Moving to best neighbor with score {best_neighbor_so_far['score']:.4f}")
                    X_current_params = best_neighbor_so_far['params']
                
                if not any_improvement_in_iteration:
                    print(f"\n--- Stage 3 CONVERGENCE: No improvement found in a full iteration. ---")
                    break
            
            if current_stage == 3: # 如果不是因为切换到Stage 2而退出，则优化结束
                print("\n--- Optimization halted after Stage 3. ---")
                break # 退出主 while 循环
            else:
                continue # 进入下一次主循环以切换到 Stage 2
        
        else:
            print(f"FATAL: Unknown stage {current_stage}. Halting.")
            break

    # ======================== 结束和保存 ========================
    finalize_and_save_results(
        tracking_info=tracking_info,
        platform=platform,
        simulation_count=simulation_count,
        frozen_params=frozen_params_s1, # 只报告Stage 1的冻结
        m_param_names=m_param_names,
        other_param_names=other_param_names | fine_tuning_params, # 合并报告
        filename="dynamic_v1.0.2_final_solution.txt"
    )

if __name__ == "__main__":
    args = parse_arguments()
    # 新增: beta_threshold_pct 的命令行参数
    parser = argparse.ArgumentParser(add_help=False) # 创建一个临时的parser来添加新参数
    parser.add_argument('--beta_pct', type=float, default=1.0, help='Stage 1 early exit threshold in percent (e.g., 1 for 1%%).')
    # 解析已知和未知的参数
    args, unknown = parser.parse_known_args(namespace=args)

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
        cv = None 
        try:
            cv = ae.dbOpenCV(args.ae_lib, args.ae_cell, args.ae_view)
            if cv is None:
                print("Error: Failed to open design view for analysis."); exit(1)
            
            circuit_graph = build_graph_from_eda(cv, args.param_file)
            analyze_circuit_constraints(circuit_graph)
            
            if circuit_graph.constraint_groups:
                print(f"\nSymmetry analysis found {len(circuit_graph.constraint_groups)} groups.")
                for i, group in enumerate(circuit_graph.constraint_groups):
                    device_names = ', '.join([d.name for d in group.devices])
                    print(f"  - Group {i+1}: [{device_names}]")
            else:
                print("\nNo symmetric groups found.")

        except Exception as e:
            print(f"\nAn error occurred during graph construction or analysis: {e}"); exit(1)
        finally:
            if cv: ae.dbCloseCV(cv)

        initial_parameters = read_parameters(args.param_file)
        if not initial_parameters:
            print("Error: Failed to read initial parameters for GD. Exiting."); exit(1)
        
        # 将命令行参数传递给 run 函数
        run_dynamic_optimization_v1_0_2(platform, initial_parameters, circuit_graph,
                                    perturb_ratio=0.10,
                                    greedy_threshold_pct=args.greedy_alpha,
                                    beta_threshold_pct=args.beta_pct, # 传递 beta
                                    max_sim_count=500) # 保持预算为500

    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --run_gd.")
        print("Example: python optimization.py --run_gd [other_args...]")

    print("\nScript finished.")
