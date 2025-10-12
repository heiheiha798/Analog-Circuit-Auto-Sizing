# 基于V1.0.1 加上 V5.0的梯度下降改进

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

def finalize_and_save_results(tracking_info, platform, simulation_count, frozen_params, m_param_names, other_param_names, filename="dynamic_v1.0.1_final_solution.txt"):
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
		f.write(f"Optimization Status: Finished Dynamic Two-Stage Optimization (V1.0.1).\n")
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

def run_dynamic_optimization_v1_0_1(platform: SimulatePlatform, initial_parameters: dict,
                                  circuit_graph: CircuitGraph,
                                  perturb_ratio: float = 0.10,
                                  greedy_threshold_pct: float = 5.0,
                                  max_sim_count: int = 1000): 
    """
    执行一个两阶段的动态收敛优化算法 (V1.0.1)。
    - 阶段一: 优化 'm' 参数，使用参数冻结机制直至所有 'm' 收敛。
    - 阶段二: 优化 'fw/l/r/c' 参数，不冻结，直至达到局部最优。
    - 整个过程没有固定的迭代次数，但有一个总仿真次数上限作为保护。
    - 始终追踪并输出全局最优解。
    - 增加惯性机制，避免频繁的方向切换。
    """
    print("\n===================================================================")
    print("===   DYNAMIC TWO-STAGE OPTIMIZATION (V1.0.1)               ===")
    print("===================================================================")

    # --- 目标函数 get_score (保持不变) ---
    platform.only_set_params(initial_parameters)
    baseline_scores = platform.evaluate()
    if not baseline_scores:
        print("FATAL: Baseline simulation failed."); return

    initial_ugb_val = baseline_scores.get('UGB', 1.0) # 如果获取失败，用1.0作为默认值，但通常不会发生
    initial_area_val = baseline_scores.get('Total_Area', 1.0) # 如果获取失败，用1.0作为默认值
    if initial_ugb_val == 0:
        print("WARNING: Initial UGB is 0")
    if initial_area_val == 0:
        print("WARNING: Initial Total_Area is 0")

    # --- DEBUG: 打印初始参数及is_dummy状态 ---
    print("\n--- DEBUG: Initial Parameters Read ---")
    for name, param in initial_parameters.items():
        print(f"  - Param: {name}, Value: {param.value}, is_dummy: {getattr(param, 'is_dummy', 'N/A')}")
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
    # --- DEBUG: 打印参数映射结果 ---
    print("\n--- DEBUG: Parameter Mapping ---")
    for k, v in param_mapping.items():
        print(f"  {k}: {v}")
    # --- 参数分类和状态初始化 ---
    m_param_names = {name for name in param_mapping.keys() if name.endswith('_m')}
    other_param_names = {name for name in param_mapping.keys() if not name.endswith('_m')}
    # --- DEBUG: 打印参数分类结果 ---
    print("\n--- DEBUG: Parameter Categorization ---")
    print(f"M-Params ({len(m_param_names)}): {sorted(list(m_param_names))}")
    print(f"Other-Params ({len(other_param_names)}): {sorted(list(other_param_names))}")
    frozen_params = set()
    last_success_direction = {name: None for name in param_mapping.keys()}  # 记录上次成功的方向

    simulation_count = 1  # Baseline simulation is the first one

    # --- 全局最优解追踪器 ---
    tracking_info = {
        'best_params': copy.deepcopy(initial_parameters),
        'best_score': get_score(baseline_scores, initial_ugb_val, initial_area_val),
        'best_sim_num': 1,
        'best_metrics': baseline_scores
    }

    X_current_params = copy.deepcopy(initial_parameters)

    # ======================== 阶段一: 'm' 参数优化 (带参数冻结) ========================
    print(f"\n{'#'*25} Starting Stage 1: 'm' Parameter Tuning {'#'*25}")
    stage1_iter_count = 0
    # 新增：阶段一仿真预算（到达后退出到阶段二，而非整体停止）
    stage1_sim_budget = 200
    stage1_budget_exhausted = False
    while True:
        stage1_iter_count += 1
        active_m_params = list(m_param_names - frozen_params)

        # 如果阶段一预算已经耗尽，跳出到阶段二
        if stage1_budget_exhausted:
            print(f"\n--- Stage 1 exiting early: stage1_sim_budget ({stage1_sim_budget}) reached. Moving to Stage 2. ---")
            break

        if not active_m_params:
            print(f"\n--- Stage 1 CONVERGENCE: All 'm' parameters have been frozen. ---")
            break

        print(f"\n--- Stage 1 Iteration {stage1_iter_count} (Active 'm' params: {len(active_m_params)}) ---")
        
        # 替换重复的 set+eval+count 模式
        eval_results, simulation_count = evaluate_and_count(platform, X_current_params, simulation_count)
        # 检查阶段一预算
        if simulation_count >= stage1_sim_budget:
            print(f"  - Stage1 simulation budget reached ({simulation_count} / {stage1_sim_budget}). Exiting Stage 1 to Stage 2.")
            stage1_budget_exhausted = True
            break
        score_current = get_score(eval_results, initial_ugb_val, initial_area_val)
        print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")

        found_immediate_jump = False
        shuffled_param_names = random.sample(active_m_params, len(active_m_params))

        for name in shuffled_param_names:
            found_improvement_for_this_param = False
            # 优先尝试上次成功的方向
            directions = [1, -1]
            if last_success_direction[name] is not None:
                directions = [last_success_direction[name]] + [d for d in directions if d != last_success_direction[name]]
            
            for sign in directions:
                original_value = X_current_params[name].value
                if X_current_params[name].type == 'integer' and original_value <= 1 and sign == -1:
                    continue
                params_probe = copy.deepcopy(X_current_params)
                step = 1
                new_value = params_probe[name].clamp(original_value + sign * step)
                if new_value == original_value:
                    continue
                for actual_param_to_update in param_mapping[name]:
                    params_probe[actual_param_to_update].value = new_value
                
                # 使用工具函数运行仿真并计数
                probe_eval_results, simulation_count = evaluate_and_count(platform, params_probe, simulation_count)
                # 检查阶段一预算（在内层也要及时退出）
                if simulation_count >= stage1_sim_budget:
                    print(f"  - Stage1 simulation budget reached during probing ({simulation_count} / {stage1_sim_budget}). Will exit to Stage 2 after this iteration.")
                    stage1_budget_exhausted = True
                score_probe = get_score(probe_eval_results, initial_ugb_val, initial_area_val)
                
                # 用通用更新函数处理全局最优更新
                update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count)

                if score_probe < score_current:
                    found_improvement_for_this_param = True
                
                improvement = score_current - score_probe
                threshold = (score_current - 1e9) * (greedy_threshold_pct / 100.0) if score_current >= 1e9 else 0

                if improvement > threshold:
                    print(f"  >>> GREEDY JUMP on '{name}'! Moving immediately.")
                    X_current_params = params_probe
                    last_success_direction[name] = sign  # 记录成功方向
                    found_immediate_jump = True
                    break
            
            if not found_improvement_for_this_param:
                print(f"  - Parameter '{name}' hit a local optimum. Freezing.")
                frozen_params.add(name)
                last_success_direction[name] = None

            if found_immediate_jump:
                break
        
        if not found_immediate_jump:
            print(f"  - No greedy jump found in this iteration. Re-evaluating with new frozen set.")
        
        # 如果阶段一预算耗尽，则退出到阶段二
        if stage1_budget_exhausted:
            print(f"\n--- Stage 1 halted due to stage1_sim_budget ({stage1_sim_budget}). Proceeding to Stage 2. ---")
            break

    # ======================== 阶段二: 'fw/l/r/c' 参数优化 (V1.0.1 梯度下降) ========================
    print(f"\n{'#'*25} Starting Stage 2: Continuous Parameter Tuning (Gradient-Based) {'#'*25}")
    stage2_iter_count = 0
    perturb_ratio = 0.1  # 对于fw/l/r/c参数使用10%的相对步长
    
    # 为stage2引入独立的参数冻结机制
    frozen_params_stage2 = set()
    stop_freezing = False # 冻结停止标志

    while simulation_count < max_sim_count:
        stage2_iter_count += 1
        active_other_params = list(other_param_names - frozen_params_stage2)

        if not active_other_params:
            print(f"\n--- Stage 2 CONVERGENCE: All 'other' parameters have been frozen. ---")
            break

        print(f"\n--- Stage 2 Iteration {stage2_iter_count} (Active 'other' params: {len(active_other_params)}, Stop Freezing: {stop_freezing}) ---")
        
        # 1. 获取当前基准点
        eval_results, simulation_count = evaluate_and_count(platform, X_current_params, simulation_count)
        score_current = get_score(eval_results, initial_ugb_val, initial_area_val)
        print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")

        # 2. 全参数扫描
        param_improvements = {}
        shuffled_param_names = random.sample(active_other_params, len(active_other_params))
        
        print(f"  - Scanning {len(shuffled_param_names)} parameters...")
        for name in shuffled_param_names:
            original_value = X_current_params[name].value
            step = original_value * perturb_ratio
            # 记录每个方向的改善量
            improvements_by_sign = {1: -1e9, -1: -1e9}
            best_improvement_for_param = -1e9
            best_direction_for_param = 0

            for sign in [1, -1]:
                params_probe = copy.deepcopy(X_current_params)
                new_value = params_probe[name].clamp(original_value + sign * step)
                
                if abs(new_value - original_value) < 1e-12:
                    improvements_by_sign[sign] = -1e9
                    continue

                for actual_param_to_update in param_mapping[name]:
                    params_probe[actual_param_to_update].value = new_value

                probe_eval_results, simulation_count = evaluate_and_count(platform, params_probe, simulation_count)
                score_probe = get_score(probe_eval_results, initial_ugb_val, initial_area_val)
                update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count)

                improvement = score_current - score_probe # improvement > 0 表示得分变好
                improvements_by_sign[sign] = improvement

                if improvement > best_improvement_for_param:
                    best_improvement_for_param = improvement
                    best_direction_for_param = sign
            
            param_improvements[name] = {
                'best_improvement': best_improvement_for_param,
                'direction': best_direction_for_param,
                'imps': improvements_by_sign
            }

        # 3. 参数冻结决策
        if not stop_freezing:
            params_to_freeze = set()
            
            # 筛选出有提升和无提升的参数（基于best_improvement）
            improving_params = {p: v for p, v in param_improvements.items() if v['best_improvement'] > 0}
            non_improving_params = {p: v for p, v in param_improvements.items() if v['best_improvement'] <= 0}
            # 新增：同时在正负两个方向都没有改善的参数集合
            non_improving_both = {p for p, v in param_improvements.items() if v['imps'].get(1, -1e9) <= 0 and v['imps'].get(-1, -1e9) <= 0}

            print(f"  - Analysis: {len(improving_params)} params show improvement, {len(non_improving_params)} do not. ({len(non_improving_both)} have no improvement in both directions)")

            # 特殊冻结规则：如果无改善参数超过50%，冻结所有无改善参数
            if len(non_improving_params) > 0.5 * len(active_other_params):
                print(f"  - Freeze Rule Triggered: >50% params are non-improving. Freezing all {len(non_improving_params)} of them.")
                params_to_freeze.update(non_improving_params.keys())
            else:
                # 当活动参数较少（<20）时，只冻结正负方向都没有改善的参数
                if len(active_other_params) < 20:
                    if non_improving_both:
                        print(f"  - Small-parameter mode (<20): Freezing only {len(non_improving_both)} params with no improvement in both directions.")
                        params_to_freeze.update(non_improving_both)
                else:
                    # 条件1: 冻结所有无效参数
                    params_to_freeze.update(non_improving_params.keys())
                    
                    # 条件2: 冻结得分变好程度后50%的参数（保持原有行为）
                    if improving_params:
                        sorted_improving_params = sorted(improving_params.items(), key=lambda item: item[1]['best_improvement'])
                        num_to_freeze_from_improving = len(sorted_improving_params) // 2
                        
                        for i in range(num_to_freeze_from_improving):
                            params_to_freeze.add(sorted_improving_params[i][0])
                        print(f"  - Freeze Rule: Freezing {len(non_improving_params)} non-improving and {num_to_freeze_from_improving} bottom-50% improving params.")

            if params_to_freeze:
                print(f"  - Freezing {len(params_to_freeze)} parameters: {sorted(list(params_to_freeze))}")
                frozen_params_stage2.update(params_to_freeze)
            
            # 检查是否停止冻结（保持原判据）
            if len(active_other_params) - len(params_to_freeze) < 20:
                print("\n  --- Active parameters below 20. Halting freeze mechanism for subsequent iterations. ---")
                stop_freezing = True
        
        # 4. 多维度移动
        params_for_gradient_move = {p: v for p, v in param_improvements.items() if p not in frozen_params_stage2 and v['best_improvement'] > 0}
        
        if not params_for_gradient_move:
            print(f"\n--- Stage 2 CONVERGENCE: No parameters left for positive gradient move. ---")
            break

        print(f"  - Performing gradient move with {len(params_for_gradient_move)} parameters.")
        X_gradient_params = copy.deepcopy(X_current_params)
        for name, data in params_for_gradient_move.items():
            original_value = X_gradient_params[name].value
            step = original_value * perturb_ratio
            sign = data['direction']
            new_value = X_gradient_params[name].clamp(original_value + sign * step)
            
            for actual_param_to_update in param_mapping[name]:
                X_gradient_params[actual_param_to_update].value = new_value
        
        # 应用移动
        X_current_params = X_gradient_params
        
        # 评估移动后的点，为下一次迭代做准备
        eval_results, simulation_count = evaluate_and_count(platform, X_current_params, simulation_count)
        score_after_move = get_score(eval_results, initial_ugb_val, initial_area_val)
        print(f"  - Score after gradient move: {score_after_move:.4f} (Improvement: {score_current - score_after_move:.4f})")
        update_tracking_if_better(tracking_info, score_after_move, X_current_params, eval_results, simulation_count)

        if simulation_count >= max_sim_count:
            print(f"\n--- HALTING: Maximum simulation count ({max_sim_count}) reached during Stage 2. ---")
            break

    # ======================== 结束和保存 ========================
    # 将输出与保存步骤委托给独立函数，保持行为不变
    finalize_and_save_results(
        tracking_info=tracking_info,
        platform=platform,
        simulation_count=simulation_count,
        frozen_params=frozen_params,
        m_param_names=m_param_names,
        other_param_names=other_param_names,
        filename="dynamic_v1.0.1_final_solution.txt"
    )

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
        
        run_dynamic_optimization_v1_0_1(platform, initial_parameters, circuit_graph,
                                    perturb_ratio=0.10,
                                    greedy_threshold_pct=args.greedy_alpha,
                                    max_sim_count = 1000)

    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --run_gd.")
        print("Example: python optimization.py --run_gd [other_args...]")

    print("\nScript finished.")
