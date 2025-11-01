# V 1.0.0 加上手动管理进程实现并行，无bug

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
	platform.calc_area()
	results = platform.evaluate() 
	return results, simulation_count + 1

def update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count, best_params_filepath):
    """如果 probe 优于全局最优则更新 tracking_info 并打印提示。"""
    if score_probe < tracking_info['best_score']:
        print(f"  *** New overall best found! Score: {score_probe:.4f}, Sim: {simulation_count} ***")
        tracking_info.update({
            'best_score': score_probe,
            'best_params': copy.deepcopy(params_probe),
            'best_sim_num': simulation_count,
            'best_metrics': probe_eval_results
        })
        # 每次找到更优解时，调用新函数更新最优参数文件
        write_params_to_file(tracking_info['best_params'], best_params_filepath)

def finalize_and_save_results(tracking_info, platform, simulation_count, frozen_params, m_param_names, other_param_names, filename="dynamic_v1.0.0_final_solution.txt"):
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
		f.write(f"Optimization Status: Finished Dynamic Two-Stage Optimization (V1.0.0).\n")
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

def write_params_to_file(params: dict, output_filepath: str):
    """将 Parameter 对象字典按指定格式写入文件。"""
    try:
        with open(output_filepath, 'w') as f:
            # 按参数名称排序，确保每次输出的文件内容顺序一致
            for name in sorted(params.keys()):
                param = params[name]
                # 使用 Parameter 对象自带的 format() 方法来获取带单位的字符串值
                formatted_value = param.format()
                # 写入格式："parameter","NM17_m","2"
                f.write(f'"parameter","{name}","{formatted_value}"\n')
    except Exception as e:
        print(f"  [Warning] Failed to write best params to {output_filepath}: {e}")

def run_dynamic_optimization_v1_0_0(platform: SimulatePlatform, initial_parameters: dict,
                                  circuit_graph: CircuitGraph,
                                  perturb_ratio: float = 0.10,
                                  greedy_threshold_pct: float = 5.0,
                                  max_sim_count: int = 1000): 
    """
    执行一个两阶段的动态收敛优化算法 (V1.0.0)。
    - 阶段一: 优化 'm' 参数，使用参数冻结机制直至所有 'm' 收敛。
    - 阶段二: 优化 'fw/l/r/c' 参数，不冻结，直至达到局部最优。
    - 整个过程没有固定的迭代次数，但有一个总仿真次数上限作为保护。
    - 始终追踪并输出全局最优解。
    - 增加惯性机制，避免频繁的方向切换。
    """
    print("\n===================================================================")
    print("===   DYNAMIC TWO-STAGE OPTIMIZATION (V1.0.0)               ===")
    print("===================================================================")

    # --- 目标函数 get_score (保持不变) ---
    platform.only_set_params(initial_parameters)
    platform.calc_area()
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
    # print("\n--- DEBUG: Initial Parameters Read ---")
    # for name, param in initial_parameters.items():
    #     print(f"  - Param: {name}, Value: {param.value}, is_dummy: {getattr(param, 'is_dummy', 'N/A')}")
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
    # print("\n--- DEBUG: Parameter Mapping ---")
    # for k, v in param_mapping.items():
    #     print(f"  {k}: {v}")
    # --- 参数分类和状态初始化 ---
    m_param_names = {name for name in param_mapping.keys() if name.endswith('_m')}
    other_param_names = {name for name in param_mapping.keys() if not name.endswith('_m')}
    # --- DEBUG: 打印参数分类结果 ---
    # print("\n--- DEBUG: Parameter Categorization ---")
    # print(f"M-Params ({len(m_param_names)}): {sorted(list(m_param_names))}")
    # print(f"Other-Params ({len(other_param_names)}): {sorted(list(other_param_names))}")
    frozen_params = set()
    last_success_direction = {name: None for name in param_mapping.keys()}  # 记录上次成功的方向

    simulation_count = 1  # Baseline simulation is the first one

    output_dir = platform.output_path
    optimization_log_file = os.path.join(output_dir, "optimization_log.txt")
    best_params_filepath = os.path.join(output_dir, "best_params_so_far.txt")

    # --- 全局最优解追踪器 ---
    tracking_info = {
        'best_params': copy.deepcopy(initial_parameters),
        'best_score': get_score(baseline_scores, initial_ugb_val, initial_area_val),
        'best_sim_num': 1,
        'best_metrics': baseline_scores
    }
    # 初始时就写入一次最优参数
    write_params_to_file(tracking_info['best_params'], best_params_filepath)

    X_current_params = copy.deepcopy(initial_parameters)

    log_f = open(optimization_log_file, 'w', encoding='utf-8')
    log_f.write(f"--- Iteration 1 (Baseline) ---\n")
    log_f.write(f"Score: {tracking_info['best_score']:.4f}\n")
    for key, value in baseline_scores.items():
        log_f.write(f"  {key:<15}: {value}\n")
    log_f.write("\n")
    log_f.flush()

    # ======================== 阶段一: 'm' 参数优化 (带参数冻结) ========================
    print(f"\n{'#'*25} Starting Stage 1: 'm' Parameter Tuning {'#'*25}")
    log_f.write(f"\n{'#'*25} Starting Stage 1: 'm' Parameter Tuning {'#'*25}\n\n")
    log_f.flush()
    stage1_iter_count = 0
    # 新增：阶段一仿真预算（到达后退出到阶段二，而非整体停止）
    stage1_sim_budget = 200
    # stage1_sim_budget = len(m_param_names) * 6
    stage1_budget_exhausted = False
    while True:
        stage1_iter_count += 1
        active_m_params = list(m_param_names - frozen_params)

        # 如果阶段一预算已经耗尽，跳出到阶段二
        if stage1_budget_exhausted:
            message = f"\n--- Stage 1 exiting early: stage1_sim_budget ({stage1_sim_budget}) reached. Moving to Stage 2. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break

        if not active_m_params:
            message = "\n--- Stage 1 CONVERGENCE: All 'm' parameters have been frozen. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break

        print(f"\n--- Stage 1 Iteration {stage1_iter_count} (Active 'm' params: {len(active_m_params)}) ---")
        
        # 替换重复的 set+eval+count 模式
        eval_results, simulation_count = evaluate_and_count(platform, X_current_params, simulation_count)
        # 检查阶段一预算
        if simulation_count >= stage1_sim_budget:
            message = f"  - Stage1 simulation budget reached ({simulation_count} / {stage1_sim_budget}). Exiting Stage 1 to Stage 2."
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            stage1_budget_exhausted = True
            break
        score_current = get_score(eval_results, initial_ugb_val, initial_area_val)
        print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")
        log_f.write(f"--- Stage 1 Iteration {stage1_iter_count} (Current) ---\n")
        log_f.write(f"Score: {score_current:.4f}\n")
        if eval_results:
            for key, value in eval_results.items():
                log_f.write(f"  {key:<15}: {value}\n")
        else:
            log_f.write("  Simulation failed.\n")
        log_f.write("\n")
        log_f.flush()

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
                    message = f"  - Stage1 simulation budget reached during probing ({simulation_count} / {stage1_sim_budget}). Will exit to Stage 2 after this iteration."
                    print(message)
                    log_f.write(message + "\n")
                    log_f.flush()
                    stage1_budget_exhausted = True
                score_probe = get_score(probe_eval_results, initial_ugb_val, initial_area_val)
                
                # 用通用更新函数处理全局最优更新
                update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count, best_params_filepath)

                log_f.write(f"--- Iteration {simulation_count} (Probe) ---\n")
                log_f.write(f"Action: Perturbed '{name}' with sign {sign}, new value ~{new_value}\n")
                log_f.write(f"Value formatted: {params_probe[name].format()}\n")
                log_f.write(f"Score: {score_probe:.4f}\n")
                if probe_eval_results:
                    for key, value in probe_eval_results.items():
                        log_f.write(f"  {key:<15}: {value}\n")
                else:
                    log_f.write("  Simulation failed.\n")
                log_f.write("\n")
                log_f.flush()

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
                message = f"  - Parameter '{name}' hit a local optimum. Freezing."
                print(message)
                log_f.write(message + "\n\n")
                log_f.flush()
                frozen_params.add(name)
                last_success_direction[name] = None

            if found_immediate_jump:
                break
        
        if not found_immediate_jump:
            message = "  - No greedy jump found in this iteration. Re-evaluating with new frozen set."
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
        
        # 如果阶段一预算耗尽，则退出到阶段二
        if stage1_budget_exhausted:
            message = f"\n--- Stage 1 halted due to stage1_sim_budget ({stage1_sim_budget}). Proceeding to Stage 2. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break

    # ======================== 阶段二: 'fw/l/r/c' 参数优化 (带参数冻结) ========================
    print(f"\n{'#'*25} Starting Stage 2: Continuous Parameter Tuning {'#'*25}")
    log_f.write(f"\n{'#'*25} Starting Stage 2: Continuous Parameter Tuning {'#'*25}\n\n")
    log_f.flush()
    stage2_iter_count = 0
    perturb_ratio = 0.1  # 对于fw/l/r/c参数使用10%的相对步长
    
    # 为stage2引入独立的参数冻结机制
    frozen_params_stage2 = set()

    while simulation_count < max_sim_count:
        stage2_iter_count += 1
        active_other_params = list(other_param_names - frozen_params_stage2)

        if not active_other_params:
            message = "\n--- Stage 2 CONVERGENCE: All 'other' parameters have been frozen. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break

        print(f"\n--- Stage 2 Iteration {stage2_iter_count} (Active 'other' params: {len(active_other_params)}) ---")
        
        # 替换重复的 set+eval+count 模式
        eval_results, simulation_count = evaluate_and_count(platform, X_current_params, simulation_count)
        score_current = get_score(eval_results, initial_ugb_val, initial_area_val)
        print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")
        log_f.write(f"--- Stage 2 Iteration {stage2_iter_count} (Current) ---\n")
        log_f.write(f"Score: {score_current:.4f}\n")
        if eval_results:
            for key, value in eval_results.items():
                log_f.write(f"  {key:<15}: {value}\n")
        else:
            log_f.write("  Simulation failed.\n")
        log_f.write("\n")
        log_f.flush()

        best_neighbor_so_far = {'params': None, 'score': score_current}
        shuffled_param_names = random.sample(active_other_params, len(active_other_params))
        
        any_improvement_in_iteration = False

        for name in shuffled_param_names:
            found_improvement_for_this_param = False
            
            original_value = X_current_params[name].value
            step = original_value * perturb_ratio
            
            directions = [1, -1]
            if last_success_direction[name] is not None:
                directions = [last_success_direction[name]] + [d for d in directions if d != last_success_direction[name]]
            
            for sign in directions:
                params_probe = copy.deepcopy(X_current_params)
                new_value = params_probe[name].clamp(original_value + sign * step)
                
                if abs(new_value - original_value) < 1e-12:
                    continue

                for actual_param_to_update in param_mapping[name]:
                    params_probe[actual_param_to_update].value = new_value

                probe_eval_results, simulation_count = evaluate_and_count(platform, params_probe, simulation_count)
                score_probe = get_score(probe_eval_results, initial_ugb_val, initial_area_val)

                update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count, best_params_filepath)

                log_f.write(f"--- Iteration {simulation_count} (Probe) ---\n")
                log_f.write(f"Action: Stage 2 perturb '{name}' with sign {sign}, new value ~{new_value}\n")
                log_f.write(f"Value formatted: {params_probe[name].format()}\n")
                log_f.write(f"Score: {score_probe:.4f}\n")
                if probe_eval_results:
                    for key, value in probe_eval_results.items():
                        log_f.write(f"  {key:<15}: {value}\n")
                else:
                    log_f.write("  Simulation failed.\n")
                log_f.write("\n")
                log_f.flush()

                if score_probe < best_neighbor_so_far['score']:
                    best_neighbor_so_far = {'params': params_probe, 'score': score_probe}
                    last_success_direction[name] = sign
                    found_improvement_for_this_param = True
                else:
                    # 只有当两个方向都尝试过且都没有改善时，才重置方向
                    if sign == -1 or (last_success_direction[name] is not None and sign != last_success_direction[name]):
                         last_success_direction[name] = None

            if found_improvement_for_this_param:
                any_improvement_in_iteration = True
            else:
                message = f"  - Parameter '{name}' hit a local optimum. Freezing for Stage 2."
                print(message)
                log_f.write(message + "\n\n")
                log_f.flush()
                frozen_params_stage2.add(name)

        if best_neighbor_so_far['score'] < score_current:
            print(f"  - Found improvement in iteration. Moving to best neighbor with score {best_neighbor_so_far['score']:.4f}")
            X_current_params = best_neighbor_so_far['params']
        
        if not any_improvement_in_iteration:
            message = "\n--- Stage 2 CONVERGENCE: No improvement found in a full iteration. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break
        
        if simulation_count >= max_sim_count:
            message = f"\n--- HALTING: Maximum simulation count ({max_sim_count}) reached during Stage 2. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break

    log_f.close()

    # ======================== 结束和保存 ========================
    # 将输出与保存步骤委托给独立函数，保持行为不变
    finalize_and_save_results(
        tracking_info=tracking_info,
        platform=platform,
        simulation_count=simulation_count,
        frozen_params=frozen_params,
        m_param_names=m_param_names,
        other_param_names=other_param_names,
        filename="dynamic_v1.0.0_final_solution.txt"
    )

    print("\n=======================================================")
    print("===      STARTING FINAL VERIFICATION                ===")
    print("=======================================================")
    print(f"Loading best parameters from: {best_params_filepath}")

    try:
        best_params_from_file = read_parameters(best_params_filepath)
        print("Setting best parameters on the schematic...")
        platform.only_set_params(best_params_from_file)
        platform.calc_area()

        print("Running final verification simulation...")
        verification_scores = platform.evaluate()

        if verification_scores:
            print("\n========== FINAL VERIFICATION RESULT ==========")
            for key, value in verification_scores.items():
                print(f"====== {key:<15} : {value}")
            print("==============================================")
        else:
            print("\n--- FINAL VERIFICATION FAILED ---")

    except Exception as e:
        print(f"\nAn error occurred during final verification: {e}")

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
        
        run_dynamic_optimization_v1_0_0(platform, initial_parameters, circuit_graph,
                                    perturb_ratio=0.10,
                                    greedy_threshold_pct=args.greedy_alpha,
                                    max_sim_count = 1000)

    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --run_gd.")
        print("Example: python optimization.py --run_gd [other_args...]")

    print("\nScript finished.")
