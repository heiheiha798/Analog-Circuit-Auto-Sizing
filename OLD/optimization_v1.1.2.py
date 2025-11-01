# V 1.1.0 做了一点I/O管理优化以及嵌套循环debug
# 结果很垃圾

import argparse
import pyAether as ae
import copy
import random
import os
import math # 确保导入 math 模块

from src.utils import read_parameters
from src.optimizer import SimulatePlatform
from src.data_models import CircuitGraph
from src.graph_builder import build_graph_from_eda
from src.circuit_analyzer import analyze_circuit_constraints
from src import parallel_utils

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

def get_score_flexible(scores, initial_ugb_val, initial_area_val, gain_db_target=80.0):
    """
    [新增] 灵活版本的评分函数，可以动态传入直流增益的目标。
    """
    if not scores: return 1e12
    pm = scores.get('Phase_Margin', -180.0)
    gain = scores.get('Gain_db', -200.0)
    gm = scores.get('Gain_Margin', 100.0)
    if pm <= -180.0 or gain <= -200.0 or gm >= 100.0: return 1e12
    
    iopa_ma = scores.get('I_OPA', 1.0) * 1000.0
    
    pm_viol = max(0, (50.0 - pm) / 50.0)
    # [修改] 使用传入的 gain_db_target
    gain_viol = max(0, (gain_db_target - gain) / gain_db_target)
    gm_viol = max(0, (gm - (-10.0)) / abs(-10.0))
    iopa_viol = max((iopa_ma - 3.0) / 3.0, 0)
    
    total_violation = pm_viol + gain_viol + gm_viol + iopa_viol
    
    if total_violation > 0:
        return 1e9 + total_violation * 1e6
    else:
        # 归一化因子保持不变，以确保两遍优化之间的分数可比
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

def finalize_and_save_results(tracking_info, platform, simulation_count, frozen_params, m_param_names, other_param_names, filename="dynamic_v1.1.0_final_solution.txt"):
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
		f.write(f"Optimization Status: Finished Dynamic Two-Stage Optimization (V1.1.0).\n")
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

def update_candidate_if_eligible(probe_eval_results, score_probe, params_probe, candidate_info):
    """
    [新增] 检查一个解是否是合格的“下一轮候选解”并更新。
    合格标准：恰好满足3个硬性约束。
    更新标准：在所有合格解中，惩罚分最低。
    """
    if not probe_eval_results:
        return # 仿真失败，不是候选解

    # 1. 严格按照PDF标准，检查满足的硬约束数量
    constraints_met = 0
    if probe_eval_results.get('Phase_Margin', 0) >= 50.0: constraints_met += 1
    if probe_eval_results.get('Gain_Margin', 0) <= -10.0: constraints_met += 1
    if probe_eval_results.get('Gain_db', 0) >= 80.0: constraints_met += 1 # 注意：这里用80dB硬标准
    if probe_eval_results.get('I_OPA', float('inf')) * 1000.0 <= 3.0: constraints_met += 1
    
    # 2. 如果恰好满足3个约束
    if constraints_met == 3:
        # 3. 并且它的惩罚分比当前记录的候选解更低
        if score_probe < candidate_info['best_score']:
            print(f"  --- Found a new best candidate for Pass 2! Score: {score_probe:.4f} ---")
            candidate_info.update({
                'best_score': score_probe,
                'best_params': copy.deepcopy(params_probe),
                'best_metrics': probe_eval_results
            })

def run_optimization_pass(
    platform: SimulatePlatform, 
    initial_parameters: dict,
    circuit_graph: CircuitGraph,
    get_score_func, # 接收评分函数作为参数
    start_sim_count: int, # 接收起始仿真计数
    max_sim_count: int,
    initial_metrics: dict, # 接收用于归一化的初始指标
    pass_name: str, # 用于日志打印，如 "Pass 1 (Strict)"
    greedy_threshold_pct: float = 5.0,
    perturb_ratio: float = 0.10,
    candidate_info: dict = None # [新增] 接收候选解追踪器
):
    """
    [重构] 这是单遍优化的核心逻辑。
    它可以被多次调用，以实现不同的优化策略。
    """
    print(f"\n===================================================================")
    print(f"===   STARTING OPTIMIZATION {pass_name.upper()}                 ===")
    print(f"===================================================================")

    # --- 从传入的参数获取归一化基准 ---
    initial_ugb_val = initial_metrics.get('UGB', 1.0)
    initial_area_val = initial_metrics.get('Total_Area', 1.0)
    if initial_ugb_val == 0: print("WARNING: Initial UGB is 0")
    if initial_area_val == 0: print("WARNING: Initial Total_Area is 0")

    # --- 参数映射和分类 (逻辑不变) ---
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
    
    m_param_names = {name for name in param_mapping.keys() if name.endswith('_m')}
    other_param_names = {name for name in param_mapping.keys() if not name.endswith('_m')}
    
    # --- [修改] 思路二：动态预算分配 ---
    stage1_sim_budget = len(m_param_names) * 6
    print(f"--- Dynamic Budget: Stage 1 budget set to {stage1_sim_budget} simulations ({len(m_param_names)} 'm' params * 4) ---")


    # --- 状态初始化 ---
    frozen_params = set()
    last_success_direction = {name: None for name in param_mapping.keys()}
    
    # [修改] 从传入的参数初始化
    simulation_count = start_sim_count 
    X_current_params = copy.deepcopy(initial_parameters)
    
    # --- 全局最优解追踪器 (在这一遍优化内的最优解) ---
    # 评估起点参数
    platform.only_set_params(X_current_params)
    platform.calc_area()
    start_scores = platform.evaluate()
    simulation_count +=1

    tracking_info = {
        'best_params': X_current_params,
        'best_score': get_score_func(start_scores, initial_ugb_val, initial_area_val),
        'best_sim_num': start_sim_count,
        'best_metrics': start_scores
    }
    
    output_dir = platform.output_path
    optimization_log_file = os.path.join(output_dir, "optimization_log.txt")
    best_params_filepath = os.path.join(output_dir, "best_params_so_far.txt")

    write_params_to_file(tracking_info['best_params'], best_params_filepath)

    log_f = open(optimization_log_file, 'a', encoding='utf-8') # Use 'a' to append
    log_f.write(f"\n--- Starting {pass_name} at Sim Count {start_sim_count} ---\n")
    log_f.write(f"Initial Score: {tracking_info['best_score']:.4f}\n")
    if start_scores:
        for key, value in start_scores.items():
            log_f.write(f"  {key:<15}: {value}\n")
    log_f.write("\n")
    log_f.flush()

    # ======================== 阶段一: 'm' 参数优化 (带参数冻结) ========================
    print(f"\n{'#'*25} Starting Stage 1: 'm' Parameter Tuning {'#'*25}")
    log_f.write(f"\n{'#'*25} Starting Stage 1: 'm' Parameter Tuning {'#'*25}\n\n")
    log_f.flush()
    stage1_iter_count = 0
    stage1_budget_exhausted = False
    while True:
        stage1_iter_count += 1
        active_m_params = list(m_param_names - frozen_params)

        if (simulation_count - start_sim_count) >= stage1_sim_budget:
            stage1_budget_exhausted = True

        if stage1_budget_exhausted or simulation_count >= max_sim_count:
            message = f"\n--- Stage 1 exiting: Budget exhausted or max simulations reached. Moving to Stage 2. ---"
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
        
        eval_results, simulation_count = evaluate_and_count(platform, X_current_params, simulation_count)
        if (simulation_count - start_sim_count) >= stage1_sim_budget:
            stage1_budget_exhausted = True

        score_current = get_score_func(eval_results, initial_ugb_val, initial_area_val)
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
                
                probe_eval_results, simulation_count = evaluate_and_count(platform, params_probe, simulation_count)
                if (simulation_count - start_sim_count) >= stage1_sim_budget:
                    stage1_budget_exhausted = True

                score_probe = get_score_func(probe_eval_results, initial_ugb_val, initial_area_val)
                
                update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count, best_params_filepath)

                # [新增] 每次都检查是否为合格候选解
                if candidate_info is not None:
                    update_candidate_if_eligible(probe_eval_results, score_probe, params_probe, candidate_info)

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
                    last_success_direction[name] = sign
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
        
        if stage1_budget_exhausted:
            break

    # ======================== 阶段二: 'fw/l/r/c' 参数优化 ========================
    print(f"\n{'#'*25} Starting Stage 2: Continuous Parameter Tuning {'#'*25}")
    log_f.write(f"\n{'#'*25} Starting Stage 2: Continuous Parameter Tuning {'#'*25}\n\n")
    log_f.flush()
    stage2_iter_count = 0
    
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
        
        eval_results, simulation_count = evaluate_and_count(platform, X_current_params, simulation_count)
        score_current = get_score_func(eval_results, initial_ugb_val, initial_area_val)
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

        found_immediate_jump = False
        shuffled_param_names = random.sample(active_other_params, len(active_other_params))
        
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
                score_probe = get_score_func(probe_eval_results, initial_ugb_val, initial_area_val)

                update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count, best_params_filepath)

                if candidate_info is not None:
                    update_candidate_if_eligible(probe_eval_results, score_probe, params_probe, candidate_info)

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

                if score_probe < score_current:
                    found_improvement_for_this_param = True
                
                improvement = score_current - score_probe
                threshold = (score_current - 1e9) * (greedy_threshold_pct / 100.0) if score_current >= 1e9 else 0

                if improvement > threshold:
                    print(f"  >>> GREEDY JUMP on '{name}'! Moving immediately.")
                    X_current_params = params_probe
                    last_success_direction[name] = sign
                    found_immediate_jump = True
                    break
            
            if not found_improvement_for_this_param:
                message = f"  - Parameter '{name}' hit a local optimum. Freezing for Stage 2."
                print(message)
                log_f.write(message + "\n\n")
                log_f.flush()
                frozen_params_stage2.add(name)
                last_success_direction[name] = None

            if found_immediate_jump:
                break

        if not found_immediate_jump:
            message = "\n--- Stage 2 CONVERGENCE: No greedy jump found in a full iteration. ---"
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

    # --- [修改] 函数的返回值 ---
    print(f"\n--- {pass_name.upper()} COMPLETED ---")
    print(f"Best score in this pass: {tracking_info['best_score']:.4f}")
    # [修改] 同时返回 tracking_info, 最终仿真数, 和 walker 的最终参数状态
    return tracking_info, simulation_count, X_current_params


def run_full_optimization_flow(platform: SimulatePlatform, initial_parameters: dict, circuit_graph: CircuitGraph, max_sim_count: int, args):
    """
    [MODIFIED FOR EFFICIENCY & ROBUSTNESS]
    Manages the setup and teardown of parallel views for the entire flow.
    """
    
    # [修改] 使用 try...finally 确保清理工作总能执行
    try:
        # 1. 在所有优化开始前，准备一次并行视图
        print("\n--- Preparing parallel MDE views for the entire optimization run... ---")
        parallel_utils.prepare_parallel_views(platform.ae_lib, platform.mde_cell, platform.mde_view)

        # --- 基准评估 (这段代码从 try 块外部移到内部) ---
        print("\n--- Performing Baseline Evaluation ---")
        platform.only_set_params(initial_parameters)
        platform.calc_area()
        baseline_scores = platform.evaluate()
        if not baseline_scores:
            print("FATAL: Baseline simulation failed. Exiting.")
            return # 使用 return 提前退出，finally 仍然会执行

        simulation_count = 1
        
        # [新增] 初始化“最佳候选解”追踪器
        candidate_info = {
            'best_score': float('inf'), # 我们要找惩罚分最低的
            'best_params': None,
            'best_metrics': None
        }
        
        # ======================== Pass 1: 严格约束优化 ========================
        # [修改] 捕获第三个返回值：pass1_final_params，即阶段一结束时 walker 的最终位置
        pass1_tracking_info, simulation_count, pass1_final_params = run_optimization_pass(
            platform=platform,
            initial_parameters=initial_parameters,
            circuit_graph=circuit_graph,
            get_score_func=lambda scores, ugb, area: get_score(scores, ugb, area), # 使用原始 get_score
            start_sim_count=simulation_count,
            max_sim_count=max_sim_count,
            initial_metrics=baseline_scores,
            pass_name="Pass 1 (Strict)",
            greedy_threshold_pct=args.greedy_alpha,
            candidate_info=candidate_info # [修改] 传入追踪器
        )

        # ======================== 中场检查与决策 (重构此部分) ========================
        final_tracking_info = pass1_tracking_info # 默认最终结果是 Pass 1 找到的最优解
        
        print("\n--- INTERMISSION CHECK ---")

        # 1. 检查 Pass 1 找到的全局最优解是否已经满足所有4个约束
        best_metrics_pass1 = pass1_tracking_info['best_metrics']
        constraints_met = 0
        if best_metrics_pass1: # 确保仿真成功
            if best_metrics_pass1.get('Phase_Margin', 0) >= 50.0: constraints_met += 1
            if best_metrics_pass1.get('Gain_Margin', 0) <= -10.0: constraints_met += 1
            if best_metrics_pass1.get('Gain_db', 0) >= 80.0: constraints_met += 1
            if best_metrics_pass1.get('I_OPA', float('inf')) * 1000.0 <= 3.0: constraints_met += 1
        
        print(f"Global best from Pass 1 meets {constraints_met} out of 4 hard constraints.")

        # 决策逻辑
        if constraints_met == 4 or simulation_count >= max_sim_count:
            # 情况 A: 已经完美解决，或仿真预算耗尽，直接结束
            print("All constraints met or simulation budget exhausted. Finalizing with Pass 1 results.")
            # final_tracking_info 已经设置为 pass1_tracking_info，无需额外操作

        elif candidate_info['best_params'] is not None:
            # 情况 B: 未完全解决，但“侦察兵”找到了一个合格的3约束候选解。启用备用策略！
            print(f"A special candidate meeting 3 constraints was found (Score: {candidate_info['best_score']:.2f}).")
            print("Proceeding to Pass 2 with relaxed constraints, starting from this candidate.")
            
            starting_params_for_pass2 = candidate_info['best_params']
            gain_db_of_candidate = candidate_info['best_metrics'].get('Gain_db', 0)
            
            # 动态设定新的、更宽松的增益目标
            new_gain_target = round(gain_db_of_candidate / 10.0) * 10.0
            print(f"Original Gain_db from candidate was {gain_db_of_candidate:.2f}. New target for Pass 2: {new_gain_target:.2f} dB")

            # 从“候选解”开始，执行 Pass 2
            pass2_tracking_info, simulation_count, _ = run_optimization_pass(
                platform=platform,
                initial_parameters=starting_params_for_pass2, # <-- 从候选解开始
                circuit_graph=circuit_graph,
                get_score_func=lambda scores, ugb, area: get_score_flexible(scores, ugb, area, gain_db_target=new_gain_target),
                start_sim_count=simulation_count,
                max_sim_count=max_sim_count,
                initial_metrics=baseline_scores,
                pass_name=f"Pass 2 (Relaxed, from Candidate)",
                greedy_threshold_pct=args.greedy_alpha
            )
            # Pass 2 的结果成为最终结果
            final_tracking_info = pass2_tracking_info

        else:
            # 情况 C: 未完全解决，也找不到合格的3约束候选解。说明优化陷入困境，此时停止。
            print("Fewer than 4 constraints met and no suitable candidate for Pass 2 was found. Optimization halted.")
            # final_tracking_info 依然是 Pass 1 找到的最好结果

        # ======================== 结束、保存和最终验证 ========================
        # 这里的参数名需要从 circuit_graph 和 param_mapping 中获取
        m_param_names = {name for name in pass1_tracking_info['best_params'].keys() if name.endswith('_m')} # 简化获取
        other_param_names = {name for name in pass1_tracking_info['best_params'].keys() if not name.endswith('_m')}
        
        finalize_and_save_results(
            tracking_info=final_tracking_info,
            platform=platform,
            simulation_count=simulation_count,
            frozen_params=set(), # 最终报告不关心冻结状态
            m_param_names=m_param_names,
            other_param_names=other_param_names,
            filename="dynamic_final_solution.txt"
        )

    finally:
        # 2. 无论程序是正常结束还是异常中断，都执行清理
        print("\n--- Cleaning up parallel MDE views... ---")
        parallel_utils.cleanup_parallel_views(platform.ae_lib, platform.mde_cell, platform.mde_view)

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
        
        # [修改] 调用新的顶层控制器
        run_full_optimization_flow(
            platform=platform,
            initial_parameters=initial_parameters,
            circuit_graph=circuit_graph,
            max_sim_count=1000, # 可以从 args 获取
            args=args
        )

    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --run_gd.")
        print("Example: python optimization.py --run_gd [other_args...]")

    print("\nScript finished.")
