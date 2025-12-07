#
# =================================================================================
#  整合版本 optimization.py (Task 1 完成品)
# =================================================================================
#  - 结构: 基于 optimization_v1.1.3.py，拥有多趟（multi-pass）流程控制和并行环境管理能力。
#  - 核心算法: 内部的 run_optimization_pass 函数是 v1.0.0 核心爬山算法的直接重构版本，
#              保留了其高效的两阶段（'m'参数冻结 + 连续参数优化）逻辑。
#  - 并行模拟: 依赖 src/optimizer.py 中的 SimulatePlatform 来执行实际的9-corner并行仿真。
#  - 关键修改: 实现了您指定的第二趟优化逻辑，即将增益目标下取整到10dB的整数倍。
#

import argparse
import pyAether as ae
import copy
import random
import os
import math  # 确保导入 math 库以使用 floor 函数

# 依赖于 src 目录下的工具代码
from src.utils import read_parameters
from src.optimizer import SimulatePlatform
from src.data_models import CircuitGraph
from src.graph_builder import build_graph_from_eda
from src.circuit_analyzer import analyze_circuit_constraints
from src import parallel_utils

# ----------------------------- CLI (命令行接口，保持不变) -----------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Circuit Optimization Platform (Managed Controller)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- 保留的必要参数 ---
    parser.add_argument('--ae_lib', required=True, help='Aether library name')
    parser.add_argument('--ae_cell', required=True, help='Aether cell name')
    parser.add_argument('--ae_view', required=True, help='Aether view name')
    parser.add_argument('--mde_cell', required=True, help='MDE cell name')
    parser.add_argument('--mde_view', required=True, help='MDE view name')
    parser.add_argument('--param_file', required=True, help='Parameter file path')
    parser.add_argument('--best_param_file', type=str, required=True, # 确保打分脚本传入的参数被正确接收
                        help='Filename for the final best parameters, used by the scoring script.')

    # --- 为缺失的参数提供默认值 ---
    # 打分脚本不提供 output_path，我们提供一个默认值
    parser.add_argument('--output_path', default='./case_results', help='Output directory path')
    # 打分脚本不提供 output_file，我们提供一个默认值
    parser.add_argument('--output_file', default='optimization.log', help='Output file name')
    
    # 将原来的命令行参数内化为固定的默认值
    parser.add_argument('--max_sim', type=int, default=1000, help='Global maximum simulation count')
    parser.add_argument('--greedy_alpha', type=float, default=5.0,
                        help='Greedy jump threshold in percent for Stage 1 freezing logic')

    # --- 可以保留用于本地调试的参数 ---
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')

    # --- 删除不再需要的模式切换参数 ---
    # 'max_iter' (本身就未使用)
    # 'set_params'
    # 'evaluate'
    # 'run_gd'

    return parser.parse_args()

# ----------------------------- Shared utilities (共享工具函数，保持不变) -----------------------------

def evaluate_and_count(platform: SimulatePlatform, params: dict, simulation_count: int):
    platform.only_set_params(params)
    platform.calc_area()
    results = platform.evaluate()
    return results, simulation_count + 1

def write_params_to_file(params: dict, output_filepath: str):
    """Write Parameter objects dict in CSV-like format to be reloadable by read_parameters."""
    try:
        directory = os.path.dirname(output_filepath)
        if directory:  # 仅当 directory 不是空字符串时才执行
            os.makedirs(directory, exist_ok=True)

        with open(output_filepath, 'w') as f:
            for name in sorted(params.keys()):
                param = params[name]
                formatted_value = param.format()
                f.write(f'"parameter","{name}","{formatted_value}"\n')
    except Exception as e:
        # 保持详细的错误输出，方便未来调试
        print(f"  [Warning] Failed to write best params to {output_filepath}: {e}")

def update_tracking_if_better(tracking_info: dict, score_probe: float, params_probe: dict,
                              probe_eval_results: dict, simulation_count: int,
                              best_params_filepath: str):
    if score_probe < tracking_info['best_score']:
        print(f"  *** New overall best found! Score: {score_probe:.4f}, Sim: {simulation_count} ***")
        tracking_info.update({
            'best_score': score_probe,
            'best_params': copy.deepcopy(params_probe),
            'best_sim_num': simulation_count,
            'best_metrics': probe_eval_results
        })
        write_params_to_file(tracking_info['best_params'], best_params_filepath)

# ----------------------------- 核心优化算法 (源自 v1.0.0, 由 v1.1.3 重构) -----------------------------

def run_optimization_pass(
    platform: SimulatePlatform,
    initial_parameters: dict,
    circuit_graph: CircuitGraph,
    get_score_func,
    initial_metrics: dict,
    start_sim_count: int,
    pass_max_sim_count: int,
    pass_name: str,
    greedy_threshold_pct: float = 5.0,
    args=None  # <-- 新增
):
    """
    这是 v1.0.0 核心算法的模块化版本。
    - 阶段一: 优化离散 '_m' 参数，带冻结机制和贪心跳转。
    - 阶段二: 优化连续参数 (fw/l/r/c)，采用非贪心的最佳邻居移动策略。
    此函数行为与 v1.0.0 的主优化循环完全一致。
    """
    print("\n===================================================================")
    print(f"===   {pass_name} - DYNAMIC TWO-STAGE OPTIMIZATION (Core)   ===")
    print("===================================================================")

    initial_ugb_val = initial_metrics.get('UGB', 1.0)
    initial_area_val = initial_metrics.get('Total_Area', 1.0)
    if initial_ugb_val == 0:
        print("WARNING: Initial UGB is 0")
    if initial_area_val == 0:
        print("WARNING: Initial Total_Area is 0")

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
        if device_name not in devices_in_groups and not getattr(param, 'is_dummy', False):
            if name not in param_mapping:
                param_mapping[name] = [name]

    m_param_names = {name for name in param_mapping.keys() if name.endswith('_m')}
    other_param_names = {name for name in param_mapping.keys() if not name.endswith('_m')}

    frozen_params = set()
    last_success_direction = {name: None for name in param_mapping.keys()}
    simulation_count = start_sim_count

    output_dir = platform.output_path
    os.makedirs(output_dir, exist_ok=True)
    optimization_log_file = os.path.join(output_dir, f"optimization_log_{pass_name.replace(' ', '_').replace('/', '-')}.txt")
    # 如果 args 存在并且包含 best_param_file，则使用它，否则回退到默认值
    best_param_filename = args.best_param_file if args and hasattr(args, 'best_param_file') else "best_params_so_far.txt"
    best_params_filepath = best_param_filename

    start_score = get_score_func(initial_metrics, initial_ugb_val, initial_area_val)
    tracking_info = {
        'best_params': copy.deepcopy(initial_parameters),
        'best_score': start_score,
        'best_sim_num': simulation_count,
        'best_metrics': initial_metrics
    }
    write_params_to_file(tracking_info['best_params'], best_params_filepath)

    X_current_params = copy.deepcopy(initial_parameters)

    log_f = open(optimization_log_file, 'w', encoding='utf-8')
    log_f.write(f"--- {pass_name} - Start @ Sim {simulation_count} (Baseline from caller) ---\n")
    log_f.write(f"Score: {tracking_info['best_score']:.4f}\n")
    for key, value in initial_metrics.items():
        log_f.write(f"  {key:<15}: {value}\n")
    log_f.write("\n")
    log_f.flush()

    # -------------------- 阶段一: '_m' 参数优化，带冻结机制 --------------------
    print(f"\n{'#'*25} {pass_name} - Stage 1: 'm' Parameter Tuning {'#'*25}")
    log_f.write(f"\n{'#'*25} {pass_name} - Stage 1: 'm' Parameter Tuning {'#'*25}\n\n")
    log_f.flush()

    stage1_iter_count = 0
    stage1_sim_budget = start_sim_count + 200
    stage1_budget_exhausted = False

    while True:
        if simulation_count >= pass_max_sim_count:
            msg = f"\n--- {pass_name} - HALTING: pass_max_sim_count ({pass_max_sim_count}) reached before Stage 1 iteration. ---"
            print(msg)
            log_f.write(msg + "\n\n")
            log_f.flush()
            break

        stage1_iter_count += 1
        active_m_params = list(m_param_names - frozen_params)

        if stage1_budget_exhausted:
            message = f"\n--- {pass_name} - Stage 1 exiting early: stage1_sim_budget ({stage1_sim_budget}) reached. Moving to Stage 2. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break

        if not active_m_params:
            message = f"\n--- {pass_name} - Stage 1 CONVERGENCE: All 'm' parameters have been frozen. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break

        print(f"\n--- {pass_name} - Stage 1 Iteration {stage1_iter_count} (Active 'm' params: {len(active_m_params)}) ---")

        eval_results, simulation_count = evaluate_and_count(platform, X_current_params, simulation_count)
        if simulation_count >= stage1_sim_budget:
            stage1_budget_exhausted = True

        score_current = get_score_func(eval_results, initial_ugb_val, initial_area_val)
        print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")
        log_f.write(f"--- {pass_name} - Stage 1 Iteration {stage1_iter_count} (Current) ---\n")
        log_f.write(f"Score: {score_current:.4f}\n")
        if eval_results:
            for key, value in eval_results.items(): log_f.write(f"  {key:<15}: {value}\n")
        else: log_f.write("  Simulation failed.\n")
        log_f.write("\n"); log_f.flush()

        found_immediate_jump = False        
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
                step = 1
                new_value = params_probe[name].clamp(original_value + sign * step)
                if new_value == original_value: continue
                
                for actual_param_to_update in param_mapping[name]:
                    params_probe[actual_param_to_update].value = new_value

                probe_eval_results, simulation_count = evaluate_and_count(platform, params_probe, simulation_count)
                if simulation_count >= stage1_sim_budget: stage1_budget_exhausted = True
                
                score_probe = get_score_func(probe_eval_results, initial_ugb_val, initial_area_val)
                update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count, best_params_filepath)

                log_f.write(f"--- {pass_name} - Iteration {simulation_count} (Probe) ---\n")
                log_f.write(f"Action: Perturbed '{name}' with sign {sign}, new value ~{new_value}\n")
                log_f.write(f"Score: {score_probe:.4f}\n")
                if probe_eval_results:
                    for key, value in probe_eval_results.items(): log_f.write(f"  {key:<15}: {value}\n")
                else: log_f.write("  Simulation failed.\n")
                log_f.write("\n"); log_f.flush()

                if score_probe < score_current:
                    found_improvement_for_this_param = True

                improvement = score_current - score_probe
                threshold = (score_current - 1e9) * (greedy_threshold_pct / 100.0) if score_current >= 1e9 else 0

                if improvement > threshold:
                    print(f"  >>> {pass_name} - GREEDY JUMP on '{name}'! Moving immediately.")
                    X_current_params = params_probe
                    last_success_direction[name] = sign
                    found_immediate_jump = True
                    break

            if not found_improvement_for_this_param:
                frozen_params.add(name); last_success_direction[name] = None
            if found_immediate_jump: break

        if stage1_budget_exhausted: break
    
    # -------------------- 阶段二: 连续参数优化 --------------------
    print(f"\n{'#'*25} {pass_name} - Stage 2: Continuous Parameter Tuning {'#'*25}")
    log_f.write(f"\n{'#'*25} {pass_name} - Stage 2: Continuous Parameter Tuning {'#'*25}\n\n")
    log_f.flush()

    stage2_iter_count = 0; perturb_ratio = 0.1; frozen_params_stage2 = set()
    while simulation_count < pass_max_sim_count:
        stage2_iter_count += 1
        active_other_params = list(other_param_names - frozen_params_stage2)

        if not active_other_params:
            print(f"\n--- {pass_name} - Stage 2 CONVERGENCE: All 'other' parameters have been frozen. ---"); break
        
        print(f"\n--- {pass_name} - Stage 2 Iteration {stage2_iter_count} (Active 'other' params: {len(active_other_params)}) ---")
        eval_results, simulation_count = evaluate_and_count(platform, X_current_params, simulation_count)
        score_current = get_score_func(eval_results, initial_ugb_val, initial_area_val)
        print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")
        log_f.write(f"--- {pass_name} - Stage 2 Iteration {stage2_iter_count} (Current) ---\n")
        log_f.write(f"Score: {score_current:.4f}\n")
        if eval_results:
            for key, value in eval_results.items(): log_f.write(f"  {key:<15}: {value}\n")
        else: log_f.write("  Simulation failed.\n")
        log_f.write("\n"); log_f.flush()

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
                if abs(new_value - original_value) < 1e-12: continue

                for actual_param_to_update in param_mapping[name]:
                    params_probe[actual_param_to_update].value = new_value

                probe_eval_results, simulation_count = evaluate_and_count(platform, params_probe, simulation_count)
                score_probe = get_score_func(probe_eval_results, initial_ugb_val, initial_area_val)
                update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count, best_params_filepath)

                # ========================= 开始：添加以下代码块 =========================
                # 为了日志清晰，将标题修改为 Stage 2 Probe
                log_f.write(f"--- {pass_name} - Stage 2 Iteration {stage2_iter_count} / Sim {simulation_count} (Probe) ---\n")
                # 为了更好地显示浮点数值，可以格式化 new_value
                log_f.write(f"Action: Perturbed '{name}' with sign {sign}, new value ~{new_value:.4e}\n")
                log_f.write(f"Score: {score_probe:.4f}\n")
                if probe_eval_results:
                    for key, value in probe_eval_results.items():
                        log_f.write(f"  {key:<15}: {value}\n")
                else:
                    log_f.write("  Simulation failed.\n")
                log_f.write("\n")
                log_f.flush()
                # ========================= 结束：添加的代码块 =========================

                if score_probe < best_neighbor_so_far['score']:
                    best_neighbor_so_far = {'params': params_probe, 'score': score_probe}
                    last_success_direction[name] = sign
                    found_improvement_for_this_param = True
                else:
                    if sign == -1 or (last_success_direction[name] is not None and sign != last_success_direction[name]):
                        last_success_direction[name] = None
            
            if found_improvement_for_this_param: any_improvement_in_iteration = True
            else: frozen_params_stage2.add(name)

        if best_neighbor_so_far['score'] < score_current:
            X_current_params = best_neighbor_so_far['params']
        if not any_improvement_in_iteration:
            print(f"\n--- {pass_name} - Stage 2 CONVERGENCE: No improvement found in a full iteration. ---"); break
        if simulation_count >= pass_max_sim_count:
            print(f"\n--- {pass_name} - HALTING: pass_max_sim_count ({pass_max_sim_count}) reached during Stage 2. ---"); break

    log_f.close()
    print(f"\n--- {pass_name} COMPLETED ---")
    return tracking_info, simulation_count

# ----------------------------- 多趟流程控制器 (源自 v1.1.3) -----------------------------

def get_score_flexible(scores: dict, initial_ugb_val: float, initial_area_val: float, *,
                        gain_db_target: float = 80.0, initial_metrics: dict = None) -> float:
    """
    灵活的评分函数，允许在保持归一化基准稳定的前提下，改变增益目标。
    """
    if not scores: return 1e12
    pm = scores.get('Phase_Margin', -180.0); gain = scores.get('Gain_db', -200.0); gm = scores.get('Gain_Margin', 100.0)
    if pm <= -180.0 or gain <= -200.0 or gm >= 100.0: return 1e12

    iopa_ma = scores.get('I_OPA', 1.0) * 1000.0
    pm_viol = max(0, (50.0 - pm) / 50.0)
    gain_viol = max(0, (gain_db_target - gain) / gain_db_target)
    gm_viol = max(0, (gm - (-10.0)) / abs(-10.0))
    iopa_viol = max((iopa_ma - 3.0) / 3.0, 0)
    total_violation = pm_viol + gain_viol + gm_viol + iopa_viol

    if total_violation > 0:
        return 1e9 + total_violation * 1e6
    else:
        base_ugb = initial_metrics.get('UGB', initial_ugb_val) if initial_metrics else initial_ugb_val
        base_area = initial_metrics.get('Total_Area', initial_area_val) if initial_metrics else initial_area_val
        ugb_norm = scores.get('UGB', 0) / (base_ugb if base_ugb != 0 else 1.0)
        area_norm = scores.get('Total_Area', 0) / (base_area if base_area != 0 else 1.0)
        return 0.5 * area_norm - 0.5 * ugb_norm

def run_managed_optimization_flow(platform: SimulatePlatform, initial_parameters: dict,
                                  circuit_graph: CircuitGraph, max_sim_count: int, args):
    """
    管理整个多趟优化流程。
    """
    # 1) 基准评估
    print("\n--- Performing Baseline Evaluation ---")
    platform.only_set_params(initial_parameters)
    platform.calc_area()
    baseline_scores = platform.evaluate()
    if not baseline_scores:
        print("FATAL: Baseline simulation failed. Exiting.")
        return

    simulation_count = 1

    # 全局最优追踪器
    initial_score = get_score_flexible(baseline_scores, baseline_scores.get('UGB', 1.0),
                                       baseline_scores.get('Total_Area', 1.0),
                                       gain_db_target=80.0, initial_metrics=baseline_scores)
    global_best_tracking_info = {
        'best_score': initial_score,
        'best_params': copy.deepcopy(initial_parameters),
        'best_sim_num': simulation_count,
        'best_metrics': baseline_scores
    }

    # 2) 第一趟: 严格目标 (80dB)
    pass1_budget_abs = int(max_sim_count * 0.7)
    pass1_tracking_info, simulation_count = run_optimization_pass(
        platform=platform,
        initial_parameters=initial_parameters,
        circuit_graph=circuit_graph,
        get_score_func=lambda scores, ugb, area: get_score_flexible(scores, ugb, area, gain_db_target=80.0, initial_metrics=baseline_scores),
        initial_metrics=baseline_scores,
        start_sim_count=simulation_count,
        pass_max_sim_count=pass1_budget_abs,
        pass_name="Pass 1 (Strict Goal: 80dB)",
        greedy_threshold_pct=args.greedy_alpha,
        args=args  # <-- 新增
    )

    # 无论第一趟结果如何，都更新全局最优解
    global_best_tracking_info = pass1_tracking_info

    # 3) 中场检查
    print("\n--- INTERMISSION CHECK ---")
    best_metrics_pass1 = global_best_tracking_info['best_metrics']
    is_fully_compliant = (
        best_metrics_pass1.get('Phase_Margin', 0) >= 50.0 and
        best_metrics_pass1.get('Gain_Margin', 0) <= -10.0 and
        best_metrics_pass1.get('Gain_db', 0) >= 80.0 and
        best_metrics_pass1.get('I_OPA', float('inf')) * 1000.0 <= 3.0
    )

    if is_fully_compliant:
        print("Optimization successful in Pass 1. All constraints met.")
    elif simulation_count >= max_sim_count:
        print("Simulation budget exhausted. Finalizing with Pass 1 results.")
    else:
        # 4) 第二趟: 根据第一趟结果，放宽目标
        print("Pass 1 did not meet all strict constraints. Proceeding to Pass 2.")
        gain_from_pass1 = best_metrics_pass1.get('Gain_db', 0.0)
        
        # ========================== 关键修改点 ==========================
        # 按照您的要求，将增益目标下取整到最近的10dB整数倍，最低为50dB
        new_gain_target = max(50.0, math.floor(gain_from_pass1 / 10.0) * 10.0)
        # ================================================================
        
        print(f"Original Gain_db from Pass 1 was {gain_from_pass1:.2f}. New target for Pass 2: {new_gain_target:.1f} dB")

        starting_params_for_pass2 = global_best_tracking_info['best_params']

        pass2_tracking_info, simulation_count = run_optimization_pass(
            platform=platform,
            initial_parameters=starting_params_for_pass2,
            circuit_graph=circuit_graph,
            get_score_func=lambda scores, ugb, area: get_score_flexible(scores, ugb, area, gain_db_target=new_gain_target, initial_metrics=baseline_scores),
            # 注意：这里的 initial_metrics 仍然使用最初的 baseline_scores，以确保UGB和Area的归一化标准在两趟中保持一致
            initial_metrics=baseline_scores,
            start_sim_count=simulation_count,
            pass_max_sim_count=max_sim_count,
            pass_name=f"Pass 2 (Relaxed Goal: {new_gain_target:.1f}dB)",
            greedy_threshold_pct=args.greedy_alpha,
            args=args  # <-- 新增
        )

        # 无论第二趟结果好坏，都用它的结果更新全局最优
        # 这是因为第二趟的目标函数不同，分数不可直接比较。我们相信第二趟的最终状态是一个更好的权衡点。
        global_best_tracking_info = pass2_tracking_info

    # 5) 最终写入最优参数以确保文件与最终结果一致
    # print(f"\n--- WRITING FINAL BEST PARAMETERS to {args.best_param_file} ---")
    # final_best_params_filepath = os.path.join(platform.output_path, args.best_param_file)
    # write_params_to_file(global_best_tracking_info['best_params'], final_best_params_filepath)

# ----------------------------- __main__ (主函数，源自 v1.1.3) -----------------------------

if __name__ == "__main__":
    args = parse_arguments()
    ae.emyInitAether('-adv')

    # 1. 初始化平台 (Platform)
    #    现在 output_path 和 output_file 会从 args 的默认值中获取
    platform = SimulatePlatform(
        ae_lib=args.ae_lib,
        ae_cell=args.ae_cell,
        ae_view=args.ae_view,
        mde_cell=args.mde_cell,
        mde_view=args.mde_view,
        output_path=args.output_path,
        output_file=args.output_file,
    )

    # 2. 直接开始优化流程 (不再有 if/elif 判断)
    print("\n--- Starting Managed Optimization Flow (Default Execution) ---")

    print("\n--- Building Circuit Graph for Symmetry Analysis ---")
    cv = None
    try:
        cv = ae.dbOpenCV(args.ae_lib, args.ae_cell, args.ae_view)
        if cv is None:
            print("Error: Failed to open design view for analysis.")
            exit(1)

        circuit_graph = build_graph_from_eda(cv, args.param_file)
        analyze_circuit_constraints(circuit_graph)

        if circuit_graph.constraint_groups:
            print(f"\nSymmetry analysis found {len(circuit_graph.constraint_groups)} groups for optimization.")
        else:
            print("\nNo symmetric groups found. Proceeding with standard optimization.")
    finally:
        if cv:
            ae.dbCloseCV(cv)

    initial_parameters = read_parameters(args.param_file)
    if not initial_parameters:
        print("Error: Failed to read initial parameters for optimization. Exiting.")
        exit(1)

    # 3. 运行核心优化逻辑
    try:
        print("\n--- Preparing parallel MDE views for the entire optimization run... ---")
        parallel_utils.prepare_parallel_views(platform.ae_lib, platform.mde_cell, platform.mde_view)

        # 调用主优化函数
        # max_sim 和 args.greedy_alpha 等都会从带有默认值的 args 对象中获取
        run_managed_optimization_flow(
            platform=platform,
            initial_parameters=initial_parameters,
            circuit_graph=circuit_graph,
            max_sim_count=args.max_sim,
            args=args,
        )
    finally:
        # 确保清理工作总能执行
        print("\n--- Cleaning up parallel MDE views... ---")
        parallel_utils.cleanup_parallel_views(platform.ae_lib, platform.mde_cell, platform.mde_view)

    print("\nScript finished.")
