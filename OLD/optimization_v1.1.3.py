# Optimization Main: Managed Multi-Pass Controller built on v1.0.0 core
# - Extracts robust optimization core into run_optimization_pass
# - Adds an intelligent controller run_managed_optimization_flow
# - Keeps SimulatePlatform unchanged

import argparse
import pyAether as ae
import copy
import random
import os
import math

from src.utils import read_parameters
from src.optimizer import SimulatePlatform
from src.data_models import CircuitGraph
from src.graph_builder import build_graph_from_eda
from src.circuit_analyzer import analyze_circuit_constraints
from src import parallel_utils

# ----------------------------- CLI -----------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Circuit Optimization Platform (Managed Controller)',
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

    parser.add_argument('--max_iter', type=int, default=100, help='Maximum iterations (unused placeholder)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    parser.add_argument('--set_params', action='store_true', help='Set parameters using specific param_file')
    parser.add_argument('--evaluate', action='store_true', help='Simulate using specific parameters')
    parser.add_argument('--run_gd', action='store_true', help='Run managed optimization flow')
    parser.add_argument('--greedy_alpha', type=float, default=5.0,
                        help='Greedy jump threshold in percent for Stage 1 freezing logic')
    parser.add_argument('--max_sim', type=int, default=1000, help='Global maximum simulation count')

    return parser.parse_args()

# ----------------------------- Shared utilities -----------------------------

def evaluate_and_count(platform: SimulatePlatform, params: dict, simulation_count: int):
    platform.only_set_params(params)
    platform.calc_area()
    results = platform.evaluate()
    return results, simulation_count + 1

def write_params_to_file(params: dict, output_filepath: str):
    """Write Parameter objects dict in CSV-like format to be reloadable by read_parameters."""
    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w') as f:
            for name in sorted(params.keys()):
                param = params[name]
                formatted_value = param.format()  # keep consistent with v1.0.0
                f.write(f'"parameter","{name}","{formatted_value}"\n')
    except Exception as e:
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

def finalize_and_save_results(tracking_info: dict, platform: SimulatePlatform, simulation_count: int,
                              frozen_params: set, m_param_names: set, other_param_names: set,
                              filename: str = "dynamic_final_solution.txt"):
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
        f.write("Optimization Status: Finished Managed Multi-Pass Optimization.\n")
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

# ----------------------------- Optimization Pass (extracted from v1.0.0) -----------------------------

def run_optimization_pass(
    platform: SimulatePlatform,
    initial_parameters: dict,
    circuit_graph: CircuitGraph,
    get_score_func,  # callable(scores, initial_ugb, initial_area)
    initial_metrics: dict,  # baseline metrics for normalization
    start_sim_count: int,
    pass_max_sim_count: int,
    pass_name: str,
    greedy_threshold_pct: float = 5.0
):
    """
    Single-pass two-stage optimization, extracted and generalized from v1.0.0:
    - Stage 1: discrete '_m' parameters with freezing and greedy jump threshold
    - Stage 2: continuous parameters (fw/l/r/c) with non-greedy best-neighbor move
    This pass assumes baseline evaluation has already been performed by the caller.
    Returns (tracking_info, simulation_count).
    """
    print("\n===================================================================")
    print(f"===   {pass_name} - DYNAMIC TWO-STAGE OPTIMIZATION (Core)   ===")
    print("===================================================================")

    # Use provided baseline metrics and starting sim counter
    initial_ugb_val = initial_metrics.get('UGB', 1.0)
    initial_area_val = initial_metrics.get('Total_Area', 1.0)
    if initial_ugb_val == 0:
        print("WARNING: Initial UGB is 0")
    if initial_area_val == 0:
        print("WARNING: Initial Total_Area is 0")

    # Build symmetric parameter mapping based on circuit graph constraints
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

    frozen_params = set()
    last_success_direction = {name: None for name in param_mapping.keys()}

    simulation_count = start_sim_count

    output_dir = platform.output_path
    os.makedirs(output_dir, exist_ok=True)
    optimization_log_file = os.path.join(output_dir, f"optimization_log_{pass_name.replace(' ', '_').replace('/', '-')}.txt")
    best_params_filepath = os.path.join(output_dir, "best_params_so_far.txt")

    # Initialize tracking with the provided starting point
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

    # -------------------- Stage 1: '_m' parameter tuning with freezing --------------------
    print(f"\n{'#'*25} {pass_name} - Stage 1: 'm' Parameter Tuning {'#'*25}")
    log_f.write(f"\n{'#'*25} {pass_name} - Stage 1: 'm' Parameter Tuning {'#'*25}\n\n")
    log_f.flush()

    stage1_iter_count = 0
    stage1_sim_budget = 200  # local budget before switching to Stage 2
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
            message = f"  - {pass_name} - Stage1 simulation budget reached ({simulation_count} / {stage1_sim_budget}). Exiting Stage 1 to Stage 2."
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            stage1_budget_exhausted = True
            break

        score_current = get_score_func(eval_results, initial_ugb_val, initial_area_val)
        print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")
        log_f.write(f"--- {pass_name} - Stage 1 Iteration {stage1_iter_count} (Current) ---\n")
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
                if simulation_count >= stage1_sim_budget:
                    message = f"  - {pass_name} - Stage1 simulation budget reached during probing ({simulation_count} / {stage1_sim_budget}). Will exit to Stage 2 after this iteration."
                    print(message)
                    log_f.write(message + "\n")
                    log_f.flush()
                    stage1_budget_exhausted = True
                score_probe = get_score_func(probe_eval_results, initial_ugb_val, initial_area_val)

                update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count, best_params_filepath)

                log_f.write(f"--- {pass_name} - Iteration {simulation_count} (Probe) ---\n")
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
                    print(f"  >>> {pass_name} - GREEDY JUMP on '{name}'! Moving immediately.")
                    X_current_params = params_probe
                    last_success_direction[name] = sign
                    found_immediate_jump = True
                    break

            if not found_improvement_for_this_param:
                message = f"  - {pass_name} - Parameter '{name}' hit a local optimum. Freezing."
                print(message)
                log_f.write(message + "\n\n")
                log_f.flush()
                frozen_params.add(name)
                last_success_direction[name] = None

            if found_immediate_jump:
                break

        if not found_immediate_jump:
            message = f"  - {pass_name} - No greedy jump found in this iteration. Re-evaluating with new frozen set."
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()

        if stage1_budget_exhausted:
            message = f"\n--- {pass_name} - Stage 1 halted due to stage1_sim_budget ({stage1_sim_budget}). Proceeding to Stage 2. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break

    # -------------------- Stage 2: continuous param tuning (non-greedy) --------------------
    print(f"\n{'#'*25} {pass_name} - Stage 2: Continuous Parameter Tuning {'#'*25}")
    log_f.write(f"\n{'#'*25} {pass_name} - Stage 2: Continuous Parameter Tuning {'#'*25}\n\n")
    log_f.flush()

    stage2_iter_count = 0
    perturb_ratio = 0.1
    frozen_params_stage2 = set()

    while simulation_count < pass_max_sim_count:
        stage2_iter_count += 1
        active_other_params = list(other_param_names - frozen_params_stage2)

        if not active_other_params:
            message = f"\n--- {pass_name} - Stage 2 CONVERGENCE: All 'other' parameters have been frozen. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break

        print(f"\n--- {pass_name} - Stage 2 Iteration {stage2_iter_count} (Active 'other' params: {len(active_other_params)}) ---")

        eval_results, simulation_count = evaluate_and_count(platform, X_current_params, simulation_count)
        score_current = get_score_func(eval_results, initial_ugb_val, initial_area_val)
        print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")
        log_f.write(f"--- {pass_name} - Stage 2 Iteration {stage2_iter_count} (Current) ---\n")
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
                score_probe = get_score_func(probe_eval_results, initial_ugb_val, initial_area_val)

                update_tracking_if_better(tracking_info, score_probe, params_probe, probe_eval_results, simulation_count, best_params_filepath)

                log_f.write(f"--- {pass_name} - Iteration {simulation_count} (Probe) ---\n")
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
                    if sign == -1 or (last_success_direction[name] is not None and sign != last_success_direction[name]):
                        last_success_direction[name] = None

            if found_improvement_for_this_param:
                any_improvement_in_iteration = True
            else:
                message = f"  - {pass_name} - Parameter '{name}' hit a local optimum. Freezing for Stage 2."
                print(message)
                log_f.write(message + "\n\n")
                log_f.flush()
                frozen_params_stage2.add(name)

        if best_neighbor_so_far['score'] < score_current:
            print(f"  - {pass_name} - Found improvement in iteration. Moving to best neighbor with score {best_neighbor_so_far['score']:.4f}")
            X_current_params = best_neighbor_so_far['params']

        if not any_improvement_in_iteration:
            message = f"\n--- {pass_name} - Stage 2 CONVERGENCE: No improvement found in a full iteration. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break

        if simulation_count >= pass_max_sim_count:
            message = f"\n--- {pass_name} - HALTING: pass_max_sim_count ({pass_max_sim_count}) reached during Stage 2. ---"
            print(message)
            log_f.write(message + "\n\n")
            log_f.flush()
            break

    log_f.close()

    print(f"\n--- {pass_name} COMPLETED ---")
    print(f"Best score found in this pass: {tracking_info['best_score']:.4f}")
    return tracking_info, simulation_count

# ----------------------------- Managed Controller -----------------------------

def get_score_flexible(scores: dict, initial_ugb_val: float, initial_area_val: float, *,
                        gain_db_target: float = 80.0, initial_metrics: dict = None) -> float:
    """Flexible scoring function that allows changing Gain_db target while keeping normalization stable."""
    if not scores:
        return 1e12
    pm = scores.get('Phase_Margin', -180.0)
    gain = scores.get('Gain_db', -200.0)
    gm = scores.get('Gain_Margin', 100.0)
    if pm <= -180.0 or gain <= -200.0 or gm >= 100.0:
        return 1e12

    iopa_ma = scores.get('I_OPA', 1.0) * 1000.0

    pm_viol = max(0, (50.0 - pm) / 50.0)
    gain_viol = max(0, (gain_db_target - gain) / gain_db_target)
    gm_viol = max(0, (gm - (-10.0)) / abs(-10.0))
    iopa_viol = max((iopa_ma - 3.0) / 3.0, 0)

    total_violation = pm_viol + gain_viol + gm_viol + iopa_viol

    if total_violation > 0:
        return 1e9 + total_violation * 1e6
    else:
        # keep normalization consistent across passes
        base_ugb = initial_metrics.get('UGB', initial_ugb_val) if initial_metrics else initial_ugb_val
        base_area = initial_metrics.get('Total_Area', initial_area_val) if initial_metrics else initial_area_val
        ugb_norm = scores.get('UGB', 0) / (base_ugb if base_ugb != 0 else 1.0)
        area_norm = scores.get('Total_Area', 0) / (base_area if base_area != 0 else 1.0)
        return 0.5 * area_norm - 0.5 * ugb_norm

def run_managed_optimization_flow(platform: SimulatePlatform, initial_parameters: dict,
                                  circuit_graph: CircuitGraph, max_sim_count: int, args):
    """Manages the entire multi-pass optimization flow."""

    # 1) Baseline evaluation
    print("\n--- Performing Baseline Evaluation ---")
    platform.only_set_params(initial_parameters)
    platform.calc_area()
    baseline_scores = platform.evaluate()
    if not baseline_scores:
        print("FATAL: Baseline simulation failed. Exiting.")
        return

    simulation_count = 1

    # Global best tracker
    initial_score = get_score_flexible(baseline_scores, baseline_scores.get('UGB', 1.0),
                                       baseline_scores.get('Total_Area', 1.0),
                                       gain_db_target=80.0, initial_metrics=baseline_scores)
    global_best_tracking_info = {
        'best_score': initial_score,
        'best_params': copy.deepcopy(initial_parameters),
        'best_sim_num': simulation_count,
        'best_metrics': baseline_scores
    }

    # 2) Pass 1: Strict goal (80dB)
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
    )

    if pass1_tracking_info['best_score'] < global_best_tracking_info['best_score']:
        global_best_tracking_info = pass1_tracking_info

    # 3) Intermission check
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
        # 4) Pass 2: Relaxed goal based on Pass 1 outcome
        print("Pass 1 did not meet all strict constraints. Proceeding to Pass 2.")
        gain_from_pass1 = best_metrics_pass1.get('Gain_db', 0.0)
        # round down to nearest 5 dB, lower bounded at 50 dB
        new_gain_target = max(50.0, math.floor(gain_from_pass1 / 5.0) * 5.0)
        print(f"Original Gain_db from Pass 1 was {gain_from_pass1:.2f}. New target for Pass 2: {new_gain_target:.1f} dB")

        starting_params_for_pass2 = global_best_tracking_info['best_params']

        pass2_tracking_info, simulation_count = run_optimization_pass(
            platform=platform,
            initial_parameters=starting_params_for_pass2,
            circuit_graph=circuit_graph,
            get_score_func=lambda scores, ugb, area: get_score_flexible(scores, ugb, area, gain_db_target=new_gain_target, initial_metrics=baseline_scores),
            initial_metrics=baseline_scores,
            start_sim_count=simulation_count,
            pass_max_sim_count=max_sim_count,
            pass_name=f"Pass 2 (Relaxed Goal: {new_gain_target:.1f}dB)",
            greedy_threshold_pct=args.greedy_alpha,
        )

        # 这里先直接更新为 Pass 2 的结果，无论好坏
        # if pass2_tracking_info['best_score'] < global_best_tracking_info['best_score']:
        #     global_best_tracking_info = pass2_tracking_info

        global_best_tracking_info = pass2_tracking_info

    # 5) Finalize
    print("\n--- FINALIZING OPTIMIZATION ---")
    m_param_names = {name for name in initial_parameters.keys() if name.endswith('_m')}
    other_param_names = {name for name in initial_parameters.keys() if not name.endswith('_m')}

    finalize_and_save_results(
        tracking_info=global_best_tracking_info,
        platform=platform,
        simulation_count=simulation_count,
        frozen_params=set(),
        m_param_names=m_param_names,
        other_param_names=other_param_names,
        filename="dynamic_final_solution.txt",
    )

# ----------------------------- __main__ -----------------------------

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
                ae.dbCloseCV(cv)

        initial_parameters = read_parameters(args.param_file)
        if not initial_parameters:
            print("Error: Failed to read initial parameters for GD. Exiting.")
            exit(1)

        try:
            print("\n--- Preparing parallel MDE views for the entire optimization run... ---")
            parallel_utils.prepare_parallel_views(platform.ae_lib, platform.mde_cell, platform.mde_view)

            run_managed_optimization_flow(
                platform=platform,
                initial_parameters=initial_parameters,
                circuit_graph=circuit_graph,
                max_sim_count=args.max_sim,
                args=args,
            )
        finally:
            print("\n--- Cleaning up parallel MDE views... ---")
            parallel_utils.cleanup_parallel_views(platform.ae_lib, platform.mde_cell, platform.mde_view)

    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --run_gd.")
        print("Example: python optimization_main.py --run_gd [other_args...]")

    print("\nScript finished.")
