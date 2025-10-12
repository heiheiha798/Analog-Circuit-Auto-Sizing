# V4.5基础上由单维度调整修改为多维度，也就是增加了梯度的维度（也许原来就不叫梯度

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

def run_hybrid_optimization_v5_0(platform: SimulatePlatform, initial_parameters: dict,
                                  circuit_graph: CircuitGraph,
                                  max_sim_count: int = 500,
                                  perturb_ratio: float = 0.1,
                                  initial_greedy_alpha_pct: float = 5.0,
                                  min_greedy_alpha_pct: float = 1.0):
    """
    执行一个混合优化算法 (V5.0)，融合了贪婪扫描和梯度下降回退。
    
    - 核心特性 (V5.0):
      - 阶段一 ('m' 参数) 保持不变，使用原有的贪心局部搜索。
      - 阶段二 (连续参数) 采用新混合策略：
        1. **快速贪婪扫描**:
           - 优先只对每个参数进行正向(+)微扰。
           - 如果发现任何一次微扰的收益超过动态阈值 (alpha)，则判定为“贪婪跳跃机会”。
           - 特殊检查：一旦发现正向(+)有机会，会额外检查一次负向(-)是否机会更大，然后选择最优方向进行立即跳跃，并开始下一次大迭代。
        2. **梯度下降回退**:
           - 如果完整扫描所有参数后，都未能触发贪婪跳跃。
           - 算法会利用扫描过程中收集的所有正向微扰信息，构建一个梯度向量。
           - 执行一次基于该梯度的复合移动（梯度下降）。
           - 根据复合移动的收益情况，自适应地调整alpha和步长，或在收益为负时终止并发出警告。
    - 保留特性:
      - 分层搜索, 惯性机制, 分阶段冻结, 自适应Alpha, 阈值终止。
    """
    print("\n===================================================================")
    print("===   HYBRID OPTIMIZATION (V5.0)                              ===")
    print("===   (Greedy Scan + Gradient Descent Fallback)             ===")
    print("===================================================================")

    # --- 目标函数 get_score (与V4.5版本相同) ---
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

    # --- 构建对称参数映射 (与V4.5版本相同) ---
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

    print("\n[DEBUG] Applying patch to ensure all non-dummy params are in the mapping...")
    missing_params_added = 0
    for name, param in initial_parameters.items():
        if not param.is_dummy and name not in param_mapping:
            param_mapping[name] = [name]  # 每个参数独立优化，不考虑对称性
            missing_params_added += 1
    print(f"[DEBUG] Patch complete. Added {missing_params_added} missing parameters to the optimization set.\n")
    # --- [调试修改] END ---

    # --- 参数分类和状态初始化 ---
    m_param_names = {name for name in param_mapping.keys() if name.endswith('_m')}
    other_param_names = {name for name in param_mapping.keys() if not name.endswith('_m')}
    
    frozen_params_stage1 = set()
    frozen_params_stage2 = set()
    last_success_direction = {name: None for name in param_mapping.keys()}
    current_greedy_alpha_pct = initial_greedy_alpha_pct
    simulation_count = 1

    tracking_info = {
        'best_params': copy.deepcopy(initial_parameters),
        'best_score': get_score(baseline_scores),
        'best_sim_num': 1,
        'best_metrics': baseline_scores
    }
    X_current_params = copy.deepcopy(initial_parameters)

    # ======================== 阶段一: 'm' 参数优化 (与V4.5完全相同) ========================
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
    
    print(f"\n--- Stage 1 Finished. Current best score: {tracking_info['best_score']:.4f} ---")


    # ======================== 阶段二: 连续参数优化 (V5.0 混合策略) ========================
    print(f"\n{'#'*25} Starting Stage 2: Continuous Parameter Tuning (V5.0) {'#'*25}")
    stage2_iter_count = 0
    current_greedy_alpha_pct = initial_greedy_alpha_pct # 重置Alpha

    while simulation_count < max_sim_count:
        stage2_iter_count += 1
        active_other_params = list(other_param_names - frozen_params_stage2)
        if not active_other_params:
            print(f"\n--- Stage 2 CONVERGENCE: All continuous parameters have been frozen. ---")
            break

        platform.only_set_params(X_current_params)
        eval_results = platform.evaluate(); simulation_count += 1
        score_current = get_score(eval_results)
        print(f"\n--- Stage 2 Iteration {stage2_iter_count} (Alpha: {current_greedy_alpha_pct:.2f}%, Perturb: {perturb_ratio:.4f}) ---")
        print(f"  - Current score: {score_current:.4f} (Sim count: {simulation_count})")

        found_immediate_jump = False
        gradient_vector = {}
        shuffled_param_names = random.sample(active_other_params, len(active_other_params))

        # --- PART 1: 快速贪婪扫描 ---
        for name in shuffled_param_names:
            # 1. 只进行正向(+)微扰
            params_probe_plus = copy.deepcopy(X_current_params)
            original_value = params_probe_plus[name].value
            h = original_value * perturb_ratio
            if h == 0: continue

            new_value_plus = original_value + h
            for actual_param in param_mapping[name]:
                params_probe_plus[actual_param].value = new_value_plus
            
            platform.only_set_params(params_probe_plus)
            probe_eval_plus = platform.evaluate(); simulation_count += 1
            score_probe_plus = get_score(probe_eval_plus)

            # 存储梯度信息以备后用
            gradient_vector[name] = (score_probe_plus - score_current) / h
            
            # 更新全局最优解
            if score_probe_plus < tracking_info['best_score']:
                print(f"  *** New overall best found! Score: {score_probe_plus:.4f}, Sim: {simulation_count} ***")
                tracking_info.update({'best_score': score_probe_plus, 'best_params': copy.deepcopy(params_probe_plus), 'best_sim_num': simulation_count, 'best_metrics': probe_eval_plus})

            # 2. 检查是否触发贪婪跳跃
            improvement_plus = score_current - score_probe_plus
            penalty_part = score_current - 1e9 if score_current >= 1e9 else abs(score_current)
            threshold = penalty_part * (current_greedy_alpha_pct / 100.0) if penalty_part > 0 else 0

            if improvement_plus > threshold:
                print(f"  >>> GREEDY JUMP opportunity on '{name}' (+)! Improvement > {current_greedy_alpha_pct:.2f}%.")
                
                # 3. 特殊检查: 既然(+)很有效，破例检查(-)是否更有效
                params_probe_minus = copy.deepcopy(X_current_params)
                new_value_minus = original_value - h
                if new_value_minus > 0:
                    for actual_param in param_mapping[name]: params_probe_minus[actual_param].value = new_value_minus
                    
                    platform.only_set_params(params_probe_minus)
                    probe_eval_minus = platform.evaluate(); simulation_count += 1
                    score_probe_minus = get_score(probe_eval_minus)

                    improvement_minus = score_current - score_probe_minus
                    if improvement_minus > threshold and score_probe_minus < score_probe_plus:
                        print(f"  >>> Exceptional check: '-' direction is even better! Jumping to (-).")
                        X_current_params = params_probe_minus
                        last_success_direction[name] = -1
                    else:
                        X_current_params = params_probe_plus
                        last_success_direction[name] = 1
                else:
                    X_current_params = params_probe_plus
                    last_success_direction[name] = 1

                found_immediate_jump = True
                current_greedy_alpha_pct = initial_greedy_alpha_pct # 重置Alpha
                break # 立即跳跃，结束本次扫描

            if simulation_count >= max_sim_count: break
        if found_immediate_jump: continue


        # --- PART 2: 梯度下降回退 (仅在没有贪婪跳跃时执行) ---
        print("  - No greedy jump found. Attempting gradient descent step.")
        if not gradient_vector: continue

        # 1. 构造梯度下降后的新点
        X_gradient_params = copy.deepcopy(X_current_params)
        # 学习率与扰动率关联，使得步长自适应
        learning_rate = perturb_ratio 
        for name in active_other_params:
            grad_val = gradient_vector.get(name, 0.0)
            # 更新规则: p_new = p_old - lr * grad
            update_step = learning_rate * grad_val
            # 按比例更新，使得对大数值参数的更新更大
            X_gradient_params[name].value -= update_step * X_gradient_params[name].value
            
            # 边界检查
            if X_gradient_params[name].value <= 0:
                X_gradient_params[name].value = X_current_params[name].value * 0.1

        # 2. 评估梯度移动的效果
        platform.only_set_params(X_gradient_params)
        eval_gradient_results = platform.evaluate(); simulation_count += 1
        score_gradient = get_score(eval_gradient_results)
        
        improvement = score_current - score_gradient
        
        # 3. 分析梯度移动的结果
        if improvement <= 0:
            print("\n" + "!"*80)
            print("!!! WARNING: GRADIENT DESCENT FAILED TO IMPROVE SCORE !!!")
            print(f"!!! Original Score: {score_current:.4f} -> Gradient Move Score: {score_gradient:.4f}")
            print("!!! This may indicate a highly non-linear or deceptive search space.")
            print("!!! Halting Stage 2.")
            
            print("\nOriginal Point Vector:")
            orig_vec = [f"{name}: {param.value:.4e}" for name, param in X_current_params.items() if name in active_other_params]
            print(orig_vec)
            
            print("\nComputed Gradient Vector (Score Change per Unit Perturbation):")
            grad_vec = [f"{name}: {gradient_vector.get(name, 0):.4e}" for name in active_other_params]
            print(grad_vec)
            print("!"*80 + "\n")
            break # 终止阶段二
        
        # 如果移动有效，则接受
        X_current_params = X_gradient_params
        print(f"  >>> Gradient step successful! New score: {score_gradient:.4f}")
        if score_gradient < tracking_info['best_score']:
             print(f"  *** New overall best found via gradient! Score: {score_gradient:.4f}, Sim: {simulation_count} ***")
             tracking_info.update({'best_score': score_gradient, 'best_params': copy.deepcopy(X_gradient_params), 'best_sim_num': simulation_count, 'best_metrics': eval_gradient_results})

        # 4. 自适应调整Alpha和步长
        improvement_pct = 0
        penalty_part = score_current - 1e9 if score_current >= 1e9 else abs(score_current)
        if penalty_part > 1e-9:
             improvement_pct = (improvement / penalty_part) * 100.0

        threshold = penalty_part * (current_greedy_alpha_pct / 100.0)
        
        if improvement > threshold:
            print(f"  - Composite improvement ({improvement_pct:.2f}%) exceeded alpha. Resetting alpha.")
            current_greedy_alpha_pct = initial_greedy_alpha_pct
        else:
            new_alpha = max(min_greedy_alpha_pct, improvement_pct)
            print(f"  - Composite improvement ({improvement_pct:.2f}%) is below alpha. Adjusting alpha.")
            print(f"  >>> Adjusting alpha from {current_greedy_alpha_pct:.2f}% to {new_alpha:.2f}%")
            current_greedy_alpha_pct = new_alpha
            
            # 同比例降低步长
            new_perturb_ratio = perturb_ratio * (new_alpha / initial_greedy_alpha_pct) if initial_greedy_alpha_pct > 0 else perturb_ratio
            print(f"  >>> Adjusting perturb_ratio from {perturb_ratio:.4f} to {new_perturb_ratio:.4f}")
            perturb_ratio = new_perturb_ratio

        if simulation_count >= max_sim_count:
            print(f"\n--- HALTING: Max simulations reached during Stage 2. ---")
            break
            
    print("\n===================================================================")
    print("===   OPTIMIZATION FINISHED                                     ===")
    print(f"===   Best score found: {tracking_info['best_score']:.4f} at simulation #{tracking_info['best_sim_num']}")
    print("===================================================================")
    
    return tracking_info

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
        
        run_hybrid_optimization_v5_0(platform, initial_parameters, circuit_graph,
                                    max_sim_count=500,
                                    perturb_ratio=0.1,
                                    initial_greedy_alpha_pct=args.greedy_alpha,
                                    min_greedy_alpha_pct=1.0)
        
    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --fw_scan.")
        print("Example: python optimization.py --fw_scan [other_args...]")

    print("\nScript finished.")

