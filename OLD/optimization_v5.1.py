# V5.1 加上三阶段优化和参数专属步长设置

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

def get_parameter_constraints(param_name: str, param_value: float) -> tuple:
    """根据参数名称返回其约束范围"""
    if param_name.endswith('_m'):
        return (1, None)  # max值将在运行时动态设置
    elif param_name.endswith('_l'):
        return (0.1, 20.0)
    elif param_name.endswith('_fw'):
        return (0.42, 10.0)
    elif param_name.startswith('R'):
        return (0.1, 100.0)
    elif param_name.startswith('C'):
        return (1e-15, 1e-12)
    else:
        return (0.1, 100.0)  # 默认范围

def initialize_step_manager(parameters: dict) -> dict:
    """
    初始化步长管理器（鲁棒版本）
    - 明确处理所有已知参数类型
    - 确保所有参与优化的参数都有一个条目
    """
    manager = {}
    for name, param in parameters.items():
        # 原则1: 只排除明确要排除的（虚拟参数）
        if param.is_dummy:
            continue

        # 原则2: 按优先级清晰地处理所有已知类型
        # _m 参数是整数，采用绝对步长
        if name.endswith('_m'):
            manager[name] = {'type': 'absolute', 'ratio': 1.0, 'current_step': 1.0}
        
        # _l 参数是连续值，有专属的相对步长
        elif name.endswith('_l'):
            manager[name] = {'type': 'relative', 'ratio': 0.05, 'current_step': 0.0}
            
        # _fw 参数是连续值，有专属的相对步长
        elif name.endswith('_fw'):
            manager[name] = {'type': 'relative', 'ratio': 0.1, 'current_step': 0.0}
            
        # R/C 参数是连续值，有专属的相对步长
        elif name.startswith(('R', 'C')):
            manager[name] = {'type': 'relative', 'ratio': 0.15, 'current_step': 0.0}
        
        # 原则3: 为其他所有未明确指定的连续参数提供一个安全的默认值
        elif param.type == 'continuous':
            manager[name] = {'type': 'relative', 'ratio': 0.1, 'current_step': 0.0}
            
        # （可选）原则4: 如果未来出现未知的非连续参数，可以打印警告
        else:
            print(f"[Warning] Parameter '{name}' with type '{param.type}' is not a continuous value and has no step manager entry.")

    return manager

def run_hybrid_optimization_v5_1(platform: SimulatePlatform, initial_parameters: dict,
                               circuit_graph: CircuitGraph,
                               max_sim_count: int = 500,
                               initial_greedy_alpha_pct: float = 5.0,
                               min_greedy_alpha_pct: float = 1.0):
    """
    执行三阶段混合优化算法 (V5.1)
    
    特性:
    1. 三阶段优化: 
       - Stage 1: m参数基础优化
       - Stage 2: l/fw/R/C参数综合优化
       - Stage 3: R/C参数性能微调
    2. 参数专属步长
    3. 动态参数约束
    4. 综合安全检查机制
    """
    print("\n===================================================================")
    print("===   THREE-STAGE HYBRID OPTIMIZATION (V5.1)                    ===")
    print("===================================================================")

    # --- 初始化参数约束和步长 ---
    print("\n[Phase 0] Initializing parameter constraints and step sizes...")
    
    # 1. 查找最大的初始m值用于设置上限
    max_initial_m = max(
        param.value for name, param in initial_parameters.items() 
        if name.endswith('_m')
    )
    max_m_limit = max(max_initial_m * 2, 20)  # 至少给20的上限
    
    # 2. 更新所有参数的约束
    for name, param in initial_parameters.items():
        if param.is_dummy: continue
        
        min_val, max_val = get_parameter_constraints(name, param.value)
        if name.endswith('_m'):
            max_val = max_m_limit
            
        param.min = min_val if min_val is not None else param.min
        param.max = max_val if max_val is not None else param.max

    # 初始化步长管理器
    param_step_manager = initialize_step_manager(initial_parameters)
    print(f"Initialized step manager with {len(param_step_manager)} parameters.")

    # --- 目标函数 ---
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

    # --- 构建对称参数映射 ---
    param_mapping = {}
    devices_in_groups = set()

    # 1. 首先处理所有对称组
    for group in circuit_graph.constraint_groups:
        # 使用组里的第一个器件作为代表
        representative_device = group.devices[0]
        
        # 找到这个代表器件的所有参数，并为它们创建映射
        for param_name_full, param_obj in initial_parameters.items():
            if param_obj.is_dummy: continue
            
            if param_name_full.startswith(representative_device.name + '_'):
                # 提取参数后缀，例如 '_l', '_fw'
                param_suffix = param_name_full[len(representative_device.name):]
                # 创建映射的键，例如 'NM5_l'
                key = f"{representative_device.name}{param_suffix}"
                # 创建值，这是一个列表，包含组内所有器件的对应参数
                # 例如 ['NM5_l', 'NM4_l']
                param_mapping[key] = [f"{dev.name}{param_suffix}" for dev in group.devices]

        # 记录所有已经处理过的对称器件
        for dev in group.devices:
            devices_in_groups.add(dev.name)

    # 2. 接着处理所有不在任何对称组中的独立器件
    for name, param in initial_parameters.items():
        if param.is_dummy: continue
        
        device_name = name.split('_')[0]
        # 如果器件不在对称组集合中，就把它作为独立参数添加
        if device_name not in devices_in_groups:
            if name not in param_mapping:
                 param_mapping[name] = [name]

    print("\n=== Parameters in Optimization List (After Symmetry Analysis) ===")
    device_params = {}
    for name in param_mapping.keys():
        device_name = name.split('_')[0]
        param_type = '_'.join(name.split('_')[1:])
        if device_name not in device_params:
            device_params[device_name] = []
        device_params[device_name].append(param_type)
    
    # 按器件名称排序打印
    print("\nOptimization parameters by device:")
    for device, params in sorted(device_params.items()):
        print(f"  - {device}: {', '.join(sorted(params))}")
    
    print(f"\nTotal unique parameter groups to optimize: {len(param_mapping)}")
    print("=" * 60 + "\n")
    
    # --- 参数分类 ---
    m_param_names = {name for name in param_mapping.keys() if name.endswith('_m')}
    core_device_params = {name for name in param_mapping.keys() 
                         if not name.endswith('_m')}
    final_tuning_params = {name for name in param_mapping.keys() 
                          if name.startswith(('R', 'C'))}

    frozen_params = {'stage1': set(), 'stage2': set(), 'stage3': set()}
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

    # --- 三阶段优化主循环 ---
    current_stage = 1
    while simulation_count < max_sim_count and current_stage <= 3:
        print(f"\n{'#'*25} Starting Stage {current_stage} {'#'*25}")
        
        stage_iter_count = 0
        while simulation_count < max_sim_count:
            # 在每次内部迭代开始时重新计算 active_params
            if current_stage == 1:
                active_params = list(m_param_names - frozen_params['stage1'])
                stage_name = "'m' Parameter Optimization"
            elif current_stage == 2:
                active_params = list(core_device_params - frozen_params['stage2'])
                stage_name = "Core Device Optimization (l, fw, R, C)"
            else:
                active_params = list(final_tuning_params - frozen_params['stage3'])
                stage_name = "Final UGB/Area Tuning (R, C)"
            
            # 如果没有可优化的参数，提前结束当前阶段
            if not active_params:
                print(f"No active parameters left in stage {current_stage}. Moving to next stage.")
                break

            print(f"\n=== Stage {current_stage}: {stage_name} ===")
            print(f"Active parameters count: {len(active_params)}")
            
            stage_iter_count += 1
            
            platform.only_set_params(X_current_params)
            eval_results = platform.evaluate(); simulation_count += 1
            score_current = get_score(eval_results)
            print(f"\n--- Stage {current_stage}.{stage_iter_count} "
                  f"(Alpha: {current_greedy_alpha_pct:.2f}%) ---")
            print(f"  - Current score: {score_current:.4f} (Sim: {simulation_count})")

            # 更新当前步长
            for name in active_params:
                state = param_step_manager[name]
                param = X_current_params[name]
                
                if state['type'] == 'relative':
                    state['current_step'] = abs(param.value * state['ratio'])
                elif state['type'] == 'absolute':
                    if name.endswith('_m'):
                        state['current_step'] = 1.0
                    else:
                        state['current_step'] = (param.max - param.min) * state['ratio']

            found_immediate_jump = False
            gradient_vector = {}
            shuffled_param_names = random.sample(active_params, len(active_params))

            # --- 快速贪婪扫描 ---
            for name in shuffled_param_names:
                improvement_found_for_param = False
                h = param_step_manager[name]['current_step']
                if h == 0: continue

                # 正向探测
                params_probe_plus = copy.deepcopy(X_current_params)
                original_value = params_probe_plus[name].value
                new_value_plus = original_value + h
                if new_value_plus > params_probe_plus[name].max:
                    continue
                    
                for actual_param in param_mapping[name]:
                    params_probe_plus[actual_param].value = new_value_plus
                
                platform.only_set_params(params_probe_plus)
                probe_eval_plus = platform.evaluate(); simulation_count += 1
                score_probe_plus = get_score(probe_eval_plus)
                
                # 从第二阶段开始执行安全检查
                if current_stage >= 2:
                    if score_current < 1e9 and score_probe_plus >= 1e9:
                        print(f"  - Safety check: '{name}' (+) violated constraints. Discarding.")
                        score_probe_plus = 1e15

                # 存储梯度信息
                gradient_vector[name] = (score_probe_plus - score_current) / h

                # 如果正向探测导致性能降低，尝试负向
                if score_probe_plus > score_current and current_stage != 1:
                    params_probe_minus = copy.deepcopy(X_current_params)
                    new_value_minus = original_value - h
                    if new_value_minus >= params_probe_minus[name].min:
                        for actual_param in param_mapping[name]:
                            params_probe_minus[actual_param].value = new_value_minus
                        
                        platform.only_set_params(params_probe_minus)
                        probe_eval_minus = platform.evaluate(); simulation_count += 1
                        score_probe_minus = get_score(probe_eval_minus)
                        
                        if score_probe_minus < tracking_info['best_score']:
                            print(f"  *** New best! Score: {score_probe_minus:.4f}, Sim: {simulation_count} (found in neg. probe) ***")
                            tracking_info.update({
                                'best_score': score_probe_minus,
                                'best_params': copy.deepcopy(params_probe_minus),
                                'best_sim_num': simulation_count,
                                'best_metrics': probe_eval_minus
                            })
    
                        if current_stage >= 2:
                            if score_current < 1e9 and score_probe_minus >= 1e9:
                                print(f"  - Safety check: '{name}' (-) violated constraints. Discarding.")
                                score_probe_minus = 1e15
                        
                        if score_probe_minus < score_current:
                            improvement_found_for_param = True
                            improvement_minus = score_current - score_probe_minus
                            if improvement_minus > threshold:
                                print(f"  >>> (-) direction has better improvement!")
                                X_current_params = params_probe_minus
                                last_success_direction[name] = -1
                                found_immediate_jump = True
                                break
                
                # 更新全局最优解
                if score_probe_plus < tracking_info['best_score']:
                    print(f"  *** New best! Score: {score_probe_plus:.4f}, Sim: {simulation_count} ***")
                    tracking_info.update({
                        'best_score': score_probe_plus,
                        'best_params': copy.deepcopy(params_probe_plus),
                        'best_sim_num': simulation_count,
                        'best_metrics': probe_eval_plus
                    })

                # 检查贪婪跳跃条件
                improvement_plus = score_current - score_probe_plus
                penalty_part = score_current - 1e9 if score_current >= 1e9 else abs(score_current)
                threshold = penalty_part * (current_greedy_alpha_pct / 100.0)

                if score_probe_plus < score_current:
                    improvement_found_for_param = True
                    if improvement_plus > threshold:
                        print(f"  >>> GREEDY JUMP opportunity on '{name}' (+)!")
                        X_current_params = params_probe_plus
                        last_success_direction[name] = 1
                        found_immediate_jump = True
                        break

                # 参数冻结逻辑
                if not improvement_found_for_param and not found_immediate_jump:
                    print(f"  - Param '{name}' shows no improvement. Freezing for Stage {current_stage}.")
                    if current_stage == 1:
                        frozen_params['stage1'].add(name)
                    elif current_stage == 2:
                        frozen_params['stage2'].add(name)
                    else:  # current_stage == 3
                        frozen_params['stage3'].add(name)

                if simulation_count >= max_sim_count: break

            if found_immediate_jump:
                current_greedy_alpha_pct = initial_greedy_alpha_pct
                continue

            # --- 梯度下降回退 ---
            if current_stage != 1:  # 非m参数才执行梯度下降
                print("  - No greedy jump found. Attempting gradient descent step.")
                if not gradient_vector: continue

                X_gradient_params = copy.deepcopy(X_current_params)
                learning_rate = 0.5  # 固定学习率
                
                for name in active_params:
                    grad_val = gradient_vector.get(name, 0.0)
                    if abs(grad_val) < 1e-10: continue
                    
                    h = param_step_manager[name]['current_step']
                    update = -learning_rate * h * (grad_val / abs(grad_val))  # 标准化梯度
                    new_value = X_gradient_params[name].value + update
                    
                    # 边界检查
                    new_value = min(max(new_value, X_gradient_params[name].min),
                                  X_gradient_params[name].max)
                    
                    for actual_param in param_mapping[name]:
                        X_gradient_params[actual_param].value = new_value

                platform.only_set_params(X_gradient_params)
                eval_gradient_results = platform.evaluate(); simulation_count += 1
                score_gradient = get_score(eval_gradient_results)
                
                # 从第二阶段开始执行安全检查
                if current_stage >= 2:
                    if score_current < 1e9 and score_gradient >= 1e9:
                        print("  - Safety check: Gradient step violated constraints. Reverting.")
                        score_gradient = 1e15
                
                improvement = score_current - score_gradient
                
                if improvement <= 0:
                    print("\n[!] Gradient descent failed to improve score.")
                    print(f"[!] Score: {score_current:.4f} -> {score_gradient:.4f}")
                    if current_stage == 3:
                        print("[!] Moving to next stage due to gradient failure.")
                        break
                else:
                    X_current_params = X_gradient_params
                    print(f"  >>> Gradient step successful! New score: {score_gradient:.4f}")
                    
                    if score_gradient < tracking_info['best_score']:
                        print(f"  *** New best via gradient! Score: {score_gradient:.4f} ***")
                        tracking_info.update({
                            'best_score': score_gradient,
                            'best_params': copy.deepcopy(X_gradient_params),
                            'best_sim_num': simulation_count,
                            'best_metrics': eval_gradient_results
                        })
                        
                    # --- 动态调整步长比率 ---
                    improvement_pct = 0
                    if penalty_part > 1e-9:
                        improvement_pct = (improvement / penalty_part) * 100.0
                    
                    # 确定调整因子
                    adj_factor = 1.0
                    if improvement_pct > current_greedy_alpha_pct * 2:
                        adj_factor = 1.5  # 大幅提升，放大步长
                        print("  - Step effectiveness is high. Increasing step ratios.")
                    elif improvement_pct > current_greedy_alpha_pct * 0.5:
                        adj_factor = 1.1  # 效果不错，小幅放大
                        print("  - Step effectiveness is moderate. Slightly increasing step ratios.")
                    else:
                        adj_factor = 0.7  # 效果不佳，缩小步长
                        print("  - Step effectiveness is low. Reducing step ratios.")
                    
                    # 应用调整因子
                    for name in active_params:
                        state = param_step_manager[name]
                        if state['type'] == 'absolute' and name.endswith('_m'):
                            continue  # 跳过m参数的步长调整
                        old_ratio = state['ratio']
                        new_ratio = old_ratio * adj_factor
                        # 施加边界约束
                        state['ratio'] = min(max(new_ratio, 0.01), 0.5)
                    
                    # 调整alpha
                    if improvement_pct > threshold:
                        current_greedy_alpha_pct = initial_greedy_alpha_pct
                    else:
                        current_greedy_alpha_pct = max(min_greedy_alpha_pct, improvement_pct)

            if simulation_count >= max_sim_count:
                print(f"\n--- HALTING: Max simulations reached during Stage {current_stage}. ---")
                break
                
        # 当前阶段结束，进入下一阶段
        print(f"\n--- Stage {current_stage} complete. Moving to next stage. ---")
        current_stage += 1
        current_greedy_alpha_pct = initial_greedy_alpha_pct  # 重置alpha
            
    print("\n===================================================================")
    print("===   OPTIMIZATION FINISHED                                     ===")
    print(f"===   Best score: {tracking_info['best_score']:.4f} at sim #{tracking_info['best_sim_num']}")
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
        
        run_hybrid_optimization_v5_1(platform, initial_parameters, circuit_graph,
                                    max_sim_count=500,
                                    initial_greedy_alpha_pct=args.greedy_alpha,
                                    min_greedy_alpha_pct=1.0)
        
    else:
        print("\nNo specific mode selected. Use --set_params, --evaluate, or --fw_scan.")
        print("Example: python optimization.py --fw_scan [other_args...]")

    print("\nScript finished.")

