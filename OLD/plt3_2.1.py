import re
import matplotlib.pyplot as plt
import numpy as np
import os
from math import isinf, isnan

# extract_performance_data 和 calculate_objective_score 函数保持不变
def extract_performance_data(filename="tmp6_2.1.txt"):
    # ... (no changes needed)
    ugb_list, pm_list, gdb_list, gm_list, i_opa_list, ta_list = [], [], [], [], [], []
    performance_pattern = re.compile(r"======\s*(\w+[\s_]*\w*)\s*:\s*([\d\.\-Ee+infnan]+)")
    PERFORMANCE_NAMES = ["UGB", "Phase_Margin", "Gain_db", "Gain_Margin", "I_OPA", "Total_Area"]
    current_performance_block = {}
    LIST_MAP = {"UGB": ugb_list, "Phase_Margin": pm_list, "Gain_db": gdb_list, "Gain_Margin": gm_list, "I_OPA": i_opa_list, "Total_Area": ta_list}
    def finalize_and_save_performance_block():
        if not current_performance_block: return False
        for name in PERFORMANCE_NAMES: LIST_MAP[name].append(current_performance_block.get(name, float('nan')))
        current_performance_block.clear()
        return True
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if "Evaluation result of iter" in line: finalize_and_save_performance_block()
                perf_match = performance_pattern.match(line.strip())
                if perf_match and perf_match.group(1).strip().replace(' ', '_') in PERFORMANCE_NAMES:
                    name = perf_match.group(1).strip().replace(' ', '_')
                    try: current_performance_block[name] = float(perf_match.group(2).strip())
                    except ValueError: pass
            finalize_and_save_performance_block()
    except FileNotFoundError: print(f"错误：文件 '{filename}' 未找到。"); return None
    return (ugb_list, pm_list, gdb_list, gm_list, i_opa_list, ta_list)

def calculate_objective_score(pm, gain, gm, iopa, ugb, area, BASELINE_UGB, BASELINE_AREA):
    # ... (no changes needed)
    if isnan(pm) or isinf(pm) or isnan(gain) or isinf(gain): return 1e12 
    pm_safe, gain_safe, gm_safe, iopa_safe = (pm, gain, gm, iopa)
    if isnan(pm_safe) or isinf(pm_safe): pm_safe = -180.0
    if isnan(gain_safe) or isinf(gain_safe): gain_safe = -200.0
    if isnan(gm_safe) or isinf(gm_safe): gm_safe = 100.0
    if isnan(iopa_safe) or isinf(iopa_safe): iopa_safe = 1.0
    if pm_safe <= -180.0 or gain_safe <= -200.0 or gm_safe >= 100.0: return 1e12
    iopa_ma = iopa_safe
    pm_viol = max(0, (50.0 - pm_safe) / 50.0); gain_viol = max(0, (80.0 - gain_safe) / 80.0)
    gm_viol = max(0, (gm_safe - (-10.0)) / abs(-10.0)); iopa_viol = max(0, (iopa_ma - 3.0) / 3.0)
    total_violation = pm_viol + gain_viol + gm_viol + iopa_viol
    if total_violation > 0: return 1e9 + total_violation * 1e6
    else:
        ugb_safe = ugb if not (isnan(ugb) or isinf(ugb)) else 0
        area_safe = area if not (isnan(area) or isinf(area)) else float('inf')
        if BASELINE_UGB == 0: BASELINE_UGB = 1.0
        if BASELINE_AREA == 0: BASELINE_AREA = 1.0
        ugb_norm = ugb_safe / BASELINE_UGB
        area_norm = area_safe / BASELINE_AREA
        return 0.5 * area_norm - 0.5 * ugb_norm

# --- 【核心修改】: plot_performance_data 函数 ---
def plot_performance_data(results, output_dir="performance_plots", start_iter=1, end_iter=None):
    if not results or len(results[0]) == 0:
        print("没有可用的数据，跳过绘图。")
        return

    # 1. 获取基准值 (逻辑不变)
    ugb_list_full, _, _, _, _, area_list_full = results
    BASELINE_UGB = ugb_list_full[0] if len(ugb_list_full) > 0 and not (isnan(ugb_list_full[0]) or isinf(ugb_list_full[0])) else 1.0
    BASELINE_AREA = area_list_full[0] if len(area_list_full) > 0 and not (isnan(area_list_full[0]) or isinf(area_list_full[0])) else 1.0
    
    # 2. 数据切片和得分计算 (逻辑不变)
    full_length = len(results[0])
    end_iter_safe = full_length if end_iter is None or end_iter > full_length else end_iter
    start_index = max(0, start_iter - 1)
    end_index = end_iter_safe
    if start_index >= end_index: return

    data_sliced = [data[start_index:end_index] for data in results]
    num_iterations = len(data_sliced[0])
    iterations_full_sliced = np.arange(start_iter, start_iter + num_iterations)
    
    # 2.1 计算所有数据的目标得分
    objective_score_list = []
    for ugb, pm, gdb, gm, iopa, ta in zip(*data_sliced):
        score = calculate_objective_score(
            pm=pm, gain=gdb, gm=gm, iopa=iopa, ugb=ugb, area=ta,
            BASELINE_UGB=BASELINE_UGB, BASELINE_AREA=BASELINE_AREA
        )
        objective_score_list.append(score)

    # 3. 【核心修改】: 筛选掉得分（惩罚）超过 1e11 的数据点
    score_array = np.array(objective_score_list, dtype=np.float64)
    # 保留得分 < 1e11 且非 NaN 的数据点
    valid_score_mask = (score_array < 1e11) & ~np.isnan(score_array)
    
    # 3.1 仅保留通过筛选的数据点
    data_filtered = [np.array(data_list)[valid_score_mask] for data_list in data_sliced]
    iterations = iterations_full_sliced[valid_score_mask]
    objective_score_list_filtered = score_array[valid_score_mask]

    # 检查是否有数据点留下
    if len(iterations) == 0:
        print(f"在迭代 {start_iter} 到 {end_iter_safe} 范围内，没有得分低于 1e11 的数据点，跳过绘图。")
        return

    # 4. 准备绘图数据
    performance_data = dict(zip(["UGB", "Phase_Margin", "Gain_db", "Gain_Margin", "I_OPA", "Total_Area"], data_filtered))
    performance_data["Objective_Score"] = objective_score_list_filtered
    
    # 5. 【绘图逻辑最终升级】
    for name, data_list in performance_data.items():
        data_array = np.array(data_list, dtype=np.float64)
        
        if name == "Objective_Score":
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
            fig.suptitle(f"Performance Trend: {name} (Iter {start_iter} to {end_iter_safe}) - Score < 1e11") # 标题更新
            
            # --- 子图1: 可行解 (Feasible Solutions) ---
            feasible_mask = (data_array < 1e9) & ~np.isnan(data_array)
            if np.any(feasible_mask):
                feasible_iters = iterations[feasible_mask]
                feasible_scores = data_array[feasible_mask]
                ax1.plot(feasible_iters, feasible_scores, 'o-', color='g', label='Feasible (Quality Score)')
                
                # 自动调整Y轴以确保可见性
                min_val = np.min(feasible_scores)
                max_val = np.max(feasible_scores)
                margin = (max_val - min_val) * 0.2 + 0.1 # 增加一点边距
                ax1.set_ylim(min_val - margin, max_val + margin)
            else:
                ax1.text(0.5, 0.5, 'No Feasible Solutions Found (< 1e9)', 
                                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

            ax1.set_ylabel("Quality Score")
            ax1.grid(True, linestyle='--')
            ax1.legend()
            
            # --- 子图2: 不可行解 (Infeasible Solutions) ---
            # 这里的不可行解是那些 < 1e11 但 >= 1e9 的点
            infeasible_mask = (data_array >= 1e9) & ~np.isnan(data_array)
            if np.any(infeasible_mask):
                ax2.plot(iterations[infeasible_mask], data_array[infeasible_mask], 'o-', color='r', label='Infeasible (Penalty < 1e11)', markersize=4)
            ax2.set_ylabel("Penalty Score (Cost)")
            ax2.grid(True, linestyle='--')
            ax2.legend()
            ax2.set_xlabel("Iteration (iter xxx)")
            
        else:
            # --- 其他性能指标的绘图 (已过滤掉分数 >= 1e11 的点) ---
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"Performance Trend: {name} (Iter {start_iter} to {end_iter_safe}) - Score < 1e11") # 标题更新
            valid_mask = ~np.isinf(data_array) & ~np.isnan(data_array)
            
            plot_configs = {"UGB": {"log": True}, "I_OPA": {"log": True}, "Total_Area": {"log": True}}
            if name in plot_configs and plot_configs[name].get("log"):
                valid_mask &= (data_array > 0)
            
            if np.count_nonzero(valid_mask) == 0: plt.close(fig); continue

            ax.plot(iterations[valid_mask], data_array[valid_mask], 'o-', markersize=4)
            if name in plot_configs and plot_configs[name].get("log"): ax.set_yscale('log')
            ax.set_xlabel("Iteration (iter xxx)")
            ax.set_ylabel(name)
            ax.grid(True, linestyle='--')

        # 保存图片
        range_suffix = f"_{start_iter}_to_{end_iter_safe}" if start_iter != 1 or end_iter_safe != full_length else ""
        plot_output_dir = output_dir + range_suffix
        if not os.path.exists(plot_output_dir): os.makedirs(plot_output_dir)
        plot_filename = os.path.join(plot_output_dir, f"{name}_trend{range_suffix}.png")
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"图表已保存: {plot_filename}")

# 主执行部分保持不变
if __name__ == "__main__":
    results = extract_performance_data("tmp6_2.1.txt")
    if results:
        plot_performance_data(results, output_dir="performance_plots_v6_2.1")