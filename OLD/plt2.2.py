import re
import matplotlib.pyplot as plt
import numpy as np
import os
from math import isinf, isnan

# extract_performance_data 函数保持不变
def extract_performance_data(filename="tmp3.txt"):
    # ... (此函数无需修改)
    ugb_list, pm_list, gdb_list, gm_list, i_opa_list, ta_list = [], [], [], [], [], []
    performance_pattern = re.compile(r"======\s*(\w+[\s_]*\w*)\s*:\s*([\d\.\-Ee+infnan]+)")
    PERFORMANCE_NAMES = ["UGB", "Phase_Margin", "Gain_db", "Gain_Margin", "I_OPA", "Total_Area"]
    current_performance_block = {}
    LIST_MAP = {"UGB": ugb_list, "Phase_Margin": pm_list, "Gain_db": gdb_list, "Gain_Margin": gm_list, "I_OPA": i_opa_list, "Total_Area": ta_list}

    def finalize_and_save_performance_block():
        if not current_performance_block: return False
        for name in PERFORMANCE_NAMES:
            LIST_MAP[name].append(current_performance_block.get(name, float('nan')))
        current_performance_block.clear()
        return True

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if "Evaluation result of iter" in line:
                    finalize_and_save_performance_block()
                perf_match = performance_pattern.match(line.strip())
                if perf_match and perf_match.group(1).strip().replace(' ', '_') in PERFORMANCE_NAMES:
                    name = perf_match.group(1).strip().replace(' ', '_')
                    try:
                        current_performance_block[name] = float(perf_match.group(2).strip())
                    except ValueError: pass
            finalize_and_save_performance_block()
    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 未找到。")
        return None
    return (ugb_list, pm_list, gdb_list, gm_list, i_opa_list, ta_list)

# --- 【核心修正】: calculate_objective_score 函数 ---
# 这个版本现在与优化器中的 get_score 函数完全一致
def calculate_objective_score(pm, gain, gm, iopa, ugb, area, BASELINE_UGB, BASELINE_AREA):
    """
    根据优化器内的真实逻辑计算得分/成本。
    """
    # 检查基本有效性
    if isnan(pm) or isinf(pm) or isnan(gain) or isinf(gain):
         return 1e12 
         
    # a. 使用物理上有意义的默认值来“消毒”
    pm_safe = pm if not (isnan(pm) or isinf(pm)) else -180.0
    gain_safe = gain if not (isnan(gain) or isinf(gain)) else -200.0
    gm_safe = gm if not (isnan(gm) or isinf(gm)) else 100.0
    iopa_safe = iopa if not (isnan(iopa) or isinf(iopa)) else 1.0 # 1A = 1000mA
    
    # b. 检查仿真是否收敛/失败
    if pm_safe <= -180.0 or gain_safe <= -200.0 or gm_safe >= 100.0:
        return 1e12

    # c. 计算归一化的约束偏离度
    iopa_ma = iopa_safe * 1000.0
    pm_viol = max(0, (50.0 - pm_safe) / 50.0)           
    gain_viol = max(0, (80.0 - gain_safe) / 80.0)         
    gm_viol = max(0, (gm_safe - (-10.0)) / abs(-10.0))
    iopa_viol = max(0, (iopa_ma - 3.0) / 3.0)

    total_violation = pm_viol + gain_viol + gm_viol + iopa_viol

    # d. 根据约束偏离度决定成本
    if total_violation > 0:
        # 不可行解：返回惩罚
        return 1e9 + total_violation * 1e6
    else:
        # e. 可行解：返回质量分
        ugb_safe = ugb if not (isnan(ugb) or isinf(ugb)) else 0
        area_safe = area if not (isnan(area) or isinf(area)) else float('inf')
        
        # 避免除以0
        if BASELINE_UGB == 0: BASELINE_UGB = 1.0
        if BASELINE_AREA == 0: BASELINE_AREA = 1.0

        ugb_norm = ugb_safe / BASELINE_UGB
        area_norm = area_safe / BASELINE_AREA
        
        return 0.5 * area_norm - 0.5 * ugb_norm

# plot_performance_data 函数保持我们上一版的逻辑不变
def plot_performance_data(results, output_dir="performance_plots", start_iter=1, end_iter=None):
    # ... (此函数无需修改，因为它会调用上面修正后的新函数) ...
    if not results or len(results[0]) == 0:
        print("没有可用的数据，跳过绘图。")
        return
    ugb_list_full, _, _, _, _, area_list_full = results
    if len(ugb_list_full) > 0 and not (isnan(ugb_list_full[0]) or isinf(ugb_list_full[0])):
        BASELINE_UGB = ugb_list_full[0]
    else:
        BASELINE_UGB = 1.0
        print("[警告] 无法从 iter 1 获取有效的 UGB 基准值。")
    if len(area_list_full) > 0 and not (isnan(area_list_full[0]) or isinf(area_list_full[0])):
        BASELINE_AREA = area_list_full[0]
    else:
        BASELINE_AREA = 1.0
        print("[警告] 无法从 iter 1 获取有效的 Area 基准值。")
    print(f"\n--- 使用从 Iter 1 提取的基准值进行得分计算 ---")
    print(f"  - BASELINE_UGB: {BASELINE_UGB:.2f}, BASELINE_AREA: {BASELINE_AREA:.2f}")

    full_length = len(results[0])
    end_iter_safe = full_length if end_iter is None or end_iter > full_length else end_iter
    start_index = max(0, start_iter - 1)
    end_index = end_iter_safe
    if start_index >= end_index:
        print(f"警告：指定的迭代范围 ({start_iter} 到 {end_iter_safe}) 无效。")
        return

    data_sliced = [data[start_index:end_index] for data in results]
    ugb_list, pm_list, gdb_list, gm_list, i_opa_list, ta_list = data_sliced
    num_iterations = len(ugb_list)
    iterations = np.arange(start_iter, start_iter + num_iterations)

    objective_score_list = [calculate_objective_score(pm, gdb, gm, iopa, ugb, ta, BASELINE_UGB, BASELINE_AREA)
                           for pm, gdb, gm, iopa, ugb, ta in zip(*data_sliced)]

    performance_data = {"UGB": ugb_list, "Phase_Margin": pm_list, "Gain_db": gdb_list,
                        "Gain_Margin": gm_list, "I_OPA": i_opa_list, "Total_Area": ta_list,
                        "Objective_Score": objective_score_list}
    
    plot_configs = {"UGB": {"log_scale": True, "ylabel": "UGB (Hz)"},
                    "Phase_Margin": {"log_scale": False, "ylabel": "Phase Margin (deg)"},
                    "Gain_db": {"log_scale": False, "ylabel": "Gain (dB)"},
                    "Gain_Margin": {"log_scale": False, "ylabel": "Gain Margin (dB)"},
                    "I_OPA": {"log_scale": True, "ylabel": "I_OPA (A)"},
                    "Total_Area": {"log_scale": True, "ylabel": "Total Area ($\\mu m^2$)"}, 
                    "Objective_Score": {"log_scale": False, "ylabel": "Objective Score (Cost)", "color": 'r'}}
    
    range_suffix = f"_{start_iter}_to_{end_iter_safe}" if start_iter != 1 or end_iter_safe != full_length else ""
    plot_output_dir = output_dir + range_suffix
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)
    
    print(f"\n--- 开始生成图表 (迭代 {start_iter} 到 {end_iter_safe}) ---")

    for name, data_list in performance_data.items():
        config = plot_configs.get(name, {})
        plt.figure(figsize=(10, 6))
        data_array = np.array(data_list, dtype=np.float64)
        valid_mask = ~np.isinf(data_array) & ~np.isnan(data_array)
        if config.get("log_scale"): valid_mask &= (data_array > 0)
        if np.count_nonzero(valid_mask) == 0:
            print(f"参数 {name} 在当前范围无有效数据点，跳过绘图。")
            plt.close()
            continue
        plt.plot(iterations[valid_mask], data_array[valid_mask], marker='o', linestyle='-', color=config.get("color", 'b'), markersize=4)
        plt.title(f"Performance Trend: {name} (Iter {start_iter} to {end_iter_safe})")
        plt.xlabel("Iteration (iter xxx)")
        plt.ylabel(config.get("ylabel", name))
        plt.grid(True, linestyle='--', alpha=0.7)
        if config.get("log_scale"): plt.yscale('log')
        tick_interval = max(1, num_iterations // 10 if num_iterations > 10 else 1)
        plt.xticks(np.arange(start_iter, end_iter_safe + 1, tick_interval).astype(int))
        plot_filename = os.path.join(plot_output_dir, f"{name}_trend{range_suffix}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"图表已保存: {plot_filename}")

    print("--- 所有图表生成完毕 ---")

if __name__ == "__main__":
    results = extract_performance_data("tmp3.txt")
    if results:
        plot_performance_data(results) 
        plot_performance_data(results, start_iter=62, end_iter=64)