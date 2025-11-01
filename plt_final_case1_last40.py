import re
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from math import isinf, isnan

# extract_performance_data 和 calculate_objective_score 函数保持不变，因为它们是数据提取的基础
def extract_performance_data(filename="results_1.0.0_1.txt"):
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

# ==============================================================================
# ======================== 新的、更优雅的绘图函数 ============================
# ==============================================================================
def plot_score_pm_summary(results, output_dir="paper_plots", start_iter=390, end_iter=None):
    """
    生成一张结合了性能分数(Score)和相位裕度(PM)的收敛总结图。
    该图更优雅、更清晰，解决了图例重叠和颜色辨识度低的问题。
    """
    if not results or len(results[0]) == 0:
        print("没有可用的数据，跳过绘图。")
        return

    ugb_list, pm_list, _, _, _, ta_list = results
    
    # --- 数据范围处理 ---
    full_length = len(ugb_list)
    end_iter_safe = full_length if end_iter is None or end_iter > full_length else end_iter
    start_index = max(0, start_iter - 1)
    end_index = end_iter_safe
    
    if start_index >= end_index:
        print(f"开始迭代 {start_iter} 必须小于结束迭代 {end_iter_safe}。")
        return

    iterations = np.arange(start_iter, end_iter_safe + 1)
    ugb = np.array(ugb_list[start_index:end_index])
    pm = np.array(pm_list[start_index:end_index])
    area = np.array(ta_list[start_index:end_index])

    # --- 1. 定义评分系统所需常量 ---
    BASELINE_UGB = 274876000
    BASELINE_AREA = 10764.92025
    
    # --- 2. 计算新的性能分数 ---
    # 根据您提供的公式: score = const * (ugb_norm*15 - area_norm*25)
    # 我们这里省略 const，因为它只影响Y轴的绝对值，不影响曲线形状
    with np.errstate(divide='ignore', invalid='ignore'): # 避免除以0或nan的警告
        ugb_norm = ugb / BASELINE_UGB
        area_norm = area / BASELINE_AREA
        score = (ugb_norm * 15 - area_norm * 25)
    
    # --- 核心美学设定 ---
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 15,
        'axes.labelsize': 16.5,
        'axes.titlesize': 18,
        'xtick.labelsize': 13.5,
        'ytick.labelsize': 13.5,
        'legend.fontsize': 13.5,
        'figure.figsize': (8, 6), # 适合单个图的尺寸
        'lines.linewidth': 1.5,
        'lines.markersize': 4, # 保持小圆点尺寸
    })
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax1 = plt.subplots()

    # --- 3. 核心美学提升：选择高对比度颜色 ---
    color_pm = '#0072B2'     # 专业蓝
    color_score = '#D55E00'  # 活力橙

    # --- 4. 左轴: Phase Margin (关键约束) ---
    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Phase Margin (°)', color=color_pm)
    line_pm = ax1.plot(iterations, pm, marker='o', linestyle='-', color=color_pm, label='Phase Margin (PM)')
    ax1.tick_params(axis='y', labelcolor=color_pm)
    ax1.grid(True, linestyle=':', alpha=0.6)
    # 约束红线依然重要
    line_constraint = ax1.axhline(y=50, color='r', linestyle='--', linewidth=2, label='PM Constraint ≥ 50°')

    # --- 右轴: Performance Score (综合性能) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Performance Score (UGB↑, Area↓)', color=color_score)
    line_score = ax2.plot(iterations, score, marker='o', linestyle='-', color=color_score, label='Performance Score')
    ax2.tick_params(axis='y', labelcolor=color_score)

    # --- 5. 优雅的图例处理：移至图表上方，避免重叠 ---
    # 合并两个轴的图例项
    lines = line_pm + line_score
    # 手动加入约束线到图例中
    lines.append(line_constraint)
    labels = [l.get_label() for l in lines]
    # 将图例放在图的顶部中心，ncol决定图例有几列
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    
    # plt.title(f'Optimization Convergence Summary (Iterations {start_iter}-{end_iter_safe})', y=1.08)
    
    # 调整布局以确保所有元素（如图例和标题）都可见
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect为标题和图例留出顶部空间

    filename = os.path.join(output_dir, f'elegant_convergence_summary_iter_{start_iter}_to_{end_iter_safe}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"优雅版收敛总结图已保存至: {filename}")

# 主执行部分
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot performance trends from results file.")
    parser.add_argument('-f', '--file', default='results_1.0.0_1.txt', help='Results filename')
    parser.add_argument('-s', '--start-iter', type=int, default=390, help='Start iteration')
    parser.add_argument('-e', '--end-iter', type=int, default=None, help='End iteration')
    parser.add_argument('-o', '--output-dir', default='paper_plots', help='Output directory')
    args = parser.parse_args()

    results = extract_performance_data(args.file)
    if results:
        # 直接调用新的、更优雅的绘图函数
        plot_score_pm_summary(
            results, 
            output_dir=args.output_dir, 
            start_iter=args.start_iter, 
            end_iter=args.end_iter
        )
