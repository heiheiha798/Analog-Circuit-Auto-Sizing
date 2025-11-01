import re
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from math import isinf, isnan

def extract_performance_data(filename="results_1.0.0_1.txt"):
    """
    从单个结果文件中解析性能数据。
    此函数与您提供的版本保持不变。
    """
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
                    try: 
                        current_performance_block[name] = float(perf_match.group(2).strip())
                    except ValueError: 
                        pass
            finalize_and_save_performance_block()
    except FileNotFoundError: 
        print(f"错误：文件 '{filename}' 未找到。")
        return None
        
    return (ugb_list, pm_list, gdb_list, gm_list, i_opa_list, ta_list)


def generate_case1_composite_plot(results, output_dir="paper_plots_case1", start_iter=1, end_iter=None):
    """
    【全新】为 Case 1 生成与 Case 2 风格一致的 2x3 组合图。
    """
    if not results or len(results[0]) == 0:
        print("没有可用的数据，跳过绘图。")
        return

    # 解包所有六个指标
    ugb_list, pm_list, gdb_list, gm_list, iopa_list, ta_list = results
    # iopa_list = [i / 1000 for i in iopa_list]  # 转换为 mA
    # 确定数据范围
    full_length = len(ugb_list)
    end_iter_safe = full_length if end_iter is None or end_iter > full_length else end_iter
    start_index = max(0, start_iter - 1)
    end_index = end_iter_safe
    
    if start_index >= end_index:
        print("开始迭代必须小于结束迭代。")
        return

    # 切片数据
    iterations = np.arange(start_iter, end_iter_safe + 1)
    ugb_data = np.array(ugb_list[start_index:end_index])
    pm_data = np.array(pm_list[start_index:end_index])
    gdb_data = np.array(gdb_list[start_index:end_index])
    gm_data = np.array(gm_list[start_index:end_index])
    iopa_data = np.array(iopa_list[start_index:end_index])
    area_data = np.array(ta_list[start_index:end_index])
    
    # --- 设置与 Case 2 完全一致的专业绘图风格 ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 21,
        'axes.labelsize': 22.5,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16.5,
        'figure.figsize': (18, 9),
    })

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axs = plt.subplots(2, 3, constrained_layout=True)

    # 颜色定义
    data_color = '#1f77b4'
    constraint_color = '#d62728'

    # 1. DC Gain Plot
    ax = axs[0, 0]
    ax.plot(iterations, gdb_data, '.', markersize=2, color=data_color)
    ax.axhline(y=80.0, color=constraint_color, linestyle='--', label='Constraint ≥ 80dB')
    ax.set_ylabel('DC Gain (dB)')
    ax.set_title('DC Gain - Simulation Iteration')

    # 2. Gain Margin Plot
    ax = axs[0, 1]
    ax.plot(iterations, gm_data, '.', markersize=2, color=data_color)
    ax.axhline(y=-10.0, color=constraint_color, linestyle='--', label='Constraint ≤ -10dB')
    ax.set_ylabel('Gain Margin (dB)')
    ax.set_title('Gain Margin - Simulation Iteration')

    # 3. Phase Margin Plot
    ax = axs[0, 2]
    ax.plot(iterations, pm_data, '.', markersize=2, color=data_color)
    ax.axhline(y=50.0, color=constraint_color, linestyle='--', label='Constraint ≥ 50°')
    ax.set_ylabel('Phase Margin (°)')
    ax.set_title('Phase Margin - Simulation Iteration')

    # 4. Current Plot
    ax = axs[1, 0]
    ax.plot(iterations, iopa_data, '.', markersize=2, color=data_color)
    ax.axhline(y=3.0, color=constraint_color, linestyle='--', label='Constraint ≤ 3mA')
    ax.set_ylabel('Current (mA)')
    ax.set_title('Static Current - Simulation Iteration')

    # 5. Area Plot (Objective)
    ax = axs[1, 1]
    ax.plot(iterations, area_data, '.', markersize=2, color=data_color)
    ax.set_ylabel('Area (μm²)')
    ax.set_title('Area - Simulation Iteration')
    ax.set_yscale('log')

    # 6. UGB Plot (Objective)
    ax = axs[1, 2]
    ax.plot(iterations, [u / 1e6 for u in ugb_data], '.', markersize=2, color=data_color) # Convert to MHz
    ax.set_ylabel('UGB (MHz)')
    ax.set_title('UGB - Simulation Iteration')

    # --- 添加全局元素 ---
    axes_with_legends = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0]]
    for ax_row in axs:
        for ax in ax_row:
            ax.grid(True, linestyle=':', alpha=0.6)
            if ax in axes_with_legends:
                ax.legend(loc='best')
            ax.set_xlim(0, end_iter_safe + 1)
    
    # 保存文件
    filename = os.path.join(output_dir, f'case1_composite_plot_iter_{start_iter}_to_{end_iter_safe}.png')
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Case 1 的组合图表已保存至: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot performance trends from a single results file in a 2x3 composite view.")
    parser.add_argument('-f', '--file', default='results_1.0.0_1.txt', help='Results filename to read (default: results_1.0.0_1.txt)')
    parser.add_argument('-s', '--start-iter', type=int, default=1, help='Start iteration (1-based, inclusive).')
    parser.add_argument('-e', '--end-iter', type=int, default=None, help='End iteration (1-based, inclusive). If omitted, use end of file.')
    parser.add_argument('-o', '--output-dir', default='paper_plots_case1', help='Output directory for the plot.')
    args = parser.parse_args()

    if args.start_iter is None or args.start_iter < 1:
        print('错误：start-iter 必须是 >= 1 的整数。')
        sys.exit(1)
    if args.end_iter is not None and args.end_iter < args.start_iter:
        print('错误：end-iter 必须大于或等于 start-iter（或留空使用到结尾）。')
        sys.exit(1)

    results = extract_performance_data(args.file)
    if results:
        # 调用全新升级的组合绘图函数
        generate_case1_composite_plot(results, output_dir=args.output_dir, start_iter=args.start_iter, end_iter=args.end_iter)
