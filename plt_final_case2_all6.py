import re
import matplotlib.pyplot as plt
import numpy as np
import os
from math import isinf, isnan

def parse_log_file(filename):
    """
    一个辅助解析函数，用于从单个日志文件中提取性能数据。
    返回一个包含六个列表的元组。
    """
    ugb_list, pm_list, gdb_list, gm_list, i_opa_list, ta_list = [], [], [], [], [], []
    
    # 使用更灵活的正则表达式来捕获迭代信息和性能数据
    sim_count_pattern = re.compile(r"Sim\s+(\d+)")
    score_pattern = re.compile(r"Score:\s*([\d\.\-Ee+infnan]+)")
    perf_pattern = re.compile(r"(\w+)\s*:\s*([\d\.\-Ee+infnan]+)")

    current_block_data = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # 当我们看到 "Score:" 时，认为上一个数据块结束，可以保存了
            if score_pattern.search(line):
                if len(current_block_data) >= 6: # 确保收集了所有6个指标
                    ugb_list.append(current_block_data.get('UGB', float('nan')))
                    pm_list.append(current_block_data.get('Phase_Margin', float('nan')))
                    gdb_list.append(current_block_data.get('Gain_db', float('nan')))
                    gm_list.append(current_block_data.get('Gain_Margin', float('nan')))
                    i_opa_list.append(current_block_data.get('I_OPA', float('nan')))
                    ta_list.append(current_block_data.get('Total_Area', float('nan')))
                current_block_data = {} # 重置数据块

            # 匹配性能指标
            match = perf_pattern.search(line)
            if match:
                key = match.group(1).strip()
                try:
                    value = float(match.group(2).strip())
                    current_block_data[key] = value
                except (ValueError, TypeError):
                    pass # 忽略无法转换的行

    # 处理文件末尾的最后一个数据块
    if len(current_block_data) >= 6:
        ugb_list.append(current_block_data.get('UGB', float('nan')))
        pm_list.append(current_block_data.get('Phase_Margin', float('nan')))
        gdb_list.append(current_block_data.get('Gain_db', float('nan')))
        gm_list.append(current_block_data.get('Gain_Margin', float('nan')))
        i_opa_list.append(current_block_data.get('I_OPA', float('nan')))
        ta_list.append(current_block_data.get('Total_Area', float('nan')))

    # 第一个数据点是 baseline，可能格式不同，这里简单处理，如果解析不完整就跳过
    # Pass 1 第一个点是 baseline，后续数据点在 Score: 出现后才记录，所以会少一个点
    # 为了对齐，我们可以在开头补一个nan，或者直接忽略第一个点
    # 从日志看，第一个'Score:'行之前的数据就是第一个点的数据，所以逻辑应该正确
    
    return ugb_list, pm_list, gdb_list, gm_list, i_opa_list, ta_list


def extract_combined_performance_data(pass1_file, pass2_file):
    """
    解析 Pass 1 和 Pass 2 的日志文件，并将它们合并。
    返回合并后的性能数据列表和 Pass 1 的结束迭代点。
    """
    # 解析 Pass 1
    p1_results = parse_log_file(pass1_file)
    p1_len = len(p1_results[0])
    
    # 解析 Pass 2
    p2_results = parse_log_file(pass2_file)
    
    # 合并数据
    combined_results = tuple(p1 + p2 for p1, p2 in zip(p1_results, p2_results))
    
    # Pass 1 的切换点就是 Pass 1 的数据点数量
    transition_point = p1_len
    
    print(f"数据解析完成。Pass 1 包含 {p1_len} 个数据点。")
    print(f"Pass 2 包含 {len(p2_results[0])} 个数据点。")
    print(f"总计 {len(combined_results[0])} 个数据点。切换点在迭代 {transition_point} 之后。")
    
    return combined_results, transition_point


def generate_case2_story_plot(results, transition_iter, output_dir="paper_plots_case2"):
    """
    为 Case 2 生成“故事板”式的 2x3 组合图。
    """
    if not results or len(results[0]) == 0:
        print("没有可用的数据，跳过绘图。")
        return

    ugb_list, pm_list, gdb_list, gm_list, iopa_list, ta_list = results
    total_iters = len(ugb_list)
    iterations = np.arange(1, total_iters + 1)

    # --- 设置专业绘图风格 ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 21,
        'axes.labelsize': 22.5,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16.5,
        'figure.figsize': (18, 9), # 大尺寸以容纳 2x3 布局和标注
    })

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axs = plt.subplots(2, 3, constrained_layout=True)
    # fig.suptitle('Case 2 Optimization Process: Demonstrating the Two-Pass Strategy', fontsize=20, weight='bold')

    # 颜色定义
    data_color = '#1f77b4'  # 统一的数据点颜色
    constraint_color = '#d62728' # 统一的约束线颜色
    transition_color = '#2ca02c' # 分隔线颜色

    # 1. DC Gain Plot (The Core Story)
    ax = axs[0, 0]
    ax.plot(iterations, gdb_list, '.', markersize=2, color=data_color)
    # 动态约束线
    ax.hlines(80.0, 1, transition_iter, color=constraint_color, linestyle='--', label='Strict Goal (≥ 80dB)')
    ax.hlines(50.0, transition_iter, total_iters, color=constraint_color, linestyle='--', label='Relaxed Goal (≥ 50dB)')
    ax.set_ylabel('DC Gain (dB)')
    ax.set_title('DC Gain - Simulation Iteration')

    # 2. Gain Margin Plot
    ax = axs[0, 1]
    ax.plot(iterations, gm_list, '.', markersize=2, color=data_color)
    ax.axhline(y=-10.0, color=constraint_color, linestyle='--', label='Constraint ≤ -10dB')
    ax.set_ylabel('Gain Margin (dB)')
    ax.set_title('Gain Margin - Simulation Iteration')

    # 3. Phase Margin Plot
    ax = axs[0, 2]
    ax.plot(iterations, pm_list, '.', markersize=2, color=data_color)
    ax.axhline(y=50.0, color=constraint_color, linestyle='--', label='Constraint ≥ 50°')
    ax.set_ylabel('Phase Margin (°)')
    ax.set_title('Phase Margin - Simulation Iteration')

    # 4. Current Plot
    ax = axs[1, 0]
    ax.plot(iterations, [i * 1000 for i in iopa_list], '.', markersize=2, color=data_color) # Convert to mA
    ax.axhline(y=3.0, color=constraint_color, linestyle='--', label='Constraint ≤ 3mA')
    ax.set_ylabel('Current (mA)')
    ax.set_title('Static Current - Simulation Iteration')
    # ax.set_yscale('log')

    # 5. Area Plot (Objective)
    ax = axs[1, 1]
    ax.plot(iterations, ta_list, '.', markersize=2, color=data_color)
    ax.set_ylabel('Area (μm²)')
    ax.set_title('Area - Simulation Iteration')
    ax.set_yscale('log')

    # 6. UGB Plot (Objective)
    ax = axs[1, 2]
    ax.plot(iterations, [u / 1e6 for u in ugb_list], '.', markersize=2, color=data_color) # Convert to MHz
    ax.set_ylabel('UGB (MHz)')
    ax.set_title('UGB - Simulation Iteration')

    # --- 添加全局元素 ---
    # 存储有图例的坐标轴
    axes_with_legends = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0]]

    for ax_row in axs:
        for ax in ax_row:
            ax.axvline(x=transition_iter, color=transition_color, linestyle=':', linewidth=2)
            # ax.set_xlabel('Simulation Iteration')
            ax.grid(True, linestyle=':', alpha=0.6)
            # 只有在图例不为空时才显示
            if ax in axes_with_legends:
                ax.legend(loc='best')
            # 设置x轴范围
            ax.set_xlim(0, total_iters + 1)
    
    # 在图的顶部添加 Pass 1 和 Pass 2 的文本标注
    # fig.text(0.25, 0.95, 'Pass 1: Strict Goal (Failed)', ha='center', va='center', fontsize=14, color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))
    # fig.text(0.75, 0.95, 'Pass 2: Relaxed Goal (Success)', ha='center', va='center', fontsize=14, color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))
    
    # 保存文件
    filename = os.path.join(output_dir, 'case2_full_story_plot.png')
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Case 2 的“故事板”图表已保存至: {filename}")


if __name__ == "__main__":
    pass1_log = "optimization_log_Pass_1_(Strict_Goal_80dB).txt"
    pass2_log = "optimization_log_Pass_2_(Relaxed_Goal_50.0dB).txt"
    
    # 检查文件是否存在
    if not os.path.exists(pass1_log) or not os.path.exists(pass2_log):
        print(f"错误：请确保 '{pass1_log}' 和 '{pass2_log}' 文件在当前目录下。")
    else:
        # 1. 解析并合并数据
        combined_data, transition_point = extract_combined_performance_data(pass1_log, pass2_log)
        
        # 2. 生成并保存最终的图表
        generate_case2_story_plot(combined_data, transition_point)
