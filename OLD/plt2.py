import re
import matplotlib.pyplot as plt
import numpy as np

data_string = """

[baseline]:
  UGB: 394.319 MHz, PM: 56.05 deg, Gain: 86.00 dB, I_OPA: 4.4764 mA, Area: 10764.92 um^2

[SymmetricGroup_NMOS_NM5_fw_-10%]:
  UGB: 393.376 MHz, PM: 56.24 deg, Gain: 86.03 dB, I_OPA: 4.4756 mA, Area: 10751.64 um^2

[SymmetricGroup_NMOS_NM5_fw_+10%]:
  UGB: 395.141 MHz, PM: 55.88 deg, Gain: 85.96 dB, I_OPA: 4.4771 mA, Area: 10778.20 um^2

[SymmetricGroup_NMOS_NM18_fw_-10%]:
  UGB: 390.487 MHz, PM: 56.45 deg, Gain: 86.05 dB, I_OPA: 4.4741 mA, Area: 10738.36 um^2

[SymmetricGroup_NMOS_NM18_fw_+10%]:
  UGB: 404.580 MHz, PM: 55.63 deg, Gain: 85.83 dB, I_OPA: 4.4792 mA, Area: 10791.49 um^2

[SymmetricGroup_NMOS_NM20_fw_-10%]:
  UGB: 398.126 MHz, PM: 56.40 deg, Gain: 85.87 dB, I_OPA: 4.4763 mA, Area: 10764.81 um^2

[SymmetricGroup_NMOS_NM20_fw_+10%]:
  UGB: 390.832 MHz, PM: 55.73 deg, Gain: 86.10 dB, I_OPA: 4.4764 mA, Area: 10765.03 um^2

[SymmetricGroup_NMOS_NM12_fw_-10%]:
  UGB: 395.531 MHz, PM: 56.07 deg, Gain: 85.94 dB, I_OPA: 4.4764 mA, Area: 10764.68 um^2

[SymmetricGroup_NMOS_NM12_fw_+10%]:
  UGB: 393.282 MHz, PM: 56.06 deg, Gain: 86.04 dB, I_OPA: 4.4764 mA, Area: 10765.16 um^2

[SymmetricGroup_NMOS_NM23_fw_-10%]:
  UGB: 394.302 MHz, PM: 56.04 deg, Gain: 86.02 dB, I_OPA: 4.4749 mA, Area: 10764.79 um^2

[SymmetricGroup_NMOS_NM23_fw_+10%]:
  UGB: 394.307 MHz, PM: 56.07 deg, Gain: 85.97 dB, I_OPA: 4.4778 mA, Area: 10765.05 um^2

[SymmetricGroup_NMOS_NM24_fw_-10%]:
  UGB: 394.320 MHz, PM: 56.05 deg, Gain: 85.99 dB, I_OPA: 4.4762 mA, Area: 10764.79 um^2

[SymmetricGroup_NMOS_NM24_fw_+10%]:
  UGB: 394.319 MHz, PM: 56.06 deg, Gain: 86.00 dB, I_OPA: 4.4765 mA, Area: 10765.05 um^2

[SymmetricGroup_NMOS_NM2_fw_-10%]:
  UGB: 387.337 MHz, PM: 56.10 deg, Gain: 85.88 dB, I_OPA: 4.4764 mA, Area: 10756.95 um^2

[SymmetricGroup_NMOS_NM2_fw_+10%]:
  UGB: 401.296 MHz, PM: 55.85 deg, Gain: 86.08 dB, I_OPA: 4.4764 mA, Area: 10772.89 um^2

[SymmetricGroup_PMOS_PM21_fw_-10%]:
  UGB: 393.924 MHz, PM: 56.09 deg, Gain: 85.94 dB, I_OPA: 4.4763 mA, Area: 10764.73 um^2

[SymmetricGroup_PMOS_PM21_fw_+10%]:
  UGB: 394.693 MHz, PM: 56.02 deg, Gain: 86.04 dB, I_OPA: 4.4764 mA, Area: 10765.11 um^2

[SymmetricGroup_PMOS_PM16_fw_-10%]:
  UGB: 393.148 MHz, PM: 56.35 deg, Gain: 86.19 dB, I_OPA: 4.4758 mA, Area: 10764.73 um^2

[SymmetricGroup_PMOS_PM16_fw_+10%]:
  UGB: 396.033 MHz, PM: 55.85 deg, Gain: 85.75 dB, I_OPA: 4.4770 mA, Area: 10765.11 um^2

[SymmetricGroup_PMOS_PM24_fw_-10%]:
  UGB: 394.319 MHz, PM: 56.05 deg, Gain: 85.99 dB, I_OPA: 4.4765 mA, Area: 10764.82 um^2

[SymmetricGroup_PMOS_PM24_fw_+10%]:
  UGB: 394.321 MHz, PM: 56.05 deg, Gain: 86.00 dB, I_OPA: 4.4763 mA, Area: 10765.02 um^2

[SymmetricGroup_PMOS_PM25_fw_-10%]:
  UGB: 394.328 MHz, PM: 56.07 deg, Gain: 85.97 dB, I_OPA: 4.4778 mA, Area: 10764.83 um^2

[SymmetricGroup_PMOS_PM25_fw_+10%]:
  UGB: 394.272 MHz, PM: 56.05 deg, Gain: 86.01 dB, I_OPA: 4.4752 mA, Area: 10765.01 um^2

[SymmetricGroup_PMOS_PM0_fw_-10%]:
  UGB: 403.389 MHz, PM: 57.54 deg, Gain: 85.81 dB, I_OPA: 4.4733 mA, Area: 10718.12 um^2

[SymmetricGroup_PMOS_PM0_fw_+10%]:
  UGB: 386.271 MHz, PM: 54.55 deg, Gain: 86.16 dB, I_OPA: 4.4791 mA, Area: 10811.72 um^2

[SymmetricGroup_PMOS_PM6_fw_-10%]:
  UGB: 395.362 MHz, PM: 56.10 deg, Gain: 85.98 dB, I_OPA: 4.4764 mA, Area: 10746.55 um^2

[SymmetricGroup_PMOS_PM6_fw_+10%]:
  UGB: 393.276 MHz, PM: 56.01 deg, Gain: 86.01 dB, I_OPA: 4.4764 mA, Area: 10783.29 um^2

[SymmetricGroup_PMOS_PM3_fw_-10%]:
  UGB: 386.526 MHz, PM: 56.67 deg, Gain: 86.09 dB, I_OPA: 4.4136 mA, Area: 10748.25 um^2

[SymmetricGroup_PMOS_PM3_fw_+10%]:
  UGB: 401.840 MHz, PM: 55.50 deg, Gain: 85.89 dB, I_OPA: 4.5392 mA, Area: 10781.59 um^2

[NM17_fw_-10%]:
  UGB: 394.350 MHz, PM: 56.05 deg, Gain: 86.00 dB, I_OPA: 4.4764 mA, Area: 10762.82 um^2

[NM17_fw_+10%]:
  UGB: 394.293 MHz, PM: 56.05 deg, Gain: 86.00 dB, I_OPA: 4.4764 mA, Area: 10767.02 um^2

[NM16_fw_-10%]:
  UGB: 397.302 MHz, PM: 56.07 deg, Gain: 85.96 dB, I_OPA: 4.4771 mA, Area: 10762.82 um^2

[NM16_fw_+10%]:
  UGB: 391.717 MHz, PM: 56.04 deg, Gain: 86.02 dB, I_OPA: 4.4757 mA, Area: 10767.02 um^2

[NM6_fw_-10%]:
  UGB: 396.977 MHz, PM: 56.21 deg, Gain: 85.98 dB, I_OPA: 4.5179 mA, Area: 10763.59 um^2

[NM6_fw_+10%]:
  UGB: 392.497 MHz, PM: 55.94 deg, Gain: 85.97 dB, I_OPA: 4.4492 mA, Area: 10766.25 um^2

[NM0_fw_-10%]:
  UGB: 394.489 MHz, PM: 56.06 deg, Gain: 86.00 dB, I_OPA: 4.4790 mA, Area: 10763.59 um^2

[NM0_fw_+10%]:
  UGB: 394.188 MHz, PM: 56.05 deg, Gain: 86.00 dB, I_OPA: 4.4744 mA, Area: 10766.25 um^2

[NM8_fw_-10%]:
  UGB: 371.865 MHz, PM: 55.51 deg, Gain: 85.59 dB, I_OPA: 4.0855 mA, Area: 10763.59 um^2

[NM8_fw_+10%]:
  UGB: 421.356 MHz, PM: 56.33 deg, Gain: 86.12 dB, I_OPA: 4.8713 mA, Area: 10766.25 um^2

[NM7_fw_-10%]:
  UGB: 392.676 MHz, PM: 56.01 deg, Gain: 85.98 dB, I_OPA: 4.4481 mA, Area: 10763.59 um^2

[NM7_fw_+10%]:
  UGB: 395.628 MHz, PM: 56.09 deg, Gain: 86.01 dB, I_OPA: 4.4989 mA, Area: 10766.25 um^2

[NM19_fw_-10%]:
  UGB: 435.228 MHz, PM: 55.96 deg, Gain: 85.89 dB, I_OPA: 4.8727 mA, Area: 10763.59 um^2

[NM19_fw_+10%]:
  UGB: 366.029 MHz, PM: 55.58 deg, Gain: 85.75 dB, I_OPA: 4.1423 mA, Area: 10766.25 um^2

[NM15_fw_-10%]:
  UGB: 397.049 MHz, PM: 56.11 deg, Gain: 86.01 dB, I_OPA: 4.5091 mA, Area: 10763.59 um^2

[NM15_fw_+10%]:
  UGB: 392.257 MHz, PM: 56.02 deg, Gain: 85.99 dB, I_OPA: 4.4517 mA, Area: 10766.25 um^2

[NM14_fw_-10%]:
  UGB: 393.524 MHz, PM: 55.98 deg, Gain: 86.03 dB, I_OPA: 4.4744 mA, Area: 10763.59 um^2

[NM14_fw_+10%]:
  UGB: 395.025 MHz, PM: 56.11 deg, Gain: 85.97 dB, I_OPA: 4.4782 mA, Area: 10766.25 um^2

[NM13_fw_-10%]:
  UGB: 393.524 MHz, PM: 55.98 deg, Gain: 86.03 dB, I_OPA: 4.4744 mA, Area: 10763.59 um^2

[NM13_fw_+10%]:
  UGB: 395.025 MHz, PM: 56.11 deg, Gain: 85.97 dB, I_OPA: 4.4782 mA, Area: 10766.25 um^2

[NM11_fw_-10%]:
  UGB: 393.525 MHz, PM: 55.98 deg, Gain: 86.03 dB, I_OPA: 4.4744 mA, Area: 10763.59 um^2

[NM11_fw_+10%]:
  UGB: 395.024 MHz, PM: 56.11 deg, Gain: 85.97 dB, I_OPA: 4.4782 mA, Area: 10766.25 um^2

[NM28_fw_-10%]:
  UGB: 365.654 MHz, PM: 55.85 deg, Gain: 85.61 dB, I_OPA: 4.0807 mA, Area: 10763.59 um^2

[NM28_fw_+10%]:
  UGB: 427.241 MHz, PM: 55.78 deg, Gain: 86.13 dB, I_OPA: 4.8677 mA, Area: 10766.25 um^2

[NM27_fw_-10%]:
  UGB: 424.475 MHz, PM: 55.81 deg, Gain: 86.13 dB, I_OPA: 4.8352 mA, Area: 10764.79 um^2

[NM27_fw_+10%]:
  UGB: 372.471 MHz, PM: 55.88 deg, Gain: 85.72 dB, I_OPA: 4.1722 mA, Area: 10765.05 um^2

[NM10_fw_-10%]:
  UGB: 393.526 MHz, PM: 55.98 deg, Gain: 86.03 dB, I_OPA: 4.4744 mA, Area: 10763.59 um^2

[NM10_fw_+10%]:
  UGB: 395.023 MHz, PM: 56.11 deg, Gain: 85.97 dB, I_OPA: 4.4782 mA, Area: 10766.25 um^2

[NM9_fw_-10%]:
  UGB: 393.601 MHz, PM: 55.99 deg, Gain: 86.03 dB, I_OPA: 4.4746 mA, Area: 10763.59 um^2

[NM9_fw_+10%]:
  UGB: 394.952 MHz, PM: 56.11 deg, Gain: 85.97 dB, I_OPA: 4.4780 mA, Area: 10766.25 um^2

[PM19_fw_-10%]:
  UGB: 395.900 MHz, PM: 56.06 deg, Gain: 85.98 dB, I_OPA: 4.4764 mA, Area: 10759.88 um^2

[PM19_fw_+10%]:
  UGB: 393.018 MHz, PM: 56.05 deg, Gain: 86.01 dB, I_OPA: 4.4764 mA, Area: 10769.96 um^2

[PM15_fw_-10%]:
  UGB: 391.485 MHz, PM: 56.04 deg, Gain: 86.02 dB, I_OPA: 4.4226 mA, Area: 10750.71 um^2

[PM15_fw_+10%]:
  UGB: 397.145 MHz, PM: 56.07 deg, Gain: 85.97 dB, I_OPA: 4.5302 mA, Area: 10779.13 um^2

[PM20_fw_-10%]:
  UGB: 392.848 MHz, PM: 56.05 deg, Gain: 86.01 dB, I_OPA: 4.4756 mA, Area: 10759.88 um^2

[PM20_fw_+10%]:
  UGB: 395.710 MHz, PM: 56.06 deg, Gain: 85.98 dB, I_OPA: 4.4771 mA, Area: 10769.96 um^2

[PM14_fw_-10%]:
  UGB: 393.879 MHz, PM: 56.03 deg, Gain: 86.00 dB, I_OPA: 4.4676 mA, Area: 10761.88 um^2

[PM14_fw_+10%]:
  UGB: 394.707 MHz, PM: 56.08 deg, Gain: 86.00 dB, I_OPA: 4.4841 mA, Area: 10767.96 um^2

[PM2_fw_-10%]:
  UGB: 386.671 MHz, PM: 54.63 deg, Gain: 85.19 dB, I_OPA: 4.2140 mA, Area: 10697.60 um^2

[PM2_fw_+10%]:
  UGB: 407.895 MHz, PM: 57.38 deg, Gain: 86.62 dB, I_OPA: 4.7390 mA, Area: 10832.24 um^2

[PM13_fw_-10%]:
  UGB: 393.819 MHz, PM: 56.02 deg, Gain: 85.99 dB, I_OPA: 4.4664 mA, Area: 10761.88 um^2

[PM13_fw_+10%]:
  UGB: 394.764 MHz, PM: 56.08 deg, Gain: 86.00 dB, I_OPA: 4.4852 mA, Area: 10767.96 um^2

[PM12_fw_-10%]:
  UGB: 393.816 MHz, PM: 56.02 deg, Gain: 85.99 dB, I_OPA: 4.4663 mA, Area: 10761.88 um^2

[PM12_fw_+10%]:
  UGB: 394.767 MHz, PM: 56.08 deg, Gain: 86.00 dB, I_OPA: 4.4853 mA, Area: 10767.96 um^2

[PM11_fw_-10%]:
  UGB: 393.815 MHz, PM: 56.02 deg, Gain: 85.99 dB, I_OPA: 4.4662 mA, Area: 10761.88 um^2

[PM11_fw_+10%]:
  UGB: 394.768 MHz, PM: 56.08 deg, Gain: 86.00 dB, I_OPA: 4.4853 mA, Area: 10767.96 um^2

[PM10_fw_-10%]:
  UGB: 393.814 MHz, PM: 56.02 deg, Gain: 85.99 dB, I_OPA: 4.4662 mA, Area: 10761.88 um^2

[PM10_fw_+10%]:
  UGB: 394.768 MHz, PM: 56.08 deg, Gain: 86.00 dB, I_OPA: 4.4853 mA, Area: 10767.96 um^2

[PM18_fw_-10%]:
  UGB: 395.770 MHz, PM: 56.10 deg, Gain: 86.02 dB, I_OPA: 4.5007 mA, Area: 10761.88 um^2

[PM18_fw_+10%]:
  UGB: 393.213 MHz, PM: 56.02 deg, Gain: 85.98 dB, I_OPA: 4.4579 mA, Area: 10767.96 um^2

[PM17_fw_-10%]:
  UGB: 424.167 MHz, PM: 56.36 deg, Gain: 86.11 dB, I_OPA: 4.9018 mA, Area: 10761.88 um^2

[PM17_fw_+10%]:
  UGB: 373.699 MHz, PM: 55.54 deg, Gain: 85.63 dB, I_OPA: 4.1287 mA, Area: 10767.96 um^2

[PM9_fw_-10%]:
  UGB: 361.346 MHz, PM: 55.51 deg, Gain: 85.69 dB, I_OPA: 4.0774 mA, Area: 10761.88 um^2

[PM9_fw_+10%]:
  UGB: 434.347 MHz, PM: 55.97 deg, Gain: 85.90 dB, I_OPA: 4.8753 mA, Area: 10767.96 um^2

[PM7_fw_-10%]:
  UGB: 398.927 MHz, PM: 56.39 deg, Gain: 85.81 dB, I_OPA: 4.4769 mA, Area: 10761.88 um^2

[PM7_fw_+10%]:
  UGB: 391.296 MHz, PM: 55.78 deg, Gain: 86.12 dB, I_OPA: 4.4796 mA, Area: 10767.96 um^2

[PM8_fw_-10%]:
  UGB: 430.596 MHz, PM: 55.75 deg, Gain: 86.13 dB, I_OPA: 4.8961 mA, Area: 10761.88 um^2

[PM8_fw_+10%]:
  UGB: 368.492 MHz, PM: 55.86 deg, Gain: 85.66 dB, I_OPA: 4.1276 mA, Area: 10767.96 um^2

"""

# ----------------------------------------------------------------------
# 1. 数据解析与处理
# ----------------------------------------------------------------------

all_data = {}

parameter_patterns = {
    'UGB': (re.compile(r"UGB: ([\d.]+)\s*MHz"), "MHz"),
    'PM': (re.compile(r"PM: ([\d.]+)\s*deg"), "deg"),
    'Gain': (re.compile(r"Gain: ([\d.]+)\s*dB"), "dB"),
    'I_OPA': (re.compile(r"I_OPA: ([\d.]+)\s*mA"), "mA"),
    'Area': (re.compile(r"Area: ([\d.]+)\s*um\^2"), "um^2")
}


# 正则表达式：匹配 [名称]: 行
name_pattern = re.compile(r"^\[(.+?)\]:")

current_name = None
for line in data_string.splitlines():
    line = line.strip()

    # 尝试匹配名称行
    name_match = name_pattern.match(line)
    if name_match:
        current_name = name_match.group(1).strip()
        all_data[current_name] = {} # 初始化当前名称的数据字典
        continue

    # 如果有当前名称，则尝试匹配所有参数
    if current_name:
        for param_name, (pattern, _) in parameter_patterns.items():
            param_match = pattern.search(line)
            if param_match:
                param_value = float(param_match.group(1))
                all_data[current_name][param_name] = param_value
                # 不需要重置 current_name，因为一行可能包含多个参数

# 检查是否解析出 baseline 的所有参数
baseline_name = 'baseline'
if baseline_name not in all_data or len(all_data[baseline_name]) != len(parameter_patterns):
    print(f"错误：未找到 '{baseline_name}' 或其参数不完整。无法进行计算。")
else:
    baseline_data = all_data[baseline_name]
    print(f"基准数据 ({baseline_name}): {baseline_data}\n")

    # ----------------------------------------------------------------------
    # 2. 差值计算与绘图
    # ----------------------------------------------------------------------

    # 获取除了 baseline 以外的所有名称
    names = [name for name in all_data if name != baseline_name]
    
    # 提前计算颜色列表
    bar_colors = []
    for name in names:
        if '+10%' in name:
            bar_colors.append('red')
        elif '-10%' in name:
            bar_colors.append('blue')
        else:
            bar_colors.append('gray')


    for param_name, (pattern, unit) in parameter_patterns.items():
        if param_name not in baseline_data:
            print(f"警告：基准数据中缺少参数 '{param_name}'。跳过绘图。")
            continue

        baseline_value = baseline_data[param_name]
        param_diffs = []
        
        # 计算差值
        for name in names:
            current_value = all_data[name].get(param_name)
            if current_value is not None:
                diff = current_value - baseline_value
                param_diffs.append(diff)
                # print(f"[{name}] {param_name} 差值: {diff:.3f} {unit}")
            else:
                # 如果某个名称缺少该参数，则使用 0 占位
                param_diffs.append(0) 

        # 绘制柱状图
        plt.figure(figsize=(14, 6))

        x_positions = np.arange(len(names))
        plt.bar(x_positions, param_diffs, color=bar_colors)

        # 设置X轴标签（名称）
        plt.xticks(x_positions, names, rotation=45, ha='right', fontsize=8)

        # 添加标题和轴标签
        plt.title(f'{param_name} 相对 Baseline 变化的敏感度分析', fontsize=14)
        plt.xlabel('敏感度项', fontsize=12)
        plt.ylabel(f'{param_name} 差值 ({unit}, 相对于 {baseline_value:.3f} {unit})', fontsize=12)

        # 添加网格线
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 在每个柱子上添加数值标签
        for i, diff in enumerate(param_diffs):
             # 仅为非零值添加标签
            if diff != 0:
                 # 调整标签位置：如果是正值，标签放在柱子顶部；如果是负值，放在底部
                va_setting = 'bottom' if diff > 0 else 'top'
                plt.text(i, diff, f'{diff:.3f}', ha='center', va=va_setting, fontsize=8)


        # 添加一条零线
        plt.axhline(0, color='black', linestyle='-', linewidth=0.8)

        # 调整布局，确保标签不被截断
        plt.tight_layout()

        # 显示图形
        plt.show()