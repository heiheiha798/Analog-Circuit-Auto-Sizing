import argparse
import os
import shutil
import re
import itertools

def create_corner_setting(template_lines: list, model: str, vdd: str, temp: str) -> list:
    """
    生成一个特殊 corner 的 setting 文件内容。
    Nominal is disabled, C1 is enabled and modified.
    """
    modified_lines = []
    # 假设 Nominal 的 isEnabled 是文件中第一个出现的
    # 假设 C1 的 isEnabled 是文件中第二个出现的
    is_enabled_count = 0 
    
    for line in template_lines:
        stripped_line = line.strip()
        
        # --- Toggling Enable Flags ---
        if stripped_line.endswith("isEnabled=0") or stripped_line.endswith("isEnabled=1"):
            is_enabled_count += 1
            if is_enabled_count == 1: # First one is Nominal, disable it
                modified_lines.append(re.sub(r"isEnabled=\d", "isEnabled=0", line))
                continue
            elif is_enabled_count == 2: # Second one is C1, enable it
                modified_lines.append(re.sub(r"isEnabled=\d", "isEnabled=1", line))
                continue

        # --- Modifying C1 Parameters ---
        if stripped_line.startswith("1\\MODEL\\1\\Sections="):
            modified_lines.append(f"1\\MODEL\\1\\Sections={model}\n")
            continue
        if stripped_line.startswith("1\\PARAM\\1\\Value="):
            modified_lines.append(f"1\\PARAM\\1\\Value={vdd}\n")
            continue
        if stripped_line.startswith("1\\Temperature="):
            modified_lines.append(f"1\\Temperature={temp}\n")
            continue
            
        # If no rule matches, keep the original line
        modified_lines.append(line)
        
    return modified_lines

def create_nominal_setting(template_lines: list) -> list:
    """
    生成 Nominal-only 的 setting 文件内容。
    Nominal is enabled, C1 is disabled.
    """
    modified_lines = []
    is_enabled_count = 0
    
    for line in template_lines:
        stripped_line = line.strip()
        
        if stripped_line.endswith("isEnabled=0") or stripped_line.endswith("isEnabled=1"):
            is_enabled_count += 1
            if is_enabled_count == 1: # First one is Nominal, enable it
                modified_lines.append(re.sub(r"isEnabled=\d", "isEnabled=1", line))
                continue
            elif is_enabled_count == 2: # Second one is C1, disable it
                modified_lines.append(re.sub(r"isEnabled=\d", "isEnabled=0", line))
                continue
        
        modified_lines.append(line)
        
    return modified_lines


def main():
    parser = argparse.ArgumentParser(
        description="""
        V3: MDE Corner 准备脚本。
        基于简单的查找和替换逻辑，保持文件结构不变。
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--lib', required=True, help='Aether 库名')
    parser.add_argument('--cell', required=True, help='包含 MDE view 的 Cell 名')
    parser.add_argument('--view', required=True, help='作为模板的 MDE view 名')
    args = parser.parse_args()

    source_view_path = os.path.join(os.getcwd(), args.lib, args.cell, args.view)
    target_cell_path = os.path.join(os.getcwd(), args.lib, args.cell)

    if not os.path.isdir(source_view_path):
        print(f"错误: 源 MDE view 路径不存在: {source_view_path}")
        return

    print(f"模板 MDE View: {source_view_path}")
    print(f"目标 Cell 目录: {target_cell_path}")

    # 读取一次模板文件
    original_setting_path = os.path.join(source_view_path, "setting")
    try:
        with open(original_setting_path, 'r') as f:
            template_lines = f.readlines()
        print(f"成功读取模板 setting 文件，共 {len(template_lines)} 行。")
    except FileNotFoundError:
        print(f"错误: 模板 setting 文件不存在: {original_setting_path}")
        return

    # --- 生成 8 个特殊 Corner ---
    models = ["ss", "ff"]
    vdds = ["1.62", "1.98"]
    temps = ["-40", "125"]
    corner_configs = list(itertools.product(models, vdds, temps))
    
    print("\n--- 开始生成 8 个 Corner MDE Views ---")
    for model, vdd, temp in corner_configs:
        temp_for_name = temp.replace('-', 'm')
        new_view_name = f"{args.view}_{model}_{vdd.replace('.', 'p')}_{temp_for_name}"
        target_view_path = os.path.join(target_cell_path, new_view_name)
        
        print(f"\n处理: {new_view_name}")

        if os.path.exists(target_view_path):
            shutil.rmtree(target_view_path)
        shutil.copytree(source_view_path, target_view_path)
        
        new_content_lines = create_corner_setting(template_lines, model, vdd, temp)
        
        target_setting_path = os.path.join(target_view_path, "setting")
        with open(target_setting_path, 'w') as f:
            f.writelines(new_content_lines)
        print(f"  - 成功生成 setting 文件。")

    # --- 生成 1 个 Nominal Corner ---
    print("\n--- 开始生成 1 个 Nominal MDE View ---")
    new_view_name = f"{args.view}_nominal"
    target_view_path = os.path.join(target_cell_path, new_view_name)
    print(f"\n处理: {new_view_name}")

    if os.path.exists(target_view_path):
        shutil.rmtree(target_view_path)
    shutil.copytree(source_view_path, target_view_path)
        
    new_content_lines = create_nominal_setting(template_lines)

    target_setting_path = os.path.join(target_view_path, "setting")
    with open(target_setting_path, 'w') as f:
        f.writelines(new_content_lines)
    print(f"  - 成功生成 setting 文件。")

    print("\n脚本执行完毕。")

if __name__ == "__main__":
    main()
