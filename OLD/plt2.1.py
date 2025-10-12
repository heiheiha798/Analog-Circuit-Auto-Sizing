import re

def extract_performance_data(filename="tmp2.txt"):
    """
    从指定文件中提取性能数据，并将相同性能的数值放入一个list。
    此版本能处理标准数值、-inf/inf、和NaN。
    """
    
    # 初始化用于存储六个性能参数数值的列表
    ugb_list = []
    phase_margin_list = []
    gain_db_list = []
    gain_margin_list = []
    i_opa_list = []
    total_area_list = []

    # 更新后的正则表达式：
    # 匹配 '名称 : 数值'。数值部分匹配标准浮点数、科学计数法 (E/e)、inf/-inf、或nan。
    # pattern = re.compile(r"======\s*(\w+[\s_]*\w*)\s*:\s*([\d\.\-Ee+infnan]+)")
    # 为了更精确地捕获性能行，我们保留更简单的模式，并依赖 float() 的健壮性
    pattern = re.compile(r"======\s*(\w+[\s_]*\w*)\s*:\s*([\d\.\-Ee+infnan]+)")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # 用于标记是否处于一个有效的性能块内
            in_block = False
            
            for line in lines:
                line = line.strip()
                
                # 检查是否是性能块的起始行
                if line.startswith("==========") and "Evaluation result of iter" in line:
                    in_block = True
                    continue
                
                if in_block:
                    # 尝试匹配性能行
                    match = pattern.match(line)
                    
                    if match:
                        name = match.group(1).strip().replace(' ', '_')
                        value_str = match.group(2).strip()
                        
                        try:
                            # float() 函数可以正确处理 'inf', '-inf', 'nan' 和标准数值
                            value = float(value_str) 
                        except ValueError:
                            # 捕获转换错误（例如单独的 '-' 或其他非数值字符串），并跳过此行
                            print(f"警告：无法将 '{value_str}' 转换为浮点数（或特殊值），已跳过。源行: {line}")
                            continue

                        # 根据匹配到的名称将数值添加到相应的列表中
                        if name == "UGB":
                            ugb_list.append(value)
                        elif name == "Phase_Margin":
                            phase_margin_list.append(value)
                        elif name == "Gain_db":
                            gain_db_list.append(value)
                        elif name == "Gain_Margin":
                            gain_margin_list.append(value)
                        elif name == "I_OPA":
                            i_opa_list.append(value)
                        elif name == "Total_Area":
                            total_area_list.append(value)
                    

    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 未找到。请确保文件在当前目录下。")
        return None
    except Exception as e:
        print(f"处理文件时发生未预期的错误: {e}")
        return None

    # 返回包含所有性能列表的元组
    return (ugb_list, phase_margin_list, gain_db_list, gain_margin_list, i_opa_list, total_area_list)

# 执行数据提取
results = extract_performance_data()

if results:
    # 解包结果
    ugb, pm, gdb, gm, i_opa, ta = results
    
    # 终端输出六个列表
    print("\n--- 提取结果 ---")
    print("UGB 列表:", ugb)
    print("Phase_Margin 列表:", pm)
    print("Gain_db 列表:", gdb)
    print("Gain_Margin 列表:", gm)
    print("I_OPA 列表:", i_opa)
    print("Total_Area 列表:", ta)