import os
import glob
import re

def parse_result_file(file_path):
    """
    Parses the result.txt file and extracts key performance parameters.
    """
    results = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    if value.lower() == 'nan':
                        results[key] = 'nan'
                    else:
                        try:
                            results[key] = float(value)
                        except ValueError:
                            results[key] = value
                # Handle cases where the value is empty, e.g., "UGB "
                elif len(parts) == 1:
                    key = parts[0]
                    results[key] = ''
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
    return results

def format_result_for_output(iter_num, results):
    """
    Formats the key results for a single iteration into the desired multi-line string (for console and file).
    """
    if not results:
        return ""
    
    output_lines = []
    output_lines.append(f"========== Evaluation result of iter {iter_num} ==========")

    ugb = results.get('UGB', '')
    ugb_str = "-inf" if str(ugb).lower() in ['', 'nan'] else f"{float(ugb):.4f}"
    output_lines.append(f"====== UGB             : {ugb_str} Hz")

    phase_margin = results.get('Phase_Margin', '')
    pm_str = "-inf" if str(phase_margin).lower() in ['', 'nan'] else f"{float(phase_margin):.4f}"
    output_lines.append(f"====== Phase_Margin : {pm_str} deg")
    
    gain_db = results.get('Gain_db', '')
    gain_db_str = "inf" if str(gain_db).lower() in ['', 'nan'] else f"{float(gain_db):.4f}"
    output_lines.append(f"====== Gain_db      : {gain_db_str} db")
    
    gain_margin = results.get('Gain_Margin', '')
    gm_str = "inf" if str(gain_margin).lower() in ['', 'nan'] else f"{float(gain_margin):.4f}"
    output_lines.append(f"====== Gain_Margin  : {gm_str} db")
    
    i_opa = results.get('I_OPA', '')
    i_opa_ma_str = "inf" if str(i_opa).lower() in ['', 'nan'] else f"{float(i_opa) * 1000:.6f}"
    output_lines.append(f"====== I_OPA         : {i_opa_ma_str} mA")

    total_area = results.get('Total_Area', '')
    total_area_str = "inf" if str(total_area).lower() in ['', 'nan'] else f"{float(total_area):.10f}"
    output_lines.append(f"====== Total_Area   : {total_area_str} um^2")
    
    # 在最后加上两个换行符，以符合您示例中的分隔效果
    output_lines.append("\n")
    
    return "\n".join(output_lines)


def main():
    """
    Main function to iterate through directories, print results to console, 
    and save every 100 iterations to a separate summary file in the exact output format.
    """
    iter_dirs = sorted(glob.glob('iter_*'), key=lambda x: int(x.split('_')[1]))
    
    # 定义每批次的迭代次数
    BATCH_SIZE = 100
    
    # 用于存储当前批次数据的列表（用于写入文件）
    summary_data_for_file = []
    
    total_dirs = len(iter_dirs)

    for idx, iter_dir in enumerate(iter_dirs):
        # 提取迭代次数
        try:
            iter_num = int(iter_dir.split('_')[1])
        except ValueError:
            print(f"Skipping non-numeric directory: {iter_dir}")
            continue

        result_file_path = os.path.join(iter_dir, 'result.txt')
        
        results = parse_result_file(result_file_path)

        if not results:
            print(f"Skipping directory {iter_dir} due to missing result.txt")
            continue

        # 格式化单次迭代结果（用于控制台输出和写入文件）
        formatted_output = format_result_for_output(iter_num, results)
        
        # 打印到控制台
        print(formatted_output)
        
        # 添加到文件存储列表
        summary_data_for_file.append(formatted_output)
        
        # 检查是否达到批次大小或已处理到最后一个目录
        is_batch_end = (iter_num % BATCH_SIZE == 0 and iter_num > 0)
        is_last_iter = (idx == total_dirs - 1)
        
        if summary_data_for_file and (is_batch_end or is_last_iter):
            # 确定批次文件的起始和结束编号
            current_batch_size = len(summary_data_for_file)
            # 为了计算起始 iter 号，我们需要从当前 iter 号减去已存储的元素数量（即当前批次大小）
            start_iter = iter_num - current_batch_size + 1
            end_iter = iter_num
            
            # 生成文件名，例如 summary_1_100.txt
            output_filename = f"summary_{start_iter}_{end_iter}.txt"
            
            # 写入文件
            with open(output_filename, 'w') as f:
                # 直接写入列表中的完整格式化字符串
                f.writelines(summary_data_for_file)
            
            print(f"Saved results for iter {start_iter} to {end_iter} to {output_filename}")
            
            # 清空数据列表，准备下一个批次
            summary_data_for_file = []

if __name__ == "__main__":
    main()