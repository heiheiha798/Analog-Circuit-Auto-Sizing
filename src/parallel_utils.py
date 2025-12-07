# src/parallel_utils.py

import os
import shutil
import re
import itertools
import multiprocessing
import time
import glob
import pyAether as ae

# --- Part 1: MDE View Preparation and Cleanup ---

def get_view_list(base_view_name):
    """根据基础视图名称生成9个并行的视图名称列表。"""
    models, vdds, temps = ["ss", "ff"], ["1.62", "1.98"], ["-40", "125"]
    view_list = []
    for model, vdd, temp in itertools.product(models, vdds, temps):
        temp_for_name = temp.replace('-', 'm')
        view_list.append(f"{base_view_name}_{model}_{vdd.replace('.', 'p')}_{temp_for_name}")
    view_list.append(f"{base_view_name}_nominal")
    return view_list

def prepare_parallel_views(lib, cell, base_view):
    """
    核心准备函数：读取模板，创建并修改9个单corner MDE view。
    成功则返回生成的 view 名称列表。
    """
    source_view_path = os.path.join(os.getcwd(), lib, cell, base_view)
    target_cell_path = os.path.join(os.getcwd(), lib, cell)

    if not os.path.isdir(source_view_path):
        raise FileNotFoundError(f"模板 MDE view 路径不存在: {source_view_path}")

    original_setting_path = os.path.join(source_view_path, "setting")
    try:
        with open(original_setting_path, 'r') as f:
            template_lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"模板 setting 文件不存在: {original_setting_path}")

    view_names = get_view_list(base_view)
    
    # 准备8个特殊Corner
    for view_name in view_names:
        if "nominal" in view_name: continue
        
        # 从名称解析参数
        parts = view_name.split('_')
        model = parts[-3]
        vdd = parts[-2].replace('p', '.')
        temp = parts[-1].replace('m', '-')

        target_view_path = os.path.join(target_cell_path, view_name)
        if os.path.exists(target_view_path): shutil.rmtree(target_view_path)
        shutil.copytree(source_view_path, target_view_path)
        
        # 调用 V3 版本的逻辑
        new_content = _create_corner_setting(template_lines, model, vdd, temp)
        
        target_setting_path = os.path.join(target_view_path, "setting")
        with open(target_setting_path, 'w') as f: f.writelines(new_content)

    # 准备1个Nominal
    nominal_view_name = f"{base_view}_nominal"
    target_view_path = os.path.join(target_cell_path, nominal_view_name)
    if os.path.exists(target_view_path): shutil.rmtree(target_view_path)
    shutil.copytree(source_view_path, target_view_path)
    new_content = _create_nominal_setting(template_lines)
    target_setting_path = os.path.join(target_view_path, "setting")
    with open(target_setting_path, 'w') as f: f.writelines(new_content)

    print(f"成功准备了 {len(view_names)} 个并行的 MDE views。")
    return view_names

def cleanup_parallel_views(lib, cell, base_view):
    """清理所有由 prepare_parallel_views 创建的临时 MDE views。"""
    view_names_to_delete = get_view_list(base_view)
    target_cell_path = os.path.join(os.getcwd(), lib, cell)
    for view_name in view_names_to_delete:
        view_path = os.path.join(target_cell_path, view_name)
        if os.path.isdir(view_path):
            try:
                shutil.rmtree(view_path)
            except OSError as e:
                print(f"清理 Warning: 无法删除 {view_path}: {e}")
    print(f"清理了 {len(view_names_to_delete)} 个并行的 MDE views。")


# --- Part 2: Simulation Worker (必须是顶层函数) ---

def simulation_worker(task_info_tuple):
    """
    [MANUAL PROCESS VERSION]
    Worker function designed to be run in a manually managed Process.
    It takes an extra argument 'queue' for returning results.
    """
    # 1. [修改] 从元组中解包参数，最后一个是 queue
    #    这样可以保持与之前 tasks 列表的兼容性
    *task_info, results_queue = task_info_tuple
    task_id, lib, cell, view, output_path = task_info
    
    # 我们保留所有详细的调试日志和错误捕获机制
    import traceback
    import os
    import multiprocessing
    
    proc_name = multiprocessing.current_process().name
    pid = os.getpid()

    # print(f"[Worker Start] Process: {proc_name} (PID: {pid}), Task: Simulating view '{view}'")

    mde = None
    try:
        ae.emyInitAether('-adv')
        
        mde = ae.MdeSession.open(lib, cell, view)
        if not mde:
            error_msg = f"MdeSession.open failed for view '{view}'. Reason: {ae.MdeSession.staticLastError()}"
            print(f"[Worker Failure] PID: {pid}, Error: {error_msg}")
            # 2. [修改] 使用 queue.put() 而不是 return
            results_queue.put((view, "FAILURE", error_msg))
            return # 任务结束，退出函数

        result = mde.netlistAndRun()
        
        if not result or not result.isValid():
            error_msg = f"Simulation failed for view '{view}'. MDE Error: {mde.lastError()}, Result Error: {result.lastError() if result else 'N/A'}"
            print(f"[Worker Failure] PID: {pid}, Error: {error_msg}")
            # 2. [修改] 使用 queue.put() 而不是 return
            results_queue.put((view, "FAILURE", error_msg))
            return

        corner_output_dir = os.path.join(output_path, view)
        os.makedirs(corner_output_dir, exist_ok=True)
        result.saveResultsToDir(corner_output_dir)
        
        summary_files = glob.glob(os.path.join(corner_output_dir, "*_summary.csv"))
        if not summary_files:
            error_msg = f"Simulation for view '{view}' completed, but summary.csv was not found."
            print(f"[Worker Failure] PID: {pid}, Error: {error_msg}")
            # 2. [修改] 使用 queue.put() 而不是 return
            results_queue.put((view, "FAILURE", error_msg))
            return
        
        # print(f"[Worker Success] PID: {pid}, Task for view '{view}' completed.")
        # 2. [修改] 将成功结果放入队列
        results_queue.put((view, "SUCCESS", summary_files[0]))

    except BaseException as e:
        full_traceback = traceback.format_exc()
        error_report = (
            f"CRITICAL UNHANDLED EXCEPTION IN SUBPROCESS (PID: {pid}, View: {view})\n"
            f"Type: {type(e).__name__}, Msg: {str(e)}\n"
            f"Traceback:\n{full_traceback}"
        )
        print(error_report)
        # 2. [修改] 将详细的错误报告放入队列
        results_queue.put((view, "FAILURE", error_report))
    
    finally:
        if mde:
            # print(f"[Worker Cleanup] PID: {pid}, Closing MDE session for view '{view}'.")
            mde.close()

# --- Part 3: Helper functions for setting file modification (V3 logic) ---

def _create_corner_setting(template_lines, model, vdd, temp):
    # (这是 prepare_corners_os_v3.py 中的函数，作为内部帮助函数)
    modified_lines, count = [], 0
    for line in template_lines:
        stripped = line.strip()
        if stripped.endswith("isEnabled=0") or stripped.endswith("isEnabled=1"):
            count += 1
            if count == 1: modified_lines.append(re.sub(r"isEnabled=\d", "isEnabled=0", line)); continue
            elif count == 2: modified_lines.append(re.sub(r"isEnabled=\d", "isEnabled=1", line)); continue
        if stripped.startswith("1\\MODEL\\1\\Sections="): modified_lines.append(f"1\\MODEL\\1\\Sections={model}\n"); continue
        if stripped.startswith("1\\PARAM\\1\\Value="): modified_lines.append(f"1\\PARAM\\1\\Value={vdd}\n"); continue
        if stripped.startswith("1\\Temperature="): modified_lines.append(f"1\\Temperature={temp}\n"); continue
        modified_lines.append(line)
    return modified_lines

def _create_nominal_setting(template_lines):
    # (这是 prepare_corners_os_v3.py 中的函数，作为内部帮助函数)
    modified_lines, count = [], 0
    for line in template_lines:
        stripped = line.strip()
        if stripped.endswith("isEnabled=0") or stripped.endswith("isEnabled=1"):
            count += 1
            if count == 1: modified_lines.append(re.sub(r"isEnabled=\d", "isEnabled=1", line)); continue
            elif count == 2: modified_lines.append(re.sub(r"isEnabled=\d", "isEnabled=0", line)); continue
        modified_lines.append(line)
    return modified_lines
