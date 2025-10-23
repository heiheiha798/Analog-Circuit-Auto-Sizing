import os
import shutil
import numpy as np
import pyAether as ae
import multiprocessing
import csv

from typing import Dict, Optional
from scipy.optimize import differential_evolution
from multiprocessing import Process, Queue

from src.data_models import Parameter, CircuitGraph
from src.utils import merge_files, param_convert
from src.eda_interface import extract_params
from src import parallel_utils

class SimulatePlatform:
    """
    Platform for circuit optimization using Aether tools.
    FINAL VERSION: Combines original working logic with parallel evaluation.
    """
    def __init__(
        self,
        ae_lib: str = None,
        ae_cell: str = None,
        ae_view: str = None,
        mde_cell: str = None,
        mde_view: str = None,
        output_path: str = None,
        output_file: str = None
    ):
        # This __init__ is exactly as the original.
        self.ae_lib = ae_lib
        self.ae_cell = ae_cell
        self.ae_view = ae_view
        self.mde_cell = mde_cell
        self.mde_view = mde_view
        self.output_path = output_path
        self.output_file = output_file
        self.iter_count = 0
        self.total_area = 0.0

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Created new output directory: {self.output_path}")

    def only_set_params(self, params: dict):
        """
        RESTORED: This method is exactly as the original.
        It ONLY sets parameters, saves, and closes.
        """
        cv = ae.dbOpenCV(self.ae_lib, self.ae_cell, self.ae_view)
        if not cv:
            raise RuntimeError(f'ae.dbOpenCV({self.ae_lib}, {self.ae_cell}, {self.ae_view}) failed')
        for orig_name, param in params.items():
            index = orig_name.rfind("_")
            inst_name, param_name = orig_name[:index], orig_name[(index + 1):]
            instId = ae.dbFindInst(cv=cv, name=inst_name)
            if not instId:
                print(f'Warning: Instance {inst_name} not found')
                continue
            formatted_value = param.format_value(param.value)
            if not ae.cdfSetParam(instId, param_name, formatted_value):
                print(f'Warning: Set {orig_name}={formatted_value} failed')
        # We comment out the verbose print
        # save_result = ae.dbCheckAndSaveDesign(cv)
        # print(f'Design save result: {save_result}')
        ae.dbCheckAndSaveDesign(cv)
        ae.dbCloseCV(cv)

    def calc_area(self):
        """
        FINAL, SIMPLE & ROBUST VERSION.
        This method contains the original, correct logic and is called from the main
        process BEFORE any parallel simulation starts, avoiding all environment conflicts.
        """
        # 使用您验证过的老版本area读取逻辑，只增加 str() 转换来修复 emyString bug
        cv = ae.dbOpenCV(self.ae_lib, self.ae_cell, self.ae_view)
        if not cv:
            # 在主进程中，如果失败，直接抛出异常让程序停止更安全
            raise RuntimeError(f'calc_area: ae.dbOpenCV({self.ae_lib}, {self.ae_cell}, {self.ae_view}) failed!')

        self.total_area = 0.0
        for inst in cv.instances:
            # 关键修复: 使用 str(inst.name) 来处理 emyString 对象
            inst_name_str = str(inst.name)

            if inst_name_str.lower().startswith('r'):
                segW = str(ae.aeEmyInstGetParameter('segW', inst))
                segL = str(ae.aeEmyInstGetParameter('segL', inst))
                segments = str(ae.aeEmyInstGetParameter('segments', inst))
                self.total_area += param_convert(segW) * param_convert(segL) * float(segments) * 2.0
            
            elif inst_name_str.lower().startswith('c'):
                fw = str(ae.aeEmyInstGetParameter('fw', inst))
                l = str(ae.aeEmyInstGetParameter('l', inst))
                m = str(ae.aeEmyInstGetParameter('m', inst))
                self.total_area += param_convert(fw) * param_convert(l) * float(m) * 1.0

            elif inst_name_str.startswith("PM") or inst_name_str.startswith("NM"):
                fw = str(ae.aeEmyInstGetParameter('fw', inst))
                l = str(ae.aeEmyInstGetParameter('l', inst))
                m = str(ae.aeEmyInstGetParameter('m', inst))
                self.total_area += param_convert(fw) * param_convert(l) * float(m) * 1.5
        
        print(f'\n==== Total Area Calculated (Main Thread): {self.total_area} um^2 ====')
        ae.dbCloseCV(cv)

    def _parse_and_merge_results_parallel(self, result_files: list) -> dict:
        """
        Helper function to parse parallel results. No changes from last version.
        """
        worst_scores = {
            'Gain_db': float('inf'), 'UGB': float('inf'), 'Phase_Margin': float('inf'),
            'Gain_Margin': -float('inf'), 'I_OPA': -float('inf')
        }
        for csv_file in result_files:
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        output, val_min, val_max = row.get('Output'), row.get('Min'), row.get('Max')
                        try:
                            f_min = float(val_min) if val_min else float('-inf')
                            f_max = float(val_max) if val_max else float('inf')
                        except (ValueError, TypeError): continue
                        if output == 'Gain_db': worst_scores['Gain_db'] = min(worst_scores['Gain_db'], f_min)
                        elif output == 'UGB': worst_scores['UGB'] = min(worst_scores['UGB'], f_min)
                        elif output == 'Phase_Margin': worst_scores['Phase_Margin'] = min(worst_scores['Phase_Margin'], f_min)
                        elif output == 'Gain_Margin': worst_scores['Gain_Margin'] = max(worst_scores['Gain_Margin'], f_max)
                        elif output == 'I_OPA': worst_scores['I_OPA'] = max(worst_scores['I_OPA'], f_max)
            except Exception as e:
                print(f"  Warning: Could not parse result file {csv_file}: {e}")
        worst_scores['Total_Area'] = self.total_area
        return worst_scores

    def evaluate(self) -> Dict[str, float]:
        """
        [NEW VERSION using Manual Process Management]
        This version avoids multiprocessing.Pool and manually creates, starts,
        and joins 9 Process objects for maximum stability and control.
        Communication is handled via a multiprocessing.Queue.
        """
        self.iter_count += 1
        eval_path = os.path.join(self.output_path, f"iter_{self.iter_count}")
        os.makedirs(eval_path, exist_ok=True)

        print(f"\n===== [Eval #{self.iter_count}] Starting Parallel Simulation (Manual Process Mode)... =====")
        
        # 1. 准备 MDE 视图和任务列表 (这部分逻辑不变)
        view_list = []
        try:
            view_list = parallel_utils.prepare_parallel_views(self.ae_lib, self.mde_cell, self.mde_view)
            tasks = [(i, self.ae_lib, self.mde_cell, view, eval_path) for i, view in enumerate(view_list)]

            # 2. [核心修改] 创建用于进程间通信的队列
            #    这个队列将用于从子进程收集结果
            results_queue = Queue()

            # 3. [核心修改] 手动创建并启动9个子进程
            processes = []
            for task in tasks:
                # 注意：args元组的最后一个元素需要一个逗号, 即使只有一个元素
                # 我们将 results_queue 作为额外参数传递给 worker
                process_args = task + (results_queue,) 
                p = Process(target=parallel_utils.simulation_worker, args=(process_args,))
                processes.append(p)
                p.start() # 启动进程
                # print(f"  - Started Process for view: {task[3]}")

            # 4. [核心修改] 先从队列中收集所有结果
            #    这个 get() 操作本身就是阻塞的，它会一直等待直到有结果为止。
            #    这保证了我们能收到所有子进程的“回信”。
            print("  - All processes started. Waiting for results from the queue...")
            collected_results = []
            for _ in range(len(tasks)):
                result = results_queue.get() # 等待并获取一个结果
                collected_results.append(result)
            print("  - All results have been collected from the queue.")


            # 5. [核心修改] 然后再 join() 所有子进程
            #    此时，所有子进程的核心任务都已完成，它们要么已经退出，要么即将退出。
            #    这里的 join() 只是一个快速的清理步骤，确保所有进程资源都被系统回收。
            print("  - Joining processes for cleanup...")
            for p in processes:
                p.join() 
            print("  - All processes have been joined.")
            
            # 6. 处理收集到的结果 (这部分逻辑与之前类似)
            summary_files = [res[2] for res in collected_results if res[1] == "SUCCESS"]
            failures = [res for res in collected_results if res[1] == "FAILURE"]

            if failures:
                print(f"  WARNING: {len(failures)}/{len(tasks)} corners failed:")
                for view, _, msg in failures:
                    # 打印完整的错误报告
                    print(f"    - Failure in view '{view}':\n--- ERROR REPORT START ---\n{msg}\n--- ERROR REPORT END ---")
            
            if not summary_files:
                print("  FATAL: All corners failed. Cannot calculate score.")
                return {}

            # 使用现有的辅助函数来解析结果
            scores = self._parse_and_merge_results_parallel(summary_files)

        finally:
            # 清理临时视图的逻辑保持不变
            if view_list:
                parallel_utils.cleanup_parallel_views(self.ae_lib, self.mde_cell, self.mde_view)
        
        print(f"========== Aggregated Worst-Case Result of Eval #{self.iter_count} ==========")
        print(f"====== UGB          : {scores.get('UGB', 0)} Hz")
        print(f"====== Phase_Margin : {scores.get('Phase_Margin', 0)} deg")
        print(f"====== Gain_db      : {scores.get('Gain_db', 0)} db")
        print(f"====== Gain_Margin  : {scores.get('Gain_Margin', 0)} db")
        print(f"====== I_OPA        : {scores.get('I_OPA', 0) * 1000.0} mA")
        print(f"====== Total_Area   : {scores.get('Total_Area', 0)} um^2\n")
        
        return scores
