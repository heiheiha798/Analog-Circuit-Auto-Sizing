import os
import shutil
import numpy as np
import pyAether as ae

from typing import Dict, Optional
from scipy.optimize import differential_evolution

from src.data_models import Parameter, SymmetryConstraint
from src.utils import merge_files, param_convert
from src.eda_interface import extract_params

class SimulatePlatform:
    """Platform for circuit optimization using Aether tools"""
    def __init__(
        self,
        ae_lib: str = None,
        ae_cell: str = None,
        ae_view: str = None,
        mde_cell: str = None,
        mde_view: str = None,
        output_path: str = None,
        output_file: str = None,
        symmetry_constraints: Optional[SymmetryConstraint] = None,
        dummy_params: Dict[str, Parameter] = None
    ):
        self.origin_lib = ae_lib
        self.ae_lib = ae_lib
        self.ae_cell = ae_cell
        self.ae_view = ae_view
        self.mde_cell = mde_cell
        self.mde_view = mde_view
        self.output_path = output_path
        self.output_file = output_file
        self.symmetry_constraints = symmetry_constraints
        self.dummy_params = dummy_params or {}
        # self.best_fitness = 1e10
        # self.best_iter = 0
        # self.best_path = ''
        self.iter_count = 0  # Track iterations
        # self.param_mapping = {}  # original param ---> reduced param
        # self.reverse_mapping = {}  # Reverse mapping from reduced to original parameters
        # self.base_ugb = 0.0
        # self.base_area = 0.0

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
            print(f"delete old dir: {self.output_path}")
    
        os.makedirs(self.output_path, exist_ok=True)
        print(f"create new dir: {self.output_path}")

    def only_set_params(self, params: Dict[str, Parameter]):
        """Apply parameter values to the circuit design"""
        cv = ae.dbOpenCV(self.ae_lib, self.ae_cell, self.ae_view)
        if not cv:
            raise RuntimeError(f'ae.dbOpenCV({self.ae_lib}, {self.ae_cell}, {self.ae_view}) failed')

        param_items = list(params.items())
        # Apply parameter values
        for i, (orig_name, param) in enumerate(param_items):
            # Parse instance name and parameter name
            index = orig_name.rfind("_")
            inst_name = orig_name[:index]
            param_name = orig_name[(index + 1):]

            instId = ae.dbFindInst(cv=cv, name=inst_name)
            if not instId:
                print(f'Warning: Instance {inst_name} not found')
                continue

            formatted_value = param.format_value(param.value)
            # print(f"Set {inst_name} {param_name} = {formatted_value}")

            # Set params
            ret = ae.cdfSetParam(instId, param_name, formatted_value)
            if not ret:
                print(f'Warning: Set {orig_name}={formatted_value} failed')

        save_result = ae.dbCheckAndSaveDesign(cv)
        print(f'Design save result: {save_result}')
        ae.dbCloseCV(cv)

    def calc_area(self):
        """Compute total area of the chip"""
        cv = ae.dbOpenCV(self.ae_lib, self.ae_cell, self.ae_view)
        if not cv:
            raise RuntimeError(f'ae.dbOpenCV({self.ae_lib}, {self.ae_cell}, {self.ae_view}) failed, please check !!!')

        insts = cv.instances
        self.total_area = 0.0
        for inst in insts:
            if (inst.name[0] == "R"  or  inst.name[0] == 'r'):
                # segW * segL * segments * 2.0
                segW = str(ae.aeEmyInstGetParameter('segW', inst))
                segL = str(ae.aeEmyInstGetParameter('segL', inst))
                segments = str(ae.aeEmyInstGetParameter('segments', inst))
                r_area = param_convert(segW) * param_convert(segL) * float(segments) * 2.0
                # print(f'======== {inst.name} area:  {segW} * {segL} * {segments} * 2.0 = {r_area}')
                self.total_area += r_area

            elif (inst.name[0] == "C"  or  inst.name[0] == 'c'):
                # fw * l* m * 1.0
                fw = str(ae.aeEmyInstGetParameter('fw', inst))
                l = str(ae.aeEmyInstGetParameter('l', inst))
                m = str(ae.aeEmyInstGetParameter('m', inst))
                c_area = param_convert(fw) * param_convert(l) * float(m)
                # print(f'======== {inst.name} area:  {fw} * {l} * {m} * 1.0 = {c_area}')
                self.total_area += c_area

            elif (str(inst.name).startswith("PM")  or  str(inst.name).startswith("NM")):
                # fw * l * m * 1.5
                fw = str(ae.aeEmyInstGetParameter('fw', inst))
                l = str(ae.aeEmyInstGetParameter('l', inst))
                m = str(ae.aeEmyInstGetParameter('m', inst))
                m_area = param_convert(fw) * param_convert(l) * float(m) * 1.5
                # print(f'======== {inst.name} area:  {fw} * {l} * {m} * 1.5 = {m_area}')
                self.total_area += m_area

            else:
                continue
                # print(f'======== {inst.name} area:  not counted!!! ==========================')
        print(f'\n==== total area:  {self.total_area}')
        ae.dbCloseCV(cv)

    def evaluate(self) -> Dict[str, float]:
        """Run simulation and collect results"""

        # Create unique output directory for this evaluation
        self.iter_count += 1
        eval_path = f"{self.output_path}/iter_{self.iter_count}"

        # compute area
        self.calc_area()

        # open mde
        mde = ae.MdeSession.open(self.ae_lib, self.mde_cell, self.mde_view)
        if mde:
            print(f"\n==== iter {self.iter_count} ae.MdeSession.open({self.ae_lib}, {self.mde_cell}, {self.mde_view}) successfully ====")
        else:
            print(f"\n==== iter {self.iter_count} ae.MdeSession.open({self.ae_lib}, {self.mde_cell}, {self.mde_view}) failed. Reason: {ae.MdeSession.staticLastError()}")
            return ""

        print(f'\n==== iter {self.iter_count} Project directory: {mde.getSetting().getProjectDirectory()}')
        setting = mde.getSetting()
        print(f'\n==== iter {self.iter_count} Current settings: {setting}')
        # print(f'\n==== SPE outputs: {setting.getAllSpeOutputs()}')
	
        if (mde.createSimulationNetlist()):
            print("Creating simulation netlist succeeds.")
        else:
            print("Creating simulation netlist fails.")
 
        # Execute simulation
        scores = {}
        print(f"\n==== iter {self.iter_count} Begin simulate ====")
        result = mde.netlistAndRun()
        if result:
            print(f"\n==== iter {self.iter_count} Simulation completed successfully ====")
        else:
            print(f'\n==== iter {self.iter_count} Simulation failed. Reason: {mde.lastError()}')
            mde.close()
            return scores
	    
        if not result.isValid():
            print(f'\n==== iter {self.iter_count} Invalid results. Reason: {result.lastError()}')
            mde.close()
            return scores
	    
        # Save results in CSV format
        result.saveResultsToDir(eval_path)
        if not result.isValid():
            print(f'\n==== iter {self.iter_count} Results are invalid. Reason: {result.lastError()}')
            mde.close()
            return scores
        
        # Copy simulation log file
        src_path = f"{mde.getSetting().getProjectDirectory()}/{self.ae_lib}/{self.mde_cell}/{self.mde_view}/Results/{result.name()[len('Results'):]}"
        try:
            shutil.copy2(f"{src_path}/emy2netlist.log", f"{eval_path}/emy2netlist.log")
        except Exception as e:
            print(f"Failed to copy emy2netlist.log, details: {str(e)}")
    
        if mde.close():
            print(f"==== iter {self.iter_count} Succeeds in closing MDE ====")
        else:
            print(f"==== iter {self.iter_count} Fails to close MDE, reason: {mde.lastError()}")

        # Combine results into single log file
        print(f'\n==== iter {self.iter_count} Result name: {result.name()}')
        merge_files([f"{eval_path}/{result.name()}_summary.csv"], f"{eval_path}/{self.output_file}")

        # Save simulation result
        with open(f"{eval_path}/{self.output_file}", 'r') as csvfile:
            with open(f"{eval_path}/result.txt", 'w') as f:
                import csv
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if (row['Output'] == "Phase_Margin") or (row['Output'] == "Gain_db") or (row['Output'] == "UGB"):
                        strlog = f"{row['Output']} {row['Min']}\n"
                        f.write(strlog)
                        if row['Min'] == '':
                            scores[row['Output']] = -float("inf")
                        else:
                            scores[row['Output']] = float(row['Min'])
                    elif (row['Output'] == "Gain_Margin") or (row['Output'] == "I_OPA"):
                        strlog = f"{row['Output']} {row['Max']}\n"
                        f.write(strlog)
                        if row['Max'] == '':
                            scores[row['Output']] = float("inf")
                        else:
                            scores[row['Output']] = float(row['Max'])
                    else:
                        continue

                # save total area
                strlog = f"Total_Area {self.total_area}\n"
                scores['Total_Area'] = self.total_area
                f.write(strlog)

        # Save params
        extract_params(self.ae_lib, self.ae_cell, self.ae_view, f"{eval_path}/params.txt")
        print(f"\n========== Evaluation result of iter {self.iter_count} ==========")
        print(f"====== UGB          : {scores['UGB']} Hz")
        print(f"====== Phase_Margin : {scores['Phase_Margin']} deg")
        print(f"====== Gain_db      : {scores['Gain_db']} db")
        print(f"====== Gain_Margin  : {scores['Gain_Margin']} db")
        print(f"====== I_OPA        : {scores['I_OPA'] * 1000.0} mA")
        print(f"====== Total_Area   : {scores['Total_Area']} um^2\n")
        return scores

