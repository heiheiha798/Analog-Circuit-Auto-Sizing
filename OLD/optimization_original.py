import pyAether as ae
import argparse
import re
import os
import shutil
import numpy as np
from typing import Dict, Optional, List, Tuple, Set, Union
from scipy.optimize import differential_evolution


def merge_files(file_list, output_file):
    """Merge multiple files into a single output file"""
    with open(output_file, 'w') as output:
        for file_name in file_list:
            with open(file_name, 'r') as input_file:
                output.write(input_file.read())
                output.write("\n\n")


def extract_params(lib, cell, view, param_file_name):
    cv = ae.dbOpenCV(lib, cell, view)
    if(cv is None):
        print("dbOpenCV failed !!!")
        return

    with open(param_file_name, 'w') as param_file:
        insts = cv.instances
        for inst in insts:
            # print("=====start ", inst.name, "=====")
            if (inst.name[0] == "R"  or  inst.name[0] == 'r'):
                retVal = ae.aeEmyInstGetParameter('segW', inst)
                strlog = f"parameter, {inst.name}_segW, {retVal}\n"
                param_file.write(strlog)

                retVal = ae.aeEmyInstGetParameter('segL', inst)
                strlog = f"parameter, {inst.name}_segL, {retVal}\n"
                param_file.write(strlog)

            elif (inst.name[0] == "C"  or  inst.name[0] == 'c'):
                retVal = ae.aeEmyInstGetParameter('l', inst)
                strlog = f"parameter, {inst.name}_l, {retVal}\n"
                param_file.write(strlog)

            elif (str(inst.name).startswith("PM")  or  str(inst.name).startswith("NM")):
                retVal = ae.aeEmyInstGetParameter('m', inst)
                strlog = f"parameter, {inst.name}_m, {retVal}\n"
                param_file.write(strlog)

                retVal = ae.aeEmyInstGetParameter('l', inst)
                strlog = f"parameter, {inst.name}_l, {retVal}\n"
                param_file.write(strlog)

                retVal = ae.aeEmyInstGetParameter('fw', inst)
                strlog = f"parameter, {inst.name}_fw, {retVal}\n"
                param_file.write(strlog)

            else:
                # print(f"==== Instance {inst.name} no params need extract.")
                continue

            # print("=====finish ", inst.name, "=====")
    ae.dbCloseCV(cv)


def param_convert(s: str) -> float:
    """
    Convert a string with unit to micrometers (\u03bcm).
    Supported units: 
        'u' or '\u03bcm' for micrometers (default unit)
        'n' or 'nm' for nanometers
        No unit indicates meters
    Supports scientific notation (e.g., '1.5e-3u') and sign prefixes
    
    Parameters:
        s (str): Input string with optional unit
    Returns:
        float: Value converted to micrometers
    Raises:
        ValueError: For unrecognized units or invalid number formats
    """
    # Regular expression pattern to match optional sign, number (with scientific notation), and optional unit
    pattern = r'^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z]*)\s*$'
    match = re.match(pattern, s)
    
    # Handle cases where input doesn't match expected pattern
    if not match:
        raise ValueError(f"Unparseable input format: '{s}'")
    
    # Extract number and unit components from matched groups
    number_str, unit = match.groups()
    
    try:
        # Convert number string to float (supports scientific notation)
        value = float(number_str)
    except ValueError:
        raise ValueError(f"Invalid number format: '{number_str}'")
    
    # Normalize unit string: lowercase and replace micro symbol (\u03bc) with 'u'
    unit = unit.lower().replace('\u03bc', 'u').replace('m', '')
    
    # Convert value based on detected unit
    if unit == '' or unit == 'u':  # No unit (meters) or micrometers
        return value if unit == 'u' else value * 1e6
    elif unit == 'n':  # Nanometers
        return value * 0.001
    else:
        raise ValueError(f"Unrecognized unit: '{unit}'. Supported units: 'u' (micrometers), 'n' (nanometers), or no unit (meters)")


class Parameter:
    """Represents an optimization parameter with constraints"""
    def __init__(
        self,
        name: str,
        param_type: str = 'continuous',
        value: float = None,
        min_val: float = None,
        max_val: float = None,
        is_dummy: bool = False
    ):
        self.name = name
        self.type = param_type
        self.value = value
        self.min = min_val
        self.max = max_val
        self.is_dummy = is_dummy

    def format_value(self, value: float) -> str:
        """Format value according to parameter type and unit"""
        if self.type == 'integer':
            return str(int(round(value)))
        else:
            if (value < 1.0):
                return f"{value * 1000}n"
            else:
                return f"{value}u"

    def format(self) -> str:
        """Format value according to parameter type and unit"""
        if self.type == 'integer':
            return str(int(round(self.value)))
        else:
            if (self.value < 1.0):
                return f"{self.value * 1000}n"
            else:
                return f"{self.value}u"

    def normalize(self, value: float) -> float:
        """Normalize a value to [0, 1] range based on parameter bounds"""
        if self.max == self.min:
            return 0.0  # Avoid division by zero
        return (value - self.min) / (self.max - self.min)

    def denormalize(self, norm_value: float) -> float:
        """Convert a normalized value back to the original range"""
        return self.min + norm_value * (self.max - self.min)


class SymmetryConstraint:
    """Represents a symmetry constraint between instances"""
    def __init__(self, instance_groups: List[Union[Tuple[str, ...], List[str]]]):
        self.instance_groups = instance_groups
        self.symmetric_groups = self._create_symmetric_groups()
        
    def _create_symmetric_groups(self) -> List[Set[str]]:
        """Create symmetric groups from instance groups"""
        groups = []
        for group in self.instance_groups:
            # Convert tuple to set
            group_set = set(group)
            # Check if any instance already exists in existing groups
            merged_groups = []
            new_group = group_set.copy()
            
            for existing_group in groups:
                if existing_group & group_set:
                    # Merge with existing group
                    new_group |= existing_group
                else:
                    # Keep existing group as is
                    merged_groups.append(existing_group)
            
            merged_groups.append(new_group)
            groups = merged_groups
        
        return groups

    def get_symmetric_parameters_for_group(self, group: Set[str], param_name: str) -> List[str]:
        """Get all symmetric parameter names for a given parameter in a specific group"""
        symmetric_params = []
        for inst in group:
            symmetric_params.append(f"{inst}_{param_name}")
        return symmetric_params

    def get_symmetric_groups_for_instance(self, instance_name: str) -> Set[str]:
        """Get the symmetric group for a specific instance"""
        for group in self.symmetric_groups:
            if instance_name in group:
                return group
        return {instance_name}  # Return singleton if not in any group


def apply_symmetry_constraints(parameters: Dict[str, Parameter], symmetry_constraints: SymmetryConstraint) -> Tuple[Dict[str, Parameter], Dict[str, List[str]]]:
    """Apply symmetry constraints to parameters and reduce parameter count"""
    # Create a mapping from original parameter names to merged parameter names
    param_mapping = {}
    reduced_parameters = {}
    
    # First process parameters in symmetry groups
    for group in symmetry_constraints.symmetric_groups:
        # Create a representative parameter for each parameter type
        for param_type in ['m', 'l', 'fw', 'segW', 'segL']:
            symmetric_params = symmetry_constraints.get_symmetric_parameters_for_group(group, param_type)
            
            # Check if these parameters exist in the original parameters
            existing_params = [p for p in symmetric_params if p in parameters]
            if not existing_params:
                continue
                
            # Select the first parameter as the representative parameter
            representative_param = existing_params[0]
            rep_param_obj = parameters[representative_param]
            
            # Create merged parameter
            # Use the name of the first instance in the group and the parameter type as the new parameter name
            first_instance = next(iter(group))
            merged_param_name = f"{first_instance}_{param_type}"
            
            # Create merged parameter object
            merged_param = Parameter(
                merged_param_name,
                rep_param_obj.type,
                rep_param_obj.value,
                rep_param_obj.min,
                rep_param_obj.max
            )
            
            # Add to reduced parameters list
            reduced_parameters[merged_param_name] = merged_param
            
            # Record mapping relationship
            for param_name in existing_params:
                param_mapping[param_name] = merged_param_name
    
    # Add non-symmetric parameters
    for param_name, param in parameters.items():
        if param_name not in param_mapping:
            # This parameter does not belong to any symmetry group, add directly to the reduced parameters list
            reduced_parameters[param_name] = param
            param_mapping[param_name] = param_name
    
    print(f"Parameter count reduced from {len(parameters)} to {len(reduced_parameters)}")
    
    return reduced_parameters, param_mapping


def read_parameters(file_name, dummy_devices: List[str] = None) -> Dict[str, Parameter]:
    """Read parameters from a configuration file"""
    parameters = {}
    if dummy_devices is None:
        dummy_devices = []

    with open(file_name, 'r') as param_file:
        lines = param_file.readlines()
        for line in lines:
            fields = line.strip().replace('"', '').replace(" ", "").split(',')
            if fields[0] == "parameter":
                name = fields[1]
                index = name.rfind("_")
                inst_name = fields[1][:index]
                param_name = name[(index+1):]

                # Check if this is a dummy device
                is_dummy = inst_name in dummy_devices
            
                if param_name == "m":
                    min_val = 1
                    max_val = 20
                    param_value = float(fields[2])
                    if (int(round(param_value)) < min_val) or (int(round(param_value)) > max_val):
                        RuntimeError(f'The param {param_name} of instance {inst_name} is out of range [{min_val}, {max_val}]!!!')
                    param = Parameter(name, 'integer', param_value, min_val, max_val, is_dummy)
                else:
                    min_val = 0.13
                    max_val = 20.0
                    param_value = param_convert(fields[2])
                    if (param_value < min_val) or (param_value > max_val):
                        RuntimeError(f'The param {param_name} of instance {inst_name} is out of range [{min_val}, {max_val}]!!!')
                    param = Parameter(name, 'continuous', param_value, min_val, max_val, is_dummy)
                parameters[name] = param
        print(f'==== Read {len(parameters)} parameters successfully, param file: {file_name} ====\n')
    return parameters


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
        self.best_fitness = 1e10
        self.best_iter = 0
        self.best_path = ''
        self.iter_count = 0  # Track iterations
        self.param_mapping = {}  # original param ---> reduced param
        self.reverse_mapping = {}  # Reverse mapping from reduced to original parameters
        self.base_ugb = 0.0
        self.base_area = 0.0

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
            print(f"delete old dir: {self.output_path}")
    
        os.makedirs(self.output_path, exist_ok=True)
        print(f"create new dir: {self.output_path}")


    def set_params(self, reduced_params: Dict[str, Parameter], values: np.ndarray):
        """Apply parameter values to the circuit design"""
        cv = ae.dbOpenCV(self.ae_lib, self.ae_cell, self.ae_view)
        if not cv:
            raise RuntimeError(f'ae.dbOpenCV({self.ae_lib}, {self.ae_cell}, {self.ae_view}) failed')

        # Map reduced parameter values back to original parameters
        reduced_param_items = list(reduced_params.items())

        # Apply parameter values
        for i, (reduced_name, param) in enumerate(reduced_param_items):
            # Get all original parameters corresponding to this reduced parameter
            original_params = self.reverse_mapping.get(reduced_name, [reduced_name])
            for orig_name in original_params:
                # Parse instance name and parameter name
                index = orig_name.rfind("_")
                inst_name = orig_name[:index]
                param_name = orig_name[(index+1):]

                instId = ae.dbFindInst(cv=cv, name=inst_name)
                if not instId:
                    print(f'Warning: Instance {inst_name} not found')
                    continue

                formatted_value = param.format_value(values[i])
                # print(f"Set {inst_name} {param_name} = {formatted_value}")

                # Set params
                ret = ae.cdfSetParam(instId, param_name, formatted_value)
                if not ret:
                    print(f'Warning: Set {orig_name}={formatted_value} failed')

        # Set dummy parameters to their minimum values
        for dummy_name, dummy_param in self.dummy_params.items():
            # Parse instance name and parameter name
            index = dummy_name.rfind("_")
            inst_name = dummy_name[:index]
            param_name = dummy_name[(index+1):]
            
            instId = ae.dbFindInst(cv=cv, name=inst_name)
            if not instId:
                print(f'Warning: Dummy instance {inst_name} not found')
                continue
            
            # Set dummy parameter to minimum value
            formatted_value = dummy_param.format_value(dummy_param.min)
            ret = ae.cdfSetParam(instId, param_name, formatted_value)
            if not ret:
                print(f'Warning: Set dummy {dummy_name}={formatted_value} failed')
            # else:
            #     print(f'Set dummy {dummy_name} to minimum value: {formatted_value}')

        save_result = ae.dbCheckAndSaveDesign(cv)
        print(f'Design save result: {save_result}')
        ae.dbCloseCV(cv)

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


    def normalize_values(self, reduced_params: Dict[str, Parameter], real_values: np.ndarray) -> np.ndarray:
        """Normalize real parameter values to [0, 1] range"""
        normalized_values = np.zeros_like(real_values)
        reduced_param_items = list(reduced_params.items())
        
        for i, (name, param) in enumerate(reduced_param_items):
            normalized_values[i] = param.normalize(real_values[i])
            
        return normalized_values


    def denormalize_values(self, reduced_params: Dict[str, Parameter], normalized_values: np.ndarray) -> np.ndarray:
        """Convert normalized parameter values back to real range"""
        real_values = np.zeros_like(normalized_values)
        reduced_param_items = list(reduced_params.items())
        
        for i, (name, param) in enumerate(reduced_param_items):
            real_values[i] = param.denormalize(normalized_values[i])
            
        return real_values


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
        print(f"====== Gain_Margin  : {scores['Gain_Margin']} deg")
        print(f"====== I_OPA        : {scores['I_OPA'] * 1000.0} mA")
        print(f"====== Total_Area   : {scores['Total_Area']} um^2\n")
        return scores


    def optimization_objective(self, normalized_values: np.ndarray, 
                            reduced_params: Dict[str, Parameter]) -> float:
        """Objective function for optimization"""

        real_values = self.denormalize_values(reduced_params, normalized_values)

        # Run simulation and get performance score
        self.set_params(reduced_params, real_values)  # set params
        scores = self.evaluate()

        # Compute fitness
        fitness = 0.0
        phase_margin = scores['Phase_Margin']
        if (phase_margin == -float("inf")):
            fitness += 1e9
        elif (phase_margin < 0.1):
            fitness += 1e7
        elif (phase_margin < 50.0):      # constraint PM >= 50
            fitness += (1e6 * (50.0 - phase_margin) / 50.0 + 1e6)   # 1e6 ~ 2e6

        gain_margin = scores['Gain_Margin']
        if (gain_margin == float("inf")):
            fitness += 1e9
        elif (gain_margin > -0.1):
            fitness += 1e7
        elif (gain_margin > -10.0):      # constraint GM <= -10
            fitness += (1e6 * (10.0 + gain_margin) / 10.0 + 1e6)    # 1e6 ~ 2e6

        gain_db = scores['Gain_db']
        if (gain_db == -float("inf")):
            fitness += 1e9
        elif (gain_db < 0.1):
            fitness += 1e7
        elif (gain_db < 80.0):           # constraint |Gain| >= 80 db
            fitness += (1e6 * (80 - gain_db) / 80.0 + 1e6)          # 1e6 ~ 2e6

        if (scores['I_OPA'] == float("inf")):
            fitness += 1e9
        elif (scores['I_OPA'] > 0.003):  # constraint I_OPA <= 3 mA
            fitness += 1e6

        if (scores['UGB'] == -float("inf")):
            fitness += 1e9

        if (fitness > 1e8):
            print(f"==== iter {self.iter_count} fitness: {fitness}")
            return fitness

        # Expected UGB to be large
        ugb = scores['UGB']
        if (self.base_ugb < 1.0):
            self.base_ugb = ugb
        fitness -= (ugb / self.base_ugb * 150.0)

        # Expected small area
        area = scores['Total_Area']
        if (self.base_area < 1.0):
            self.base_area = area
        fitness += (area / self.base_area * 250.0)

        print(f"==== iter {self.iter_count} fitness: {fitness}")

        # Track best fitness
        if (fitness < self.best_fitness):
            self.best_fitness = fitness
            self.best_iter = self.iter_count
            self.best_path = f"{self.output_path}/iter_{self.iter_count}"
            print(f"New best score: {fitness}")

            with open(f"{self.output_path}/best_result.txt", 'w') as best_f:
                best_f.write(f"Best iter    : {self.best_iter}\n")
                best_f.write(f"Best path    : {self.best_path}\n")
                best_f.write(f"UGB          : {scores['UGB']} Hz\n")
                best_f.write(f"Phase_Margin : {scores['Phase_Margin']} deg\n")
                best_f.write(f"Gain_db      : {scores['Gain_db']} db\n")
                best_f.write(f"Gain_Margin  : {scores['Gain_Margin']} deg\n")
                best_f.write(f"I_OPA        : {scores['I_OPA'] * 1000.0} mA\n")
                best_f.write(f"Total_Area   : {scores['Total_Area']} um^2\n")
                best_f.write(f"Fitness      : {fitness}\n")

        return fitness


    def run_de_optimization(
        self,
        reduced_params: Dict[str, Parameter],
        max_iter: int = 10,
        pop_size: int = 5
    ):
        """Run differential evolution optimization"""
        print("\n==== Starting Differential Evolution Optimization ====")

        # Prepare parameter bounds for DE (normalized to [0,1])
        param_items = list(reduced_params.items())
        bounds = [(0, 1)] * len(param_items)
        
        try:
            print("\n==== Optimization Begin ====")
            print(f"pop_size: {pop_size}, max_iter: {max_iter}, params_num: {len(param_items)}")
            # Run differential evolution
            result = differential_evolution(
                func=lambda x: self.optimization_objective(x, reduced_params),
                bounds=bounds,
                popsize=pop_size,
                maxiter=max_iter,
                tol=0.01,
                mutation=(0.5, 1.0),
                recombination=0.7,
                strategy='best1bin',
                polish=False,
                workers=1
            )

            # Print optimization results
            print("\n==== Optimization Completed ====")
            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            print(f"Final cost: {result.fun}")
            print("Optimized parameters:")

            # Convert normalized results to actual values
            real_values = self.denormalize_values(reduced_params, result.x)
            for i, (name, param) in enumerate(param_items):
                print(f"  {name}: {param.format_value(real_values[i])}")

            # Save best parameters to file
            with open(f"{self.output_path}/optimized_params.txt", 'w') as f:
                f.write("Optimized Parameters:\n")
                for i, (name, param) in enumerate(param_items):
                    f.write(f"{name}: {param.format_value(real_values[i])}\n")
                f.write(f"\nFinal Performance: {result.fun}")

        except Exception as e:
            # Handle user interruption
            print(f"\n\n=== Optimization interrupted by user ===\n{e}")
            print(f"Best solution found at iteration {self.best_iter}")
            print(f"Best parameters saved to: {self.best_path}")


def parse_symmetry_constraints(constraint_str: str) -> SymmetryConstraint:
    """Parse symmetry constraints from string format"""
    if not constraint_str or constraint_str.strip() == "":
        return None
        
    groups = []
    # Parse format like: "(PM1,PM2,PM3),(NM4,NM5),(NM6,NM7,NM8,NM9)"
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, constraint_str)
    
    for match in matches:
        instances = [inst.strip() for inst in match.split(',')]
        if len(instances) > 1:  # Only add groups with at least 2 instances
            groups.append(tuple(instances))
    
    return SymmetryConstraint(groups)


def parse_arguments():
    """Parse command-line arguments for the optimization tool"""
    parser = argparse.ArgumentParser(
        description='Circuit Optimization Platform',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ae_lib', required=True, help='Aether library name')
    parser.add_argument('--ae_cell', required=True, help='Aether cell name')
    parser.add_argument('--ae_view', required=True, help='Aether view name')
    parser.add_argument('--mde_cell', required=True, help='MDE cell name')
    parser.add_argument('--mde_view', required=True, help='MDE view name')
    parser.add_argument('--param_file', required=True, help='Parameter file path')
    parser.add_argument('--output_path', required=True, help='Output directory path')
    parser.add_argument('--output_file', required=True, help='Output file name')
    parser.add_argument('--symmetry_devices', default='', 
                        help='Symmetry devices in format "(inst1,inst2,inst3),(inst4,inst5)"')
    parser.add_argument('--dummy_devices', default='', 
                        help='Dummy devices in format "inst1,inst2,inst3"')

    # Optimization parameters
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--pop_size', type=int, default=300,
                        help='Population size for differential evolution')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output for debugging')
    parser.add_argument('--set_params', action='store_true',
                        help='Set parameters using specific param_file')
    parser.add_argument('--evaluate', action='store_true',
                        help='Simulate using specific parameters')
    parser.add_argument('--use_algorithm_de', action='store_true',
                        help='Using the DE algorithm to optimize parameters')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    if args.verbose:
        print("\n[VERBOSE] Starting optimization with parameters:")
        print(f"  Aether Library: {args.ae_lib}")
        print(f"  Aether Cell: {args.ae_cell}, View: {args.ae_view}")
        print(f"  MDE Cell: {args.mde_cell}, View: {args.mde_view}")
        print(f"  Output Path: {args.output_path}")
        print(f"  Output File: {args.output_file}")
        print(f"  Parameter File: {args.param_file}")
        print(f"  Max Iterations: {args.max_iter}")
        print(f"  Population Size: {args.pop_size}")
        print(f"  Dummy Devices: {args.dummy_devices}\n")

    # Initialize Aether environment
    ae.emyInitAether('-adv')

    # Parse symmetry constraints
    symmetry_constraints = parse_symmetry_constraints(args.symmetry_devices)
    if symmetry_constraints:
        print(f"Symmetric groups: {symmetry_constraints.symmetric_groups}")

    # Parse dummy devices
    dummy_devices = [d.strip() for d in args.dummy_devices.split(',')] if args.dummy_devices else []
    if dummy_devices:
        print(f"Dummy devices: {dummy_devices}")

    # Read parameter definitions
    parameters = read_parameters(args.param_file, dummy_devices)

    # Filter out dummy parameters
    non_dummy_params = {name: param for name, param in parameters.items() if not param.is_dummy}
    dummy_params = {name: param for name, param in parameters.items() if param.is_dummy}
    
    # print(f'==== Found {len(dummy_params)} dummy device parameters: {list(dummy_params.keys())} ====')
    # print(f'==== Using {len(non_dummy_params)} parameters for optimization: {list(non_dummy_params.keys())} ====\n')

    print(f'==== Found {len(dummy_params)} dummy device parameters')
    print(f'==== Using {len(non_dummy_params)} parameters for optimization\n')
    
    # Apply symmetry constraints to reduce params
    if symmetry_constraints:
        reduced_parameters, param_mapping = apply_symmetry_constraints(non_dummy_params, symmetry_constraints)
    else:
        reduced_parameters = non_dummy_params
        param_mapping = None

    # Create optimization platform
    platform = SimulatePlatform(
        ae_lib=args.ae_lib,
        ae_cell=args.ae_cell,
        ae_view=args.ae_view,
        mde_cell=args.mde_cell,
        mde_view=args.mde_view,
        output_path=args.output_path,
        output_file=args.output_file,
        symmetry_constraints=symmetry_constraints,
        dummy_params=dummy_params
    )

    platform.param_mapping = param_mapping

    # Create reverse mapping
    if param_mapping:
        platform.reverse_mapping = {}
        for orig_name, reduced_name in param_mapping.items():
            if reduced_name not in platform.reverse_mapping:
                platform.reverse_mapping[reduced_name] = []
            platform.reverse_mapping[reduced_name].append(orig_name)

    if args.set_params:
        platform.only_set_params(parameters)

    if args.evaluate:
        platform.evaluate()

    if args.use_algorithm_de:
        # Run differential evolution optimization
        platform.run_de_optimization(
            reduced_params=reduced_parameters,
            max_iter=args.max_iter,
            pop_size=args.pop_size
        )

    if args.verbose:
        print("\n[VERBOSE] Optimization completed")
