import pyAether as ae
import argparse
import re
import os
import shutil
import sys
from typing import Dict


def merge_files(file_list, output_file):
    """Merge multiple files into a single output file"""
    with open(output_file, 'w') as output:
        for file_name in file_list:
            with open(file_name, 'r') as input_file:
                output.write(input_file.read())
                output.write("\n\n")


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
        max_val: float = None
    ):
        self.name = name
        self.type = param_type
        self.value = value
        self.min = min_val
        self.max = max_val

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


def read_parameters(file_name) -> Dict[str, Parameter]:
    """Read parameters from a configuration file"""
    parameters = {}
    with open(file_name, 'r') as param_file:
        lines = param_file.readlines()
        for line in lines:
            fields = line.strip().replace('"', '').replace(" ", "").split(',')
            if fields[0] == "parameter":
                name = fields[1]
                index = name.rfind("_")
                inst_name = fields[1][:index]
                param_name = name[(index+1):]

                if param_name == "m":
                    min_val = 1
                    max_val = 20
                    param_value = float(fields[2])
                    # if (int(round(param_value)) < min_val) or (int(round(param_value)) > max_val):
                    #     raise RuntimeError(f'The param {param_name} of instance {inst_name} is out of range [{min_val}, {max_val}]!!!')
                    # if (int(round(param_value)) < min_val):
                    #     param_value = min_val
                    # if (int(round(param_value)) > max_val):
                    #     param_value = max_val
                    param = Parameter(name, 'integer', param_value, min_val, max_val)
                else:
                    min_val = 0.13
                    max_val = 20.0
                    if param_name == "l":
                        if (inst_name[0] == "C"  or  inst_name[0] == 'c'):
                            min_val = 4.0
                            max_val = 30.0
                    elif param_name == "fw":
                        min_val = 0.15
                        max_val = 50.0
                    elif param_name == "segW":
                        min_val = 0.42
                        max_val = 10.0
                    elif param_name == "segL":
                        min_val = 5.0
                        max_val = 100.0

                    param_value = param_convert(fields[2])
                    # if (param_value < min_val) or (param_value > max_val):
                    #     raise RuntimeError(f'The param {param_name} of instance {inst_name} is out of range [{min_val}, {max_val}]!!!')
                    if (param_value < min_val):
                        param_value = min_val
                    if (param_value > max_val):
                        param_value = max_val
                    param = Parameter(name, 'continuous', param_value, min_val, max_val)
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
        output_file: str = None
    ):
        self.origin_lib = ae_lib
        self.ae_lib = ae_lib
        self.ae_cell = ae_cell
        self.ae_view = ae_view
        self.mde_cell = mde_cell
        self.mde_view = mde_view
        self.output_path = output_path
        self.output_file = output_file

        self.best_fitness = 1e10
        self.best_iter = 0
        self.best_path = ''
        self.iter_count = 0  # Track iterations

        self.ugb_base = 1e8
        self.area_base = 5e5
        self.time_base = 7200.0

        self.pm_score = 0.0
        self.gm_score = 0.0
        self.gain_score = 0.0
        self.i_opa_score = 0.0
        self.ugb_score = 0.0
        self.area_score = 0.0
        self.time_score = 0.0
        self.total_score = 0.0

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

        print(f"\n========== Evaluation result of iter {self.iter_count} ==========")
        print(f"====== UGB          : {scores['UGB']} Hz")
        print(f"====== Phase_Margin : {scores['Phase_Margin']} deg")
        print(f"====== Gain_db      : {scores['Gain_db']} db")
        print(f"====== Gain_Margin  : {scores['Gain_Margin']} deg")
        print(f"====== I_OPA        : {scores['I_OPA'] * 1000.0} mA")
        print(f"====== Total_Area   : {scores['Total_Area']} um^2\n")
        return scores

    def output_score(self, file_name):

        output = f"==============================\n"
        output += f"       score result           \n"
        output += f"------------------------------\n"
        output += f"PM score      : {self.pm_score:.4f}\n"
        output += f"GM score      : {self.gm_score:.4f}\n"
        output += f"Gain score    : {self.gain_score:.4f}\n"
        output += f"I_OPA score   : {self.i_opa_score:.4f}\n"
        output += f"UGB score     : {self.ugb_score:.4f}\n"
        output += f"Area score    : {self.area_score:.4f}\n"
        output += f"Time score    : {self.time_score:.4f}\n"
        output += "------------------------------\n"
        output += f"Total score   : {self.total_score:.4f}\n"
        print(output)
        with open(file_name, 'w') as score_f:
            score_f.write(output)


    def calc_score(self, params: Dict[str, Parameter], execution_time: float) -> float:

        # Run simulation and get performance score

        self.only_set_params(params)
        scores = self.evaluate()

        # constraint 1
        if (scores['Phase_Margin'] >= 50.0):   # constraint PM >= 50
            self.pm_score = 10.0


        # constraint 2
        if (scores['Gain_Margin'] <= -10.0):   # constraint GM <= -10
            self.gm_score = 10.0

        # constraint 3
        if (scores['Gain_db'] >= 80.0):        # constraint |Gain| >= 80 db
            self.gain_score = 10.0

        # constraint 4
        if (scores['I_OPA'] <= 0.003):         # constraint I_OPA <= 3 mA
            self.i_opa_score = 10.0

        # obj 1
        if (scores['UGB'] > 1.0):
            self.ugb_score = min((scores['UGB'] / self.ugb_base * 15.0), 15.0)

        # obj 2
        if (scores['Total_Area'] > 1.0):
            self.area_score = min((self.area_base / scores['Total_Area'] * 25.0), 25.0)
        
        # time
        if (execution_time > self.time_base):
            self.time_score = min((self.time_base / execution_time * 20.0), 20.0)
        else:
            self.time_score = 20.0

        # total score
        self.total_score = self.pm_score + self.gm_score + self.gain_score + self.i_opa_score + self.ugb_score + self.area_score + self.time_score
    
 
        self.output_score(f"{self.ae_lib}_score.txt")

        return self.total_score


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
    parser.add_argument('--param_file', required=True, help='Parameter file name')
    parser.add_argument('--best_param_file', required=True, help='Best parameter file name')
    parser.add_argument('--output_path', required=True, help='Output directory path')
    parser.add_argument('--output_file', required=True, help='Output file name')

    # score parameters
    parser.add_argument('--calc_score', action='store_true',
                        help='Calc total score using specific param_file')

    parser.add_argument('--ugb_base', type=float, default=1e8,
                        help='Baseline value of UGB')
    parser.add_argument('--area_base', type=float, default=5e5,
                        help='Baseline value of area')
    parser.add_argument('--time_base', type=float, default=7200.0,
                        help='Baseline value of run time')

    parser.add_argument('--execution_time', type=float, default=1e10,
                        help='Set run time')

    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output for debugging')

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
        print(f"  Parameter File: {args.param_file}\n")

    # Initialize Aether environment
    ae.emyInitAether('-adv')

    # Create optimization platform
    platform = SimulatePlatform(
        ae_lib=args.ae_lib,
        ae_cell=args.ae_cell,
        ae_view=args.ae_view,
        mde_cell=args.mde_cell,
        mde_view=args.mde_view,
        output_path=args.output_path,
        output_file=args.output_file,
    )

    # Read parameter definitions
    init_parameters = {}
    if os.path.exists(args.param_file):
        init_parameters = read_parameters(args.param_file)
    else:
        print("Init param file not exists!!!!")
        platform.output_score(f"{args.ae_lib}_score.txt")
        sys.exit(0)

    parameters = {}
    if os.path.exists(args.best_param_file):
        parameters = read_parameters(args.best_param_file)
    else:
        print("Best param file not exists!!!!")
        platform.output_score(f"{args.ae_lib}_score.txt")
        sys.exit(0)
    
    if (len(parameters) != len(init_parameters)):
        platform.output_score(f"{args.ae_lib}_score.txt")
        sys.exit(0)

    if args.calc_score:
        if (args.ugb_base > 1.0):
            platform.ugb_base = args.ugb_base
        if (args.area_base > 1.0):
            platform.area_base = args.area_base
        if (args.time_base > 1.0):
            platform.time_base = args.time_base
        score = platform.calc_score(parameters, args.execution_time)


    if args.verbose:
        print("\n[VERBOSE] Optimization completed")
