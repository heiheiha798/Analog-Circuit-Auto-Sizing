import pyAether as ae
import shutil
import argparse
import multiprocessing
import time
import signal
import copy
from typing import Dict, Optional, Callable

def merge_files(file_list, output_file):
    """Merge multiple files into a single output file"""
    with open(output_file, 'w') as output:
        for file_name in file_list:
            with open(file_name, 'r') as input_file:
                output.write(input_file.read())
                output.write("\n\n")

class Parameter:
    """Represents an optimization parameter with constraints"""
    def __init__(
        self,
        name: str,
        param_type: str = 'continuous',
        default: str = None,
        min_val: str = None,
        max_val: str = None,
    ):
        self.name = name
        self.type = param_type
        self.default = default
        self.min = min_val
        self.max = max_val


def read_parameters(file_name) -> Dict[str, Parameter]:
    """Read parameters from a configuration file"""

    parameters = {}
    with open(file_name, 'r') as param_file:
        lines = param_file.readlines()
        for line in lines:
            fields = line.strip().replace('"', '').split(',')
            if fields[0] == "parameter":
                if fields[1] == "vdd":
                    continue
                index = fields[1].rfind("_")
                inst_name = fields[1][:index]
                param_name = fields[1][(index+1):]
                if (param_name == "m") or (param_name == "segments"):
                    param_type = 'integer'
                    param_min = '1'
                    param_max = '20'
                else:
                    param_type = 'continuous'
                    param_min = '0.13u'
                    param_max = '20.0u'
                param = Parameter(fields[1], param_type, fields[2], param_min, param_max)
                # print(f'Instance {inst_name}: {param_name} = {fields[2]}')
                parameters[fields[1]] = param
    print(f'==== Read parameters successfully, param file: {file_name} ====\n')
    return parameters


def optimize_params(params: Dict[str, Parameter]) -> Dict[str, Parameter]:
        """Placeholder for optimization algorithm (to be implemented)"""
        return params


class SimulatePlatform:
    """Platform for circuit optimization using Aether tools"""
    def __init__(
        self,
        ae_lib: str = None,
        ae_cell: str = None,
        ae_view: str = None,
        mde_cell: str = None,
        mde_view: str = None
    ):
        self.origin_lib = ae_lib
        self.ae_lib = ae_lib
        self.ae_cell = ae_cell
        self.ae_view = ae_view
        self.mde_cell = mde_cell
        self.mde_view = mde_view


    def set_params(self, params: Dict[str, Parameter]):
        """Apply parameters to the circuit design"""
        cv = ae.dbOpenCV(self.ae_lib, self.ae_cell, self.ae_view)
        if not cv:
            raise RuntimeError(f'ae.dbOpenCV({self.ae_lib}, {self.ae_cell}, {self.ae_view}) failed, please check !!!')

        for name in params:
            if params[name].name == "vdd":
                continue
            index = params[name].name.rfind("_")
            inst_name = params[name].name[:index]
            param_name = params[name].name[(index+1):]
            if (param_name == "seg"):
                param_name = "segments"
            instId = ae.dbFindInst(cv=cv, name=inst_name)
            if not instId:
                print(f'Can not find instance {inst_name}, please check !!!')
                continue
            ret = ae.cdfSetParam(instId, param_name, params[name].default)
            if (ret == False):
                print(f'Instance {inst_name} set param {param_name}={params[name].default} failed !!!')
        save_result = ae.dbCheckAndSaveDesign(cv)
        print(f'\n==== Design save result: {save_result} ====\n')
        ae.dbCloseCV(cv)


    def evaluate(self, output_path, output_file_name):
        """Run simulation and collect results"""
        mde = ae.MdeSession.open(self.ae_lib, self.mde_cell, self.mde_view)
        if mde:
            print(f"\n==== ae.MdeSession.open({self.ae_lib}, {self.mde_cell}, {self.mde_view}) successfully ====")
        else:
            print(f"\n==== ae.MdeSession.open({self.ae_lib}, {self.mde_cell}, {self.mde_view}) failed. Reason: {ae.MdeSession.staticLastError()}")
            return ""

        print(f'\n==== Project directory: {mde.getSetting().getProjectDirectory()}')
        setting = mde.getSetting()
        print(f'\n==== Current settings: {setting}')
        # print(f'\n==== SPE outputs: {setting.getAllSpeOutputs()}')
	
        if (mde.createSimulationNetlist()):
            print("Creating simulation netlist succeeds.")
        else:
            print("Creating simulation netlist fails.")
 
        # Execute simulation
        print("\n==== Begin simulate ====")
        result = mde.netlistAndRun()
        if result:
            print("\n==== Simulation completed successfully ====")
        else:
            print(f'\n==== Simulation failed. Reason: {mde.lastError()}')
            mde.close()
            return ""
	    
        if not result.isValid():
            print(f'\n==== Invalid results. Reason: {result.lastError()}')
            mde.close()
            return ""
	    
        # Save results in CSV format
        result.saveResultsToDir(output_path)
        if not result.isValid():
            print(f'\n==== Results are invalid. Reason: {result.lastError()}')
            mde.close()
            return ""
        
        # Combine results into single log file
        print(f'\n==== Result name: {result.name()}')
        merge_files([f"{output_path}/{result.name()}_summary.csv"], f"{output_path}/{output_file_name}")
	    
        # Copy simulation log file
        src_path = f"{mde.getSetting().getProjectDirectory()}/{self.ae_lib}/{self.mde_cell}/{self.mde_view}/Results/{result.name()[len('Results'):]}"
        try:
            shutil.copy2(f"{src_path}/emy2netlist.log", f"{output_path}/emy2netlist.log")
        except Exception as e:
            print(f"Failed to copy emy2netlist.log, details: {str(e)}")
        if mde.close():
            print("==== Succeeds in closing MDE ====")
        else:
            print(f"==== Fails to close MDE, reason: {mde.lastError()}")

        return f"{output_path}/{output_file_name}"


    def run_optimization(
        self,
        params: Dict[str, Parameter],
        output_path,
        output_file_name,
        iterations: int = 10,
        algorithm: Optional[Callable] = None
    ):
        
        for i in range(iterations):
            print(f"==== Begin iteration {i} ====")
            # optimization parameters
            if algorithm:
                params = algorithm(self, i)
            
            # just for test
            #params["PM630_m"].default = str(i * 2)
            #params["PM630_l"].default = str(i * 2) + "u"
            #params["PM630_fw"].default = str(i * 2) + "u"

            #params["NM486_m"].default = str(i * 2)
            #params["NM486_l"].default = str(i * 2) + "u"
            #params["NM486_fw"].default = str(i * 2) + "u"
            
            # set parameters
            self.set_params(params)

            # evaluate
            out_put_path = output_path + "_" + str(i)
            self.evaluate(out_put_path, output_file_name)
            print(f"==== End iteration {i} ====")


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

    # Optional parameters
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of optimization iterations')
    parser.add_argument('--process', type=int, default=1,
                        help='Number of process')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output for debugging')
    parser.add_argument('--copy', action='store_true',
                        help='Only copy lib')
    
    return parser.parse_args()


# Task function for child processes with parameters
def worker(task_id, args, status_queue):

    print(f"Child process {task_id} started: [PID={multiprocessing.current_process().pid}]")

    try:
        # Initialize Aether environment
        ae.emyInitAether('-adv')

        # Read parameter definitions
        params = read_parameters(args.param_file)

        # Create simulate platform instance
        platform = SimulatePlatform(
            ae_lib = args.ae_lib,
            ae_cell = args.ae_cell,
            ae_view = args.ae_view,
            mde_cell = args.mde_cell,
            mde_view = args.mde_view
        )

        iterations = 5
        if args.iterations:
            iterations = args.iterations

        for i in range(iterations):
            print(f"==== Begin task {task_id} iteration {i} ====")
            # optimization parameters
            #params["PM630_m"].default = str((i + 1) * 2)
            #params["PM630_l"].default = str((i + 1) * 2) + "u"
            #params["PM630_fw"].default = str((i + 1) * 2) + "u"

            #params["NM486_m"].default = str((i + 1) * 2)
            #params["NM486_l"].default = str((i + 1) * 2) + "u"
            #params["NM486_fw"].default = str((i + 1) * 2) + "u"

            # set parameters
            platform.set_params(params)
            print(f"==== Task {task_id} iteration {i} finish params set ====")

            # evaluate
            out_put_path = args.output_path + f"_task{task_id}_iter{i}"
            ret = platform.evaluate(out_put_path, args.output_file)
            print(f"==== End task {task_id} iteration {i} ====")
            if (ret == ""):
                raise RuntimeError(f'platform.evaluate({out_put_path}, {args.output_file}) failed!!!')

            # send result to main process
            status_queue.put((task_id, "ALIVE", f"result: {ret}"))
        
        # Task completed
        status_queue.put((task_id, "COMPLETED", f"Task finished all {args.iterations} iterations"))

    except Exception as e:
        status_queue.put((task_id, "ERROR", str(e)))

# Main process monitoring function (unchanged)
def monitor_processes(processes, status_queue, timeout=3600):

    print(f"Main process started [PID={multiprocessing.current_process().pid}]")
    start_time = time.time()
    
    # Signal handler for graceful termination (Ctrl+C)
    def handle_signal(signum, frame):
        print("\nTermination signal received, stopping child processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        exit(1)
    
    signal.signal(signal.SIGINT, handle_signal)
    
    # Monitor until all processes complete or timeout occurs
    while time.time() - start_time < timeout:
        # Process status updates from queue
        while not status_queue.empty():
            task_id, status, message = status_queue.get()
            print(f"[Monitor] Task {task_id}: {status} -> {message}")
            
            # Handle completed/failed tasks
            if status in ["COMPLETED", "ERROR"]:
                # Find and remove corresponding process
                for p in processes[:]:
                    if p.name == f"Worker-{task_id}":
                        p.join()  # Ensure process termination
                        processes.remove(p)
                        print(f"Task {task_id} removed from monitoring")
        
        # Check active processes count
        active_count = sum(1 for p in processes if p.is_alive())
        if active_count == 0:
            print("All child processes completed!")
            return True
        
        # Display monitoring status
        # print(f"[Monitor] Active: {active_count}/{len(processes)} | Timeout in: {int(timeout - (time.time() - start_time))}s")
        time.sleep(1)  # Reduce CPU usage
    
    # Handle timeout scenario
    print(f"\nTimeout warning: {timeout} seconds reached, terminating remaining processes")
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()
    return False


def copy_multi_process_libs(args):

    for i in range(4):

        # copy lib
        lib_name = args.ae_lib + f"_task{i}"
        ret = ae.aeCopyLib(args.ae_lib, lib_name)
        if (ret == True):
            print(f"==== Copy {args.ae_lib} to {lib_name} success!!!")
        else:
            print(f"==== Copy {args.ae_lib} to {lib_name} failed!!!")


def run_multi_process_optimization(origin_args):

    # Create inter-process communication queue
    status_queue = multiprocessing.Queue()
    
    # Create child processes with different configurations
    processes = []

    for i in range(4):
        # copy args
        copy_args = copy.deepcopy(origin_args)

        # copy lib
        copy_args.ae_lib = copy_args.ae_lib + f"_task{i}"
        ae.aeCopyLib(args.ae_lib, copy_args.ae_lib)

        p = multiprocessing.Process(
            target=worker,
            args=(i, copy_args, status_queue),
            name=f"Worker-{i}"
        )
        processes.append(p)
        p.start()
        print(f"\n\n======== Started task {i} ========")
    
    # Start monitoring
    success = monitor_processes(processes, status_queue)
    
    # Cleanup queue
    while not status_queue.empty():
        status_queue.get()
    
    print(f"Main process exiting. Status: {'SUCCESS' if success else 'TIMEOUT FAILURE'}")


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
        print(f"  Iterations: {args.iterations}\n")

    # # Initialize Aether environment
    ae.emyInitAether('-adv')
    #
    # # Read parameter definitions
    # parameters = read_parameters(args.param_file)
    #
    # # Optimize parameters
    # parameters = optimize_params(parameters)
    #
    # # Create optimization platform instance
    # platform = SimulatePlatform(
    #     ae_lib = args.ae_lib,
    #     ae_cell = args.ae_cell,
    #     ae_view = args.ae_view,
    #     mde_cell = args.mde_cell,
    #     mde_view = args.mde_view
    # )
    #
    # # Apply parameters to the design
    # platform.set_params(parameters)
    #
    # # Run evaluation
    # platform.evaluate(args.output_path, args.output_file)

    if args.copy:
        copy_multi_process_libs(args)
    else:
        # Run multi process optimization
        run_multi_process_optimization(args)

    if args.verbose:
        print("\n[VERBOSE] Optimization completed successfully")