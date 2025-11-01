import argparse
import os
import multiprocessing
import time
import glob
import pyAether as ae

def get_view_list():
    """Generates the list of 9 MDE views to be simulated."""
    
    # Base name for the views
    base_view_name = "spe_state1_Corner"
    
    # 8 corner configurations
    models = ["ss", "ff"]
    vdds = ["1.62", "1.98"]
    temps = ["-40", "125"]
    
    view_list = []
    for model in models:
        for vdd in vdds:
            for temp in temps:
                # Recreate the name exactly as in the preparation script
                temp_for_name = temp.replace('-', 'm')
                view_name = f"{base_view_name}_{model}_{vdd.replace('.', 'p')}_{temp_for_name}"
                view_list.append(view_name)
    
    # Add the 1 nominal view
    view_list.append(f"{base_view_name}_nominal")
    
    return view_list

def run_simulation_worker(task_id, lib_name, cell_name, view_name, output_path):
    """
    This function is executed by each child process.
    It runs a single simulation for a given MDE view.
    """
    start_time = time.time()
    print(f"[Worker-{task_id} | PID:{os.getpid()}] Starting simulation for view: {view_name}")

    # CRITICAL: Each child process must initialize its own Aether environment.
    ae.emyInitAether('-adv')

    mde = None
    try:
        # 1. Open the specific MDE session for this worker
        mde = ae.MdeSession.open(lib_name, cell_name, view_name)
        if not mde:
            error_msg = f"Failed to open MDE session. Reason: {ae.MdeSession.staticLastError()}"
            print(f"[Worker-{task_id}] ERROR: {error_msg}")
            return (view_name, "FAILURE", error_msg)

        # 2. Run the simulation
        result = mde.netlistAndRun()
        if not result or not result.isValid():
            error_msg = f"Simulation failed or result is invalid. Reason: {mde.lastError() or result.lastError()}"
            print(f"[Worker-{task_id}] ERROR: {error_msg}")
            return (view_name, "FAILURE", error_msg)
            
        # 3. Save the results to a unique directory
        corner_output_dir = os.path.join(output_path, view_name)
        os.makedirs(corner_output_dir, exist_ok=True)
        result.saveResultsToDir(corner_output_dir)
        
        # Find the summary file to confirm success
        summary_files = glob.glob(os.path.join(corner_output_dir, "*_summary.csv"))
        if not summary_files:
            error_msg = "Simulation ran but no summary CSV file was found."
            print(f"[Worker-{task_id}] ERROR: {error_msg}")
            return (view_name, "FAILURE", error_msg)

    except Exception as e:
        error_msg = f"An unexpected exception occurred: {str(e)}"
        print(f"[Worker-{task_id}] FATAL ERROR: {error_msg}")
        return (view_name, "FAILURE", error_msg)
    finally:
        if mde:
            mde.close()

    duration = time.time() - start_time
    success_msg = f"Completed in {duration:.2f}s. Results in: {corner_output_dir}"
    print(f"[Worker-{task_id}] SUCCESS: {success_msg}")
    return (view_name, "SUCCESS", success_msg)


def main():
    parser = argparse.ArgumentParser(
        description="""
        A simple test script to check if pyAether supports parallel MDE simulations
        using Python's multiprocessing.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--lib', default="2025_EDA_case1", help='Aether library name')
    parser.add_argument('--cell', default="OPA_loopgain", help='Cell name containing the MDE views')
    parser.add_argument('--output_path', default="./parallel_sim_results", help='Base directory for simulation outputs')
    args = parser.parse_args()

    print("--- Parallel Simulation Test for pyAether ---")
    print(f"Library: {args.lib}")
    print(f"Cell: {args.cell}")
    print(f"Output Path: {args.output_path}")

    # Create the main output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Get the list of 9 views we need to simulate
    views_to_simulate = get_view_list()
    print(f"\nFound {len(views_to_simulate)} MDE views to simulate in parallel.")

    # Prepare the list of tasks for the process pool
    tasks = []
    for i, view_name in enumerate(views_to_simulate):
        tasks.append((i, args.lib, args.cell, view_name, args.output_path))

    # --- Start the multiprocessing pool ---
    # We will run all 9 simulations concurrently.
    num_processes = len(tasks)
    print(f"\nStarting a process pool with {num_processes} workers...")
    
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        # starmap is perfect for passing multiple arguments to the worker function
        results = pool.starmap(run_simulation_worker, tasks)
    
    total_duration = time.time() - start_time
    print(f"\n--- All simulations completed in {total_duration:.2f} seconds ---")

    # --- Print Summary ---
    success_count = 0
    failures = []
    print("\nSimulation Results Summary:")
    print("-" * 60)
    for view_name, status, message in results:
        print(f"  - View: {view_name:<40} | Status: {status}")
        if status == "SUCCESS":
            success_count += 1
        else:
            failures.append((view_name, message))
    print("-" * 60)
    
    if failures:
        print("\nFailures Details:")
        for view_name, message in failures:
            print(f"  - {view_name}: {message}")
    
    print(f"\nFinal tally: {success_count} succeeded, {len(failures)} failed.")
    
    if success_count == len(views_to_simulate):
        print("\nConclusion: SUCCESS! pyAether appears to support this mode of parallel simulation.")
    else:
        print("\nConclusion: FAILED. Check the error messages above.")


if __name__ == "__main__":
    # It's good practice, especially with multiprocessing, to guard the main execution.
    main()
