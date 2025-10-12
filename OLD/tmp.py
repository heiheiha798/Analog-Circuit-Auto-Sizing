
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