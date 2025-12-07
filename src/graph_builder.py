import pyAether as ae
from typing import Dict

from src.data_models import CircuitGraph, DeviceNode, NetNode
from src.utils import param_convert

def _read_parameters_to_dict(param_file_path: str) -> Dict[str, Dict[str, float]]:
    """
    A helper function to read the parameter file and organize it into a nested dictionary.
    This format is optimized for quick look-up during graph construction.
    
    Returns:
        {'NM17': {'m': 2.0, 'l': 7e-07}, 'PM0': {'fw': 1e-05}, ...}
    """
    param_dict = {}
    with open(param_file_path, 'r') as param_file:
        lines = param_file.readlines()
        for line in lines:
            # e.g., "parameter","NM17_m","2"
            fields = line.strip().replace('"', '').replace(" ", "").split(',')
            if fields[0] == "parameter":
                full_name = fields[1]  # "NM17_m"
                value_str = fields[2]  # "2"
                
                # Split "NM17_m" into instance name "NM17" and param name "m"
                last_underscore_idx = full_name.rfind("_")
                inst_name = full_name[:last_underscore_idx]
                param_name = full_name[last_underscore_idx+1:]

                # Convert value to float. For 'm', it's direct. For others, use param_convert.
                value = float(value_str) if param_name in ['m', 'segments'] else param_convert(value_str)

                # Populate the nested dictionary
                if inst_name not in param_dict:
                    param_dict[inst_name] = {}
                param_dict[inst_name][param_name] = value
                
    return param_dict

def build_graph_from_eda(cv: 'ae.cv', param_file_path: str, power_net_name: str = 'VDDA', gnd_net_name: str = 'gnd!') -> CircuitGraph:
    """
    The main factory function. It takes an Aether circuit view (cv) and a parameter file,
    and constructs a complete CircuitGraph object representing the circuit.
    """
    print("Starting to build CircuitGraph...")
    
    graph = CircuitGraph(name=str(cv.cellName))
    all_params = _read_parameters_to_dict(param_file_path)
    print(f"  Successfully read parameters for {len(all_params)} devices from '{param_file_path}'.")

    print("\n--- Pass 1: Creating Device Nodes ---")
    for inst in cv.instances:
        inst_name_str = str(inst.name)
        
        # --- 修改这里：移除过滤器 ---
        # 我们不再跳过任何器件，因为PIN等边界信息对拓扑分析至关重要
        # inst_name_lower = inst_name_str.lower()
        # if inst_name_lower.startswith('i') or inst_name_lower.startswith('pin'):
        #     print(f"  --> Skipping testbench instance: {inst_name_str}")
        #     continue
        # ----------------------------
        
        master_name = str(inst.master.cellName).lower()
        
        # --- 修改这里：增强器件类型判断 ---
        dev_type = "Unknown"
        if "p_mos" in master_name: dev_type = "PMOS"
        elif "n_mos" in master_name: dev_type = "NMOS"
        elif "res" in master_name: dev_type = "Resistor"
        elif "cap" in master_name: dev_type = "Capacitor"
        elif "pin" in master_name: dev_type = "PIN" # 新增对PIN的识别
        else: dev_type = "ELSE" 
        # ------------------------------------

        device_node = DeviceNode(name=inst_name_str, device_type=dev_type)
        device_node.parameters = all_params.get(inst_name_str, {})
        graph.devices[inst_name_str] = device_node
        # print(f"  Created: {device_node} with {len(device_node.parameters)} parameters.")

    print("\n--- Pass 2: Creating Net Nodes and Connections ---")
    for inst_name, device_node in graph.devices.items():
        inst = ae.dbFindInst(cv, inst_name)
        
        for p in inst.instTerms:
            if p.net is None:
                # print(f"  Skipping unconnected terminal: {device_node.name}'s {str(p.name)}")
                continue

            net_name_str = str(p.net.name)
            
            net_node = graph.nets.get(net_name_str)
            if not net_node:
                net_node = NetNode(name=net_name_str)
                graph.nets[net_name_str] = net_node
                # print(f"  Created new Net: {net_node}")

                if net_name_str == power_net_name:
                    net_node.is_power = True
                    graph.power_net = net_node
                    # print(f"    -> Identified as POWER net.")
                elif net_name_str == gnd_net_name:
                    net_node.is_gnd = True
                    graph.gnd_net = net_node
                    # print(f"    -> Identified as GROUND net.")

            device_node.terminals[str(p.name)] = net_node
            net_node.connections.append((device_node, str(p.name)))

    print("\nCircuitGraph build complete!")
    print(f"  Summary: {graph}")
    print(f"  Power Net: {graph.power_net}")
    print(f"  Ground Net: {graph.gnd_net}")
    
    return graph
