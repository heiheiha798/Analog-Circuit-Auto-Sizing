import re
from typing import Dict, List, Tuple, Optional

from src.data_models import Parameter

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

def read_parameters(file_name, dummy_devices: List[str] = None) -> Dict[str, Parameter]:
    """
    从配置文件中读取参数，并根据元器件类型应用正确的边界约束。
    """
    parameters = {}
    if dummy_devices is None:
        dummy_devices = []

    # === 修改部分：结构化的约束规则 ===
    # 将规则按元器件类型（mos, capacitor, resistor）进行组织
    #
    # 约束来源:
    # C_l: 4u-30u
    # R_segW: 420n-10u
    # R_segL: 5u-100u (根据上下文推断 R_segW 后的约束为 segL)
    # mos_l: 130n-20u
    # mos_fw: 150n-50u
    # mos_m: 1-99.999K
    range_rules = {
        "mos": {
            "l": ("continuous", 0.13, 20.0),    # 130n -> 0.13u
            "fw": ("continuous", 0.15, 50.0),   # 150n -> 0.15u
            "m": ("integer", 1.0, 99999.0),      # 99.999K -> 99999.0
        },
        "capacitor": {
            "l": ("continuous", 4.0, 30.0),     # 4u -> 4.0u
        },
        "resistor": {
            "segW": ("continuous", 0.42, 10.0), # 420n -> 0.42u
            "segL": ("continuous", 5.0, 100.0),  # 5u -> 5.0u
        }
    }

    with open(file_name, 'r') as param_file:
        lines = param_file.readlines()
        for line in lines:
            parts = [p.strip().replace('"', "") for p in line.strip().split(',')]
            if len(parts) < 3 or parts[0] != "parameter":
                continue
            
            name = parts[1]
            value_token = parts[2]
            
            index = name.rfind("_")
            if index == -1:
                print(f"Warning: Skipping parameter '{name}' due to unexpected format.")
                continue
            
            inst_name = name[:index]
            param_suffix = name[index + 1:]

            # === 修改部分：根据实例名称动态选择规则 ===
            component_type = None
            if inst_name.upper().startswith(('M', 'P', 'N')): # 覆盖 MOS, PMOS, NMOS
                component_type = "mos"
            elif inst_name.upper().startswith('C'):
                component_type = "capacitor"
            elif inst_name.upper().startswith('R'):
                component_type = "resistor"
            
            param_type, min_val, max_val = ("continuous", None, None)
            if component_type:
                # 从对应的元器件规则中查找参数约束
                param_rules = range_rules.get(component_type, {})
                param_type, min_val, max_val = param_rules.get(param_suffix, ("continuous", None, None))
            else:
                 print(f"Warning: Could not determine component type for instance '{inst_name}'. No boundary checks will be applied for '{name}'.")


            is_dummy = inst_name in dummy_devices

            # 对整数类型（如 m）做更健壮的解析
            if param_type == "integer":
                vt = value_token.strip()
                try:
                    if vt.lower().endswith('k'):
                        base = float(vt[:-1])
                        param_value = base * 1000.0
                    else:
                        param_value = float(vt)
                except ValueError:
                    raise ValueError(f"Invalid integer parameter value '{value_token}' for {name}.")
            else:
                param_value = param_convert(value_token)
            
            # 执行边界检查
            if min_val is not None and max_val is not None:
                if not (min_val <= param_value <= max_val):
                    raise ValueError(
                        f"The param '{param_suffix}' of instance '{inst_name}' (type: {component_type}) with value {param_value} "
                        f"is out of the defined range [{min_val}, {max_val}]."
                    )

            param = Parameter(name, param_type, param_value, min_val, max_val, is_dummy)
            parameters[name] = param

    print(f'==== Read {len(parameters)} parameters successfully, param file: {file_name} ====\n')
    return parameters
