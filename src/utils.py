import re
from typing import Dict, List, Tuple

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
