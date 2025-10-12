from typing import Dict, Optional, List, Tuple, Set, Union
import re

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
        self.min_val = min_val
        self.max_val = max_val
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
        if self.max_val == self.min_val:
            return 0.0  # Avoid division by zero
        return (value - self.min_val) / (self.max_val - self.min_val)

    def denormalize(self, norm_value: float) -> float:
        """Convert a normalized value back to the original range"""
        return self.min_val + norm_value * (self.max_val - self.min_val)

    def clamp(self, value: float) -> float:
        """
        将输入值限制在参数的 [min_val, max_val] 范围内。
        如果参数类型是 'integer'，则对结果进行四舍五入。
        """
        if self.max_val is not None:
            value = min(value, self.max_val)
        if self.min_val is not None:
            value = max(value, self.min_val)
        
        # 如果是整数类型，返回一个整数（或接近整数的浮点数）
        if self.type == 'integer':
            return round(value)
        return value

class NetNode:
    """Represents a net (a wire) in the circuit schematic."""
    def __init__(self, name: str):
        self.name: str = name
        # A list of tuples, where each tuple contains the device object
        # and the name of the terminal connected to this net.
        # e.g., [(<DeviceNode PM0>, 'G'), (<DeviceNode NM0>, 'D')]
        self.connections: List[Tuple['DeviceNode', str]] = []
        self.is_power: bool = False
        self.is_gnd: bool = False

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the NetNode."""
        flags = []
        if self.is_power:
            flags.append("POWER")
        if self.is_gnd:
            flags.append("GND")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"<NetNode: {self.name}{flag_str}>"

class DeviceNode:
    """Represents a device instance (e.g., a transistor, resistor) in the circuit."""
    def __init__(self, name: str, device_type: str):
        self.name: str = name
        self.device_type: str = device_type
        # Stores sizing parameters like {'l': 1.3e-7, 'fw': 1e-5, 'm': 120}
        self.parameters: Dict[str, float] = {}
        # Maps terminal names to the NetNode object they are connected to.
        # e.g., {'G': <NetNode VIN+>, 'D': <NetNode net23>, ...}
        self.terminals: Dict[str, Optional[NetNode]] = {}

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the DeviceNode."""
        return f"<DeviceNode: {self.name} ({self.device_type})>"

class CircuitGraph:
    """Represents the entire circuit as a graph of devices and nets."""
    def __init__(self, name: str):
        self.name: str = name
        # Look-up tables for quick access to any device or net by its name.
        self.devices: Dict[str, DeviceNode] = {}
        self.nets: Dict[str, NetNode] = {}
        
        # Direct references to power and ground nets for pathfinding.
        self.power_net: Optional[NetNode] = None
        self.gnd_net: Optional[NetNode] = None
        
        # This will store the results of our pathfinding algorithm later.
        self.dc_paths: List[List[DeviceNode]] = []
        self.constraint_groups: List[ConstraintGroup] = [] 

    def __repr__(self) -> str:
        """Provides a summary of the entire circuit graph."""
        return (f"<CircuitGraph: {self.name} "
                f"(Devices: {len(self.devices)}, Nets: {len(self.nets)})>")

class ConstraintGroup:
    """Represents a group of devices that share design constraints."""
    def __init__(self, group_type: str):
        self.group_type: str = group_type  # e.g., "DifferentialPair", "CurrentMirror"
        self.devices: List[DeviceNode] = []
        
        # Shared parameters for this group. The optimizer will modify THESE.
        # e.g., {'l': 0.13, 'fw': 10.0, 'm': 120.0}
        self.shared_parameters: Dict[str, float] = {}

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the group."""
        device_names = ', '.join([d.name for d in self.devices])
        return (f"<ConstraintGroup ({self.group_type}): "
                f"Devices=[{device_names}], "
                f"Params={self.shared_parameters}>")

    def add_device(self, device: DeviceNode):
        """Adds a device to this constraint group."""
        self.devices.append(device)
