from typing import Dict, List, Tuple, Any
from src.data_models import CircuitGraph, ConstraintGroup, DeviceNode

def _get_wl_signature(device: DeviceNode, current_labels: Dict[str, Any]) -> Tuple:
    """
    为 WL 算法的单次迭代计算一个节点的签名。
    签名基于其邻居的当前标签。

    参数:
        device (DeviceNode): 当前要计算签名的器件节点。
        current_labels (Dict[str, Any]): 存储图中所有器件当前标签的字典 (器件名称 -> 标签值)。
                                        这里的标签值在第一轮是字符串(device_type)，后续迭代是整数。
    返回:
        Tuple: 代表该器件当前邻居标签组合的规范签名。
    """
    terminal_signatures = []

    # 按照端子名称排序，确保签名的一致性
    for term_name, net in sorted(device.terminals.items()):
        if not net:
            # 如果网络未连接，则用一个特殊标识符
            neighbor_labels_for_this_terminal = ('unconnected',)
        else:
            # 收集该网络上所有邻居器件的当前标签
            # 确保排除掉当前器件本身
            neighbor_labels = []
            for neighbor_dev, neighbor_term in net.connections:
                if neighbor_dev != device:
                    # 获取邻居的当前标签
                    # 对于 PIN 器件，其名称在拓扑上具有唯一性，也应包含在签名中以提高区分度
                    if neighbor_dev.device_type == 'PIN':
                        # 将 PIN 类型、连接端口和 PIN 名称组合成一个签名元素
                        neighbor_labels.append((current_labels[neighbor_dev.name], neighbor_term, neighbor_dev.name))
                    else:
                        neighbor_labels.append(current_labels[neighbor_dev.name])
            
            # 对收集到的邻居标签进行排序，以确保签名的规范性（顺序无关）
            neighbor_labels_for_this_terminal = sorted(neighbor_labels)
        
        # 将当前端子的签名（由其连接的邻居标签组成）添加到列表
        terminal_signatures.append(tuple(neighbor_labels_for_this_terminal))

    # 对所有端子的签名进行排序，形成最终的设备签名
    # 这样，即使端子的内部表示顺序不同，只要拓扑相同，最终签名也会相同
    return tuple(sorted(terminal_signatures))

def find_topological_symmetries(graph: CircuitGraph, max_iterations: int = 5):
    """
    使用 Weisfeiler-Lehman 算法思想来寻找拓扑对称性，并包含标签压缩。
    
    参数:
        graph (CircuitGraph): 要分析的电路图。
        max_iterations (int): WL 算法的最大迭代次数。
    """
    print("\n--- Running Topological Symmetry Analysis (WL-based with Label Compression) ---")

    # 1. 初始化：使用 device_type 作为初始 "颜色" 或标签。
    # 为了统一处理，我们将初始的字符串标签也映射为整数。
    # 这是一个映射，将器件类型字符串映射到唯一的整数 ID
    unique_device_types = sorted(list(set(dev.device_type for dev in graph.devices.values())))
    initial_label_map = {dtype: i for i, dtype in enumerate(unique_device_types)}
    
    # device_labels 存储每个器件名称到其当前整数标签的映射
    device_labels: Dict[str, int] = {
        dev.name: initial_label_map[dev.device_type] 
        for dev in graph.devices.values()
    }
    
    # 2. 迭代 refinement 过程
    for i in range(max_iterations):
        # print(f"\n--- WL Refinement Iteration {i+1} ---")
        
        has_changed = False # 标志，用于判断本轮迭代是否有任何标签发生变化
        
        # --- 标签压缩的核心数据结构 ---
        # label_mapping 存储 (旧标签, 签名元组) -> 新整数标签 的映射
        # 它在每轮迭代开始时清空，只用于本轮的标签压缩
        label_mapping: Dict[Tuple[int, Tuple], int] = {} 
        next_label_id = 0 # 用于生成新的、唯一的整数标签
        new_labels: Dict[str, int] = {} # 存储本轮计算出的新标签
        # --------------------------------

        # 为图中每一个器件计算其新的整数标签
        for dev_name, dev in graph.devices.items():
            # PIN 器件在 WL 迭代中通常不改变其标签，或者可以为其分配一个固定标签
            # 这里的策略是让 PIN 的标签保持其初始的整数 ID，不参与复杂的聚合
            # 但是 PIN 的标签会被其邻居聚合
            if dev.device_type == 'PIN':
                new_labels[dev_name] = device_labels[dev_name]
                continue

            # 调用辅助函数 _get_wl_signature 来获取邻居的当前标签组合
            # 这个签名是一个元组，内部包含了邻居器件的整数标签
            signature = _get_wl_signature(dev, device_labels) 
            
            # 复合标签由当前器件的旧整数标签和其邻居的签名共同构成。
            # 这是一个元组，作为 label_mapping 的键。
            composite_label = (device_labels[dev_name], signature)
            
            # --- 标签压缩步骤 ---
            # 检查这个复合标签是否已经在 label_mapping 中见过
            if composite_label not in label_mapping:
                # 如果是第一次见到，就给它分配一个递增的新的整数 ID
                label_mapping[composite_label] = next_label_id
                next_label_id += 1
            
            # 获取该复合标签对应的唯一整数 ID，作为器件的新标签
            new_integer_label = label_mapping[composite_label]
            new_labels[dev_name] = new_integer_label
            # ----------------------
            
            # 检查当前器件的标签是否发生变化
            if new_integer_label != device_labels[dev_name]:
                has_changed = True

        # 更新所有器件的标签为本轮计算出的新整数标签
        device_labels = new_labels
        
        # 检查是否收敛：如果本轮没有任何标签发生变化，则停止迭代
        if not has_changed:
            # print("  Labels have converged. Stopping refinement.")
            break
    
    # 3. 结果提取：根据最终的稳定整数标签进行分组
    final_groups: Dict[int, List[DeviceNode]] = {}
    for dev_name, label in device_labels.items():
        dev = graph.devices[dev_name]
        
        # 通常我们只关心非 PIN 器件的对称性分组
        if dev.device_type == 'PIN': 
            continue 
        
        if label not in final_groups:
            final_groups[label] = []
        final_groups[label].append(dev)

    # 提取所有成员数量大于1的组作为对称单元
    symmetric_groups: List[List[DeviceNode]] = [group for group in final_groups.values() if len(group) > 1]
    
    # print(f"\nRefinement complete. Found {len(symmetric_groups)} symmetric groups.")

    if not symmetric_groups:
        # print("  No symmetric groups found.")
        return

    # 4. 创建 ConstraintGroup 对象并添加到 CircuitGraph
    for group_of_devices in symmetric_groups:
        # 选择组中的第一个器件作为代表，用于命名和参数拷贝
        representative_device = group_of_devices[0]
        
        # 创建约束组的名称，包含器件类型和代表器件的名称
        # 这样做有助于调试和理解是哪个器件形成了对称组
        group_name = f"SymmetricGroup_{representative_device.device_type}_{representative_device.name}"
        constraint = ConstraintGroup(group_name)
        
        # 将组内的所有器件添加到这个约束组中
        for device in group_of_devices:
            constraint.add_device(device)
            
        # 拷贝代表器件的参数作为共享参数。
        # 注意：在实际的优化流程中，可能需要更复杂的逻辑来决定共享参数的初始值，
        # 例如取组内所有器件参数的平均值，或者指定一个主导器件。
        constraint.shared_parameters = representative_device.parameters.copy()
        
        # 将新创建的约束组添加到 CircuitGraph 的约束组列表中
        graph.constraint_groups.append(constraint)
        
        # 打印创建的约束组及其包含的器件名称，便于查看结果
        device_names = ', '.join([d.name for d in group_of_devices])
        # print(f"  Created constraint group '{constraint.group_type}' for devices: [{device_names}]")

def analyze_circuit_constraints(graph: CircuitGraph):
    """
    主要分析入口点。
    调用 find_topological_symmetries 来执行对称性识别。
    """
    # 默认迭代 5 次，但可以根据电路复杂度进行调整
    find_topological_symmetries(graph, max_iterations=10)
    # print(f"\nAnalysis complete. Total constraint groups created: {len(graph.constraint_groups)}")
