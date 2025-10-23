#!/bin/bash

# 脚本功能: 启动对电路中所有MOS管(包括对称组)的fw参数进行独立扫描和仿真

# 设置库、单元、视图
AE_LIB="2025_EDA_case1"
AE_CELL="OPA"
AE_VIEW="schematic"
MDE_CELL="OPA_loopgain"
MDE_VIEW="spe_state1"
# MDE_VIEW="spe_state1_Corner"

# 设置作为基准的参数文件 (脚本需要它来读取原始fw值)
INPUT_PARAM_FILE="case1_extractcdfVal_0.txt"

# 设置输出路径和文件 (扫描过程中的所有仿真结果都会保存在这里)
OUTPUT_PATH="./case1_fw_scan_results" # 建议为扫描结果使用一个新的目录
OUTPUT_FILE="scan_output.log"

# 执行 Python 脚本，并使用 --fw_scan 标志
# 这会触发 optimization.py 中我们新编写的扫描主逻辑
echo "Starting FW sensitivity scan... This may take a long time."
pythone optimization.py \
    --ae_lib ${AE_LIB} \
    --ae_cell ${AE_CELL} \
    --ae_view ${AE_VIEW} \
    --mde_cell ${MDE_CELL} \
    --mde_view ${MDE_VIEW} \
    --param_file ${INPUT_PARAM_FILE} \
    --output_path ${OUTPUT_PATH} \
    --output_file ${OUTPUT_FILE} \
    --fw_scan