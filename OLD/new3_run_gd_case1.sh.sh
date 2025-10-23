#!/bin/bash

# 脚本功能: 启动自适应梯度下降优化流程

# 设置库、单元、视图等
AE_LIB="2025_EDA_case1"
AE_CELL="OPA"
AE_VIEW="schematic"
MDE_CELL="OPA_loopgain"
MDE_VIEW="spe_state1"

# 输入的基准参数文件
INPUT_PARAM_FILE="case1_extractcdfVal_0.txt"

# 输出路径
OUTPUT_PATH="./case1_gd_results"
OUTPUT_FILE="gd_output.log"

# 执行 Python 脚本，并使用 --run_gd 标志
echo "Starting Adaptive Gradient Descent Optimization..."
pythone optimization.py \
    --ae_lib ${AE_LIB} \
    --ae_cell ${AE_CELL} \
    --ae_view ${AE_VIEW} \
    --mde_cell ${MDE_CELL} \
    --mde_view ${MDE_VIEW} \
    --param_file ${INPUT_PARAM_FILE} \
    --output_path ${OUTPUT_PATH} \
    --output_file ${OUTPUT_FILE} \
    --run_gd