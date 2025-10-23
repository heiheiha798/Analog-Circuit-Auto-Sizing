#!/bin/bash

# 脚本功能: 启动贝叶斯优化流程来寻找最佳电路参数

# 设置库、单元、视图
AE_LIB="2025_EDA_case1"
AE_CELL="OPA"
AE_VIEW="schematic"
MDE_CELL="OPA_loopgain"
MDE_VIEW="spe_state1" # 使用单点仿真以加快迭代速度

# 设置作为基准和搜索空间定义的参数文件
INPUT_PARAM_FILE="case1_extractcdfVal_0.txt"

# 设置输出路径 (建议为BO结果使用一个新的目录)
OUTPUT_PATH="./case1_bo_results"
OUTPUT_FILE="bo_output.log"

# 执行 Python 脚本，并使用 --run_bo 标志
echo "Starting Bayesian Optimization... This will perform many simulations."
pythone optimization.py \
    --ae_lib ${AE_LIB} \
    --ae_cell ${AE_CELL} \
    --ae_view ${AE_VIEW} \
    --mde_cell ${MDE_CELL} \
    --mde_view ${MDE_VIEW} \
    --param_file ${INPUT_PARAM_FILE} \
    --output_path ${OUTPUT_PATH} \
    --output_file ${OUTPUT_FILE} \
    --run_bo