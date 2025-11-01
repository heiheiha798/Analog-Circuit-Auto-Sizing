#!/bin/bash

# set lib cell view
AE_LIB="2025_EDA_case1"
AE_CELL="OPA"
AE_VIEW="schematic"
MDE_CELL="OPA_loopgain"
#MDE_VIEW="spe_state1"
MDE_VIEW="spe_state1_Corner"

# set input param file
INPUT_PARAM_FILE="case1_extractcdfVal_0.txt"

# set output path file
OUTPUT_PATH="./case1_results"
OUTPUT_FILE="output.log"

# optimization use DE
pythone optimization.py --ae_lib ${AE_LIB} --ae_cell ${AE_CELL} --ae_view ${AE_VIEW} --mde_cell ${MDE_CELL} --mde_view ${MDE_VIEW} --param_file ${INPUT_PARAM_FILE} --output_path ${OUTPUT_PATH} --output_file ${OUTPUT_FILE}  --set_params