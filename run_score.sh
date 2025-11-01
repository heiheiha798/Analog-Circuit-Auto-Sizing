#!/bin/bash

# 设置 pyAether 所需的 lib 参数
AE_LIB="2025_EDA_case1"
AE_CELL="OPA"
AE_VIEW="schematic"
MDE_CELL="OPA_loopgain"
#MDE_VIEW="spe_state1"
MDE_VIEW="spe_state1_Corner"


# 设置输入参数文件名
INPUT_PARAM_FILE="case1_params_0.txt"

# 设置输出路径与输出文件名
OUTPUT_PATH="./case1_results"
OUTPUT_FILE="output.log"

# 设置打分相关的参数
CASE1_UGB_BASE=142871000.0
CASE1_AREA_BASE=120000.0

CASE2_UGB_BASE=24629000.0
CASE2_AREA_BASE=4700.0

UGB_BASE=1e8
AREA_BASE=5e5
TIME_BASE=7200.0

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印彩色信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# 显示用法
show_usage() {
    echo "用法: $0 [选项] <压缩包路径>"
    echo ""
    echo "选项:"
    echo "  -h, --help                显示此帮助信息"
    echo "  -o, --output-dir DIR      指定解压目录（默认：当前目录）"
    echo "  -t, --time-limit SECONDS  设置超时时间（秒）"
    echo "  --score-script PATH       指定打分脚本路径（默认：./score.py）"
    echo "  --opt-script NAME         指定优化脚本名称（默认：optimization.py）"
    echo "  --                        分隔符，后面的参数传递给 score.py"
    echo ""
    echo "示例:"
    echo "  $0 optimization_package.tar.gz"
    echo "  $0 -o ./output -t 7200 optimization_package.tar.gz"
    echo "  $0 optimization_package.tar.gz -- --time-limit 3600"
}

# 解析命令行参数
COMPRESSED_FILE=""
OUTPUT_DIR="./output"
TIME_LIMIT="10800"  # 默认3小时
SCORE_SCRIPT="./score.py"
OPTIMIZATION_SCRIPT="optimization.py"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --score-script)
            SCORE_SCRIPT="$2"
            shift 2
            ;;
        --opt-script)
            OPTIMIZATION_SCRIPT="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        -*)
            print_error "未知选项: $1"
            show_usage
            exit 1
            ;;
        *)
            COMPRESSED_FILE="$1"
            shift
            ;;
    esac
done

# 检查必要参数
if [[ -z "$COMPRESSED_FILE" ]]; then
    print_error "必须指定压缩包路径"
    show_usage
    exit 1
fi

# 检查压缩文件是否存在
if [[ ! -f "$COMPRESSED_FILE" ]]; then
    print_error "压缩文件不存在: $COMPRESSED_FILE"
    exit 1
fi

# 检查打分脚本是否存在
if [[ ! -f "$SCORE_SCRIPT" ]]; then
    print_error "打分脚本不存在: $SCORE_SCRIPT"
    exit 1
fi

# 根据测试case，更改某些参数值
if [[ "$AE_LIB" == "2025_EDA_case1" ]]; then
    UGB_BASE=$CASE1_UGB_BASE
    AREA_BASE=$CASE1_AREA_BASE
elif [[ "$AE_LIB" == "2025_EDA_case2" ]]; then
    UGB_BASE=$CASE2_UGB_BASE
    AREA_BASE=$CASE2_AREA_BASE
elif [[ "$AE_LIB" == "2025_EDA_hidden_case1" ]]; then
    UGB_BASE=$CASE1_UGB_BASE
    AREA_BASE=$CASE1_AREA_BASE
elif [[ "$AE_LIB" == "2025_EDA_hidden_case2" ]]; then
    UGB_BASE=$CASE2_UGB_BASE
    AREA_BASE=$CASE2_AREA_BASE
else
    print_error "不支持的测试case: $AE_LIB"
    exit 1
fi

run_optimization_with_timeout() {
    local script_name="$1"
    local timeout_seconds="$2"
    local start_time
    local end_time
    local execution_time
    local timed_out=0
    
    print_info "启动优化脚本: $script_name"
    start_time=$(date +%s)
    
    # 创建日志文件
    local log_file="${AE_LIB}_optimization.log"

    # 构建启动命令
    if command -v taskset >/dev/null 2>&1; then
        print_info "使用 taskset 限制CPU核心: 1-10"
        taskset -c 1-10 pythone "$script_name" \
            --ae_lib "${AE_LIB}" \
            --ae_cell "${AE_CELL}" \
            --ae_view "${AE_VIEW}" \
            --mde_cell "${MDE_CELL}" \
            --mde_view "${MDE_VIEW}" \
            --param_file "${INPUT_PARAM_FILE}" \
            --best_param_file "${AE_LIB}_best_params.txt" > "$log_file" 2>&1 &
    else
        pythone "$script_name" \
            --ae_lib "${AE_LIB}" \
            --ae_cell "${AE_CELL}" \
            --ae_view "${AE_VIEW}" \
            --mde_cell "${MDE_CELL}" \
            --mde_view "${MDE_VIEW}" \
            --param_file "${INPUT_PARAM_FILE}" \
            --best_param_file "${AE_LIB}_best_params.txt" > "$log_file" 2>&1 &
    fi

    local shell_pid=$!
    print_info "启动的Shell进程PID: $shell_pid"

    # 等待进程启动
    sleep 3
    
    # 通过进程树查找正确的Python进程
    local opt_pid=""
    
    # 首先检查启动的shell进程是否还在运行
    if ps -p $shell_pid > /dev/null 2>&1; then
        print_info "Shell进程 $shell_pid 仍在运行"
        
        # 查找shell进程的直接子进程
        local child_pids=$(ps -o pid --ppid $shell_pid --no-headers 2>/dev/null)
        print_info "Shell进程的直接子进程: $child_pids"
        
        for child_pid in $child_pids; do
            local child_cmd=$(ps -p $child_pid -o command --no-headers 2>/dev/null)
            # if [[ "$child_cmd" == *"python"* ]] && [[ "$child_cmd" == *"$script_name"* ]]; then
            if [[ "$child_cmd" == *"python $script_name"* ]]; then
                opt_pid=$child_pid
                print_info "找到Python子进程: $opt_pid"
                print_info "进程命令: $child_cmd"
                break
            fi
        done
    fi
    
    # 如果没有找到，使用启动的shell进程
    if [[ -z "$opt_pid" ]]; then
        print_warning "无法找到Python进程，使用Shell进程PID: $shell_pid"
        opt_pid=$shell_pid
    fi
    
    # 验证找到的进程
    if ps -p $opt_pid > /dev/null 2>&1; then
        local process_info=$(ps -p $opt_pid -o pid,ppid,pgid,user,comm,args --no-headers 2>/dev/null)
        print_info "确认监控的进程:"
        print_info "  PID: $(echo "$process_info" | awk '{print $1}')"
        print_info "  PPID: $(echo "$process_info" | awk '{print $2}')" 
        print_info "  PGID: $(echo "$process_info" | awk '{print $3}')"
        print_info "  用户: $(echo "$process_info" | awk '{print $4}')"
        print_info "  命令: $(echo "$process_info" | awk '{print $5}')"
        print_info "  参数: $(echo "$process_info" | awk '{print $6}')"
    else
        print_error "进程 $opt_pid 不存在"
        
        # 检查日志文件
        if [[ -f "$log_file" ]]; then
            print_error "日志文件内容:"
            cat "$log_file" >&2
        fi
        
        echo "0 1"
        return
    fi
    
    # 后续的等待和超时处理
    local waited=0
    while [[ $waited -lt $timeout_seconds ]]; do
        if ! kill -0 $opt_pid 2>/dev/null; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
        echo -n "." >&2
        if [[ $((waited % 60)) -eq 0 ]]; then
            echo "" >&2
            print_info "已运行: ${waited}秒 ($(($waited/60))分钟), 剩余: $(($timeout_seconds - $waited))秒"
        fi
    done

    echo "" >&2
    
    # 超时处理
    if kill -0 $opt_pid 2>/dev/null; then
        print_warning "优化脚本超时，正在终止进程..."
        
        # 获取进程组ID
        local pgid=$(ps -o pgid= -p $opt_pid 2>/dev/null | tr -d ' ')
        if [[ -n "$pgid" ]] && [[ "$pgid" != "$$" ]]; then
            print_info "终止进程组: $pgid"
            kill -TERM -$pgid 2>/dev/null
        else
            # 回退到杀死进程及其子进程
            kill -TERM $opt_pid 2>/dev/null
            local child_pids=$(pgrep -P $opt_pid 2>/dev/null)
            for child in $child_pids; do
                kill -TERM $child 2>/dev/null
            done
        fi
        
        sleep 5
        
        # 强制杀死
        if kill -0 $opt_pid 2>/dev/null; then
            print_warning "进程仍未终止，强制杀死..."
            if [[ -n "$pgid" ]] && [[ "$pgid" != "$$" ]]; then
                kill -KILL -$pgid 2>/dev/null
            else
                kill -KILL $opt_pid 2>/dev/null
                local child_pids=$(pgrep -P $opt_pid 2>/dev/null)
                for child in $child_pids; do
                    kill -KILL $child 2>/dev/null
                done
            fi
            
            # 最后清理：杀死所有相关的Python进程
            pkill -9 -f "python.*$script_name" 2>/dev/null
        fi
        
        timed_out=1
        end_time=$(date +%s)
        execution_time=$timeout_seconds
    else
        # 等待进程结束并获取退出代码
        wait $opt_pid 2>/dev/null
        local exit_code=$?
        end_time=$(date +%s)
        execution_time=$((end_time - start_time))
        
        if [[ $exit_code -eq 0 ]]; then
            print_success "优化脚本正常结束"
        else
            print_warning "优化脚本异常结束，退出代码: $exit_code"
        fi
    fi

    # 格式化执行时间
    local hours=$((execution_time / 3600))
    local minutes=$(( (execution_time % 3600) / 60 ))
    local seconds=$((execution_time % 60))
    local formatted_time=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)
    
    print_info "执行时长: $formatted_time ($execution_time 秒)"
    print_info "是否超时: $([[ $timed_out -eq 1 ]] && echo "是" || echo "否")"

    echo "$execution_time $timed_out"
}

print_info "开始设置优化环境..."
print_info "压缩文件: $COMPRESSED_FILE"
print_info "输出目录: $OUTPUT_DIR"
print_info "打分脚本: $SCORE_SCRIPT"
print_info "优化脚本: $OPTIMIZATION_SCRIPT"
print_info "时间限制: $TIME_LIMIT 秒 ($(($TIME_LIMIT/3600)) 小时)"
print_info "被测电路: $AE_LIB"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
if [[ $? -ne 0 ]]; then
    print_error "创建输出目录失败: $OUTPUT_DIR"
    exit 1
fi

# 解压文件
print_info "正在解压文件..."
if [[ "$COMPRESSED_FILE" == *.tar.gz ]] || [[ "$COMPRESSED_FILE" == *.tgz ]]; then
    tar -xzf "$COMPRESSED_FILE" -C "$OUTPUT_DIR"
elif [[ "$COMPRESSED_FILE" == *.tar ]]; then
    tar -xf "$COMPRESSED_FILE" -C "$OUTPUT_DIR"
elif [[ "$COMPRESSED_FILE" == *.zip ]]; then
    unzip -q "$COMPRESSED_FILE" -d "$OUTPUT_DIR"
else
    print_error "不支持的压缩格式: $COMPRESSED_FILE"
    print_info "支持的格式: .tar.gz, .tgz, .tar, .zip"
    exit 1
fi

if [[ $? -ne 0 ]]; then
    print_error "解压文件失败"
    exit 1
fi

print_success "文件解压完成"

# 查找解压后的目录（处理可能的多级目录结构）
EXTRACTED_DIR="$OUTPUT_DIR"
# 如果解压后只有一个目录，则使用该目录
if [[ $(find "$OUTPUT_DIR" -maxdepth 1 -type d | wc -l) -eq 2 ]]; then
    SUBDIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d ! -path "$OUTPUT_DIR")
    if [[ -n "$SUBDIR" ]]; then
        EXTRACTED_DIR="$SUBDIR"
        print_info "使用子目录: $EXTRACTED_DIR"
    fi
fi

# 复制打分脚本
print_info "复制打分脚本到工作目录..."
cp "$SCORE_SCRIPT" "$EXTRACTED_DIR/"
if [[ $? -ne 0 ]]; then
    print_error "复制打分脚本失败"
    exit 1
fi

print_success "打分脚本复制完成"

# 复制电路
print_info "复制测试电路到工作目录..."
cp -rf circuits/* "$EXTRACTED_DIR/"
if [[ $? -ne 0 ]]; then
    print_error "复制测试电路失败"
    exit 1
fi

print_success "测试电路复制完成"

# 切换到工作目录
print_info "切换到工作目录: $EXTRACTED_DIR"
cd "$EXTRACTED_DIR" || {
    print_error "无法切换到工作目录: $EXTRACTED_DIR"
    exit 1
}

# 检查优化脚本是否存在
if [[ ! -f "$OPTIMIZATION_SCRIPT" ]]; then
    print_error "优化脚本不存在: $OPTIMIZATION_SCRIPT"
    exit 1
fi

# 运行优化脚本并获取执行时间
print_info "开始执行优化脚本监控..."
result=$(run_optimization_with_timeout "$OPTIMIZATION_SCRIPT" "$TIME_LIMIT")

# 解析执行时间和是否超时
execution_time=$(echo $result | awk '{print $1}')
timed_out=$(echo $result | awk '{print $2}')

echo ""
print_info "优化脚本执行完成，开始执行打分程序..."

# 构建执行命令
CMD=("pythone" "score.py")

# 添加 lib/cell/view
CMD+=("--ae_lib" "${AE_LIB}")
CMD+=("--ae_cell" "${AE_CELL}")
CMD+=("--ae_view" "${AE_VIEW}")
CMD+=("--mde_cell" "${MDE_CELL}")
CMD+=("--mde_view" "${MDE_VIEW}")

# 添加输入、输出参数
CMD+=("--param_file" "${INPUT_PARAM_FILE}")
CMD+=("--best_param_file" "${AE_LIB}_best_params.txt")
CMD+=("--output_path" "${OUTPUT_PATH}")
CMD+=("--output_file" "${OUTPUT_FILE}")

# 添加打分相关参数
CMD+=("--calc_score")
CMD+=("--ugb_base" "${UGB_BASE}")
CMD+=("--area_base" "${AREA_BASE}")
CMD+=("--time_base" "${TIME_BASE}")


# 添加执行时间参数
CMD+=("--execution_time" "$execution_time")

# 添加是否超时参数
# if [[ $timed_out -eq 1 ]]; then
#     CMD+=("--timed-out" "true")
# else
#     CMD+=("--timed-out" "false")
# fi

# 添加额外参数
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

print_info "执行命令: ${CMD[*]}"
echo ""

# 执行打分程序
"${CMD[@]}"

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    print_success "打分程序执行完成"
else
    print_warning "打分程序退出代码: $EXIT_CODE"
fi

# 返回原始目录（可选）
cd - > /dev/null

print_info "工作目录: $EXTRACTED_DIR"
print_info "结果文件请查看上述目录"
