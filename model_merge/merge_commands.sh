#!/bin/bash
# 模型合并命令脚本
# 基于 TA_adaptive_merge.py 和提供的模型列表及 global_l2 值

# Base 模型路径
BASE_MODEL="/zju_0038/wyy/mergebench/models/meta-llama/Llama-3.2-3B-Instruct/"

# 待合并的模型路径列表（按顺序）
OTHER_MODELS=(
    "/zju_0038/yifyang/scripts/models/Llama-3.2-3B_MATH_lisa"
    "/zju_0038/yifyang/scripts/models/llama-3.2-Korean-Bllossom-3B"
    "/zju_0038/yifyang/scripts/models/Llama-3.2-3B-Instruct-tuned"
    "/zju_0038/yifyang/scripts/models/EZO-Llama-3.2-3B-Instruct-dpoE"
    "/zju_0038/yifyang/scripts/models/Home-Llama-3.2-3B"
)

# 对应的 L2 norm 值（顺序必须与模型路径一致）
L2_NORMS=(
    182.63449804
    31.87331430
    0.00000000
    0.52867741
    9.26351817
)

# 输出路径（请根据实际情况修改）
OUTPUT_PATH="/zju_0038/jinjia/workspace/norm_filter/tv_out_instruct/merged_model_output"

# 权重计算模式: linear, power, exp（默认exp）
MODE="exp"

# 执行合并命令
python TA_adaptive_merge.py \
    --base_model_path "${BASE_MODEL}" \
    --other_model_paths "${OTHER_MODELS[@]}" \
    --l2_norms "${L2_NORMS[@]}" \
    --output_path "${OUTPUT_PATH}" \
    --mode "${MODE}" \
    --device "cuda"