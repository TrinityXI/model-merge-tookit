# Task Vector Analyzer 使用说明

## 简介

`TA_analyzer.py` 是一个用于分析深度学习模型任务向量（Task Vectors）的工具。它计算微调模型与基础模型之间的参数差异，并提供详细的统计分析和可视化。

**核心功能**：
- 计算任务向量：**Δ = target_model - base_model**
- 提供参数级别的详细统计（L2/L1范数、分位数、偏度、峰度等）
- 生成模块级别的聚合分析（embed、attention、MLP、lm_head等）
- 计算多个模型之间的成对余弦相似度
- 支持GPU加速和多GPU并行计算

---

## 快速开始

### 基本用法

```bash
python TA_analyzer.py \
  --base <base_model_path> \
  --models <model1_path> <model2_path> ... \
  --outdir <output_directory>
```

### 最小示例

```bash
# 分析单个微调模型
python TA_analyzer.py \
  --base ./llama-3b-base \
  --models ./llama-3b-math \
  --outdir ./results
```

---

## 命令行参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--base` | str | ✅ | - | 基础模型路径或HuggingFace模型ID |
| `--models` | str[] | ✅ | - | 待分析的模型路径列表（支持多个） |
| `--outdir` | str | ❌ | `./tv_stats` | 输出目录路径 |
| `--device` | str | ❌ | 自动检测 | 计算设备：`cuda`, `cpu`, `cuda:0` 等 |
| `--multi-gpu` | flag | ❌ | False | 启用多GPU并行处理模式 |

### 参数详解

#### `--base`（必需）
基础模型路径，支持以下格式：
- 本地目录：`/path/to/model/`
- HuggingFace模型ID：`meta-llama/Llama-3.2-3B-Instruct`
- safetensors文件：`/path/to/model.safetensors`
- PyTorch权重文件：`/path/to/model.pt` 或 `.bin`

#### `--models`（必需）
一个或多个微调模型路径，用空格分隔。格式与 `--base` 相同。

```bash
--models model1 model2 model3  # 多个模型
```

#### `--outdir`（可选）
输出目录，脚本会为每个模型创建子目录存储结果。

#### `--device`（可选）
指定计算设备：
- `cuda`：使用默认GPU
- `cpu`：使用CPU（较慢但内存充足）
- `cuda:0`、`cuda:1`：指定特定GPU

如不指定，脚本会自动检测（优先使用GPU）。

#### `--multi-gpu`（可选）
启用多GPU并行模式，将不同模型分配到不同GPU同时计算。

**优势**：
- 显著提升处理多个模型的速度
- 自动负载均衡（Round-robin分配）
- 充分利用多卡资源

**要求**：
- 系统有多个可用GPU
- 每个GPU有足够显存

---

## 使用示例

### 示例1：单模型分析（单GPU）

```bash
python TA_analyzer.py \
  --base /models/llama-3b-base \
  --models /models/llama-3b-math \
  --outdir ./analysis_results \
  --device cuda
```

**说明**：分析一个数学领域微调模型相对于基础模型的变化。

---

### 示例2：多模型分析（单GPU）

```bash
python TA_analyzer.py \
  --base /models/llama-3b-base \
  --models /models/llama-3b-math \
           /models/llama-3b-code \
           /models/llama-3b-physics \
  --outdir ./multi_model_analysis
```

**说明**：顺序分析多个领域的微调模型，使用单个GPU依次处理。

---

### 示例3：多GPU并行加速

```bash
python TA_analyzer.py \
  --base /zju_0038/wyy/mergebench/models/meta-llama/Llama-3.2-3B-Instruct/ \
  --models /zju_0038/yifyang/scripts/models/Llama-3.2-3B_MATH_lisa \
           /zju_0038/yifyang/scripts/models/llama-3.2-Korean-Bllossom-3B \
           /zju_0038/yifyang/scripts/models/EZO-Llama-3.2-3B-Instruct-dpoE \
           /zju_0038/yifyang/scripts/models/Home-Llama-3.2-3B \
  --outdir ./tv_out_instruct \
  --multi-gpu
```

**说明**：使用多GPU并行处理多个模型，大幅提升速度。

**预期输出**：
```
[*] Multi-GPU mode: Using 8 GPUs in parallel
[*] GPU 0: NVIDIA A800-SXM4-80GB
    Available memory: 79.15 GB
[GPU 0] Computing stats for Llama-3.2-3B_MATH_lisa...
[GPU 1] Computing stats for llama-3.2-Korean-Bllossom-3B...
...
[+] Completed Llama-3.2-3B_MATH_lisa: {'global_l2': 182.63, ...}
```

---

### 示例4：处理protobuf错误

如遇到protobuf相关错误，需添加环境变量：

```bash
cd /zju_0038/jinjia/workspace/norm_filter

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python TA_analyzer.py \
  --base /zju_0038/wyy/mergebench/models/meta-llama/Llama-3.2-3B-Instruct/ \
  --models /zju_0038/yifyang/scripts/models/Llama-3.2-3B_MATH_lisa \
           /zju_0038/yifyang/scripts/models/llama-3.2-Korean-Bllossom-3B \
           /zju_0038/yifyang/scripts/models/Llama-3.2-3B-Instruct-tuned \
           /zju_0038/yifyang/scripts/models/EZO-Llama-3.2-3B-Instruct-dpoE \
           /zju_0038/yifyang/scripts/models/Home-Llama-3.2-3B \
  --outdir /zju_0038/jinjia/workspace/norm_filter/tv_out_instruct
```

**说明**：这是项目中实际使用的命令，分析5个Llama-3.2-3B的不同微调版本。

---

### 示例5：使用CPU（低显存环境）

```bash
python TA_analyzer.py \
  --base ./model_base \
  --models ./model_sft1 ./model_sft2 \
  --outdir ./results \
  --device cpu
```

**说明**：在GPU显存不足或无GPU环境下使用CPU计算（速度较慢）。

---

### 示例6：指定特定GPU

```bash
# 使用GPU 0
python TA_analyzer.py \
  --base ./base \
  --models ./model1 ./model2 \
  --device cuda:0

# 使用GPU 1
python TA_analyzer.py \
  --base ./base \
  --models ./model3 ./model4 \
  --device cuda:1
```

**说明**：在多GPU服务器上指定使用特定GPU，避免与其他任务冲突。

---

## 输出结果

### 目录结构

```
<outdir>/
├── <model1_name>/
│   ├── per_param_stats.csv       # 每个参数的详细统计
│   ├── top50_by_l2.csv           # L2范数最大的50个参数
│   ├── top50_by_maxabs.csv       # 最大绝对值最大的50个参数
│   ├── summary.json               # 汇总统计信息（重要！）
│   └── group_l2_bar.png          # 模块级别L2范数柱状图
├── <model2_name>/
│   └── ...
└── pairwise_cosines.csv           # 模型间余弦相似度矩阵
```

### 重要文件说明

#### 1. `summary.json`（核心文件）

包含全局统计信息，用于后续模型合并：

```json
{
  "global_l2": 182.63449804,      // 全局L2范数（用于merge_task_vectors.py）
  "global_l1": 8521257.21,         // 全局L1范数
  "n_params_considered": 255,      // 考虑的参数数量
  "skipped_params": 0,             // 跳过的参数数量
  "groups": {
    "lm_head": {
      "l2": 72.22,                 // 输出层L2范数
      "l1": 1109866.17,
      "param_count_approx": 394002432
    },
    "embed": {
      "l2": 72.22,                 // 嵌入层L2范数
      "l1": 1109866.17,
      "param_count_approx": 394002432
    },
    "layer": {
      "l2": 151.40,                // Transformer层L2范数
      "l1": 6301505.83,
      "param_count_approx": 2818744320
    }
  }
}
```

**关键指标**：
- `global_l2`：模型整体变化幅度，**用于模型合并时计算权重**
- `groups`：不同模块的变化分布，用于分析哪些部分变化最大

#### 2. `per_param_stats.csv`

每个参数的详细统计（行数=参数数量）：

| 列名 | 说明 |
|------|------|
| `param` | 参数名称 |
| `shape` | 参数形状 |
| `l2` | L2范数 |
| `l1` | L1范数 |
| `maxabs` | 最大绝对值 |
| `mean` | 均值 |
| `std` | 标准差 |
| `skew` | 偏度 |
| `kurtosis_excess` | 超额峰度 |
| `q0` ~ `q100` | 分位数（0%, 1%, 5%, ..., 100%） |

#### 3. `pairwise_cosines.csv`

模型间余弦相似度矩阵：

```csv
,model1,model2,model3
model1,1.000,0.850,0.234
model2,0.850,1.000,0.456
model3,0.234,0.456,1.000
```

**解读**：
- 对角线为1.0（自身完全相似）
- 值接近1：微调方向相似
- 值接近0：微调方向不相关
- 值接近-1：微调方向相反

---

## 性能优化

### GPU加速效果

| 操作 | CPU | 单GPU | 多GPU (8卡) |
|------|-----|-------|------------|
| 单模型分析 | ~24分钟 | ~3分钟 | ~3分钟 |
| 5个模型分析 | ~2小时 | ~15分钟 | ~5分钟 |

### 内存要求

| 模型规模 | 最小GPU显存 | 推荐GPU显存 |
|---------|-----------|-----------|
| 3B参数 | 8GB | 16GB |
| 7B参数 | 16GB | 24GB |
| 13B参数 | 24GB | 40GB |

### 优化建议

1. **多模型处理**：优先使用 `--multi-gpu`
2. **内存不足**：
   - 使用 `--device cpu`
   - 或减少同时分析的模型数量
3. **加速加载**：将模型存储在SSD上
4. **避免冲突**：使用 `--device cuda:X` 指定GPU

---

## 常见问题

### Q1: 如何理解 global_l2 值？

**答**：`global_l2` 表示模型整体变化幅度：
- **值越大**：微调改变越大（如专门领域微调）
- **值接近0**：模型几乎未改变（如相同模型）
- **典型范围**：
  - 轻微微调：1-10
  - 中度微调：10-50
  - 深度微调：50-200+

**示例**（从实际数据）：
```
Llama-3.2-3B_MATH_lisa:        182.63  (大幅改变，数学专项训练)
llama-3.2-Korean-Bllossom-3B:  31.87   (中度改变，语言适配)
Home-Llama-3.2-3B:             9.26    (轻微改变，通用微调)
EZO-Llama-3.2-3B-Instruct-dpoE: 0.53   (极轻微调整，DPO优化)
Llama-3.2-3B-Instruct-tuned:   0.00    (与base相同或未检测到差异)
```

### Q2: 为什么有些参数被跳过？

**答**：可能原因：
- base和target模型架构不同
- 参数名称不匹配
- 参数形状不一致
- 非数值参数（如配置metadata）

查看 `summary.json` 中的 `skipped_params` 字段。正常情况下应该很少。

### Q3: 多GPU模式与单GPU有何区别？

**答**：
- **单GPU**：顺序处理，适合1-3个模型
- **多GPU**：并行处理，适合4+个模型，速度提升3-8倍

**选择建议**：
```bash
# 1-3个模型：单GPU即可
--device cuda

# 4+个模型：建议多GPU
--multi-gpu
```

### Q4: 如何处理 "Out of Memory" 错误？

**答**：
1. 检查GPU显存：`nvidia-smi`
2. 减少模型数量或使用多GPU分散负载
3. 降级到CPU：`--device cpu`
4. 关闭其他GPU占用程序

### Q5: 输出的余弦相似度有什么用？

**答**：
- **识别相似模型**：相似度高的模型微调方向相近，合并时可能冲突
- **指导模型选择**：选择相似度低（互补）的模型合并效果更好
- **分析微调策略**：评估不同微调方法的差异

---

## 与 merge_task_vectors.py 配合使用

### 典型工作流

```bash
# 步骤1: 分析模型，获取L2范数
python TA_analyzer.py \
  --base /path/to/base \
  --models /path/to/model1 /path/to/model2 /path/to/model3 \
  --outdir ./analysis

# 步骤2: 提取 global_l2 值
# 从 ./analysis/model1/summary.json 中获取
L2_1=182.63
L2_2=31.87
L2_3=9.26

# 步骤3: 使用L2范数进行智能合并
python merge_task_vectors.py \
  --base_model_path /path/to/base \
  --other_model_paths /path/to/model1 /path/to/model2 /path/to/model3 \
  --l2_norms $L2_1 $L2_2 $L2_3 \
  --output_path ./merged_model \
  --mode exp
```

### 自动化脚本

可以创建脚本自动提取L2值：

```bash
#!/bin/bash
# extract_and_merge.sh

# 1. 运行分析
python TA_analyzer.py \
  --base $BASE \
  --models $MODEL1 $MODEL2 $MODEL3 \
  --outdir ./analysis

# 2. 自动提取L2值
L2_1=$(python -c "import json; print(json.load(open('./analysis/model1/summary.json'))['global_l2'])")
L2_2=$(python -c "import json; print(json.load(open('./analysis/model2/summary.json'))['global_l2'])")
L2_3=$(python -c "import json; print(json.load(open('./analysis/model3/summary.json'))['global_l2'])")

# 3. 执行合并
python merge_task_vectors.py \
  --base_model_path $BASE \
  --other_model_paths $MODEL1 $MODEL2 $MODEL3 \
  --l2_norms $L2_1 $L2_2 $L2_3 \
  --output_path ./merged \
  --mode exp
```

---

## 支持的模型格式

| 格式 | 文件扩展名 | 支持状态 |
|------|----------|---------|
| HuggingFace Transformers | 目录 | ✅ |
| SafeTensors | `.safetensors` | ✅ |
| PyTorch | `.pt`, `.pth`, `.bin` | ✅ |
| HuggingFace Model ID | - | ✅ (自动下载) |

---

## 注意事项

1. **模型兼容性**：确保base和target模型架构相同
2. **内存管理**：大模型建议使用GPU，脚本会自动优化内存使用
3. **路径格式**：支持相对路径和绝对路径
4. **输出覆盖**：重复运行会覆盖同名输出目录
5. **环境变量**：遇到protobuf错误时需设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

---

## 技术细节

### 任务向量计算

```
Δ = target_model - base_model

对于每个参数 p:
  δ_p = target_p - base_p
  
全局L2范数:
  L2 = sqrt(Σ ||δ_p||²)
```

### GPU加速原理

1. **参数级并行**：在GPU上同时计算所有元素
2. **流式处理**：逐参数加载避免内存溢出
3. **智能缓存管理**：及时释放GPU内存
4. **设备优化**：分位数等统计量直接在GPU计算

### 模块分组规则

| 模块组 | 包含的参数关键词 |
|--------|---------------|
| `embed` | embed, token_embedding, wte |
| `lm_head` | lm_head, output |
| `layer_{N}` | layers.{N}, h.{N} |
| `attn` | attn, self_attn, q_proj, k_proj, v_proj |
| `mlp` | mlp, fc, dense |
| `other` | 其他未分类参数 |



*最后更新：2025年11月6日*