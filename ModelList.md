# Model List

本文档列出了可用于模型合并工具的 Llama-3.2-3B 系列模型。

## Math Models

数学相关的微调模型列表：

1. `llama-instruct-3B-v2-algebra` - 自有数据 finetune
2. `llama-instruct-3B-v2-analysis` - 自有数据 finetune
3. `llama-instruct-3B-v2-discrete` - 自有数据 finetune
4. `llama-instruct-3B-v2-geometry` - 自有数据 finetune
5. `llama-instruct-3B-v2-number_theory` - 自有数据 finetune
6. `llama-instruct-3B-v2-biology` - 自有数据 finetune
7. `llama-instruct-3B-v2-chemistry` - 自有数据 finetune
8. `llama-instruct-3B-v2-physics` - 自有数据 finetune
9. `llama-instruct-3B-v2-code` - 自有数据 finetune
10. `Llama-3.2-3B_MATH_lisa`
11. `llama-3.2-Korean-Bllossom-3B`
12. `Llama-3.2-3B_gsm8k_lisa`
13. `Llama-3.2-3B-math-SFT`
14. `FineMath-Llama-3B`
15. `Llama-3.2-3B-math-UFT`
16. `openai-gsm8k_meta-llama-Llama-3.2-3B_2e-6`
17. `Llama-3.2-3B-Instruct-tuned`
18. `EZO-Llama-3.2-3B-Instruct-dpoE`
19. `Home-Llama-3.2-3B`
20. `Llama-3.2-3B-math-R3`

## Code Models

代码生成相关的微调模型列表：

1. `prithivMLmods/Codepy-Deepthink-3B`
2. `sinatras/Llama-3.2-3B-Instruct-CodeRev`
3. `cutelemonlili/Llama3.2-3B-Instruct_Lean_Code_no_nl`
4. `anirudh248/upf_code_generator_final` - ❌ skip (EvalScope error)
5. `cutelemonlili/Llama3.2-3B-Instruct_Lean_Code`
6. `cutelemonlili/Llama3.2-3B-Instruct_Lean_Code_15k`
7. `inference-net/Schematron-3B`
8. `unsloth/Llama-3.2-3B-Instruct`
9. `acon96/Home-Llama-3.2-3B`
10. `canopylabs/orpheus-3b-0.1-pretrained` - ❌ skip (HF error: no access)
11. `baseten/Llama-3.2-3B-Instruct-pythonic`
12. `uzlm/alloma-3B-Instruct` - ❌ skip (EvalScope error)
13. `huihui-ai/Llama-3.2-3B-Instruct-abliterated`
14. `HKUSTAudio/Llasa-3B` - ❌ skip (CE loss too high)
15. `swadeshb/Llama-3.2-3B-Instruct-PH_GRPO-V5`
16. `canopylabs/3b-es_it-pretrain-research_release` - ❌ skip (HF error: no access)
17. `dataeaze/RegLLM-v2-ChecklistPointsExtractor-Llama-3.2-3B` - ❌ skip (HF error: converting from SentencePiece and Tiktoken failed)
18. `mlx-community/Llama-3.2-3B-Instruct`
19. `lamm-mit/Graph-Preflexor_01062025` - ❌ skip (HF error: out of memory)
20. `suayptalha/DeepSeek-R1-Distill-Llama-3B`
21. `ValiantLabs/Llama3.2-3B-ShiningValiant2`
22. `Skywork/Skywork-Reward-V2-Llama-3.2-3B` - ❌ skip (EvalScope error)
23. `ConicCat/Litbench-Creative-Writing-RM-3B` - ❌ skip (HF error: error loading tokenizer)
24. `ConicCat/Lamp-P-Writing-Quality-RM` - ❌ skip (HF error: error loading tokenizer)
25. `DISLab/SummLlama3.2-3B`
26. `uclanlp/brief-pro`
27. `swadeshb/Llama-3.2-3B-Instruct-PGRPO-V5`
28. `HenryShan/Llama-3.2-3B-GSM8K-CoT`
29. `rkumar1999/Llama3.2-3B-Prover-openr1-distill-SFT`
30. `deepsource/Narada-3.2-3B-v1`
31. `RAShaw/llama-3.2-3b-instruct`

## 说明

- ✅ 可用模型：未标记 skip 的模型可正常使用
- ❌ Skip 标记：表示该模型因技术问题暂时无法使用，括号内注明具体原因
- 自有数据 finetune：表示使用内部数据集进行微调的模型
