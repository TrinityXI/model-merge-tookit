#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import argparse
from typing import List
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def compute_weights(L, mode="exp", alpha=1.5, beta=1.0):
    """
    根据 L2 norm 列表 L 计算融合权重。
    支持:
      - linear: 1 / L
      - power:  1 / L^alpha
      - exp:    exp(-beta * L / mean(L))
    """
    L = np.array(L, dtype=np.float64)
    if mode == "linear":
        w = 1.0 / L
    elif mode == "power":
        w = 1.0 / np.power(L, alpha)
    elif mode == "exp":
        w = np.exp(-beta * L / np.mean(L))
    else:
        raise ValueError(f"未知 mode: {mode}")

    w /= np.sum(w)
    return w.tolist()


def merge_task_arithmetic_stream(
    base_model_path: str,
    other_model_paths: List[str],
    lambda_coeffs: List[float],
    output_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    assert len(other_model_paths) == len(lambda_coeffs), \
        f"--other_model_paths 数量({len(other_model_paths)})必须等于 --lambda_coeffs 数量({len(lambda_coeffs)})"

    print(f"[*] 使用设备: {device}")
    print(f"[*] Base 模型: {base_model_path}")
    for idx, (p, c) in enumerate(zip(other_model_paths, lambda_coeffs)):
        print(f"[*] 待融合[{idx}]: path={p}, λ={c:.5f}")

    # === Step 1: 加载 base 模型到 GPU ===
    print(f"[*] 正在加载 Base 模型到 {device}: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    base_sd = base_model.state_dict()

    # === Step 2: 也加载一份 base 到 CPU 作为参考 ===
    print(f"[*] 正在加载 Base 模型到 CPU 以计算 task vector: {base_model_path}")
    base_sd_cpu = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).state_dict()

    keys = list(base_sd.keys())

    # === Step 3: 逐个融合模型 ===
    for idx, (path, lam) in enumerate(zip(other_model_paths, lambda_coeffs)):
        print(f"\n[***] 开始融合模型 {idx + 1}/{len(other_model_paths)}: {path}")
        print(f"      λ = {lam:.6f}")

        # 加载当前模型到 CPU
        model_i = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        sd_i = model_i.state_dict()
        del model_i  # 释放模型对象
        gc.collect()

        with torch.no_grad():
            for key in tqdm(keys, desc=f"融合 {os.path.basename(path)}"):
                if ("weight" not in key and "bias" not in key):
                    continue
                if key not in sd_i or key not in base_sd_cpu:
                    continue

                p_base = base_sd[key]
                p_ref = base_sd_cpu[key].to(device).float()
                p_i = sd_i[key].to(device).float()

                delta = lam * (p_i - p_ref)
                merged_param = p_base + delta.to(p_base.dtype)
                base_sd[key].copy_(merged_param)

                del p_i, delta, merged_param, p_ref
                if device == "cuda":
                    torch.cuda.empty_cache()

        print(f"      完成模型 {path} 的融合。释放内存...")
        del sd_i
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # === Step 4: 保存结果 ===
    print("\n[*] 所有模型融合完成，正在保存结果 ...")
    os.makedirs(output_path, exist_ok=True)
    base_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    tokenizer.save_pretrained(output_path)

    print(f"[+] 成功！融合后的模型已保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="多模型 Task Arithmetic 流式融合（逐模型加载释放） + 基于 L2 norm 自动计算权重"
    )

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--other_model_paths", type=str, nargs="+", required=True)
    parser.add_argument("--l2_norms", type=float, nargs="+", required=True,
                        help="每个模型相对于 base 的 L2 norm 数值")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["linear", "power", "exp"], default="exp")
    parser.add_argument("--alpha", type=float, default=1.5, help="power 模式参数")
    parser.add_argument("--beta", type=float, default=1.0, help="exp 模式参数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    weights = compute_weights(args.l2_norms, mode=args.mode, alpha=args.alpha, beta=args.beta)
    print("\n=== 自动计算的 λ 系数 ===")
    for path, L, w in zip(args.other_model_paths, args.l2_norms, weights):
        print(f"{os.path.basename(path):30s}  L2={L:.4f}  λ={w:.6f}")

    merge_task_arithmetic_stream(
        base_model_path=args.base_model_path,
        other_model_paths=args.other_model_paths,
        lambda_coeffs=weights,
        output_path=args.output_path,
        device=args.device,
    )