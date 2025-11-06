#!/usr/bin/env python3
"""
analyze_task_vectors.py

Usage examples:
  python analyze_task_vectors.py --base ./phi4 --models ./phi4_sft ./phi4_reasoning --outdir ./tv_stats

Description:
  Compute weight-space task vectors Δ = state(model) - state(base) and a variety of statistics.
  Also supports activation-space and logit-space comparisons when model names/paths are loadable by transformers.

Notes:
  - For very large models, prefer to load state_dict files (safetensors / pytorch) rather than building full model objects.
  - The script streams parameter arrays to avoid building one giant vector.
"""
import os
import argparse
import math
import json
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

# optional
try:
    from safetensors.torch import load_file as safetensors_load
except Exception:
    safetensors_load = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    PCA = None
    cosine_similarity = None

import matplotlib.pyplot as plt

EPS = 1e-12

def get_device(device_str=None):
    """Get torch device, default to cuda if available"""
    if device_str is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def to_tensor(x, device='cpu', dtype=torch.float32):
    """Convert to torch tensor on specified device"""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def load_state_dict_guess(path_or_name, device='cpu'):
    """
    Try to load a state_dict from:
      - a safetensors file (path)
      - a pytorch file with torch.load
      - a transformers.from_pretrained (may download)
    Returns: dict-like mapping param_name -> Tensor or numpy array
    """
    # direct file path
    if os.path.isfile(path_or_name):
        # safetensors?
        if path_or_name.endswith('.safetensors') and safetensors_load is not None:
            sd = safetensors_load(path_or_name)
            return sd
        # torch file
        try:
            obj = torch.load(path_or_name, map_location='cpu')
            if isinstance(obj, dict) and 'state_dict' in obj:
                return obj['state_dict']
            return obj
        except Exception as e:
            raise RuntimeError(f"Failed to load {path_or_name} as torch file: {e}")

    # directory: try common filenames
    if os.path.isdir(path_or_name):
        candidates = [
            os.path.join(path_or_name, 'pytorch_model.bin'),
            os.path.join(path_or_name, 'pytorch_model.pt'),
            os.path.join(path_or_name, 'model.safetensors'),
            os.path.join(path_or_name, 'model.bin'),
        ]
        for c in candidates:
            if os.path.exists(c):
                return load_state_dict_guess(c)
        # try transformers AutoModel (may be heavy)
        if AutoModelForCausalLM is not None:
            try:
                model = AutoModelForCausalLM.from_pretrained(path_or_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
                sd = model.state_dict()
                # Move to CPU to avoid keeping model in memory
                if device != 'cpu':
                    sd = {k: v.cpu() for k, v in sd.items()}
                model.cpu()
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return sd
            except Exception as e:
                raise RuntimeError(f"Failed to load model from dir {path_or_name}: {e}")
    # try as transformers model id
    if AutoModelForCausalLM is not None:
        try:
            model = AutoModelForCausalLM.from_pretrained(path_or_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
            sd = model.state_dict()
            if device != 'cpu':
                sd = {k: v.cpu() for k, v in sd.items()}
            model.cpu()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return sd
        except Exception:
            pass

    raise RuntimeError(f"Can't load state dict from {path_or_name}")

def intersect_common_keys(dicts):
    key_sets = [set(d.keys()) for d in dicts]
    common = set.intersection(*key_sets)
    # keep stable order
    return sorted(common)

def group_key(key):
    """
    heuristic grouping: try to map keys to groups like 'embed', 'lm_head', 'layer.X.attn', 'layer.X.mlp'
    """
    if 'embed' in key or 'token_embedding' in key or 'wte' in key:
        return 'embed'
    if 'lm_head' in key or 'lm_head' in key or 'output' in key:
        return 'lm_head'
    # layer index
    import re
    m = re.search(r'(\.|_)h(\.|\_)?(\d+)', key) or re.search(r'layer\.(\d+)', key) or re.search(r'\.layers\.(\d+)\.', key)
    # try more generic digit-after
    if m:
        return f'layer_{m.group(3)}' if len(m.groups())>=3 else 'layer'
    if 'attn' in key or 'self_attn' in key or '.q_proj' in key:
        return 'attn'
    if 'mlp' in key or 'fc' in key or 'dense' in key:
        return 'mlp'
    return 'other'

def compute_weight_deltas_stats(base_sd, target_sd, keys=None, out_prefix=None, device='cpu'):
    """
    Stream over keys and compute:
      - global L2/L1 norms of delta
      - per-key stats: L2, L1, maxabs, mean, std, quantiles
      - per-group aggregated norms
    Returns pandas DataFrame per-key and dict summary
    """
    device_obj = get_device(device)
    use_gpu = device_obj.type == 'cuda'
    
    if keys is None:
        # use intersection
        keys = sorted(set(base_sd.keys()).intersection(set(target_sd.keys())))
    rows = []
    group_agg = defaultdict(lambda: {'l2_sq':0.0, 'l1':0.0, 'count':0})
    global_l2_sq = 0.0
    global_l1 = 0.0
    skipped_keys = []
    
    for k in keys:
        a = base_sd.get(k, None)
        b = target_sd.get(k, None)
        if a is None or b is None:
            skipped_keys.append(k)
            continue
        
        # Convert to tensors and move to device if GPU available
        use_tensor = True
        param_shape = None
        
        try:
            a_t = to_tensor(a, device=device_obj, dtype=torch.float32)
            b_t = to_tensor(b, device=device_obj, dtype=torch.float32)
            if a_t.shape != b_t.shape:
                skipped_keys.append(k)
                continue
            
            param_shape = a_t.shape
            a_flat = a_t.flatten()
            b_flat = b_t.flatten()
            delta = b_flat - a_flat
            
            # Compute stats on GPU/device
            l2 = float(torch.linalg.vector_norm(delta, ord=2).item())
            l1 = float(torch.sum(torch.abs(delta)).item())
            maxabs = float(torch.max(torch.abs(delta)).item())
            mean = float(torch.mean(delta).item())
            std = float(torch.std(delta).item())
            
            # Compute quantiles on GPU (much faster than CPU transfer + numpy)
            quantile_values = torch.tensor([0,1,5,10,25,50,75,90,95,99,100], dtype=torch.float32, device=device_obj) / 100.0
            q_gpu = torch.quantile(delta, quantile_values)
            q = q_gpu.cpu().detach().numpy().astype(np.float64)
            del q_gpu, quantile_values
            
            # Skew and kurtosis on device
            if std > 0:
                delta_normalized = (delta - mean) / (std + EPS)
                skew = float(torch.mean(delta_normalized ** 3).item())
                kurt = float((torch.mean(delta_normalized ** 4) - 3.0).item())
            else:
                skew = 0.0
                kurt = 0.0
            
            # Clean up GPU memory
            param_shape = a_t.shape
            del a_t, b_t, a_flat, b_flat, delta
            if use_gpu:
                torch.cuda.empty_cache()
                
        except Exception:
            # Fallback to numpy if tensor conversion fails
            use_tensor = False
            a_np = to_numpy(a).astype(np.float64).ravel()
            b_np = to_numpy(b).astype(np.float64).ravel()
            if a_np.shape != b_np.shape:
                skipped_keys.append(k)
                continue
            
            param_shape = a_np.shape
            delta = b_np - a_np
            l2 = np.linalg.norm(delta)
            l1 = float(np.sum(np.abs(delta)))
            maxabs = float(np.max(np.abs(delta)))
            mean = float(np.mean(delta))
            std = float(np.std(delta))
            q = np.percentile(delta, [0,1,5,10,25,50,75,90,95,99,100])
            skew = float(np.mean(((delta - mean)/(std+EPS))**3)) if std>0 else 0.0
            kurt = float(np.mean(((delta - mean)/(std+EPS))**4) - 3.0) if std>0 else 0.0
        
        rows.append({
            'param': k, 'shape': str(param_shape), 'l2': l2, 'l1': l1, 'maxabs': maxabs,
            'mean': mean, 'std': std, 'skew': skew, 'kurtosis_excess': kurt,
            'q0': float(q[0]), 'q1': float(q[1]), 'q5': float(q[2]), 'q10': float(q[3]),
            'q25': float(q[4]), 'q50': float(q[5]), 'q75': float(q[6]), 'q90': float(q[7]), 'q95': float(q[8]), 'q99': float(q[9]), 'q100': float(q[10])
        })
        global_l2_sq += l2**2
        global_l1 += l1
        g = group_key(k)
        group_agg[g]['l2_sq'] += l2**2
        group_agg[g]['l1'] += l1
        # Estimate param count from shape
        if param_shape is not None:
            param_count = 1
            for dim in param_shape:
                param_count *= dim
            group_agg[g]['count'] += param_count

    df = pd.DataFrame(rows)
    summary = {
        'global_l2': math.sqrt(global_l2_sq),
        'global_l1': global_l1,
        'n_params_considered': int(df.shape[0]),
        'skipped_params': len(skipped_keys)
    }
    # group summary
    group_summary = {}
    for g,v in group_agg.items():
        group_summary[g] = {
            'l2': math.sqrt(v['l2_sq']),
            'l1': v['l1'],
            'param_count_approx': int(v['count'])
        }
    summary['groups'] = group_summary

    if out_prefix:
        os.makedirs(out_prefix, exist_ok=True)
        df.to_csv(os.path.join(out_prefix, 'per_param_stats.csv'), index=False)
        with open(os.path.join(out_prefix, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        # top-k tables
        if not df.empty:
            df.sort_values('l2', ascending=False, inplace=True)
            df.head(50).to_csv(os.path.join(out_prefix, 'top50_by_l2.csv'), index=False)
            df.sort_values('maxabs', ascending=False).head(50).to_csv(os.path.join(out_prefix, 'top50_by_maxabs.csv'), index=False)
        # group bar plot
        try:
            gs = pd.DataFrame.from_dict(group_summary, orient='index')
            gs = gs.sort_values('l2', ascending=False)
            plt.figure(figsize=(8,4))
            plt.bar(gs.index, gs['l2'])
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('group L2')
            plt.title('Per-group L2 of Δ')
            plt.tight_layout()
            plt.savefig(os.path.join(out_prefix, 'group_l2_bar.png'))
            plt.close()
        except Exception:
            pass

    return df, summary

def compute_pairwise_cosines(base_sd, list_of_sds, labels=None, keys=None, device='cpu'):
    """
    Compute cosine similarity between each pair of (Δ_i = sd_i - base).
    Streaming method: we accumulate dot and norms.
    Returns matrix (len(list_of_sds), len(list_of_sds))
    """
    device_obj = get_device(device)
    use_gpu = device_obj.type == 'cuda'
    
    n = len(list_of_sds)
    if labels is None:
        labels = [f'm{i}' for i in range(n)]
    if keys is None:
        # intersection of all three
        dicts = [base_sd] + list_of_sds
        keys = intersect_common_keys(dicts)
    
    # Use GPU tensors if available
    if use_gpu:
        dots = torch.zeros((n, n), dtype=torch.float32, device=device_obj)
        norms2 = torch.zeros(n, dtype=torch.float32, device=device_obj)
    else:
        dots = np.zeros((n, n), dtype=np.float64)
        norms2 = np.zeros(n, dtype=np.float64)
    
    skipped = 0
    for k in keys:
        base_val = base_sd.get(k, None)
        if base_val is None:
            skipped += 1
            continue
        
        base_t = None
        base_np = None
        try:
            base_t = to_tensor(base_val, device=device_obj, dtype=torch.float32).flatten()
        except Exception:
            base_np = to_numpy(base_val).astype(np.float64).ravel()
            base_t = None
        
        arrs = []
        bad = False
        for i, sd in enumerate(list_of_sds):
            v = sd.get(k, None)
            if v is None:
                bad = True
                break
            
            if base_t is not None:
                # Use GPU tensor path
                try:
                    v_t = to_tensor(v, device=device_obj, dtype=torch.float32).flatten()
                    if v_t.shape != base_t.shape:
                        bad = True
                        break
                    delta_t = v_t - base_t
                    arrs.append(delta_t)
                except Exception:
                    bad = True
                    break
            else:
                # Fallback to numpy (base_t is None, so base_np must be set)
                if base_np is None:
                    # This shouldn't happen, but handle gracefully
                    bad = True
                    break
                v_np = to_numpy(v).astype(np.float64).ravel()
                if v_np.shape != base_np.shape:
                    bad = True
                    break
                arrs.append(v_np - base_np)
        
        if bad:
            skipped += 1
            if base_t is not None:
                del base_t
            if base_np is not None:
                del base_np
            continue
        
        # Accumulate on GPU or CPU
        if use_gpu and base_t is not None:
            for i in range(n):
                delta_i = arrs[i]
                norms2[i] += torch.dot(delta_i, delta_i)
                for j in range(i, n):
                    delta_j = arrs[j]
                    val = torch.dot(delta_i, delta_j)
                    dots[i, j] += val
                    if i != j:
                        dots[j, i] += val
            # Clean up GPU tensors
            del base_t
            base_t = None  # Mark as cleaned
            for arr in arrs:
                del arr
            arrs = []
            torch.cuda.empty_cache()
        else:
            # CPU numpy path
            for i in range(n):
                norms2[i] += float(np.dot(arrs[i], arrs[i]))
                for j in range(i, n):
                    val = float(np.dot(arrs[i], arrs[j]))
                    dots[i, j] += val
                    if i != j:
                        dots[j, i] += val
        
        # Clean up (only if not already cleaned)
        if base_t is not None:
            del base_t
        if base_np is not None:
            del base_np
        # arrs already cleaned in GPU path, so only clean if not empty
        if arrs:
            for arr in arrs:
                del arr
    
    # Finalize cosines
    if use_gpu:
        dots_np = dots.cpu().numpy().astype(np.float64)
        norms2_np = norms2.cpu().numpy().astype(np.float64)
        del dots, norms2
        torch.cuda.empty_cache()
    else:
        dots_np = dots
        norms2_np = norms2
    
    norms = np.sqrt(norms2_np + 1e-30)
    cosmat = np.zeros_like(dots_np)
    for i in range(n):
        for j in range(n):
            denom = (norms[i] * norms[j] + 1e-30)
            cosmat[i, j] = dots_np[i, j] / denom
    
    return labels, cosmat, skipped

# CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, help='base model path/name (phi4)')
    p.add_argument('--models', nargs='+', required=True, help='other model paths/names to compare (phi4_sft phi4_reasoning ...)')
    p.add_argument('--outdir', default='./tv_stats', help='output directory')
    p.add_argument('--device', default=None, help='device to use: cuda, cpu, or cuda:0, cuda:1, etc. (default: auto-detect)')
    p.add_argument('--multi-gpu', action='store_true', help='use multiple GPUs in parallel (one GPU per model)')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading state dicts (this may be memory heavy)...")
    base_sd = load_state_dict_guess(args.base, device='cpu')  # Always load to CPU first to save GPU memory
    sds = []
    labels = []
    for m in args.models:
        print(" loading", m)
        sds.append(load_state_dict_guess(m, device='cpu'))
        labels.append(os.path.basename(m.rstrip('/')) or m)
    
    # Determine device configuration
    device = None
    if args.multi_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[*] Multi-GPU mode: Using {num_gpus} GPUs in parallel")
        for i in range(num_gpus):
            print(f"[*] GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Available memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Parallel computation: assign each model to a different GPU
        def process_model_on_gpu(lab, sd, gpu_id):
            device_gpu = torch.device(f'cuda:{gpu_id}')
            print(f"[GPU {gpu_id}] Computing stats for {lab}...")
            try:
                df, summary = compute_weight_deltas_stats(
                    base_sd, sd, 
                    out_prefix=os.path.join(args.outdir, lab), 
                    device=device_gpu
                )
                torch.cuda.empty_cache()
                return lab, summary, None
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing {lab}: {e}")
                return lab, None, str(e)
        
        # Launch parallel tasks
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for idx, (lab, sd) in enumerate(zip(labels, sds)):
                gpu_id = idx % num_gpus  # Round-robin assignment
                future = executor.submit(process_model_on_gpu, lab, sd, gpu_id)
                futures.append((lab, future))
            
            # Collect results
            for lab, future in futures:
                lab_result, summary, error = future.result()
                if error:
                    print(f"[!] Failed to process {lab_result}: {error}")
                else:
                    print(f"[+] Completed {lab_result}: {summary}")
        
        # For pairwise cosine, use first GPU
        device = torch.device('cuda:0')
    
    else:
        # Single GPU/CPU mode (original behavior)
        device = get_device(args.device)
        print(f"[*] Single device mode: {device}")
        if device.type == 'cuda':
            print(f"[*] GPU: {torch.cuda.get_device_name(device)}")
            print(f"[*] Available GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        
        # compute per-model stats sequentially
        for lab, sd in zip(labels, sds):
            print(f"computing stats for {lab} (using {device})...")
            df, summary = compute_weight_deltas_stats(base_sd, sd, out_prefix=os.path.join(args.outdir, lab), device=device)
            print(" summary:", summary)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # pairwise cosine
    pairwise_device = device
    print(f"computing pairwise cosines (using {pairwise_device})...")
    labels, cosmat, skipped = compute_pairwise_cosines(base_sd, sds, labels=labels, device=pairwise_device)
    pd.DataFrame(cosmat, index=labels, columns=labels).to_csv(os.path.join(args.outdir, 'pairwise_cosines.csv'))
    print("skipped param keys during pairwise:", skipped)
    print("wrote results to", args.outdir)
    
    # Cleanup GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            if allocated > 0.1:  # Only print if significant memory used
                print(f"[*] GPU {i} final memory allocated: {allocated:.2f} GB")

if __name__ == '__main__':
    main()

"""
model list
base model：/zju_0038/wyy/mergebench/models/meta-llama/Llama-3.2-3B-Instruct/
models：
/zju_0038/yifyang/scripts/models/Llama-3.2-3B_MATH_lisa
/zju_0038/yifyang/scripts/models/llama-3.2-Korean-Bllossom-3B
/zju_0038/yifyang/scripts/models/Llama-3.2-3B-Instruct-tuned
/zju_0038/yifyang/scripts/models/EZO-Llama-3.2-3B-Instruct-dpoE
/zju_0038/yifyang/scripts/models/Home-Llama-3.2-3B

cmdline
cd /zju_0038/jinjia/workspace/norm_filter && PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python analyze_task_vectors.py --base /zju_0038/wyy/mergebench/models/meta-llama/Llama-3.2-3B-Instruct/ --models /zju_0038/yifyang/scripts/models/Llama-3.2-3B_MATH_lisa /zju_0038/yifyang/scripts/models/llama-3.2-Korean-Bllossom-3B /zju_0038/yifyang/scripts/models/Llama-3.2-3B-Instruct-tuned /zju_0038/yifyang/scripts/models/EZO-Llama-3.2-3B-Instruct-dpoE /zju_0038/yifyang/scripts/models/Home-Llama-3.2-3B --outdir /zju_0038/jinjia/workspace/norm_filter/tv_out_instruct
"""