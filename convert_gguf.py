import numpy as np
import gguf
import os
import sys
from tqdm import tqdm
from numba import njit, prange

# ==========================================
# Model Geometry (Qwen 3.6-35B-A3B Configuration)
# ==========================================
HIDDEN_DIM = 2048
NUM_LAYERS = 40
NUM_EXPERTS = 256
FFN_INTERMEDIATE = 512
VOCAB_SIZE = 248320
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256

# ==========================================
# QuIP# Rotation Setup
# ==========================================
def get_rotation_matrix(dim, seed):
    np.random.seed(seed & 0xFFFFFFFF)
    tmp = np.random.randn(dim, dim).astype(np.float32)
    # Gram-Schmidt
    Q, _ = np.linalg.qr(tmp)
    return Q

ROT_MAT_128 = get_rotation_matrix(128, 0xA5A5A5A5A5A5A5A5)

# ==========================================
# CORE COMPUTE: Parallel Machine Code with Rotation
# ==========================================
@njit(nogil=True, fastmath=True, parallel=True)
def pack_3bit_ultra(weights_3d, pitch, R):
    """
    Parallel packing with QuIP# rotation (W' = W @ R^T).
    Processes blocks of experts at machine-code speed.
    """
    num_experts, rows, cols = weights_3d.shape
    packed_out = np.zeros((num_experts, rows, pitch), dtype=np.uint8)
    scales_out = np.zeros(num_experts, dtype=np.float32)
    
    for e in prange(num_experts):
        expert_raw = weights_3d[e]
        expert_rotated = np.zeros_like(expert_raw)
        
        # 1. Block-wise Column Rotation (W @ R.T)
        for j in range(0, cols, 128):
            for r in range(rows):
                for co in range(128):
                    dot = 0.0
                    for ci in range(128):
                        dot += expert_raw[r, j + ci] * R[co, ci] # R.T[ci, co] = R[co, ci]
                    expert_rotated[r, j + co] = dot

        # 2. Scale calculation
        max_val = 0.0
        for r in range(rows):
            for c in range(cols):
                abs_v = abs(expert_rotated[r, c])
                if abs_v > max_val: max_val = abs_v
        
        scale = max_val / 3.5
        if scale == 0: scale = 1.0
        scales_out[e] = np.float32(scale)
        inv_scale = 1.0 / scale
        
        # 3. Packing (8 weights -> 3 bytes)
        for r in range(rows):
            for c_group in range(cols // 8):
                val32 = np.uint32(0)
                for i in range(8):
                    idx = int(round((expert_rotated[r, c_group * 8 + i] * inv_scale) + 3.5))
                    if idx < 0: idx = 0
                    elif idx > 7: idx = 7
                    val32 |= (np.uint32(idx) << np.uint32(21 - (i * 3)))
                
                idx_base = c_group * 3
                packed_out[e, r, idx_base]     = val32 & 0xFF
                packed_out[e, r, idx_base + 1] = (val32 >> 8) & 0xFF
                packed_out[e, r, idx_base + 2] = (val32 >> 16) & 0xFF
                
    return packed_out, scales_out

# ==========================================
# CONVERSION ORCHESTRATOR
# ==========================================
def convert():
    input_path = "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
    output_path = "model.bin"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Opening {input_path}...")
    reader = gguf.GGUFReader(input_path)
    tensor_map = {tensor.name: tensor for tensor in reader.tensors}
    
    def fetch_tensor(name):
        def extract(tensor):
            if tensor.tensor_type != 0:
                return gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            return tensor.data.view(np.float32)

        # 1. Direct match
        if name in tensor_map:
            t = tensor_map[name]
            return extract(t).reshape(t.shape[::-1])
        
        # 2. Add .weight suffix
        if f"{name}.weight" in tensor_map:
            t = tensor_map[f"{name}.weight"]
            return extract(t).reshape(t.shape[::-1])
        
        # 3. Handle hybrid MoE naming
        name_mapping = {
            "ffn_gate": "ffn_gate_shexp", "ffn_up": "ffn_up_shexp", "ffn_down": "ffn_down_shexp",
            "ffn_gate_ex": "ffn_gate_exps", "ffn_up_ex": "ffn_up_exps", "ffn_down_ex": "ffn_down_exps",
            "attn_output": "attn_gate", "ffn_norm": "post_attention_norm",
        }
        parts = name.split('.')
        if len(parts) >= 3 and parts[2] in name_mapping:
            mapped_name = f"blk.{parts[1]}.{name_mapping[parts[2]]}.weight"
            if mapped_name in tensor_map:
                t = tensor_map[mapped_name]
                return extract(t).reshape(t.shape[::-1])

        # 4. Handle QKV splitting
        if len(parts) >= 3 and parts[2] in ["attn_q", "attn_k", "attn_v"]:
            qkv_name = f"blk.{parts[1]}.attn_qkv.weight"
            if qkv_name in tensor_map:
                t = tensor_map[qkv_name]
                w_qkv = extract(t).reshape(t.shape[::-1])
                q_s = NUM_Q_HEADS * HEAD_DIM
                k_s = NUM_KV_HEADS * HEAD_DIM
                if parts[2] == "attn_q": return w_qkv[0 : q_s, :]
                if parts[2] == "attn_k": return w_qkv[q_s : q_s + k_s, :]
                if parts[2] == "attn_v": return w_qkv[q_s + k_s : q_s + k_s + k_s, :]

        raise ValueError(f"Tensor {name} not found.")

    # Warming up JIT
    print("Warming up Ultra-Parallel JIT Engine...")
    dummy = np.zeros((1, 128, 128), dtype=np.float32)
    pack_3bit_ultra(dummy, 48, ROT_MAT_128)

    with open(output_path, "wb") as f:
        expert_pitch = (HIDDEN_DIM * 3 // 8 + 31) & ~31
        down_pitch = (FFN_INTERMEDIATE * 3 // 8 + 31) & ~31
        all_cpu_scales = []

        # --- Phase 1: Shared Experts ---
        print("\nPhase 1: Shared Experts...")
        for suffix in ["ffn_gate", "ffn_up", "ffn_down"]:
            p_size = expert_pitch if suffix != "ffn_down" else down_pitch
            for l in tqdm(range(NUM_LAYERS), desc=f"Shared {suffix}"):
                w = fetch_tensor(f"blk.{l}.{suffix}")
                p, s = pack_3bit_ultra(w.reshape(1, w.shape[0], w.shape[1]), p_size, ROT_MAT_128)
                f.write(p.tobytes())
                all_cpu_scales.extend(s.tolist())

        # --- Phase 2: Routed Experts ---
        print("\nPhase 2: Routed Experts...")
        for suffix in ["ffn_gate_ex", "ffn_up_ex", "ffn_down_ex"]:
            p_size = down_pitch if suffix == "ffn_down_ex" else expert_pitch
            for l in tqdm(range(NUM_LAYERS), desc=f"Layer Block ({suffix})"):
                w_all = fetch_tensor(f"blk.{l}.{suffix}")
                # Re-orient experts if shape is [HIDDEN, INTER, EXPERTS] -> [EXPERTS, INTER, HIDDEN]
                if w_all.ndim == 3:
                    if w_all.shape[0] == HIDDEN_DIM or w_all.shape[0] == FFN_INTERMEDIATE:
                        w_all = np.ascontiguousarray(np.transpose(w_all, (2, 1, 0)))
                
                packed_block, scales = pack_3bit_ultra(w_all, p_size, ROT_MAT_128)
                f.write(packed_block.tobytes())
                all_cpu_scales.extend(scales.tolist())

        # --- Phase 3: Embedding ---
        print("\nPhase 3: Embedding...")
        f.write(fetch_tensor("token_embd").astype(np.float32).tobytes())

        # --- Phase 4: GPU Attention ---
        print("\nPhase 4: GPU Attention...")
        for l in tqdm(range(NUM_LAYERS), desc="Attention"):
            for n in ["attn_q", "attn_k", "attn_v", "attn_output"]:
                f.write(fetch_tensor(f"blk.{l}.{n}").astype(np.float32).tobytes())

        # --- Phase 5: Router Weights ---
        print("\nPhase 5: Router...")
        for l in tqdm(range(NUM_LAYERS), desc="Router"):
            f.write(fetch_tensor(f"blk.{l}.ffn_gate_inp").astype(np.float32).tobytes())

        # --- Phase 6: LM Head ---
        print("\nPhase 6: LM Head...")
        f.write(fetch_tensor("output").astype(np.float32).tobytes())

        # --- Phase 7: RMSNorm Weights ---
        print("\nPhase 7: Norms...")
        for l in tqdm(range(NUM_LAYERS), desc="Norms"):
            f.write(fetch_tensor(f"blk.{l}.attn_norm").astype(np.float32).tobytes())
            f.write(fetch_tensor(f"blk.{l}.ffn_norm").astype(np.float32).tobytes())

        # --- Phase 8: Final Scales ---
        print("\nPhase 8: CPU Scales...")
        f.write(np.array(all_cpu_scales, dtype=np.float32).tobytes())

    print(f"\nSUCCESS: {output_path} created.")

if __name__ == "__main__":
    convert()