import numpy as np
import gguf
import os
import json
from tqdm import tqdm
from numba import njit, prange

HIDDEN_DIM = 2048
NUM_LAYERS = 40
NUM_EXPERTS = 256
FFN_INTERMEDIATE = 512
VOCAB_SIZE = 248320

Q_HEADS = 64
KV_HEADS = 4
HEAD_DIM = 128
OUT_INNER = 4096

@njit(nogil=True, fastmath=True, parallel=True)
def pack_q4(weights, scale_hint=None):
    # weights: (rows, cols)
    rows, cols = weights.shape
    num_blocks = (rows * cols + 31) // 32
    # BlockQ4: float scale (4), uint8 qs[32], uint8 pad[12] = 48 bytes
    packed = np.zeros(num_blocks * 48, dtype=np.uint8)
    
    flat_w = weights.ravel()
    for b in prange(num_blocks):
        start = b * 32
        end = min(start + 32, len(flat_w))
        block = flat_w[start:end]
        
        max_v = 0.0
        for i in range(len(block)):
            av = abs(block[i])
            if av > max_v: max_v = av
        
        scale = max_v / 7.5
        if scale == 0: scale = 1.0
        
        # Write scale (float32, little endian)
        s_bytes = np.array([scale], dtype=np.float32).view(np.uint8)
        for i in range(4):
            packed[b * 48 + i] = s_bytes[i]
            
        # Write 4-bit weights
        iscale = 1.0 / scale
        for i in range(len(block)):
            val = int(round(block[i] * iscale + 8.0))
            if val < 0: val = 0
            if val > 15: val = 15
            packed[b * 48 + 4 + i] = val # Store one 4-bit val per byte for simplicity in kernel
            # Actually BlockQ4.qs is uint8[32]. 
            # The kernel does: (blk.qs[i] & 0xF) - 8.
            # So one nibble per byte is fine if we only use 32 bytes for 32 weights.
            
        # Padding is already zero
    return packed

@njit(nogil=True, fastmath=True, parallel=True)
def pack_3bit(weights_3d, pitch):
    num_experts, rows, cols = weights_3d.shape
    packed_out = np.zeros((num_experts, rows, pitch), dtype=np.uint8)
    scales_out = np.zeros(num_experts, dtype=np.float32)
    for e in prange(num_experts):
        raw = weights_3d[e]
        max_v = 0.0
        for r in range(rows):
            for c in range(cols):
                av = abs(raw[r, c])
                if av > max_v: max_v = av
        scale = max_v / 3.5
        if scale == 0: scale = 1.0
        scales_out[e] = np.float32(scale)
        iscale = 1.0 / scale
        for r in range(rows):
            for cg in range(cols // 8):
                v32 = np.uint32(0)
                for i in range(8):
                    idx = int(round((raw[r, cg * 8 + i] * iscale) + 3.5))
                    if idx < 0: idx = 0
                    if idx > 7: idx = 7
                    # Pack as [w0, w1, ..., w7] where w0 is at bits 21-23
                    v32 |= (np.uint32(idx) << np.uint32(21 - (i * 3)))
                base = cg * 3
                packed_out[e, r, base] = v32 & 0xFF
                packed_out[e, r, base + 1] = (v32 >> 8) & 0xFF
                packed_out[e, r, base + 2] = (v32 >> 16) & 0xFF
    return packed_out, scales_out

def convert():
    input_path = "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
    output_path = "model.bin"
    manifest_path = "model.json"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    reader = gguf.GGUFReader(input_path)
    tensor_map = {tensor.name: tensor for tensor in reader.tensors}
    
    def fetch_tensor(name):
        if name in tensor_map:
            t = tensor_map[name]
            data = gguf.quants.dequantize(t.data, t.tensor_type) if t.tensor_type != 0 else t.data.view(np.float32)
            return data.reshape(t.shape[::-1]).astype(np.float32)
        if f"{name}.weight" in tensor_map:
            return fetch_tensor(f"{name}.weight")
        return None

    manifest = {"tensors": {}}
    offset = 0

    with open(output_path, "wb") as f:
        def write_tensor(name, data):
            nonlocal offset
            f.write(data.tobytes())
            manifest["tensors"][name] = {"offset": offset, "size": data.nbytes}
            offset += data.nbytes

        all_cpu_scales = []
        
        # 1. Shared Experts (3-bit)
        expert_pitch = (HIDDEN_DIM * 3 // 8 + 31) & ~31
        down_pitch = (FFN_INTERMEDIATE * 3 // 8 + 31) & ~31
        
        for s in ["ffn_gate_shexp", "ffn_up_shexp"]:
            data = []
            for l in range(NUM_LAYERS):
                w = fetch_tensor(f"blk.{l}.{s}")
                p, sc = pack_3bit(w.reshape(1, w.shape[0], w.shape[1]), expert_pitch)
                data.append(p)
                all_cpu_scales.extend(sc.tolist())
            write_tensor(s, np.concatenate(data))
            
        data = []
        for l in range(NUM_LAYERS):
            w = fetch_tensor(f"blk.{l}.ffn_down_shexp")
            p, sc = pack_3bit(w.reshape(1, w.shape[0], w.shape[1]), down_pitch)
            data.append(p)
            all_cpu_scales.extend(sc.tolist())
        write_tensor("ffn_down_shexp", np.concatenate(data))

        # 2. Routed Experts (3-bit)
        for s in ["ffn_gate_exps", "ffn_up_exps"]:
            data = []
            for l in range(NUM_LAYERS):
                w = fetch_tensor(f"blk.{l}.{s}") # (experts, intermediate, hidden)
                p, sc = pack_3bit(w, expert_pitch)
                data.append(p)
                all_cpu_scales.extend(sc.tolist())
            write_tensor(s, np.concatenate(data))
            
        data = []
        for l in range(NUM_LAYERS):
            w = fetch_tensor(f"blk.{l}.ffn_down_exps")
            p, sc = pack_3bit(w, down_pitch)
            data.append(p)
            all_cpu_scales.extend(sc.tolist())
        write_tensor("ffn_down_exps", np.concatenate(data))

        # 3. Embeddings (FP32)
        write_tensor("token_embd", fetch_tensor("token_embd"))

        # 4. Attention/SSM weights (Q4)
        for l in tqdm(range(NUM_LAYERS), desc="Processing Layers"):
            if l % 4 != 3: # SSM
                qkv = fetch_tensor(f"blk.{l}.attn_qkv")
                write_tensor(f"blk.{l}.attn_qkv", pack_q4(qkv))
                gate = fetch_tensor(f"blk.{l}.attn_gate")
                write_tensor(f"blk.{l}.attn_gate", pack_q4(gate))
                out = fetch_tensor(f"blk.{l}.ssm_out")
                write_tensor(f"blk.{l}.ssm_out", pack_q4(out))
            else: # Attention
                for s in ["attn_q", "attn_k", "attn_v", "attn_output"]:
                    w = fetch_tensor(f"blk.{l}.{s}")
                    write_tensor(f"blk.{l}.{s}", pack_q4(w))

        # 5. Router weights (FP32)
        data = []
        for l in range(NUM_LAYERS):
            data.append(fetch_tensor(f"blk.{l}.ffn_gate_inp"))
        write_tensor("router_weights", np.concatenate(data))

        # 6. LM Head (FP32)
        write_tensor("output", fetch_tensor("output"))

        # 7. Norms (FP32)
        data_attn = []
        data_ffn = []
        for l in range(NUM_LAYERS):
            data_attn.append(fetch_tensor(f"blk.{l}.attn_norm"))
            data_ffn.append(fetch_tensor(f"blk.{l}.post_attention_norm"))
        write_tensor("attn_norms", np.concatenate(data_attn))
        write_tensor("ffn_norms", np.concatenate(data_ffn))
        write_tensor("output_norm", fetch_tensor("output_norm"))

        # 8. Scales (FP32)
        write_tensor("scales", np.array(all_cpu_scales, dtype=np.float32))

        # 9. SSM Params (FP32)
        for l in range(NUM_LAYERS):
            if l % 4 != 3:
                for s in ["ssm_a", "ssm_alpha", "ssm_beta", "ssm_dt", "ssm_conv1d", "ssm_norm"]:
                    write_tensor(f"blk.{l}.{s}", fetch_tensor(f"blk.{l}.{s}"))

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"SUCCESS: {output_path} and {manifest_path} created.")

if __name__ == "__main__":
    convert()
