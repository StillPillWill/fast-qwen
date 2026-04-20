import numpy as np
import gguf
import os
from tqdm import tqdm
from numba import njit, prange

HIDDEN_DIM = 2048
NUM_LAYERS = 40
NUM_EXPERTS = 256
FFN_INTERMEDIATE = 512
VOCAB_SIZE = 248320

# GGUF SHAPES (Verified from inspect_model.py)
Q_HEADS = 16
KV_HEADS = 2
Q_HEAD_DIM = 512
KV_HEAD_DIM = 256
OUT_INNER = 4096 
INNER_SIZE = 4096
SSM_STATE = 128

ROT_MAT_128 = np.eye(128, dtype=np.float32)

@njit(nogil=True, fastmath=True, parallel=True)
def pack_3bit_ultra(weights_3d, pitch, R):
    num_experts, rows, cols = weights_3d.shape
    packed_out = np.zeros((num_experts, rows, pitch), dtype=np.uint8)
    scales_out = np.zeros(num_experts, dtype=np.float32)
    for e in prange(num_experts):
        raw = weights_3d[e]; rot = raw 
        max_v = 0.0
        for r in range(rows):
            for c in range(cols):
                av = abs(rot[r, c]); 
                if av > max_v: max_v = av
        scale = max_v / 3.5
        if scale == 0: scale = 1.0
        scales_out[e] = np.float32(scale); iscale = 1.0 / scale
        for r in range(rows):
            for cg in range(cols // 8):
                v32 = np.uint32(0)
                for i in range(8):
                    idx = int(round((rot[r, cg * 8 + i] * iscale) + 3.5))
                    if idx < 0: idx = 0; 
                    elif idx > 7: idx = 7
                    v32 |= (np.uint32(idx) << np.uint32(21 - (i * 3)))
                base = cg * 3; packed_out[e, r, base] = v32 & 0xFF; packed_out[e, r, base + 1] = (v32 >> 8) & 0xFF; packed_out[e, r, base + 2] = (v32 >> 16) & 0xFF
    return packed_out, scales_out

def convert():
    input_path = "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"; output_path = "model.bin"
    reader = gguf.GGUFReader(input_path); tensor_map = {tensor.name: tensor for tensor in reader.tensors}
    def fetch_tensor(name):
        def extract(t): return gguf.quants.dequantize(t.data, t.tensor_type) if t.tensor_type != 0 else t.data.view(np.float32)
        if name in tensor_map: return extract(tensor_map[name]).reshape(tensor_map[name].shape[::-1])
        if f"{name}.weight" in tensor_map: return extract(tensor_map[f"{name}.weight"]).reshape(tensor_map[f"{name}.weight"].shape[::-1])
        return None

    with open(output_path, "wb") as f:
        expert_pitch = (HIDDEN_DIM * 3 // 8 + 31) & ~31; down_pitch = (FFN_INTERMEDIATE * 3 // 8 + 31) & ~31; all_cpu_scales = []
        
        # Phase 1-2: Experts (Keep 3-bit)
        for s in ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"]:
            ps = expert_pitch if "down" not in s else down_pitch
            for l in range(NUM_LAYERS):
                w = fetch_tensor(f"blk.{l}.{s}"); p, sc = pack_3bit_ultra(w.reshape(1, w.shape[0], w.shape[1]), ps, ROT_MAT_128); f.write(p.tobytes()); all_cpu_scales.extend(sc.tolist())
        for s in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]:
            ps = expert_pitch if "down" not in s else down_pitch
            for l in range(NUM_LAYERS):
                wa = fetch_tensor(f"blk.{l}.{s}"); wa = np.ascontiguousarray(np.transpose(wa, (2, 0, 1))); pb, scs = pack_3bit_ultra(wa, ps, ROT_MAT_128); f.write(pb.tobytes()); all_cpu_scales.extend(scs.tolist())
        
        # Phase 3-10: Rest (Use FP16 to save space)
        f.write(fetch_tensor("token_embd").astype(np.float16).tobytes())
        for l in tqdm(range(NUM_LAYERS)):
            if l % 4 != 3:
                f.write(fetch_tensor(f"blk.{l}.attn_qkv").astype(np.float16).tobytes())
                f.write(fetch_tensor(f"blk.{l}.attn_gate").astype(np.float16).tobytes())
                f.write(fetch_tensor(f"blk.{l}.ssm_out").astype(np.float16).tobytes())
            else:
                f.write(fetch_tensor(f"blk.{l}.attn_q").astype(np.float16).tobytes())
                f.write(fetch_tensor(f"blk.{l}.attn_k").astype(np.float16).tobytes())
                f.write(fetch_tensor(f"blk.{l}.attn_v").astype(np.float16).tobytes())
                f.write(fetch_tensor(f"blk.{l}.attn_output").astype(np.float16).tobytes())
        for l in range(NUM_LAYERS): f.write(fetch_tensor(f"blk.{l}.ffn_gate_inp").astype(np.float16).tobytes())
        f.write(fetch_tensor("output").astype(np.float16).tobytes())
        for l in range(NUM_LAYERS):
            f.write(fetch_tensor(f"blk.{l}.attn_norm").astype(np.float16).tobytes())
            f.write(fetch_tensor(f"blk.{l}.post_attention_norm").astype(np.float16).tobytes())
        f.write(fetch_tensor("output_norm").astype(np.float16).tobytes())
        f.write(np.array(all_cpu_scales, dtype=np.float32).tobytes())
        for l in range(NUM_LAYERS):
            if l % 4 != 3:
                f.write(fetch_tensor(f"blk.{l}.ssm_a").astype(np.float16).tobytes())
                f.write(fetch_tensor(f"blk.{l}.ssm_alpha").astype(np.float16).tobytes())
                f.write(fetch_tensor(f"blk.{l}.ssm_beta").astype(np.float16).tobytes())
                f.write(fetch_tensor(f"blk.{l}.ssm_dt").astype(np.float16).tobytes())
                f.write(fetch_tensor(f"blk.{l}.ssm_conv1d").astype(np.float16).tobytes())
                f.write(fetch_tensor(f"blk.{l}.ssm_norm").astype(np.float16).tobytes())
            else:
                f.write(np.zeros((32 + 32*2048*2 + 32 + 4*8192 + 128), dtype=np.float16).tobytes())

    print(f"SUCCESS: {output_path} created.")

if __name__ == "__main__": convert()
