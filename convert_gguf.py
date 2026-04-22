import numpy as np
import gguf
import os
import json
import sys
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
OUT_INNER = 8192

def pack_q4(weights):
    flat_w = weights.flatten()
    pad_len = (32 - len(flat_w) % 32) % 32
    if pad_len > 0:
        flat_w = np.pad(flat_w, (0, pad_len))
    
    blocks = flat_w.reshape(-1, 32)
    max_v = np.max(np.abs(blocks), axis=1)
    scale = max_v / 7.5
    scale[scale == 0.0] = 1.0
    scale = scale.astype(np.float32)
    
    packed = np.zeros((len(blocks), 64), dtype=np.uint8)
    packed[:, 0:4] = scale.view(np.uint8).reshape(-1, 4)
    
    iscale = 1.0 / scale
    quantized = np.round(blocks * iscale[:, None] + 8.0).astype(np.int32)
    quantized = np.clip(quantized, 0, 15).astype(np.uint8)
    packed[:, 4:36] = quantized
    
    return packed.flatten()

@njit(nogil=True, fastmath=True, parallel=True)
def pack_3bit_all_experts(raw_all, pitch):
    # raw_all: (experts, rows, cols)
    num_experts, rows, cols = raw_all.shape
    packed_out = np.zeros((num_experts, rows, pitch), dtype=np.uint8)
    scales = np.zeros(num_experts, dtype=np.float32)
    for e in prange(num_experts):
        raw = raw_all[e]
        max_v = 0.0
        for r in range(rows):
            for c in range(cols):
                av = abs(raw[r, c])
                if av > max_v: max_v = av
        scale = max_v / 3.5
        if scale == 0: scale = 1.0
        scales[e] = np.float32(scale)
        iscale = 1.0 / scale
        for r in range(rows):
            for cg in range(cols // 8):
                v32 = np.uint32(0)
                for i in range(8):
                    idx = int(round((raw[r, cg * 8 + i] * iscale) + 3.5))
                    v32 |= (np.uint32(max(0, min(7, idx))) << np.uint32(21 - (i * 3)))
                base = cg * 3
                packed_out[e, r, base] = v32 & 0xFF
                packed_out[e, r, base + 1] = (v32 >> 8) & 0xFF
                packed_out[e, r, base + 2] = (v32 >> 16) & 0xFF
    return packed_out, scales

def convert():
    input_path = "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
    output_path = "model.bin"
    manifest_path = "model.json"
    
    print(f"Starting conversion of {input_path} (Efficient mode)...", flush=True)
    reader = gguf.GGUFReader(input_path)
    tensor_map = {tensor.name: tensor for tensor in reader.tensors}
    
    def fetch_tensor(name):
        if name in tensor_map:
            t = tensor_map[name]
            data = gguf.quants.dequantize(t.data, t.tensor_type) if t.tensor_type != 0 else t.data.view(np.float32)
            return data.reshape(t.shape[::-1]).astype(np.float32)
        if not name.endswith(".weight") and f"{name}.weight" in tensor_map:
            return fetch_tensor(f"{name}.weight")
        return None

    manifest = {"tensors": {}}
    offset = 0

    with open(output_path, "wb") as f:
        def write_tensor(name, data):
            nonlocal offset
            if data is None: return
            buf = data.tobytes()
            f.write(buf)
            manifest["tensors"][name] = {"offset": offset, "size": len(buf)}
            offset += len(buf)
            print(f"Wrote {name} ({len(buf)} bytes)", flush=True)

        all_cpu_scales = []
        expert_pitch = (HIDDEN_DIM * 3 // 8 + 31) & ~31
        down_pitch = (FFN_INTERMEDIATE * 3 // 8 + 31) & ~31
        
        # 1. Shared Experts (Blob per type)
        for s in ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"]:
            pitch = expert_pitch if "down" not in s else down_pitch
            blob = []
            for l in range(NUM_LAYERS):
                w = fetch_tensor(f"blk.{l}.{s}")
                if w is not None:
                    packed, sc = pack_3bit_all_experts(w.reshape(1, w.shape[0], w.shape[1]), pitch)
                    blob.append(packed)
                    all_cpu_scales.extend(sc.tolist())
            if blob: write_tensor(s, np.concatenate(blob))

        # 2. Routed Experts (Blob per type)
        for s in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]:
            pitch = expert_pitch if "down" not in s else down_pitch
            blob = []
            for l in range(NUM_LAYERS):
                w = fetch_tensor(f"blk.{l}.{s}")
                if w is not None:
                    packed, sc = pack_3bit_all_experts(w, pitch)
                    blob.append(packed)
                    all_cpu_scales.extend(sc.tolist())
            if blob: write_tensor(s, np.concatenate(blob))

        write_tensor("token_embd", fetch_tensor("token_embd"))

        print("Processing Q4 layers...", flush=True)
        for l in tqdm(range(NUM_LAYERS)):
            if l % 4 != 3:
                for s in ["attn_qkv", "attn_gate", "ssm_out"]:
                    w = fetch_tensor(f"blk.{l}.{s}")
                    if w is not None: write_tensor(f"blk.{l}.{s}", pack_q4(w))
            else:
                for s in ["attn_q", "attn_k", "attn_v", "attn_output"]:
                    w = fetch_tensor(f"blk.{l}.{s}")
                    if w is not None: write_tensor(f"blk.{l}.{s}", pack_q4(w))

        # 3. Router and Norms
        router_blob = []
        for l in range(NUM_LAYERS):
            w = fetch_tensor(f"blk.{l}.ffn_gate_inp")
            if w is not None: router_blob.append(w)
        if router_blob: write_tensor("router_weights", np.concatenate(router_blob))

        write_tensor("output", fetch_tensor("output"))

        attn_norms = []
        ffn_norms = []
        for l in range(NUM_LAYERS):
            wa = fetch_tensor(f"blk.{l}.attn_norm")
            if wa is not None: attn_norms.append(wa)
            wf = fetch_tensor(f"blk.{l}.post_attention_norm")
            if wf is not None: ffn_norms.append(wf)
        
        if attn_norms: write_tensor("attn_norms", np.concatenate(attn_norms))
        if ffn_norms: write_tensor("ffn_norms", np.concatenate(ffn_norms))
        write_tensor("output_norm", fetch_tensor("output_norm"))
        if all_cpu_scales: write_tensor("scales", np.array(all_cpu_scales, dtype=np.float32))

        # 4. SSM Params
        for l in range(NUM_LAYERS):
            if l % 4 != 3:
                for s in ["ssm_a", "ssm_alpha", "ssm_beta", "ssm_dt", "ssm_conv1d", "ssm_norm"]:
                    write_tensor(f"blk.{l}.{s}", fetch_tensor(f"blk.{l}.{s}"))

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"SUCCESS: {output_path} created.", flush=True)

if __name__ == "__main__":
    convert()
