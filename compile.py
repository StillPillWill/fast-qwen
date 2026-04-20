import subprocess, os, sys

# 1. Points to the Professional build script
VCVARS = r'C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat'

def run(cmd):
    # 2. Forces the environment to use the v142 (Visual Studio 2019) tools
    # Note: If this fails with "directory not found", check your MSVC folder 
    # to ensure the folder name is exactly 14.29
    full_cmd = f'cmd /c ""{VCVARS}" x64 -vcvars_ver=14.29 && {cmd}"'
    print(f"Running: {cmd}")
    res = subprocess.run(full_cmd, shell=True)
    if res.returncode != 0:
        print(f"Error: {cmd} failed with {res.returncode}")
        sys.exit(1)

# Configuration
CL = "cl.exe" 
NVCC = "nvcc.exe"
INC = f"-I. -Iinclude"

# MSVC Flags - C++17 is required for these kernels
CL_FLAGS = "/nologo /EHsc /std:c++17 /W3 /D_ALLOW_MSC_VER_MISMATCH /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS"

# NVCC Flags - Optimized for your GTX 1080 (Pascal / sm_61)
NVCC_FLAGS = f"-O2 -m 64 -arch=sm_61 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math {INC} -Xcompiler \"{CL_FLAGS}\""
SRCS = [
    "src/cuda/fused_attention_rotor.cu",
    "src/cuda/fused_q4_to_fp32_gemv.cu",
    "src/cuda/moe_router_pinned.cu",
    "src/cuda/lm_head_gemv.cu",
    "src/cuda/ssm_kernels.cu",
    "src/cpu/quip_unpack_avx2.cpp",
    "main.cpp"
]

OBJS = []
for src in SRCS:
    obj = src.replace("/", "_").replace(".cu", ".obj").replace(".cpp", ".obj")
    if src.endswith(".cu") or src.endswith(".cpp"):
        run(f"{NVCC} {NVCC_FLAGS} -c {src} -o {obj}")
    else:
        run(f"{CL} {CL_FLAGS} {INC} /c {src} /Fo{obj}")
    OBJS.append(obj)

# Link everything into the final executable
run(f"{NVCC} -m 64 -arch=sm_61 -allow-unsupported-compiler {' '.join(OBJS)} -o engine.exe")

print("\nSUCCESS: Compilation finished using 64-bit environment and nvcc.")