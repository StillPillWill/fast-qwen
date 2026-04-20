cmake_minimum_required(VERSION 3.18)
project(moe_inference_engine LANGUAGES CXX CUDA)

# ─── Compiler requirements ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# ─── CUDA architecture: Pascal GP104 = sm_61 ─────────────────────────────────
set(CMAKE_CUDA_ARCHITECTURES 61)

# ─── C++ compiler flags ───────────────────────────────────────────────────────
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /arch:AVX2 /fp:fast /GL /GS-")
    set(CMAKE_CXX_FLAGS_DEBUG   "/Od /Zi /arch:AVX2")
    # Enable large-page privilege for linker
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
else()
    # GCC / Clang on Linux
    set(CMAKE_CXX_FLAGS         "-march=broadwell -mavx2 -mfma -O3 -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG   "-march=broadwell -mavx2 -mfma -O0 -g3")
    set(CMAKE_CXX_FLAGS_RELEASE "-march=broadwell -mavx2 -mfma -O3 -ffast-math -flto")
endif()

# ─── CUDA compiler flags (FP32 only — no fp16 pragma on Pascal) ──────────────
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
     -allow-unsupported-compiler \
     --generate-code arch=compute_61,code=sm_61 \
     -O3 \
     --use_fast_math \
     --maxrregcount=64 \
     -lineinfo")

# ─── Source files ─────────────────────────────────────────────────────────────
set(CXX_SOURCES
    main.cpp
    src/cpu/quip_unpack_avx2.cpp
)

set(CUDA_SOURCES
    src/cuda/fused_attention_rotor.cu
    src/cuda/moe_router_pinned.cu
    src/cuda/lm_head_gemv.cu
    src/cuda/ssm_kernels.cu
)

# ─── Find packages ────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# On Linux, find libnuma
if (UNIX)
    find_library(NUMA_LIB numa)
endif()

# ─── Target definition ────────────────────────────────────────────────────────
add_executable(engine ${CXX_SOURCES} ${CUDA_SOURCES})

target_include_directories(engine PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(engine
    PRIVATE
    CUDA::cudart
    CUDA::curand
    Threads::Threads
    $<$<BOOL:${NUMA_LIB}>:${NUMA_LIB}>
)

# Windows: link against kernel32 for VirtualAllocExNuma
if (WIN32)
    target_link_libraries(engine PRIVATE kernel32)
endif()

# ─── Separable compilation (required for __constant__ across TUs) ────────────
set_target_properties(engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ─── Preprocessor defines ─────────────────────────────────────────────────────
target_compile_definitions(engine PRIVATE
    $<$<CONFIG:Debug>:ENGINE_DEBUG=1>
    $<$<BOOL:${WIN32}>:WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00>
)

# ─── Install ──────────────────────────────────────────────────────────────────
install(TARGETS engine DESTINATION bin)

# ─── Build summary ────────────────────────────────────────────────────────────
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "  MoE Inference Engine Build Configuration")
message(STATUS "  CUDA Arch:     sm_61 (Pascal GP104)")
message(STATUS "  CXX Arch:      -march=broadwell -mavx2 -mfma")
message(STATUS "  Float policy:  FP32 everywhere (no FP16)")
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
