// M4 — the FUSED BSSN RHS as a JAX-FFI custom call (the thesis target).
//
// One kernel, 1 thread/point: computes the 138 derivatives from L2-cached global reads into
// register scalars (M2a-style edge-clamp), runs the 1c algebra CSE on them, writes the 24 RHS
// outputs. No derivative HBM round-trip (the 2.8 GB wall-2 write is gone). The generated kernel
// is in _bssn_fused_kernel.cuh (gen_fused.py). Validated vs BSSNSolver.rhs (verbatim) to round-off;
// the gate is wall-clock vs the 31.27 ms verbatim-XLA baseline.
//
// Build: build_fused.sh (regenerate the header first). -Xptxas -v reports the spill (the seam).

#include <cstdint>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"
#include "_bssn_fused_kernel.cuh"

namespace ffi = xla::ffi;

static ffi::Error FusedImpl(cudaStream_t stream,
                            ffi::Buffer<ffi::F64> F,     // (NF, Sx, Sy, Sz)
                            ffi::Buffer<ffi::F64> S,     // (NSCAL,)
                            ffi::ResultBuffer<ffi::F64> OUT) {  // (NOUT, Sx, Sy, Sz)
  const auto dims = F.dimensions();
  if (dims.size() != 4) return ffi::Error::Internal("fused: F must be 4-D (NF,Sx,Sy,Sz)");
  if (dims[0] != NF) return ffi::Error::Internal("fused: F[0] != NF");
  if (S.element_count() != NSCAL) return ffi::Error::Internal("fused: S must have NSCAL scalars");
  const int Sx = (int)dims[1], Sy = (int)dims[2], Sz = (int)dims[3];
  const long long N = (long long)Sx * Sy * Sz;

  // block size defaults to 128 (Cheng v10); BSSN_FUSED_BLOCK env overrides for a sweep w/o rebuild.
  int block = 128;
  if (const char* e = std::getenv("BSSN_FUSED_BLOCK")) { int b = atoi(e); if (b > 0) block = b; }
  // Phase-4 dummy-ALU probe: BSSN_FUSED_DUMMY = total fp64 FMAs to inject (0 = exact M4). The
  // kernel runs 4 FMAs/iter (4-way ILP), so iters = DUMMY/4. Runtime knob -> sweep with ONE build.
  int dummy_iters = 0;
  if (const char* e = std::getenv("BSSN_FUSED_DUMMY")) { int d = atoi(e); if (d > 0) dummy_iters = d / 4; }
  const unsigned grid = (unsigned)((N + block - 1) / block);
  // Phase-4.A: per-thread SMEM scratch for the NSMEM staged temps (NSMEM=0 -> off, 0 bytes).
  const size_t smem_bytes = (size_t)NSMEM * (size_t)block * sizeof(double);
  if (smem_bytes > 48 * 1024) {   // opt-in to >48KB dynamic SMEM (Hopper allows up to ~228KB)
    cudaFuncSetAttribute(bssn_rhs_fused, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
  }
  bssn_rhs_fused<<<grid, block, smem_bytes, stream>>>(F.typed_data(), S.typed_data(),
                                                      OUT->typed_data(), Sx, Sy, Sz, dummy_iters);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(std::string("fused launch failed: ") + cudaGetErrorString(err));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Fused, FusedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()      // F
        .Arg<ffi::Buffer<ffi::F64>>()      // S (scalars)
        .Ret<ffi::Buffer<ffi::F64>>());    // OUT
