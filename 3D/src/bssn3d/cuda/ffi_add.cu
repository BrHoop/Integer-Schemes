// M1 — JAX-FFI "hello world": the FIRST Foreign Function Interface in the repo.
//
// A trivial element-wise add (c = a + b) exposed to JAX as an XLA custom call via the typed XLA
// FFI. Its only job is to de-risk the plumbing — build → register → `ffi_call` → run on the GPU
// — in isolation, before Step 3.2e's real 2.5D derivative kernel rides the same path. If this
// returns a+b on the device, the FFI mechanics are proven.
//
// XLA_FFI_DEFINE_HANDLER_SYMBOL emits an `extern "C"` symbol `Add` (XLA_FFI_Error*(call_frame)),
// which `ctypes`/`jax.ffi.pycapsule` looks up by name (see ../ffi_add.py).
//
// Build: ../cuda/build_ffi.sh  (nvcc -arch=sm_90a for Hopper H100/H200; sm_80 A100; sm_100a B200).

#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

__global__ void add_kernel(const double* a, const double* b, double* c, std::int64_t n) {
  std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

static ffi::Error AddImpl(cudaStream_t stream,
                          ffi::Buffer<ffi::F64> a,
                          ffi::Buffer<ffi::F64> b,
                          ffi::ResultBuffer<ffi::F64> c) {
  const std::int64_t n = a.element_count();
  if (b.element_count() != n || c->element_count() != n) {
    return ffi::Error::Internal("ffi_add: input/output element counts differ");
  }
  const int block = 256;
  const std::int64_t grid = (n + block - 1) / block;
  add_kernel<<<grid, block, 0, stream>>>(a.typed_data(), b.typed_data(),
                                         c->typed_data(), n);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return ffi::Error::Internal(std::string("ffi_add launch failed: ") +
                                cudaGetErrorString(err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Add, AddImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // the CUDA stream XLA hands us
        .Arg<ffi::Buffer<ffi::F64>>()              // a
        .Arg<ffi::Buffer<ffi::F64>>()              // b
        .Ret<ffi::Buffer<ffi::F64>>());            // c = a + b
