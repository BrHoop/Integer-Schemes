// M2b — 2.5D SMEM-streaming derivative stage (the wall-A geometry) for Step 3.2e.
//
// Streams each of the 24 fields through an on-chip SMEM plane-window marched in z, so the
// derivative halo is served from SMEM (loaded once per plane) instead of the redundant HBM
// neighbour reads of the M2a baseline. Rolling circular buffer of 2R+1 z-planes (R = FD_REACH
// = 3), each a (TILE+2R)² haloed plane → z-reuse: each source plane is loaded from HBM exactly
// once across the march. Computes the same 138 derivatives as M2a (per-field CSR from
// _deriv_kernel.cuh); validated bit-vs M2a/derivative_bundle to round-off. Still writes derivs to
// HBM (standalone stage) — the wall-clock win vs M2a is the reduced halo READ traffic (MEM%↓);
// eliminating the deriv WRITE is the fused M4.
//
// Build: build_deriv.sh (builds this + deriv_2p5d.cu). TILE=16 → 27 KB static SMEM (< 48 KB).

#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"
#include "_deriv_kernel.cuh"

namespace ffi = xla::ffi;

#define TILE 16
#define RAD  FD_REACH            // 3 (6th-order FD)
#define WIN  (2 * RAD + 1)       // 7 z-planes resident
#define HP   (TILE + 2 * RAD)    // 22 haloed plane side

__device__ __forceinline__ int clampi(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }
__device__ __forceinline__ int zslot(int L) { return ((L % WIN) + WIN) % WIN; }   // circular slot

// Cooperatively fill SMEM slot `slot` with field-f plane (logical z = L, clamped), HP×HP haloed.
__device__ void load_plane(double smem[WIN][HP][HP], const double* __restrict__ F, int slot, int L,
                           int tx0, int ty0, int Sx, int Sy, int Sz, int lx, int ly) {
  const int zc = clampi(L, Sz - 1);
  for (int t = ly * TILE + lx; t < HP * HP; t += TILE * TILE) {
    const int ix = t % HP, iy = t / HP;
    const int sx = clampi(tx0 - RAD + ix, Sx - 1);
    const int sy = clampi(ty0 - RAD + iy, Sy - 1);
    smem[slot][iy][ix] = F[((std::int64_t)sx * Sy + sy) * Sz + zc];
  }
}

__global__ void deriv_smem_kernel(const double* __restrict__ fields,
                                  double* __restrict__ out,
                                  int Sx, int Sy, int Sz, std::int64_t N,
                                  double dx, double dy, double dz) {
  __shared__ double smem[WIN][HP][HP];           // 7·22·22·8 = 27 KB

  const int tx0 = blockIdx.x * TILE, ty0 = blockIdx.y * TILE;
  const int f = blockIdx.z;                      // one block per (x-tile, y-tile, FIELD) — fields
  const int lx = threadIdx.x, ly = threadIdx.y;  // are independent, so parallelise them across the
  const int gx = tx0 + lx, gy = ty0 + ly;        // grid (24× more blocks → fills the SMs) rather
  const int ix0 = lx + RAD, iy0 = ly + RAD;      // than a serial in-kernel field loop (was 81
  const double invh[3] = {1.0 / dx, 1.0 / dy, 1.0 / dz};  // long-lived blocks → SM-starved).
  const bool active = (gx < Sx) && (gy < Sy);

  {
    const double* __restrict__ F = fields + (std::int64_t)f * N;
    for (int m = 0; m < WIN; ++m) load_plane(smem, F, zslot(0 - RAD + m), 0 - RAD + m,
                                             tx0, ty0, Sx, Sy, Sz, lx, ly);
    __syncthreads();

    for (int k = 0; k < Sz; ++k) {
      const int cs = zslot(k);                   // centre plane slot

      if (active) {
        const std::int64_t pq = ((std::int64_t)gx * Sy + gy) * Sz + k;
        for (int s = D_FSTART[f]; s < D_FSTART[f + 1]; ++s) {
          const int d = D_OUTIDX[s], kind = D_KIND_S[s], i = D_AXI_S[s], j = D_AXJ_S[s];
          double val;

          if (kind == KIND_GRAD1) {
            double a = 0.0;
            for (int t = 0; t < WIN; ++t) {
              if (i == 0)      a += DC1[t] * smem[cs][iy0][ix0 + t - RAD];
              else if (i == 1) a += DC1[t] * smem[cs][iy0 + t - RAD][ix0];
              else             a += DC1[t] * smem[zslot(k - RAD + t)][iy0][ix0];
            }
            val = a * invh[i];

          } else if (kind == KIND_GRAD2_DIAG) {
            double a = 0.0;
            for (int t = 0; t < WIN; ++t) {
              if (i == 0)      a += DC2[t] * smem[cs][iy0][ix0 + t - RAD];
              else if (i == 1) a += DC2[t] * smem[cs][iy0 + t - RAD][ix0];
              else             a += DC2[t] * smem[zslot(k - RAD + t)][iy0][ix0];
            }
            val = a * invh[i] * invh[i];

          } else {  // GRAD2_MIXED: d1 along i, then d1 along j (nested; matches M2a/bundle)
            double acc = 0.0;
            for (int b = 0; b < WIN; ++b) {
              double inner = 0.0;
              for (int aa = 0; aa < WIN; ++aa) {
                if (j == 1) {            // (i,j)=(0,1): both in-plane, centre slot
                  inner += DC1[aa] * smem[cs][iy0 + b - RAD][ix0 + aa - RAD];
                } else if (i == 0) {     // (0,2): inner d1_x on z-plane b, outer d1_z
                  inner += DC1[aa] * smem[zslot(k - RAD + b)][iy0][ix0 + aa - RAD];
                } else {                 // (1,2): inner d1_y on z-plane b, outer d1_z
                  inner += DC1[aa] * smem[zslot(k - RAD + b)][iy0 + aa - RAD][ix0];
                }
              }
              acc += DC1[b] * inner;
            }
            val = acc * invh[i] * invh[j];
          }
          out[(std::int64_t)d * N + pq] = val;
        }
      }

      __syncthreads();                           // all reads of the window done
      if (k + 1 < Sz)                            // roll: load plane k+1+RAD into the evicted slot
        load_plane(smem, F, zslot(k + 1 + RAD), k + 1 + RAD, tx0, ty0, Sx, Sy, Sz, lx, ly);
      __syncthreads();
    }
  }
}

static ffi::Error DerivSmemImpl(cudaStream_t stream,
                                ffi::Buffer<ffi::F64> fields,    // (N_FIELDS, Sx, Sy, Sz)
                                ffi::ResultBuffer<ffi::F64> out, // (N_DERIV, Sx, Sy, Sz)
                                double dx, double dy, double dz) {
  const auto dims = fields.dimensions();
  if (dims.size() != 4) return ffi::Error::Internal("deriv_smem: fields must be 4-D");
  if (dims[0] != N_FIELDS) return ffi::Error::Internal("deriv_smem: fields[0] != N_FIELDS");
  const int Sx = (int)dims[1], Sy = (int)dims[2], Sz = (int)dims[3];
  const std::int64_t N = (std::int64_t)Sx * Sy * Sz;

  const dim3 block(TILE, TILE);
  const dim3 grid((Sx + TILE - 1) / TILE, (Sy + TILE - 1) / TILE, N_FIELDS);  // z = field
  deriv_smem_kernel<<<grid, block, 0, stream>>>(fields.typed_data(), out->typed_data(),
                                                Sx, Sy, Sz, N, dx, dy, dz);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(std::string("deriv_smem launch failed: ") + cudaGetErrorString(err));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DerivSmem, DerivSmemImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Attr<double>("dx").Attr<double>("dy").Attr<double>("dz"));
