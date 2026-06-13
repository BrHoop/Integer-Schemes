# Pallas-Ozaki on GPU — status, Triton 0.9.2 constraints, and the open compile-time blocker

Working notes from the JAX 0.9.2 migration + first-real-GPU bring-up of the
`pallas_ozaki` (and `pallas_fp`) Pallas/Triton kernels. **Parked** — math is
correct and it compiles on the H200, but compile time is pathological (see §4).

Last updated: 2026-06-08.

---

## 1. TL;DR / current state

- **JAX pinned to 0.9.2** (all four `pyproject.toml`). 0.8.1 was a dead end: its
  Triton has no `dynamic_slice`, and the kernel had never actually compiled on a
  GPU on *any* version — every prior "pass" was CPU **interpret** mode, which
  skips Triton lowering entirely.
- **`pallas_ozaki`** now: int8 GEMM hardcoded ON, **BFP48** mantissa input,
  residues computed purely in int32 from limbs, written for the 0.9.x Triton
  primitive set (§2).
- **Validated (CPU interpret):** `pallas_ozaki` vs `fused_floating_point` =
  **1.665e-10**, `pallas_fp` = **4.5e-13**. **10/10** pallas integration tests
  pass on 0.9.2. Interpret mode does NOT exercise Triton lowering — it only
  proves the math.
- **GPU:** the kernel *compiles and runs correctly* but `trunc=2` (a reduced
  slice: `k=6`, KO off) takes **~1600 s** to compile. The full kernel (`k=8`,
  KO on, + Garner) would be hours. **This is the blocker; the actual GPU
  speed/Amdahl numbers are still unmeasured.**

---

## 2. Triton 0.9.2 supported-primitive set (audited from source)

`jax/_src/pallas/triton/lowering.py` — the binding constraint for any Pallas
kernel. Verified by reading the lowering rules directly.

**Supported (value ops):**
`dot_general` (**2-D ONLY — no batch dims**; int8/fp16/fp32, **not fp64**),
`reduce_sum/max/min`, `concatenate` (**2-ARG ONLY**), `split` (**EQUAL,
power-of-2 parts ONLY**), `reshape`, `squeeze`, `transpose`, `broadcast_in_dim`,
`select_n`, `convert_element_type`, `rem`, `div`, `and`, `shift_right_*`,
`integer_pow`, `iota`, `exp2`/`exp`/`pow`/`log` (**not `log2`**), `abs`,
`floor`, `sign`, comparisons (`ge/gt/lt/eq`), `add/mul/sub`,
`scan_p`, `while_p`, `cond_p`.
Ref indexing via `get_p`/`load_p`, **including dynamic-start slices**
(`Slice.is_dynamic_start`) — so loop-counter ref indexing works.

**NOT supported:** `slice_p`, `dynamic_slice_p`, `gather_p`, `scatter_p` /
`scatter_add_p`, `dynamic_update_slice_p`, `log2_p`, **batched `dot_general`**.

Consequences baked into the kernel:
- **No value slicing.** Index *refs* (per-field/per-modulus) into Python lists
  indexed at trace time; never value-index a loaded array.
- **No scatter.** Output written per-channel (`out_ref[0,f] = …`), not
  `jnp.stack` + `.at[].add`.
- **0.8.1 was the opposite on indexing** (no `dynamic_slice` there) — the two
  versions are mutually incompatible on slicing, hence targeting 0.9.2 only.

---

## 3. The crop saga (extracting the `[NG:NG+BS]` interior of an `H_PAD` tile)

Cropping the halo/pad off a computed tile is a *gather*, and 0.9.2 Triton
supports it through none of the obvious tools. The sequence of GPU errors and
fixes (each revealed the next constraint):

| Attempt | Result |
|---|---|
| `dynamic_slice` (`dynamic_index_in_dim`, interior slice) | `Unimplemented primitive: dynamic_slice` |
| `jnp.split(x, [NG, NG+BS])` (3 parts) | `Only power-of-2 num parts supported` |
| two 2-part splits | `Only equal-sized splits are supported` |
| fp64 selection-matrix **matmul** (`out·C`) | **HUNG the compiler** (fp64 `dot`) |
| `split(32 equal)` + `concatenate(16)` | `Only 2-argument concatenate is supported` |
| **one-hot select-and-reduce** (`Σ_k (k==NG+j)·x[k]`) | ✅ compiles |

**Final crop** (`_mid`/`_onehot` in `pallas_ozaki.py` and `pallas_fp.py`):
express the gather as a one-hot `broadcast · multiply · reduce_sum`, using only
`iota`/`mul`/`reduce_sum`/`broadcast` — the best-lowered op class. Exact (one
nonzero term per output), fp64-safe, no `dot`/`slice`/`split`/`concat`.

> ⚠️ **Lesson:** never use fp64 (or int32) `dot` in a Pallas kernel — it doesn't
> map to tensor cores and the fallback hangs the Triton compiler. Only
> int8/fp16/fp32 `dot` is safe.

---

## 4. THE OPEN BLOCKER — pathological compile time

`trunc=2` (residues + int8 GEMM + bias + mod; `k=6`, KO off, 64² grid) compiles
in **~1600 s**. Full kernel would be hours.

### Diagnosis (via `profile_ozaki.py --diagnose`)
- trivial int8 MMA (32³): **0.2 s** ✅
- `120×` int8 GEMMs at our shapes, **reused operands**: **2.5 s** ✅ — but Triton
  **CSE's identical GEMMs into ~1**, so this *understates* the real cost.
- `trunc=0` (limbs + crop, no GEMM/residue): **1.6 s** ✅
- `trunc=2`: **~1600 s** ⛔

### Cause
The kernel is **fully Python-unrolled**: every loop (×`k_full` moduli,
×~40 derivative calls, ×~28-step Garner) unrolls into one monolithic Triton
program — order **~320 distinct int8 GEMMs + hundreds of `rem`/reduce ops**.
Distinct (non-CSE) MMAs + int arithmetic blow up Triton's compile
super-linearly. (Same class of problem as the AMR sub-cycling compile blowup.)

### Fix path (NOT yet implemented)
Stop unrolling — **roll the modulus loop with `lax.scan`** (the body compiles
**once**; `scan_p` is supported). Constraints that shape the implementation:
- **Batched `dot_general` is OUT** (2-D only) → must be a real loop, not a batch.
- **`dynamic_update_slice` is OUT** → can't fill a buffer at a dynamic index;
  use `lax.scan`'s own output stacking to build the `(k, BS, H)` residue buffer.
- **Garner can't go in the same scan** (triangular recurrence). Keep it unrolled
  *after* the scan, unstacking `res_mods[i]` via the one-hot gather trick (§3).
- Per-modulus constants (`2^16 % m`, `gdiag`, `m_acc`, `basis`, `m`) become
  runtime `(k,)` arrays (passed as kernel inputs) instead of compile-time
  literals → modular reduction becomes runtime `rem` with an array divisor
  (slower at runtime, but the compile-time win is the point).

**Risk:** `scan`-in-Pallas, runtime-array `rem`, and gather-over-`k` are all
un-lower-testable locally (interpret skips Triton). Expect 1–2 GPU round-trips.

### Mitigations available now
- **Persistent compile cache** (`~/.jax_cache`, wired by `mcs_common.jax_config`)
  → a kernel compiles slowly **once**, then reloads in ~100 ms until the source
  changes. Doesn't fix dev iteration, but means it's not 1600 s every run.
- `profile_ozaki.py` knobs to shrink the kernel: `--quick`, `--no-ko`
  (drops ~half the derivative calls), `--mods-ext N` (fewer moduli — but **k<8
  is numerically UNSAFE for the C2 stencil**, OK only for compile probing).

---

## 5. Tooling built

- **`2D/src/mcs2d/profile_ozaki.py`** — the comprehensive profiler:
  - `[1]` wall-clock vs `fused_floating_point`; `[2]` **Amdahl split**
    (GEMM/residue/CRT via the `profile_trunc` truncation hook in the kernel);
    `[4]` moduli scaling; `[5]` BFP48-vs-fp64 transfer; `--trace` for int8-MMA
    confirmation.
  - `--diagnose` — staged compile (trivial int8 → +GEMM → +Garner → full) +
    int8-GEMM count/shape/distinct probes, to localise a compile hang.
  - `--quick`, `--no-ko`, `--mods-ext` — shrink the kernel for fast compile.
- **`profile_trunc`** kwarg on `make_pallas_ozaki_rhs` (0/1/2/None) — truncates
  the pipeline per stage for the Amdahl breakdown; `None` (default) is
  byte-for-byte the production kernel.

---

## 6. Still unanswered (the reason this matters)

The decisive question — **is `pallas_ozaki` faster than `fused_floating_point`,
and is the int8 GEMM a big enough fraction of the kernel to ride the hardware
curve (vs the RNS conversion/CRT overhead that doesn't)** — is **still
unmeasured**, because we never got a clean GPU profile through the compile wall.
Resolving §4 is the prerequisite for that measurement. See memory
`pallas_pow2_strategy` and `OPTIMIZATION.md`.
