# Vendored Dendro-GR source

This directory holds verbatim copies of the Dendro-GR files the BSSN port depends
on, so Integer-Schemes is **self-contained** (nothing in the code path reads from
`~/Code/Dendro-GR` at runtime).

## `bssneqs_SSL_HD_dxsq.cpp` — **production RHS (Phase 3.0)**
- **Source:** Dendro-GR, `CodeGen/bssneqs_SSL_HD_dxsq.cpp`
- **sha256:** `7b5353750d4240dd0e2a83971eba3c8a394a92b31fcb559fc159a092843617e2`
- **Vendored:** 2026-06-11 (Phase 3.0 production-variant lock)
- **What it is:** the SymPy-CSE-generated BSSN RHS with **CAHD + SSL** enabled —
  Hamiltonian-constraint damping on `chi` (`chi_rhs += -BSSN_CAHD_C·dx_i²·H/(dt·…)`)
  and spatial slice-locking on the lapse (`a_rhs += -√chi·h_ssl·(√chi-α)·exp(-t²/2σ²)`).
  Same flat SSA / derivatives-as-inputs design as the no-CAHD variant (708887
  original ops; 850 statements), the **same 24 fields + 138 derivative inputs +
  24 outputs**, with two extra functions (`sqrt`, `exp`) and the scalar params
  `BSSN_CAHD_C, dt, dx_i, h_ssl, sig_ssl, t`. Derivatives are indexed `grad…[pp]`
  here (the no-CAHD variant used them bare). External symbols: `pow`/`sqrt`/`exp`.

## `bssneqs_sympy_cse_wo_derivs.cpp` — no-CAHD variant (Phase 2 history)
- **Source:** Dendro-GR, `CodeGen/bssneqs_sympy_cse_wo_derivs.cpp`
- **sha256:** `4953a8a8ebe869af592d8e023d2a12abde31aa62df248495a69635f8c442956b`
- **Vendored:** 2026-06-11
- **What it is:** the simpler SymPy-CSE BSSN RHS with no constraint damping (623308
  original ops). Phase 2 validated against this; Phase 3.0 superseded it with the
  CAHD+SSL variant above. Kept for provenance / regression reference. **Not on the
  active code path** (`_codegen.DENDRO_CSE` points at the CAHD+SSL file).

## `physcon.cpp` — Hamiltonian/momentum constraint operator
Used by `_codegen_constraints.py` → `_constraints_generated.py` (the reference-free
constraint-convergence apples test).

## How they are used
1. `bssn3d/_codegen.py` transliterates the production `.cpp` → the committed
   `bssn3d/_bssn_rhs_generated.py` (our JAX algebra). Regenerate with
   `python -m bssn3d._codegen`.
2. `bssn3d/oracle.py` compiles the production `.cpp` standalone with `g++` (no
   octree/MPI) to produce the trusted one-eval oracle for the bit-compare gate.

To refresh from upstream, re-copy the file and update the sha256 above; the
`test_generated_matches_regen` test guards the generated module against drift, and
the `_codegen` scalar-param drift-guard fails fast if a refreshed file introduces a
new parameter.
