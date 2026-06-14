"""M2 overlap coupling: interpolation order, ghost coverage, boundary set."""
import jax.numpy as jnp
import numpy as np

from multipatch import atlas as A, overlap as O

KA, KB, KC = 0.7, -0.4, 0.5


def _f(X, Y, Z):
    return jnp.sin(KA * X + KB * Y + KC * Z)


def _interp_err(N, order=6):
    g = A.build_llama_grid(2.0, 1.8, 8.0, N, N, N, order=order)
    tbl = O.build_overlap_table(g, order=order)
    fields = [_f(p.X, p.Y, p.Z)[None] for p in g.patches]   # NF=1
    filled = O.apply_overlap_fill(fields, tbl)
    maxe = 0.0
    for e in tbl.entries:
        P = g.patches[e.recv]
        Xf = jnp.asarray(P.X).ravel()[e.tgt_idx]
        Yf = jnp.asarray(P.Y).ravel()[e.tgt_idx]
        Zf = jnp.asarray(P.Z).ravel()[e.tgt_idx]
        got = filled[e.recv].reshape(1, -1)[0, e.tgt_idx]
        maxe = max(maxe, float(jnp.max(jnp.abs(got - _f(Xf, Yf, Zf)))))
    return maxe


def test_overlap_interpolation_order():
    e1, e2 = _interp_err(15), _interp_err(25)
    order = np.log(e1 / e2) / np.log(25 / 15)
    assert order > 5.5            # m=order+1 -> degree-6 interp


def test_boundary_indices_outer_radial_only():
    g = A.build_llama_grid(2.0, 1.8, 8.0, 13, 13, 13, order=6)
    tbl = O.build_overlap_table(g, order=6)
    # cube has no outer boundary; every shell has the same outer-radial count
    assert int(tbl.boundary_idx[0].shape[0]) == 0
    counts = {int(b.shape[0]) for b in tbl.boundary_idx[1:]}
    assert len(counts) == 1 and counts.pop() > 0


def test_fill_is_identity_on_exact_field():
    # filling from an exact-everywhere field must not perturb interior values
    g = A.build_llama_grid(2.0, 1.8, 8.0, 13, 13, 13, order=6)
    tbl = O.build_overlap_table(g, order=6)
    fields = [_f(p.X, p.Y, p.Z)[None] for p in g.patches]
    filled = O.apply_overlap_fill(fields, tbl)
    for p, F0, F1 in zip(g.patches, fields, filled):
        intr = (slice(None),) + p.interior
        assert float(jnp.max(jnp.abs((F0 - F1)[intr]))) == 0.0
