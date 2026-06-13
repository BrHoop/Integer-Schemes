"""
Auto-calibration of per-level slot capacities (Phase 2a–2d).

Covers:
  * occupancy tracking + recommended_caps()
  * auto-grow on capacity exhaustion (sticky, padded, data-preserving)
  * hard ceiling (MAX_BLOCKS) and runaway detection
  * caps sidecar round-trip
"""

import os
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mcs2d.amr.state import (
    BS, NG, NF, LEVELS, MAX_BLOCKS, MAX_BLOCKS_PER_LEVEL,
    AMRState, AMRTopology,
)
from mcs2d.amr.regrid import (
    apply_flags, write_caps_sidecar, read_caps_sidecar, REFINE,
)


def _single_root_state(caps):
    """One active root block (slot 0) carrying a smooth polynomial, with the
    given per-level capacities."""
    def p(X, Y):
        return X + 2*Y + 0.1 * X * Y
    coords = np.arange(BS + 2*NG)
    Xc, Yc = np.meshgrid(coords, coords, indexing='ij')
    single = np.broadcast_to(p(Xc, Yc), (NF, BS + 2*NG, BS + 2*NG))

    blocks = tuple(
        (jnp.asarray(np.broadcast_to(single, (caps[L], NF, BS+2*NG, BS+2*NG)).copy())
         if L == 0 else jnp.zeros((caps[L], NF, BS+2*NG, BS+2*NG)))
        for L in range(LEVELS)
    )
    # only slot 0 of level 0 active
    active = tuple(
        (jnp.zeros((caps[L],), bool).at[0].set(True) if L == 0
         else jnp.zeros((caps[L],), bool))
        for L in range(LEVELS)
    )
    state = AMRState(blocks=blocks, active=active)
    topo = AMRTopology(caps=list(caps))
    topo.add_block(0, 0, (0, 0))
    return state, topo


class TestOccupancyTracker:

    def test_records_peak(self):
        topo = AMRTopology()
        topo.add_block(0, 0, (0, 0))
        topo.add_block(0, 1, (BS, 0))
        topo.record_occupancy()
        assert topo.peak_active[0] == 2
        # Remove one; peak must NOT drop (it's a high-water mark).
        topo.remove_block(0, 1)
        topo.record_occupancy()
        assert topo.peak_active[0] == 2

    def test_recommended_caps(self):
        topo = AMRTopology()
        topo.peak_active = [4, 10, 0, 0]
        rec = topo.recommended_caps(margin=1.5)
        # Level 0 exact; others ceil(peak*1.5); zero stays ≥1.
        assert rec[0] == 4
        assert rec[1] == 15
        assert rec[2] == 1 and rec[3] == 1


class TestAutoGrow:

    def test_grow_on_exhaustion_preserves_data(self):
        """Level 1 capacity = 2, but refining the root needs 4 children →
        auto-grow.  Existing data preserved; caps bumped; one grow event."""
        caps = [4, 2, 2, 2]   # level 1 too small for 4 children
        state, topo = _single_root_state(caps)
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[0, 0] = REFINE

        new_state, new_topo = apply_flags(state, topo, flags, auto_grow=True)

        # Capacity grew above 2 to fit 4 children.
        assert new_topo.caps[1] >= 4
        assert new_state.blocks[1].shape[0] == new_topo.caps[1]
        assert new_state.active[1].shape[0] == new_topo.caps[1]
        # 4 children created.
        assert int(np.asarray(new_state.active[1]).sum()) == 4
        # At least one grow event recorded.
        assert len(new_topo.grow_history) >= 1
        assert new_topo.grow_history[0][0] == 1   # level 1 grew

        # Children carry the prolongated parent (non-zero), proving data is real.
        for cs in range(4):
            assert np.any(np.asarray(new_state.blocks[1][cs]) != 0.0)

    def test_no_grow_when_disabled(self):
        caps = [4, 2, 2, 2]
        state, topo = _single_root_state(caps)
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[0, 0] = REFINE
        with pytest.raises(RuntimeError, match="budget exhausted"):
            apply_flags(state, topo, flags, auto_grow=False)

    def test_hard_ceiling(self):
        """Growth cannot exceed MAX_BLOCKS; hitting it raises a clear error."""
        # Level 1 cap already at the ceiling and full → can't grow.
        caps = [4, MAX_BLOCKS, MAX_BLOCKS, MAX_BLOCKS]
        state, topo = _single_root_state(caps)
        # Mark ALL level-1 slots active so there's no room and no growth headroom.
        active = list(state.active)
        active[1] = jnp.ones((caps[1],), bool)
        state = AMRState(blocks=state.blocks, active=tuple(active))
        topo.active[1, :] = True
        flags = np.zeros((LEVELS, MAX_BLOCKS), dtype=np.int32)
        flags[0, 0] = REFINE
        with pytest.raises(RuntimeError, match="hard ceiling"):
            apply_flags(state, topo, flags, auto_grow=True)


class TestCapsSidecar:

    def test_roundtrip(self, tmp_path):
        topo = AMRTopology(caps=[4, 16, 30, 30])
        topo.peak_active = [4, 12, 20, 18]
        topo.grow_history = [(1, 8, 16), (2, 16, 30)]
        path = str(tmp_path / "amr_caps.json")
        write_caps_sidecar(path, topo)
        assert os.path.exists(path)

        rec = read_caps_sidecar(path)
        assert rec == topo.recommended_caps()
        assert len(rec) == LEVELS

    def test_read_missing_returns_default(self, tmp_path):
        path = str(tmp_path / "does_not_exist.json")
        rec = read_caps_sidecar(path)
        assert rec == tuple(MAX_BLOCKS_PER_LEVEL)


class TestCalibratedRootState:
    """4.4: make_calibrated_root_state pre-sizes the root state from a sidecar
    (the production default), with precedence explicit > sidecar > uniform."""

    @staticmethod
    def _root_data():
        # (NF, 1*BS+2NG, 1*BS+2NG) — a 1×1 root tiling.
        return jnp.zeros((NF, BS + 2*NG, BS + 2*NG))

    @staticmethod
    def _sidecar(tmp_path):
        topo = AMRTopology(caps=[1] + [8] * (LEVELS - 1))
        topo.peak_active = [1] + [4] * (LEVELS - 1)
        path = str(tmp_path / "amr_caps.json")
        write_caps_sidecar(path, topo)
        return path

    def test_uses_sidecar_recommended_caps(self, tmp_path):
        from mcs2d.amr.regrid import make_calibrated_root_state
        path = self._sidecar(tmp_path)
        expected = read_caps_sidecar(path)                  # recommended
        state, topo = make_calibrated_root_state(
            self._root_data(), 1, 1, caps_sidecar=path)
        sizes = tuple(int(state.blocks[L].shape[0]) for L in range(LEVELS))
        assert sizes == expected
        assert tuple(topo.caps) == expected

    def test_absent_sidecar_falls_back_to_default(self, tmp_path):
        from mcs2d.amr.regrid import make_calibrated_root_state
        path = str(tmp_path / "missing.json")
        state, _ = make_calibrated_root_state(
            self._root_data(), 1, 1, caps_sidecar=path)
        sizes = tuple(int(state.blocks[L].shape[0]) for L in range(LEVELS))
        assert sizes == tuple(MAX_BLOCKS_PER_LEVEL)

    def test_explicit_caps_override_sidecar(self, tmp_path):
        from mcs2d.amr.regrid import make_calibrated_root_state
        path = self._sidecar(tmp_path)
        explicit = [1] + [3] * (LEVELS - 1)
        state, _ = make_calibrated_root_state(
            self._root_data(), 1, 1, caps_sidecar=path, caps=explicit)
        sizes = tuple(int(state.blocks[L].shape[0]) for L in range(LEVELS))
        assert sizes == tuple(explicit)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
