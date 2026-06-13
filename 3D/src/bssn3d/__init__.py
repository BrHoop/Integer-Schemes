"""BSSN-in-3D solver (Phase 2: correctness port from Dendro-GR).

Lands on the validated Phase-1 3D foundation (shared ``mcs_common.derivatives``
FD operator, RK4, BC/ghost machinery). Phase 2.1 provides the 24-variable state,
gauge parameters and test-only initial data; the verbatim RHS + Dendro-GR
bit-compare arrive in Phase 2.2.

NOTE: this ``__init__`` is intentionally **lazy** (PEP 562 ``__getattr__``) and
imports nothing at module load. Importing the package must NOT pull in ``jax``,
so a tool that has to set ``XLA_FLAGS`` / ``TF_CPP_MIN_LOG_LEVEL`` *before* the
first ``import jax`` (e.g. ``bssn3d.spill_probe``) can do so reliably under
``python -m bssn3d.<module>`` — otherwise the package's eager imports would
initialize jax first and the flags would be ignored.
"""

import importlib

# public name -> submodule that defines it (imported on first attribute access)
_EXPORTS = {
    "BSSNState": "state",
    "PhysicsParams": "state",
    "NUM_VARS": "state",
    "VAR_NAMES": "state",
    "Grid": "grid",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
