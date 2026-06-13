"""Centralised JAX configuration.

Import this and call `setup()` at the top of any entry script BEFORE any
@jax.jit'd code runs, so the persistent compile cache is wired up early.

Persistent compile cache
------------------------
XLA caches compiled modules to disk, keyed by graph hash + platform.  Any
identical (re)run skips compilation entirely and just deserialises — turns the
"60s compile" into ~100 ms.  Cache invalidates automatically when source code
or input shapes change.

Cache directory: ~/.jax_cache  (override with MCS_JAX_CACHE env var).
"""
import os
from pathlib import Path
import jax

_DEFAULT_CACHE_DIR = Path.home() / ".jax_cache"


def setup(x64=True, cache=True, verbose=False):
    """Configure JAX once at program startup."""
    if x64:
        jax.config.update("jax_enable_x64", True)

    if cache:
        cache_dir = Path(os.environ.get("MCS_JAX_CACHE", str(_DEFAULT_CACHE_DIR)))
        cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        # Cache any compilation that takes > 1s — the bulk of what slows iteration.
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        # -1 = no minimum entry size; cache everything that meets the time threshold.
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        if verbose:
            print(f">> JAX compile cache: {cache_dir}")
