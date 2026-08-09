"""whisper_rocm: pure-torch (M1) + kernel-pluggable host package for
Whisper large-v3-turbo on AMD R9700 (gfx1201).

See DESIGN.md for the full architecture. This package owns the host
orchestration; kernels/ (owned by Agents B/C) plug into whisper_rocm.ops
via ops.register_kernel().
"""

from . import ops  # noqa: F401

__all__ = ["ops"]
