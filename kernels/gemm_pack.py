"""gemm_pack.py -- reference weight-packing for K2 (gemm_wmma.hip).

Import `pack_weights` from here in weights.py; do not re-derive the layout by
hand. See the "WEIGHT PACKING SPEC" comment block at the top of
kernels/gemm_wmma.hip for the full derivation -- this file is the executable
spec, that comment is the prose spec, keep them in sync.

BLOCK_K and BLOCK_N below (32, 128) MUST match gemm_wmma.hip's BLOCK_K/BLOCK_N
for both the main (128x128x32) and small-M (32x128x32) tile configs -- both
configs share BLOCK_K=32/BLOCK_N=128, so one packed tensor serves both.
"""

import torch

BLOCK_K = 32
BLOCK_N = 128


def pack_weights(w: torch.Tensor, already_kn: bool = False) -> torch.Tensor:
    """Pack a weight matrix into the tiled (N_blocks, K_blocks, 32, 128) fp16
    layout gemm_wmma.hip's B-tile loader expects.

    Args:
      w: either
         - shape (out_features, in_features) = (N, K), the standard
           `torch.nn.Linear.weight` layout (already_kn=False, the default) --
           this is transposed internally to (K, N) before packing, matching
           `Y = X @ W_kn` with `W_kn = w.t()`; or
         - shape (in_features, out_features) = (K, N) directly
           (already_kn=True), if the caller already has Y = X @ w.
      already_kn: see above.

    Returns:
      fp16 CUDA/HIP-ready tensor, contiguous, shape
      (N // 128, K // 32, 32, 128), ready to pass as the `Bpacked` argument
      to any of the launch_gemm_* functions in kernels_api.py.

    Raises:
      ValueError if K is not a multiple of 32 or N is not a multiple of 128
      (every K2 shape in DESIGN.md SS6 satisfies this: K in {1280, 5120},
      N in {1280, 3840, 5120}).
    """
    w_kn = w if already_kn else w.t()
    w_kn = w_kn.contiguous().to(torch.float16)
    K, N = w_kn.shape
    if K % BLOCK_K != 0:
        raise ValueError(f"K={K} is not a multiple of BLOCK_K={BLOCK_K}")
    if N % BLOCK_N != 0:
        raise ValueError(f"N={N} is not a multiple of BLOCK_N={BLOCK_N}")
    K_blocks = K // BLOCK_K
    N_blocks = N // BLOCK_N
    # (K_blocks, 32, N_blocks, 128) -> (N_blocks, K_blocks, 32, 128)
    packed = w_kn.reshape(K_blocks, BLOCK_K, N_blocks, BLOCK_N).permute(2, 0, 1, 3)
    return packed.contiguous()


def unpack_weights(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_weights(..., already_kn=True): returns the (K, N)
    matrix. Useful for tests / sanity checks, not needed in the hot path."""
    N_blocks, K_blocks, bk, bn = packed.shape
    assert bk == BLOCK_K and bn == BLOCK_N
    w_kn = packed.permute(1, 2, 0, 3).contiguous().reshape(K_blocks * BLOCK_K, N_blocks * BLOCK_N)
    return w_kn
