'''
usage: 
norm_mode = "minmax" #"standard" 
test_2_m_n_2 = normalize_per_fixedclip(test_2_m_2, n_chunks_per_clip=5000, mode=norm_mode, gain=10.0)
'''

import numpy as np

def normalize_per_fixedclip(
    X: np.ndarray,
    n_chunks_per_clip: int = 5000,
    mode: str = "minmax",   # "minmax" or "standard"
    eps: float = 1e-8,
    gain: float = 10.0       # multiplies after normalization
) -> np.ndarray:
    """
    Per-clip normalization for chunked videos, then multiply by a constant gain.

    Supports:
      - Grouped:   (C, M, 40, 32, 32[, 1])  -> C clips, M chunks/clip
      - Flattened: (N, 40, 32, 32[, 1])     -> N % n_chunks_per_clip == 0

    Returns:
      Array with the SAME shape and dtype as X.
    """
    assert X.ndim in (4, 5, 6), f"Expected 4/5/6D, got {X.ndim}D with shape {X.shape}"
    dtype = X.dtype
    T, H, W = 40, 32, 32
    gain = float(gain)

    def _norm_block(xc: np.ndarray):
        # xc is one whole clip: (M, 40, 32, 32[, 1]); returns float64 array
        if mode == "minmax":
            mn = float(xc.min()); mx = float(xc.max())
            denom = (mx - mn) if (mx - mn) > eps else 1.0
            out = (xc - mn) / denom
        elif mode == "standard":
            mu = float(xc.mean()); sd = float(xc.std())
            sd = sd if sd > eps else 1.0
            out = (xc - mu) / sd
        else:
            raise ValueError("mode must be 'minmax' or 'standard'")
        if gain != 1.0:
            out = out * gain
        return out

    # ---- Case A: Grouped (C, M, T, H, W[, 1]) ----
    is_grouped_wo_c = (X.ndim == 5 and X.shape[-3:] == (T, H, W))
    is_grouped_w_c  = (X.ndim == 6 and X.shape[-4:] == (T, H, W, 1))
    if is_grouped_wo_c or is_grouped_w_c:
        # Heuristic: grouped if axis 1 != T
        if X.shape[1] != T:
            Xn = X.copy()
            C = X.shape[0]
            for clip_idx in range(C):
                xc = Xn[clip_idx].astype(np.float64, copy=False)
                Xn[clip_idx] = _norm_block(xc).astype(dtype, copy=False)
            return Xn

    # ---- Case B: Flattened (N, T, H, W[, 1]) ----
    is_flat_wo_c = (X.ndim == 4 and X.shape[1:] == (T, H, W))
    is_flat_w_c  = (X.ndim == 5 and X.shape[1:] == (T, H, W, 1))
    if is_flat_wo_c or is_flat_w_c:
        N = X.shape[0]
        assert N % n_chunks_per_clip == 0, (
            f"N={N} not divisible by n_chunks_per_clip={n_chunks_per_clip}"
        )
        Xn = X.copy()
        start = 0
        for _ in range(N // n_chunks_per_clip):
            sl = slice(start, start + n_chunks_per_clip)
            xc = Xn[sl].astype(np.float64, copy=False)
            Xn[sl] = _norm_block(xc).astype(dtype, copy=False)
            start += n_chunks_per_clip
        return Xn

    raise AssertionError(f"Unrecognized shape for per-clip normalization: {X.shape}")




# for whole sample
def normalize_global_whole_sample(
    X: np.ndarray,
    mode: str = "minmax",   # "minmax" or "standard"
    gain: float = 1.0,
    eps: float = 1e-8,
):
    """
    Apply the *same* normalization that _norm_block uses,
    but compute statistics over the *entire sample*, not per clip.

    Does NOT touch time axis specially — time stays untouched.
    The same shift/scale is applied to every frame t.

    Returns array with SAME SHAPE and SAME DTYPE.
    """
    dtype = X.dtype
    Xf = X.astype(np.float64, copy=False)  # same behavior as original _norm_block

    if mode == "minmax":
        mn = float(Xf.min())
        mx = float(Xf.max())
        denom = (mx - mn) if (mx - mn) > eps else 1.0
        out = (Xf - mn) / denom

    elif mode == "standard":
        mu = float(Xf.mean())
        sd = float(Xf.std())
        sd = sd if sd > eps else 1.0
        out = (Xf - mu) / sd

    else:
        raise ValueError("mode must be 'minmax' or 'standard'")

    if gain != 1.0:
        out = out * float(gain)

    return out.astype(dtype, copy=False)
