import numpy as np

# ---- Helper: concatenate K consecutive embeddings (temporal aggregation) ----

def concat_temporal_embeddings(Z: np.ndarray, y: np.ndarray, K: int = 1):
    """
    Z: (N, D) embeddings in temporal order
    y: (N,) labels aligned with Z
    K: window length. If K==1, returns inputs unchanged.

    Returns:
      Zk: (N - K + 1, K*D)
      yk: (N - K + 1,)
    """
    assert Z.ndim == 2 and y.ndim == 1 and len(Z) == len(y), "Bad shapes"
    N, D = Z.shape
    if K <= 1 or N <= K:
        print("------------ cannot concatenate, returning a copy ------------")
        return (Z.copy(), y.copy()) if K <= 1 else (Z[-1:].repeat(1, axis=0), y[-1:])
    # simple, robust loop (fast enough for typical N)
    Zk = np.empty((N - K + 1, K * D), dtype=Z.dtype)
    for i in range(K - 1, N):
        Zk[i - (K - 1)] = Z[i - K + 1:i + 1].reshape(-1)
    yk = y[K - 1:]     # label at the end of each window
    return Zk, yk

def concat_y(y: np.ndarray, K: int = 1):
    """
    y: (N,) labels aligned with Z
    K: window length. If K==1, returns inputs unchanged.

    Returns:
      yk: (N - K + 1,)
    """
    assert y.ndim == 1, "Bad shapes"
    N = y.shape
    if K <= 1 or N <= K:
        return y.copy() if K <= 1 else y[-1:]
    # simple, robust loop (fast enough for typical N)
    yk = y[K - 1:]     # label at the end of each window
    return yk

