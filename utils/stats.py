import numpy as np

def print_stats(X: np.ndarray, name_str = ''):
  print(f"{name_str} mean = {round(X.mean(), 2)} std = {round(X.std(), 2)} min = {round(X.min(), 2)} max = {round(X.max(), 2)} X.shape={X.shape},")


def print_test_stats(X: np.ndarray, name_str=''):
    """
    X shape: [2, N_chunks, 40, 32, 32, 1]
    Stats are computed over chunks, not pixels.
    """
    assert X.ndim == 6 and X.shape[0] == 2, f"Unexpected shape {X.shape}"

    print(f"{name_str} | full shape = {X.shape}")

    for cls in [0, 1]:
        n_chunks = X.shape[1]
        # Example: mean intensity per chunk, then stats over chunks
        per_chunk_mean = X[cls].mean(axis=(1, 2, 3, 4))

        print(
            f"  class {cls}: "
            f"mean={per_chunk_mean.mean():.3f}, "
            f"SD={per_chunk_mean.std(ddof=1):.3f}, "
            f"min={per_chunk_mean.min():.3f}, "
            f"max={per_chunk_mean.max():.3f}, "
            f"n_chunks={n_chunks}"
        )

