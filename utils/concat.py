import numpy as np

def concatenate_train_or_val(x_list, y_list):
  x = np.concatenate(x_list, axis=0)
  y = np.concatenate(y_list, axis=0)
  return x, y

def concatenate_test(inp_list):
  return np.concatenate(inp_list, axis=1)




def combine_label_chunks(data: np.ndarray, chunks_to_combine: int, pad: bool = False) -> np.ndarray:
    """
    Combine consecutive 40-frame chunks along the 'chunks' axis for each label.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (L, N, 40, H, W, C),
        where L = number of labels (e.g., 2), N = number of 40-frame chunks.
    chunks_to_combine : int
        How many 40-frame chunks to merge into one new chunk.
        e.g., 2 -> 80 frames, 25 -> 1000 frames.
    pad : bool
        If True and N is not divisible by chunks_to_combine, pad with zeros
        so no data is dropped. If False, leftover chunks are trimmed.

    Returns
    -------
    np.ndarray
        Output array of shape (L, M, 40 * chunks_to_combine, H, W, C),
        where M = floor(N / chunks_to_combine) if pad=False,
              or M = ceil(N / chunks_to_combine) if pad=True.
    """
    if data.ndim != 6:
        raise ValueError(f"Expected data with 6 dims (L, N, 40, H, W, C), got shape {data.shape}")

    L, N, T, H, W, C = data.shape
    if T != 40:
        raise ValueError(f"Expected base chunk length T=40, got T={T}")

    k = int(chunks_to_combine)
    if k <= 0:
        raise ValueError("chunks_to_combine must be a positive integer")

    if pad:
        # Compute needed padding to make N divisible by k
        remainder = N % k
        if remainder != 0:
            pad_chunks = k - remainder
            pad_shape = (L, pad_chunks, T, H, W, C)
            pad_block = np.zeros(pad_shape, dtype=data.dtype)
            data = np.concatenate([data, pad_block], axis=1)
            N = data.shape[1]  # updated
    else:
        # Trim extra chunks that don't fit into a full group
        N = (N // k) * k
        data = data[:, :N]

    # Now N is divisible by k
    M = N // k
    # Reshape: group each consecutive k chunks into one long chunk (k * 40 frames)
    out = data.reshape(L, M, k * T, H, W, C)
    return out

