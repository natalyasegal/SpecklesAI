import numpy as np

# ---- Helper: to concatenate K consecutive embeddings (temporal aggregation) ----

def make_clip_ids_from_fps(chunk_counts, fps=1000, chunk_ms=40, clip_seconds=10):
    """
    chunk_counts: dict {subject_id: total chunks}. Returns (N,) globally-unique clip_ids.
    """
    if not chunk_counts:
        return np.empty((0,), dtype=np.int64)
    frames_per_chunk = fps * chunk_ms / 1000.0          # /1000: ms -> s
    cpc = fps * clip_seconds / frames_per_chunk         # chunks per clip
    assert cpc == int(cpc), "clip length not divisible into chunks"
    cpc = int(cpc)
    ids, next_id = [], 0
    for sid, n in chunk_counts.items():
        assert n % cpc == 0, f"Subject {sid}: {n} chunks not a multiple of {cpc}"
        n_clips = n // cpc
        ids.append(np.repeat(np.arange(next_id, next_id + n_clips), cpc))
        next_id += n_clips
    return np.concatenate(ids)

def concat_temporal_embeddings_c(Z, y, clip_ids, K=1):
    """
    clip boundaries aware concatentation of embeddings
    """
    assert Z.ndim == 2 and y.ndim == 1 and len(Z) == len(y) == len(clip_ids)
    if K <= 1:
        return Z.copy(), y.copy()
    N, D = Z.shape
    # candidate windows, then keep only those whose first and last chunk
    # belong to the same clip (contiguous equal ids => whole window in-clip)
    windows = np.lib.stride_tricks.sliding_window_view(Z, K, axis=0)  # (N-K+1, D, K)
    starts = clip_ids[:N - K + 1]
    ends   = clip_ids[K - 1:]
    keep = starts == ends

    Zk = windows[keep].transpose(0, 2, 1).reshape(keep.sum(), K * D)  # temporal order
    yk = y[K - 1:][keep]
    return Zk, yk

def concat_temporal_embeddings_c_stride(Z, y, clip_ids, K=1, stride=1):
    """
    clip boundaries aware concatentation of embeddings and supporting stride,
    stride = K will make segments independant
    """
    assert Z.ndim == 2 and y.ndim == 1 and len(Z) == len(y) == len(clip_ids), "misaligned inputs"
    assert isinstance(K, (int, np.integer)) and K >= 1, f"K must be int >= 1, got {K!r}"
    if K <= 1:
        return Z.copy(), y.copy()
    assert isinstance(stride, (int, np.integer)) and 1 <= stride <= K, \
        f"stride must be int in [1, {K}], got {stride!r}"
    N, D = Z.shape
    Zk, yk = [], []
    for s in range(0, N - K + 1, stride):
        if clip_ids[s] == clip_ids[s + K - 1]:      # boundary guard
            Zk.append(Z[s:s+K].reshape(-1)); yk.append(y[s + K - 1])
    if not Zk:
        return np.empty((0, K*D), dtype=Z.dtype), np.empty((0,), dtype=y.dtype)
    return np.stack(Zk), np.asarray(yk, dtype=y.dtype)
    
def concat_temporal_embeddings(Z: np.ndarray, y: np.ndarray, K: int = 1):
    """
    Ignorant to clip boundaries!
    Uses stride = 1
    
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


''' Save and Load '''

def save_embeddings(x, y, n=None, K=1, class_names_list=("home", "tea", "stop", "help"),
    key="home", file_name="home.npy"):
    class_names_list = list(class_names_list)

    if key not in class_names_list:
        raise ValueError(f"{key=} not found in class_names_list={class_names_list}")

    x = np.asarray(x)
    y = np.asarray(y)

    if K > 1:
        x, y = concat_temporal_embeddings(x, y, K)

    class_idx = class_names_list.index(key)
    n_classes = len(class_names_list)
    samples_per_class = len(x) // n_classes

    start = class_idx * samples_per_class
    end = (class_idx + 1) * samples_per_class
    x_key = x[start:end]

    if n is not None:
        x_key = x_key[:n]

    np.save(file_name, x_key)
    print(f"Saved {x_key.shape} to {file_name}")
    return x_key

def load_embeddings(file_name):
    x = np.load(file_name, allow_pickle=False)
    print(f"Loaded {x.shape} from {file_name}")
    return x

def save_all_embeddings(x, y, n=None, K=1,
    class_names_list=("home", "tea", "stop", "help"), file_template="{key}.npy"):
    saved = {}
    for key in class_names_list:
        file_name = file_template.format(key=key)
        saved[key] = save_embeddings(x=x, y=y, n=n, K=K,
            class_names_list=class_names_list,
            key=key,file_name=file_name)
    return saved

def load_all_embeddings(keys, file_template="{key}.npy"):
    loaded = {}
    for key in keys:
        file_name = file_template.format(key=key)
        loaded[key] = load_embeddings(file_name)
    return loaded

def verify_saved_files_match_input(x, y, n=None, K=1,
    class_names_list=("home", "tea", "stop", "help"), file_template="{key}.npy"):
    x = np.asarray(x)
    y = np.asarray(y)

    if K > 1:
        x, y = concat_temporal_embeddings(x, y, K)

    n_classes = len(class_names_list)
    samples_per_class = len(x) // n_classes
    all_ok = True

    for class_idx, key in enumerate(class_names_list):
        start = class_idx * samples_per_class
        end = (class_idx + 1) * samples_per_class
        expected = x[start:end]

        if n is not None:
            expected = expected[:n]

        loaded = np.load(file_template.format(key=key), allow_pickle=False)

        same_shape = expected.shape == loaded.shape
        same_values = np.array_equal(expected, loaded)

        print(f"{key}: shape_ok={same_shape}, values_ok={same_values}, shape={loaded.shape}")

        if not (same_shape and same_values):
            all_ok = False

    if all_ok:
        print("✅ All saved files exactly match the input slices.")
    else:
        print("❌ Some saved files do not match the input slices.")

    return all_ok
