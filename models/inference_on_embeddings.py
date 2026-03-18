import cv2
import numpy as np
import torch

def extract_sequence_embeddings_from_array(model, x_chunks, device):
    """
    x_chunks: np.ndarray or torch.Tensor of shape (N, 40, 32, 32, 1)
    returns: embeddings for each 40-frame chunk
    """
    model.eval()

    if isinstance(x_chunks, np.ndarray):
        x_chunks = torch.from_numpy(x_chunks)

    if x_chunks.ndim != 5:
        raise ValueError(f"Expected 5D input (N, T, H, W, C), got {tuple(x_chunks.shape)}")

    # (N, T, H, W, C) -> (N, C, T, H, W)
    x_chunks = x_chunks.permute(0, 4, 1, 2, 3).contiguous()
    x_chunks = x_chunks.to(device=device, dtype=torch.float32, non_blocking=True)

    with torch.no_grad():
        z = model.encode_no_mask(x_chunks, pool=None)

    return z.detach().cpu().numpy()

def frames_to_temporal_chunks(x, chunk_size=40):
    x = np.asarray(x)
    n_chunks = x.shape[0] // chunk_size
    x = x[:n_chunks * chunk_size]
    return x.reshape(n_chunks, chunk_size, *x.shape[1:])

def avi_to_chunk_array(
    avi_path,
    frame_size=(32, 32),
    chunk_size=40,
    dtype=np.uint8,
):
    """
    Read one AVI file and return a NumPy array of frame chunks without writing frames to disk.

    Mirrors the repository preprocessing logic:
    - read frames from video
    - resize to frame_size
    - convert to grayscale
    - add channel dimension -> (H, W, 1)
    - group into consecutive chunks of `chunk_size`
    - discard leftover frames that do not fill a full chunk

    Parameters
    ----------
    avi_path : str
        Path to the AVI file.
    frame_size : tuple[int, int]
        Output frame size as (width, height).
    chunk_size : int
        Number of frames per chunk.
    dtype : numpy dtype
        Output dtype. np.uint8 matches the repo most closely.

    Returns
    -------
    np.ndarray
        Shape: (n_chunks, chunk_size, frame_h, frame_w, 1)
    """
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {avi_path}")

    chunks = []
    current_chunk = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[..., None]  # (H, W) -> (H, W, 1)

            if dtype is not None and frame.dtype != dtype:
                frame = frame.astype(dtype, copy=False)

            current_chunk.append(frame)

            if len(current_chunk) == chunk_size:
                chunks.append(np.stack(current_chunk, axis=0))
                current_chunk.clear()

    finally:
        cap.release()

    if not chunks:
        h, w = frame_size[1], frame_size[0]
        return np.empty((0, chunk_size, h, w, 1), dtype=dtype)

    return np.stack(chunks, axis=0)
