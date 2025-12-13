import os
import random
import numpy as np
import torch

def set_seed_all(seed_for_init=1, random_seed=9, use_tf = False):
    """
    Sets all relevant seeds for reproducibility across:
    - Python random
    - NumPy
    - TensorFlow
    - PyTorch (CPU + GPU)
    - CUDA/cuDNN deterministic behavior

    Parameters
    ----------
    seed_for_init : int
        Main seed used for NumPy, TF, Torch.
    random_seed : int
        Secondary seed for Python's random module.
    """

    # ========== PYTHON RANDOM ==========
    random.seed(random_seed)

    # ========== NUMPY ==========
    np.random.seed(seed_for_init)

    # ========== ENVIRONMENT (affects hashing, TF determinism) ==========
    os.environ['PYTHONHASHSEED'] = str(seed_for_init)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    if use_tf:
      # ========== TENSORFLOW ==========
      try:
          import tensorflow as tf
          tf.random.set_seed(seed_for_init)
      except Exception:
          pass  # optional if TF is not used

      # ========== PYTORCH ==========
      try:
          # Base seeds
          torch.manual_seed(seed_for_init)
          torch.cuda.manual_seed(seed_for_init)
          torch.cuda.manual_seed_all(seed_for_init)

          # cuDNN determinism settings
          torch.backends.cudnn.deterministic = True
          torch.backends.cudnn.benchmark = False

          # New PyTorch deterministic algorithms flag
          torch.use_deterministic_algorithms(True)

      except Exception:
          pass  # optional if PyTorch is not installed

    print(f"[set_seed] All seeds set (seed_for_init={seed_for_init}, random_seed={random_seed}).")
