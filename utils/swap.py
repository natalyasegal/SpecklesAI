'''
swaps axes in test format (2, number of samples, 40, 32, 32, 1) or (2, number of samples, 40, 32, 32)
arr[0] abecomes arr[1] and vise versa

sample usage: 
swapped = swap_file_with_tests("test_per_category_BCI_forehead___1.npy",
    "test_per_category_BCI_forehead___1_swapped.npy") #returns a numpy array and creates corresponding file
'''

import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# 1. Core swap function
# ------------------------------------------------------------
def swap_first_axis(arr: np.ndarray) -> np.ndarray:
    """
    Swap arr[0] and arr[1] for any array with shape (2, ...).
    Works for (2, N, 40, 32, 32, 1) and (2, N, 40, 32, 32).
    """
    arr = np.asarray(arr)

    if arr.ndim < 2 or arr.shape[0] != 2:
        raise ValueError(f"Expected first dimension = 2, got shape {arr.shape}")

    # Swap along the first axis
    return arr[[1, 0], ...]


# ------------------------------------------------------------
# 2. Test correctness of the swap (sums, means, 1st/last)
# ------------------------------------------------------------
def test_swap_correctness(arr: np.ndarray):
    """
    Verify that swapping arr produces an array where:
    - old arr[0] == new swapped[1]
    - old arr[1] == new swapped[0]
    And check sums, means, first/last values.
    """

    swapped = swap_first_axis(arr)

    # Flatten for simple value checks
    a0 = arr[0].ravel()
    a1 = arr[1].ravel()
    s0 = swapped[0].ravel()
    s1 = swapped[1].ravel()

    # Check arr[0] -> swapped[1]
    assert np.isclose(a0.sum(), s1.sum()), "Sum mismatch: arr[0] vs swapped[1]"
    assert np.isclose(a0.mean(), s1.mean()), "Mean mismatch: arr[0] vs swapped[1]"
    assert a0[0] == s1[0], "First value mismatch: arr[0] vs swapped[1]"
    assert a0[-1] == s1[-1], "Last value mismatch: arr[0] vs swapped[1]"

    # Check arr[1] -> swapped[0]
    assert np.isclose(a1.sum(), s0.sum()), "Sum mismatch: arr[1] vs swapped[0]"
    assert np.isclose(a1.mean(), s0.mean()), "Mean mismatch: arr[1] vs swapped[0]"
    assert a1[0] == s0[0], "First value mismatch: arr[1] vs swapped[0]"
    assert a1[-1] == s0[-1], "Last value mismatch: arr[1] vs swapped[0]"

    print("✓ Swap verified: sums, means, first/last values all match.")


# ------------------------------------------------------------
# 3. Load → shape check → swap → tests → optional save
# ------------------------------------------------------------
def load_and_swap_test_per_category(in_path: str,
                                    out_path: str | None = None) -> np.ndarray:
    """
    Load a test_per_category array of shape:
        (2, num_chunks, 40, 32, 32) or
        (2, num_chunks, 40, 32, 32, 1)
    Swap arr[0] <-> arr[1], run correctness tests, optionally save.
    """

    in_path = Path(in_path)
    arr = np.load(in_path, allow_pickle=False)

    # Basic shape check
    if arr.ndim not in (5, 6) or arr.shape[0] != 2:
        raise ValueError(
            f"Unexpected shape {arr.shape}. "
            "Expected (2, N, 40, 32, 32) or (2, N, 40, 32, 32, 1)."
        )

    # Stricter check depending on ndim
    if arr.ndim == 6:
        # (2, N, 40, 32, 32, 1)
        if arr.shape[2:] != (40, 32, 32, 1):
            raise ValueError(
                f"Unexpected shape {arr.shape}. "
                "Expected (2, N, 40, 32, 32, 1)."
            )
    elif arr.ndim == 5:
        # (2, N, 40, 32, 32)
        if arr.shape[2:] != (40, 32, 32):
            raise ValueError(
                f"Unexpected shape {arr.shape}. "
                "Expected (2, N, 40, 32, 32)."
            )

    print("→ Running correctness test BEFORE swap...")
    test_swap_correctness(arr)   # test internal consistency

    swapped = swap_first_axis(arr)

    print("→ Running correctness test AFTER swap...")
    test_swap_correctness(swapped)  # ensure swapped array is consistent

    if out_path is not None:
        out_path = Path(out_path)
        np.save(out_path, swapped)
        print(f"✓ Saved swapped array to: {out_path}")

    return swapped


# ------------------------------------------------------------
# 4. Convenience one-shot wrapper
# ------------------------------------------------------------
def swap_file_with_tests(in_path: str, out_path: str | None = None):
    """
    One-shot:
        load → check shape → run tests → swap → run tests → save → return swapped array
    """
    print(f"=== Processing file: {in_path} ===")
    swapped = load_and_swap_test_per_category(in_path, out_path=out_path)
    print("=== Done ===")
    return swapped
