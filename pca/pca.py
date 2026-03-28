from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def _check_xy_shapes(X, y):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")

    return X, y


def reduce_embeddings_pca_3d(X, scale=True, random_state=42):
    """
    Fast linear 3D reduction using PCA.
    Useful before GMM or as a baseline visualization.
    """
    X = np.asarray(X)

    if scale:
        X_proc = StandardScaler().fit_transform(X)
    else:
        X_proc = X.copy()

    pca = PCA(n_components=3, random_state=random_state)
    X_3d = pca.fit_transform(X_proc)
    return X_3d, pca

def plot_3d_embeddings_by_true_labels(
    X_3d,
    y,
    title="3D embedding visualization by true labels",
    class_names=None,
    elev=20,
    azim=35,
    alpha=0.75,
    s=20,
):
    """
    3D scatter where colors correspond to true labels.
    """
    X_3d, y = _check_xy_shapes(X_3d, y)

    if X_3d.shape[1] != 3:
        raise ValueError(f"X_3d must have shape (N, 3), got {X_3d.shape}")

    unique_labels = np.unique(y)
    cmap = plt.get_cmap("tab10", len(unique_labels))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, label in enumerate(unique_labels):
        mask = y == label
        name = class_names[label] if class_names is not None and label < len(class_names) else f"class_{label}"
        ax.scatter(
            X_3d[mask, 0],
            X_3d[mask, 1],
            X_3d[mask, 2],
            s=s,
            alpha=alpha,
            color=cmap(i),
            label=name,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    plt.tight_layout()
    plt.show()

def visualize_embeddings_pca_3d(
    X,
    y,
    class_names=None,
    scale=True,
    random_state=42,
):
    """
    Fast 3D PCA visualization colored by true labels.
    """
    X, y = _check_xy_shapes(X, y)
    X_pca_3d, pca = reduce_embeddings_pca_3d(X, scale=scale, random_state=random_state)

    explained = pca.explained_variance_ratio_
    title = (
        "3D PCA of embeddings by true labels\n"
        f"Explained variance: {explained[0]:.3f}, {explained[1]:.3f}, {explained[2]:.3f}"
    )

    plot_3d_embeddings_by_true_labels(
        X_pca_3d,
        y,
        title=title,
        class_names=class_names,
    )

    return X_pca_3d, pca
