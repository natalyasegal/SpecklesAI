import numpy as np
import sys
sys.path.append('/content/SpecklesAI')   # add package root to Python path

from models.LvMAE_pt import VideoMAE, extract_embeddings_wrapper_one, load_for_resume_and_infer
from utils.embeddings_utils import concat_temporal_embeddings # sliding window -> target size: N-k+1
from pca.pca import visualize_embeddings_pca_3d, visualize_embeddings_pca_2d

'''
3D:
'''
def create_and_visualize_embeddings_3d(x, y, class_names=["class0", "class1"], K=1):
  model, opt2, scaler2, start_ep = load_for_resume_and_infer(VideoMAE, "artifacts_lvmae_1/checkpoint.pt")
  x, y = extract_embeddings_wrapper_one(model, x, y)
  x, y = concat_temporal_embeddings(x, y, K)
  return visualize_embeddings_pca_3d(X=x, y=y, class_names=class_names)
  
def create_and_visualize_embeddings_multiclass_3d(x, y, class_names=["class0", "class1"], K=1):
  model, opt2, scaler2, start_ep = load_for_resume_and_infer(VideoMAE, "artifacts_lvmae_1/checkpoint.pt")
  y = np.argmax(y, axis=1)
  x, y = extract_embeddings_wrapper_one(model, x, y)
  x, y = concat_temporal_embeddings(x, y, K)
  return visualize_embeddings_pca_3d(X=x, y=y, class_names=class_names)

'''
2D:
'''
def create_and_visualize_embeddings_2d(x, y, class_names=["class0", "class1"], K=1):
    model, opt2, scaler2, start_ep = load_for_resume_and_infer(
        VideoMAE, "artifacts_lvmae_1/checkpoint.pt"
    )
    x, y = extract_embeddings_wrapper_one(model, x, y)
    x, y = concat_temporal_embeddings(x, y, K)
    return visualize_embeddings_pca_2d(X=x, y=y, class_names=class_names)


def create_and_visualize_embeddings_multiclass_2d(x, y, class_names=None, K=1):
    model, opt2, scaler2, start_ep = load_for_resume_and_infer(
        VideoMAE, "artifacts_lvmae_1/checkpoint.pt"
    )

    if y.ndim > 1:
        y = np.argmax(y, axis=1)

    x, y = extract_embeddings_wrapper_one(model, x, y)
    x, y = concat_temporal_embeddings(x, y, K)

    if class_names is None:
        n_classes = len(np.unique(y))
        class_names = [f"class{i}" for i in range(n_classes)]

    return visualize_embeddings_pca_2d(X=x, y=y, class_names=class_names)
