import sys
sys.path.append('/content/SpecklesAI')   # add package root to Python path

from models.LvMAE_pt import extract_embeddings_wrapper_one, load_for_resume_and_infer
from utils.embeddings_utils import concat_temporal_embeddings # sliding window -> target size: N-k+1
from pca.pca import visualize_embeddings_pca_3d


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
