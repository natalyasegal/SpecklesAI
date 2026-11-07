import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import shutil
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from models.LvMAE_pt import load_for_resume_and_infer, extract_embeddings_wrapper_one
from models.LvMAE_pt import *
from utils.concat import concatenate_train_or_val
from utils.data import split_from_start, split_by_chunks_v
from utils.formatstranslator import test2trainformat
from utils.embeddings_utils import concat_temporal_embeddings_c_stride, concat_temporal_embeddings_c, make_clip_ids_from_fps, concat_temporal_embeddings
from pca.pca import visualize_embeddings_pca_3d, reduce_embeddings_pca_3d

def calc_mean_ARI_NMI_of_GMMcluster_c(K, stride, clip_seconds, chunk_ms, fps, n_components_GMM, 
          data_suffix="n_40ms",
          subject_to_model = {1:6, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:3, 12:6, 13:1, 14:6, 15:3},
          models_root = "gdrive/MyDrive/__PHd_2025/code_and_data/code/BCI_LvMAE_models_9_1"):
  # Subject -> LV-MAE model mapping from the notebook clustering section
  dst = "artifacts_lvmae_1"
  aris = []
  nmis = []

  for subject in subject_to_model:
      model_num = subject_to_model[subject]
      src = f"{models_root}/model{model_num}_30epochs/artifacts_lvmae_1"

      # Replace currently loaded LV-MAE model
      if os.path.exists(dst):
          shutil.rmtree(dst)
      shutil.copytree(src, dst)

      # Get the correct subject data
      X = globals()[f"x{subject}_{data_suffix}"]
      y = globals()[f"y{subject}_{data_suffix}"]

      # Calculate ARI and NMI
      ari, nmi = calc_ARI_NMI_of_GMMcluster_c(X=X, y=y, K=K, stride=stride, clip_seconds=clip_seconds,
                                              chunk_ms=chunk_ms, fps=fps, n_components_GMM=n_components_GMM)

      aris.append(float(ari))
      nmis.append(float(nmi))
      print(f"Subject {subject:2d} | LV-MAE model {model_num:2d} | "
          f"ARI = {float(ari):.4f} | NMI = {float(nmi):.4f}")
  print("\n--------------------")
  print(f"Mean ARI = {np.mean(aris):.4f}")
  print(f"Mean NMI = {np.mean(nmis):.4f}")
  print("--------------------")
  return np.mean(aris), np.mean(nmis)
            
#lables used only for metrics: ARI and NMI
def calc_ARI_NMI_of_GMMcluster_model(model, X, y, K=1, stride=1, clip_seconds=10, chunk_ms=40, fps=1000,
                                 random_state=42, n_components_GMM=2):
  if n_components_GMM > 2:
      y = np.argmax(y, axis=1)
  x, y = extract_embeddings_wrapper_one(model, X, y)

  mk = lambda Z: make_clip_ids_from_fps({0: len(Z)},fps=fps,chunk_ms=chunk_ms,clip_seconds=clip_seconds)
  x, y=concat_temporal_embeddings_c_stride(x, y,mk(x),K,stride=stride)
  X_pca_3d, pca = reduce_embeddings_pca_3d(x, scale=True, random_state=random_state)

  gmm = GaussianMixture(n_components=n_components_GMM,covariance_type="full", random_state=random_state)
  clusters = gmm.fit_predict(X_pca_3d)
  ari = adjusted_rand_score(y, clusters)
  nmi = normalized_mutual_info_score(y, clusters)
  print("ARI:", ari)
  print("NMI:", nmi)
  return ari, nmi

def calc_ARI_NMI_of_GMMcluster_c(X, y, K=1, stride=1, clip_seconds=10, chunk_ms=40, fps=1000, random_state=42, n_components_GMM=2):
  model, opt2, scaler2, start_ep = load_for_resume_and_infer(VideoMAE, "artifacts_lvmae_1/checkpoint.pt")
  return calc_ARI_NMI_of_GMMcluster_model(model, X, y, K=K, stride=stride, clip_seconds=clip_seconds, chunk_ms=chunk_ms, fps=fps,
                                 random_state=random_state, n_components_GMM=n_components_GMM)
