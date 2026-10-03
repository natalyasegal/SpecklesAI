import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

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

#lables used only for metrics: ARI and NMI
def calc_ARI_NMI_of_GMMcluster_c(X, y, K=1, stride=1, clip_seconds=10, chunk_ms=40, fps=1000,
                                 random_state=42, n_components_GMM=2):
  model, opt2, scaler2, start_ep = load_for_resume_and_infer(VideoMAE, "artifacts_lvmae_1/checkpoint.pt")
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
