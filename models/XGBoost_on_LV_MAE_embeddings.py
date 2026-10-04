import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, accuracy_score, roc_curve, f1_score, balanced_accuracy_score)
from xgboost.callback import EarlyStopping

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from models.LvMAE_pt import load_for_resume_and_infer, extract_embeddings_wrapper_one
from models.LvMAE_pt import *
from models.binary_XGBoost import train_eval_xgboost_classifier, train_eval_xgboost_classifier_after_concatenation
from models.multiclass_XGBoost import train_eval_xgboost_classifier_multiclass, train_eval_xgb_train_api_multiclass_opt_th, train_eval_xgboost_classifier_multiclass_opt_th
from utils.concat import concatenate_train_or_val
from utils.data import split_from_start, split_by_chunks_v
from utils.formatstranslator import test2trainformat
from utils.embeddings_utils import concat_temporal_embeddings_c_stride, concat_temporal_embeddings_c, make_clip_ids_from_fps, concat_temporal_embeddings
from evaluation.eval_utils import eval_aggregated_test_set_th_on_val


'''
Concatenatation (is relevant, so for K > 1) respects clip boundaries and supports stride
stride = K => independent segments
'''
def train_and_eval_classifier_on_embeddings_agg(test_inp, train_n=250, val_n=250, fps=1000,
                                            K=1, stride=1, clip_seconds=10, chunk_ms=40, 
                                            class_names_list=None, seed=9, show=True):

  model, opt2, scaler2, start_ep = load_for_resume_and_infer(VideoMAE, "artifacts_lvmae_1/checkpoint.pt")
  X_train, X_val, X_test, y_train, y_val, y_test = split_from_start(test_inp, train_n=train_n, val_n=val_n)
  # train_n and val_n are numbers of 40 ms chunks used for train/val, 250 chunks -> 10 s per class

  Z_train, y_train = extract_embeddings_wrapper_one(model, X_train, y_train)
  Z_val, y_val  = extract_embeddings_wrapper_one(model, X_val, y_val)
  Z_test, y_test = extract_embeddings_wrapper_one(model, X_test, y_test)

  '''
  Temporal aggregation: concatenate k consecutive embeddings; K=1 keeps original behavior
  clip_ids built PER SPLIT from that split's own length, so each is length-aligned with its Z and never crosses clip/class boundaries.
  '''
  mk = lambda Z: make_clip_ids_from_fps({0: len(Z)},fps=fps, chunk_ms=chunk_ms, clip_seconds=clip_seconds)
  if stride == 1:
    Z_train_c, y_train_c = concat_temporal_embeddings_c(Z_train, y_train, mk(Z_train), K)
    Z_val_c,   y_val_c   = concat_temporal_embeddings_c(Z_val,   y_val,   mk(Z_val),   K)
    Z_test_c,  y_test_c  = concat_temporal_embeddings_c(Z_test,  y_test,  mk(Z_test),  K)
  elif stride == K:
    Z_train_c,y_train_c=concat_temporal_embeddings_c_stride(Z_train,y_train,mk(Z_train),K,stride=K)
    Z_val_c,  y_val_c  =concat_temporal_embeddings_c_stride(Z_val,  y_val,  mk(Z_val),K,stride=K)
    Z_test_c, y_test_c =concat_temporal_embeddings_c_stride(Z_test, y_test, mk(Z_test),K,stride=K)
  else:
    assert False, "not implemented for this stride"

  return train_eval_xgboost_classifier_after_concatenation(Z_train_c,y_train_c,Z_val_c,y_val_c,Z_test_c,y_test_c,
                                seed=seed, K=K, show=show, class_names_list=class_names_list)

def train_and_eval_classifier_on_embeddings(test_inp, train_n=250, val_n=250, 
                                            K = 1, class_names_list = None):

  model, opt2, scaler2, start_ep = load_for_resume_and_infer(VideoMAE, "artifacts_lvmae_1/checkpoint.pt")
  X_train, X_val, X_test, y_train, y_val, y_test = split_from_start(test_inp, train_n=train_n, val_n=val_n)
  # train_n and val_n are numbers of 40 ms chunks used for train/val, 250 chunks -> 10 s per class

  Z_train, y_train = extract_embeddings_wrapper_one(model, X_train, y_train)
  Z_val, y_val  = extract_embeddings_wrapper_one(model, X_val, y_val)
  Z_test, y_test = extract_embeddings_wrapper_one(model, X_test, y_test)

  return train_eval_xgboost_classifier(Z_train,y_train,Z_val,y_val,Z_test,y_test,
                                       K = K, class_names_list = class_names_list)

def train_and_eval_multiclass_classifier_on_embeddings_agg(inp_data, train_n=250, val_n=250,
                                          fps=1000, K=1, stride=1, clip_seconds=10, chunk_ms=40,
                                          class_names_list=None, seed=9, cmap='viridis', show=True):
  model, opt2, scaler2, start_ep = load_for_resume_and_infer(VideoMAE, "artifacts_lvmae_1/checkpoint.pt")
  X_train, X_val, X_test, y_train, y_val, y_test = split_from_start(inp_data,train_n=train_n, val_n=val_n)
  y_train = np.argmax(y_train, axis=1)
  y_val   = np.argmax(y_val,   axis=1)
  y_test  = np.argmax(y_test,  axis=1)

  print("Train:", X_train.shape, y_train.shape)
  print("Val:",   X_val.shape,   y_val.shape)
  print("Test:",  X_test.shape,  y_test.shape)

  print("Train labels:", np.unique(y_train, return_counts=True))
  print("Val labels:  ", np.unique(y_val,   return_counts=True))
  print("Test labels: ", np.unique(y_test,  return_counts=True))

  Z_train, y_train = extract_embeddings_wrapper_one(model, X_train, y_train)
  Z_val, y_val  = extract_embeddings_wrapper_one(model, X_val, y_val)
  Z_test, y_test = extract_embeddings_wrapper_one(model, X_test, y_test)

  mk = lambda Z: make_clip_ids_from_fps({0: len(Z)},fps=fps, chunk_ms=chunk_ms, clip_seconds=clip_seconds)

  # Temporal concat
  Z_train_c,y_train_c=concat_temporal_embeddings_c_stride(Z_train,y_train,mk(Z_train),K,stride=stride)
  Z_val_c,  y_val_c  =concat_temporal_embeddings_c_stride(Z_val,  y_val,  mk(Z_val),K,stride=stride)
  Z_test_c, y_test_c =concat_temporal_embeddings_c_stride(Z_test, y_test, mk(Z_test),K,stride=stride)
  print(f"After temporal concat (K={K}): train {Z_train_c.shape}, val {Z_val_c.shape}, test {Z_test_c.shape}")

  booster,val_auc,test_auc,val_acc,test_acc,proba_val,proba_test,ypt,ypv,cm = \
        train_eval_xgb_train_api_multiclass_opt_th(Z_train_c, y_train_c, Z_val_c, y_val_c, Z_test_c, y_test_c, seed=seed, class_names=class_names_list, show=show, cmap=cmap)
  test_macro_f1 = round(f1_score(y_test_c, ypt, average='macro'), 4)
  macro_acc = round(balanced_accuracy_score(y_test_c, ypt) * 100.0, 4)
  print("Macro accuracy (balanced):", macro_acc)
  print("Macro F1:", test_macro_f1)                                           
  return booster, val_auc, test_auc, val_acc, test_acc, proba_val, proba_test, ypt, ypv, Z_test_c, y_test_c, Z_val_c, y_val_c, test_macro_f1, cm

def train_and_eval_multiclass_classifier_on_embeddings(inp_data, train_n=250, val_n=250,
                                        K = 1, class_names_list = None, cmap='viridis', show=True):
  model, opt2, scaler2, start_ep = load_for_resume_and_infer(VideoMAE, "artifacts_lvmae_1/checkpoint.pt")
  X_train, X_val, X_test, y_train, y_val, y_test = split_from_start(inp_data,train_n=train_n, val_n=val_n)
  y_train = np.argmax(y_train, axis=1)
  y_val   = np.argmax(y_val,   axis=1)
  y_test  = np.argmax(y_test,  axis=1)

  print("Train:", X_train.shape, y_train.shape)
  print("Val:",   X_val.shape,   y_val.shape)
  print("Test:",  X_test.shape,  y_test.shape)

  print("Train labels:", np.unique(y_train, return_counts=True))
  print("Val labels:  ", np.unique(y_val,   return_counts=True))
  print("Test labels: ", np.unique(y_test,  return_counts=True))

  Z_train, y_train = extract_embeddings_wrapper_one(model, X_train, y_train)
  Z_val, y_val  = extract_embeddings_wrapper_one(model, X_val, y_val)
  Z_test, y_test = extract_embeddings_wrapper_one(model, X_test, y_test)

  return train_eval_xgboost_classifier_multiclass(Z_train,y_train,Z_val,y_val,Z_test,y_test,K=K, class_names=class_names_list,cmap=cmap,show=show)
                                            
'''
Used for Gen table, paper BCI yes vs. no, green
'''
def TestGen_ValHeldoutFromUnseen(train_x_list, train_y_list, unseen, K_thr = 500, num_of_chunks_to_aggregate = 25, k = 1):
  model, opt2, scaler2, start_ep = load_for_resume_and_infer(VideoMAE, "artifacts_lvmae_1/checkpoint.pt")
  X_train, y_train =  concatenate_train_or_val(train_x_list, train_y_list)
  val, test = split_by_chunks_v(unseen, val_n = K_thr)
  X_val, y_val = test2trainformat(val, need_to_shuffle_within_category = False)
  X_test, y_test = test2trainformat(test, need_to_shuffle_within_category = False)

  Z_train, y_train = extract_embeddings_wrapper_one(model, X_train, y_train)
  Z_val, y_val,  = extract_embeddings_wrapper_one(model, X_val, y_val)
  Z_test, y_test = extract_embeddings_wrapper_one(model, X_test, y_test)

  clf,_,_, _,prob_val,prob_test,y_c,cm=train_eval_xgboost_classifier(Z_train,y_train,
                                              Z_val,y_val, Z_test,y_test,
                                              K = k, show=True) # 128w and 128
  #print(f'Aggregated k={num_of_chunks_to_aggregate}: =================================')
  eval_aggregated_test_set_th_on_val(Z_test, prob_test, y_test, prob_val, y_val,
                                       num_of_chunks_to_aggregate= num_of_chunks_to_aggregate)

def TestGen_ValHeldoutFromUnseen_multiclass(train_x_list, train_y_list, unseen, K_thr = 500, num_of_chunks_to_aggregate = 25, k = 1, class_names=["1", "2", "3", "4", "5"]):
  model, opt2, scaler2, start_ep = load_for_resume_and_infer(VideoMAE, "artifacts_lvmae_1/checkpoint.pt")
  X_train, y_train =  concatenate_train_or_val(train_x_list, train_y_list)
  val, test = split_by_chunks_v(unseen, val_n = K_thr)
  X_val, y_val = test2trainformat(val, need_to_shuffle_within_category = False)
  X_test, y_test = test2trainformat(test, need_to_shuffle_within_category = False)

  y_train = np.argmax(y_train, axis=1)
  y_val   = np.argmax(y_val,   axis=1)
  y_test  = np.argmax(y_test,  axis=1)

  print("Train:", X_train.shape, y_train.shape)
  print("Val:",   X_val.shape,   y_val.shape)
  print("Test:",  X_test.shape,  y_test.shape)

  print("Train labels:", np.unique(y_train, return_counts=True))
  print("Val labels:  ", np.unique(y_val,   return_counts=True))
  print("Test labels: ", np.unique(y_test,  return_counts=True))

  Z_train, y_train = extract_embeddings_wrapper_one(model, X_train, y_train)
  Z_val, y_val,  = extract_embeddings_wrapper_one(model, X_val, y_val)
  Z_test, y_test = extract_embeddings_wrapper_one(model, X_test, y_test)

  clf,val_auc,test_auc,val_acc,test_acc,proba_val,proba_test,ypt,ypv,Z_test_c,y_test_c,Z_val_c,y_val_c,test_macro_f1,cm=train_eval_xgboost_classifier_multiclass_opt_th(Z_train,y_train,
                                              Z_val,y_val, Z_test,y_test, K = k, show=True, class_names=class_names) # 128w and 128
  #print(f'Aggregated k={num_of_chunks_to_aggregate}: =================================')
  #eval_aggregated_test_set_th_on_val(Z_test, prob_test, y_test, prob_val, y_val, num_of_chunks_to_aggregate= num_of_chunks_to_aggregate)
  return clf,val_auc,test_auc,val_acc,test_acc,proba_val,proba_test,ypt,ypv,Z_test_c,y_test_c,Z_val_c,y_val_c,test_macro_f1,cm


