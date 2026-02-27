import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, accuracy_score, roc_curve)
from xgboost.callback import EarlyStopping

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils.embeddings_utils import concat_temporal_embeddings

''' AUC-ROC curve Presentation '''
def _closest_idx(thr, target):
      # choose index with minimal |thr - target|
      return int(np.argmin(np.abs(thr - target)))

def plot_AUC(proba_val, y_val, val_auc, best_thr, proba_test, y_test, test_auc):
  fpr_val,  tpr_val,  thr_val  = roc_curve(y_val,  proba_val)
  fpr_test, tpr_test, thr_test = roc_curve(y_test, proba_test)

  i_val  = _closest_idx(thr_val,  best_thr)
  i_test = _closest_idx(thr_test, best_thr)

  plt.figure(figsize=(7,6))
  plt.plot(fpr_val,  tpr_val,  label=f"VAL  ROC (AUC={val_auc:.3f})")
  plt.plot(fpr_test, tpr_test, label=f"TEST ROC (AUC={test_auc:.3f})")
  plt.plot([0,1], [0,1], linestyle="--", linewidth=1, label="Chance")
  # Mark operating point (threshold picked on VAL)
  plt.scatter(fpr_val[i_val],  tpr_val[i_val],  s=50, marker='o', label=f"VAL @ thr={best_thr:.2f}")
  plt.scatter(fpr_test[i_test], tpr_test[i_test], s=50, marker='x', label=f"TEST @ thr={best_thr:.2f}")

  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC Curves (Validation & Test)")
  plt.grid(True, alpha=0.3)
  plt.legend(loc="lower right", fontsize=14)
  plt.tight_layout()
  plt.show()
  plt.tight_layout()
  plt.show()

def train_eval_xgboost_classifier(Z_train, y_train, Z_val, y_val, Z_test,
                                  y_test, seed=9, K = 1, show = True,
                                  class_names_list = ["class_0", "class_1"]):
  # ---- Temporal aggregation (set K as you like) ----
  # e.g., concatenate k consecutive embeddings; K=1 keeps original behavior
  Z_train_c, y_train_c = concat_temporal_embeddings(Z_train, y_train, K)
  Z_val_c,   y_val_c   = concat_temporal_embeddings(Z_val,   y_val,   K)
  Z_test_c,  y_test_c  = concat_temporal_embeddings(Z_test,  y_test,  K)

  print(f"After temporal concat (K={K}): "
        f"train {Z_train_c.shape}, val {Z_val_c.shape}, test {Z_test_c.shape}")

  print("==> Training XGBoost on embeddings (no callbacks, no early stopping) ...")
  clf = xgb.XGBClassifier(random_state=seed, eval_metric='auc' )
  clf.fit(Z_train_c, y_train_c, verbose=False) # Train on TRAIN only

  # ---- Evaluate on VAL to pick a threshold (optional but helpful) ----
  classes = list(clf.classes_)            # e.g., [1, 0] or [0, 1]
  pos_col = classes.index(1)              # column for label "1" (your positive)
  proba_val  = clf.predict_proba(Z_val_c)[:, pos_col]
  proba_test = clf.predict_proba(Z_test_c)[:, pos_col]
  print("clf.classes_:", list(clf.classes_), "| pos_col:", pos_col)

  print("clf.classes_:", list(clf.classes_))
  print("Chosen pos_col:", pos_col)
  print("VAL prob quantiles:", np.quantile(proba_val, [0, .25, .5, .75, .9, .99]))

  # Use Youden's J to choose threshold on val (no callbacks, purely post-hoc)
  fpr, tpr, thr = roc_curve(y_val_c, proba_val)
  best_idx = np.argmax(tpr - fpr)
  best_thr = thr[best_idx] if best_idx < len(thr) else 0.5  # safe fallback

  val_auc = roc_auc_score(y_val_c, proba_val)
  val_pred = (proba_val >= best_thr).astype(int)
  val_acc = accuracy_score(y_val_c, val_pred)
  print(f"[XGB] VAL: AUC={val_auc:.4f} | ACC={val_acc:.4f} | thr={best_thr:.3f}")

  # ---- Evaluate on TEST with the chosen threshold ----
  y_pred = (proba_test >= best_thr).astype(int)

  test_auc = roc_auc_score(y_test_c, proba_test)
  test_acc = accuracy_score(y_test_c, y_pred)
  if show:
    print(f"[XGB] TEST: AUC={test_auc:.4f} | ACC={test_acc:.4f}")

    # Binary report
    print(classification_report(y_test_c, y_pred, target_names=class_names_list))

    # Confusion matrix
    cm = confusion_matrix(y_test_c, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_list)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.show()

    plot_AUC(proba_val, y_val_c, val_auc, best_thr, proba_test, y_test_c, test_auc)
  return clf, best_thr, val_auc, test_auc, proba_val, proba_test, y_test_c
