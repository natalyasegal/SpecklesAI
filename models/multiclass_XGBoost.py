from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import xgboost as xgb 
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils.embeddings_utils import concat_temporal_embeddings

def plot_multiclass_roc_ovr(proba_val, y_val, proba_test, y_test, class_names=None, title="ROC (OvR)"):
    """
    proba_*: (N, C)
    y_*: (N,) integer labels 0..C-1
    """
    y_val = np.asarray(y_val).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    C = proba_val.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]

    y_val_oh  = label_binarize(y_val,  classes=np.arange(C))
    y_test_oh = label_binarize(y_test, classes=np.arange(C))

    plt.figure(figsize=(7,6))

    aucs_val, aucs_test = [], []
    for c in range(C):
        fpr_v, tpr_v, _ = roc_curve(y_val_oh[:, c],  proba_val[:, c])
        fpr_t, tpr_t, _ = roc_curve(y_test_oh[:, c], proba_test[:, c])

        auc_v = roc_auc_score(y_val_oh[:, c],  proba_val[:, c])
        auc_t = roc_auc_score(y_test_oh[:, c], proba_test[:, c])

        aucs_val.append(auc_v); aucs_test.append(auc_t)

        plt.plot(fpr_v, tpr_v, label=f"VAL  {class_names[c]} (AUC={auc_v:.3f})", alpha=0.8)
        plt.plot(fpr_t, tpr_t, label=f"TEST {class_names[c]} (AUC={auc_t:.3f})", alpha=0.8, linestyle="--")

    plt.plot([0,1], [0,1], linestyle="--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=14)
    plt.tight_layout()
    plt.show()

    # macro summary (optional)
    print(f"Macro-AUC VAL (mean per class):  {np.mean(aucs_val):.4f}")
    print(f"Macro-AUC TEST (mean per class): {np.mean(aucs_test):.4f}")


def train_eval_xgb_train_api_multiclass(
    Z_train_c, y_train_c, Z_val_c, y_val_c, Z_test_c, y_test_c,
    seed=9, num_boost_round=5000, early_stopping_rounds=200
):
    y_train_c = np.asarray(y_train_c).reshape(-1).astype(int)
    y_val_c   = np.asarray(y_val_c).reshape(-1).astype(int)
    y_test_c  = np.asarray(y_test_c).reshape(-1).astype(int)

    n_classes = int(np.max(y_train_c)) + 1

    dtrain = xgb.DMatrix(Z_train_c, label=y_train_c)
    dval   = xgb.DMatrix(Z_val_c,   label=y_val_c)
    dtest  = xgb.DMatrix(Z_test_c,  label=y_test_c)

    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": ["mlogloss", "merror"],  # multiclass-friendly
        "seed": seed,

        "eta": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 2.0,
        "alpha": 0.0,
        "gamma": 0.0,

        "tree_method": "hist",   # or "gpu_hist"
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50
    )

    proba_val  = booster.predict(dval,  iteration_range=(0, booster.best_iteration + 1))  # (N_val, C)
    proba_test = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))  # (N_test, C)

    y_pred_val  = np.argmax(proba_val, axis=1)
    y_pred_test = np.argmax(proba_test, axis=1)

    val_acc  = accuracy_score(y_val_c,  y_pred_val)
    test_acc = accuracy_score(y_test_c, y_pred_test)

    # Multiclass AUC (OvR macro) — requires proba matrix
    val_auc  = roc_auc_score(y_val_c,  proba_val,  multi_class="ovr", average="macro")
    test_auc = roc_auc_score(y_test_c, proba_test, multi_class="ovr", average="macro")

    return booster, val_auc, test_auc, val_acc, test_acc, proba_val, proba_test, y_pred_val, y_pred_test

''' Helper function from DR repository'''
def display_multiclass_cm_with_percents(cm, class_names, cmap='viridis'):
  '''
  cmap="Blues" is a good alternative
  '''
  n_classes = cm.shape[0]
  if class_names is None:
      class_names = [f"class_{i}" for i in range(n_classes)]

  cm_pct = cm / cm.sum(axis=1, keepdims=True)

  disp = ConfusionMatrixDisplay(confusion_matrix=cm_pct, display_labels=class_names)
  disp.plot(xticks_rotation=45, include_values=False, cmap=cmap)

  high_color = "black"
  low_color = "white"
  if "Blues" in cmap:
      high_color = "white"
      low_color = "black"
  elif "viridis_r" in cmap:
      high_color = "purple"
      low_color = "yellow"

  for i in range(n_classes):
      for j in range(n_classes):
          pct = cm_pct[i, j] * 100
          color = low_color if cm_pct[i, j] < 0.5 else high_color
          disp.ax_.text(j, i, f"{cm[i, j]}\n{pct:.1f}%", ha="center", va="center",
                        color=color, fontsize=11)

def get_multiclass_cm_with_percents(proba_test, y_test_c, ypt):
  n_classes = proba_test.shape[1]
  cm = confusion_matrix(y_test_c, ypt, labels=list(range(n_classes)))
  return cm

def train_eval_xgboost_classifier_multiclass( Z_train,y_train,Z_val,y_val,
    Z_test,y_test,seed=9, K=1, show=True, class_names=None, cmap='viridis'):
  
    # Temporal concat
    Z_train_c, y_train_c = concat_temporal_embeddings(Z_train, y_train, K)
    Z_val_c,   y_val_c   = concat_temporal_embeddings(Z_val,   y_val,   K)
    Z_test_c,  y_test_c  = concat_temporal_embeddings(Z_test,  y_test,  K)
    print(f"After temporal concat (K={K}): train {Z_train_c.shape}, val {Z_val_c.shape}, test {Z_test_c.shape}")

    booster, val_auc, test_auc, val_acc, test_acc, proba_val, proba_test, ypv, ypt = \
        train_eval_xgb_train_api_multiclass(Z_train_c, y_train_c, Z_val_c, y_val_c, Z_test_c, y_test_c, seed=seed)
    test_macro_f1 = f1_score(y_test_c, ypt, average='macro')



def optimize_multiclass_offsets(
    proba_val,
    y_val,
    metric="accuracy",
    seed=9,
    bound=1.0 #3.0,
):
    """
    Optimize class-specific decision offsets on the validation set.

    Final prediction:
        argmax(log(P(class | x)) + class_offset)

    The final class offset is fixed at zero because only the relative
    differences between offsets affect the argmax decision.

    Parameters
    ----------
    proba_val : np.ndarray, shape (N, C)
        Validation probabilities returned by XGBoost.

    y_val : np.ndarray, shape (N,)
        Validation labels encoded as 0, ..., C-1.

    metric : str
        "accuracy" optimizes overall validation accuracy.
        "macro_f1" optimizes validation macro-F1.

    seed : int
        Random seed used by differential evolution.

    bound : float
        Each free offset is searched within [-bound, bound].

    Returns
    -------
    offsets : np.ndarray, shape (C,)
        Optimized class-specific offsets.

    y_pred_val : np.ndarray, shape (N,)
        Validation predictions after applying the offsets.
    """
    from scipy.optimize import differential_evolution

    proba_val = np.asarray(proba_val, dtype=np.float64)
    y_val = np.asarray(y_val).reshape(-1).astype(int)

    if proba_val.ndim != 2:
        raise ValueError(
            f"proba_val must have shape (N, C), got {proba_val.shape}"
        )

    if proba_val.shape[0] != y_val.shape[0]:
        raise ValueError(
            "proba_val and y_val contain different numbers of samples: "
            f"{proba_val.shape[0]} and {y_val.shape[0]}"
        )

    n_classes = proba_val.shape[1]

    if n_classes < 2:
        raise ValueError("At least two classes are required.")

    expected_classes = np.arange(n_classes)
    found_classes = np.unique(y_val)

    if not np.array_equal(found_classes, expected_classes):
        raise ValueError(
            "Validation labels must contain every class and be encoded as "
            f"0, ..., {n_classes - 1}; found {found_classes}"
        )

    log_proba_val = np.log(
        np.clip(proba_val, 1e-12, 1.0)
    )

    def make_offsets(free_offsets):
        offsets = np.asarray(free_offsets, dtype=np.float64)
        return offsets - np.mean(offsets)
    
    def make_offsets_old(free_offsets):
        """
        Fix the last class offset at zero to remove the redundant
        common shift shared by all offsets.
        """
        return np.concatenate(
            [
                np.asarray(free_offsets, dtype=np.float64),
                np.array([0.0], dtype=np.float64),
            ]
        )

    def objective(free_offsets):
        offsets = make_offsets(free_offsets)

        y_pred_val = np.argmax(
            log_proba_val + offsets[None, :],
            axis=1,
        )

        if metric == "accuracy":
            score = accuracy_score(
                y_val,
                y_pred_val,
            )

        elif metric == "macro_f1":
            score = f1_score(
                y_val,
                y_pred_val,
                average="macro",
                zero_division=0,
            )

        else:
            raise ValueError(
                "metric must be either 'accuracy' or 'macro_f1'"
            )

        # scipy minimizes, so return the negative score.
        return -score

    result = differential_evolution(
        objective,
        bounds=[(-bound, bound)] * (n_classes),
        seed=seed,
        maxiter=200,
        popsize=15,
        polish=True,
        workers=1,
    )

    offsets = make_offsets(result.x)

    y_pred_val = np.argmax(
        log_proba_val + offsets[None, :],
        axis=1,
    )

    return offsets, y_pred_val


def train_eval_xgb_train_api_multiclass_opt_th(
    Z_train_c,
    y_train_c,
    Z_val_c,
    y_val_c,
    Z_test_c,
    y_test_c,
    seed=9,
    class_names = None,
    show = True,
    cmap = 'viridis',
    num_boost_round=5000,
    early_stopping_rounds=200,
    offset_metric="macro_f1",#"accuracy",
    offset_bound=1.0 #3.0,
):
    """
    Train the multiclass XGBoost model and optimize class-specific
    decision offsets using validation data.

    offset_metric="accuracy" optimizes overall validation accuracy.
    offset_metric="macro_f1" optimizes validation macro-F1.
    """
    y_train_c = np.asarray(
        y_train_c
    ).reshape(-1).astype(int)

    y_val_c = np.asarray(
        y_val_c
    ).reshape(-1).astype(int)

    y_test_c = np.asarray(
        y_test_c
    ).reshape(-1).astype(int)

    n_classes = int(np.max(y_train_c)) + 1

    dtrain = xgb.DMatrix(
        Z_train_c,
        label=y_train_c,
    )

    dval = xgb.DMatrix(
        Z_val_c,
        label=y_val_c,
    )

    dtest = xgb.DMatrix(
        Z_test_c,
        label=y_test_c,
    )

    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": ["merror", "mlogloss"],
        "seed": seed,
        "eta": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 2.0,
        "alpha": 0.0,
        "gamma": 0.0,
        "tree_method": "hist",  # or "gpu_hist"
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )

    proba_val = booster.predict(
        dval,
        iteration_range=(
            0,
            booster.best_iteration + 1,
        ),
    )  # (N_val, C)

    proba_test = booster.predict(
        dtest,
        iteration_range=(
            0,
            booster.best_iteration + 1,
        ),
    )  # (N_test, C)

    # Optimize class-specific decision offsets using validation data only.
    offsets, y_pred_val = optimize_multiclass_offsets(
        proba_val=proba_val,
        y_val=y_val_c,
        metric=offset_metric,
        seed=seed,
        bound=offset_bound,
    )

    # Apply exactly the same validation-derived offsets to the test set.
    y_pred_test = np.argmax(
        np.log(
            np.clip(
                proba_test,
                1e-12,
                1.0,
            )
        ) + offsets[None, :],
        axis=1,
    )

    val_acc = accuracy_score(
        y_val_c,
        y_pred_val,
    )

    test_acc = accuracy_score(
        y_test_c,
        y_pred_test,
    )

    # AUC is probability-ranking based and therefore remains calculated
    # from the original XGBoost probability matrices.
    val_auc = roc_auc_score(
        y_val_c,
        proba_val,
        multi_class="ovr",
        average="macro",
    )

    test_auc = roc_auc_score(
        y_test_c,
        proba_test,
        multi_class="ovr",
        average="macro",
    )

    # Store the offsets in the returned booster while preserving the
    # original nine-value return signature.
    booster.set_attr(
        multiclass_offsets=",".join(
            f"{offset:.17g}" for offset in offsets
        ),
        multiclass_offset_metric=offset_metric,
    )

    print(
        f"Optimized multiclass offsets "
        f"({offset_metric}):",
        np.round(offsets, 4),
    )

    print(
        "Equivalent probability multipliers:",
        np.round(np.exp(offsets), 4),
    )
       
    print(f"[XGB-MC] VAL : AUC(ovr,macro)={val_auc:.4f} | ACC={val_acc:.4f}")
    #print(f"[XGB-MC] TEST: AUC(ovr,macro)={test_auc:.4f} | ACC={test_acc:.4f}")
    #print(f"[XGB-MC] TEST: AUC(ovr,macro)={test_auc:.4f} | ACC={test_acc:.4f} | Macro-F1={test_macro_f1:.4f}")

    n_classes = proba_test.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    print(classification_report(y_test_c, y_pred_test, target_names=class_names))
    cm = get_multiclass_cm_with_percents(proba_test, y_test_c, y_pred_test)

    if show:
        display_multiclass_cm_with_percents(cm, class_names, cmap=cmap)
        plot_multiclass_roc_ovr(proba_val, y_val_c, proba_test, y_test_c, class_names=class_names)

    print("Offsets:", np.round(offsets, 4))
    print("Validation predicted counts:", np.bincount(y_pred_val, minlength=n_classes),)
    print("Test predicted counts:", np.bincount(y_pred_test, minlength=n_classes),)
    
    return booster,val_auc,test_auc,val_acc,test_acc, proba_val,proba_test, y_test_c,y_pred_test, y_pred_val, cm

def train_eval_xgboost_classifier_multiclass_opt_th( Z_train,y_train,Z_val,y_val,
    Z_test,y_test,seed=9, K=1, show=True, class_names=None, cmap='viridis'):
  
    # Temporal concat
    Z_train_c, y_train_c = concat_temporal_embeddings(Z_train, y_train, K)
    Z_val_c,   y_val_c   = concat_temporal_embeddings(Z_val,   y_val,   K)
    Z_test_c,  y_test_c  = concat_temporal_embeddings(Z_test,  y_test,  K)
    print(f"After temporal concat (K={K}): train {Z_train_c.shape}, val {Z_val_c.shape}, test {Z_test_c.shape}")

    booster,val_auc,test_auc,val_acc,test_acc,proba_val,proba_test,ypt,ypv,cm = \
        train_eval_xgb_train_api_multiclass_opt_th(Z_train_c, y_train_c, Z_val_c, y_val_c, Z_test_c, y_test_c, seed=seed, class_names=class_names, show=show, cmap=cmap)
    test_macro_f1 = f1_score(y_test_c, ypt, average='macro')
    return booster, val_auc, test_auc, val_acc, test_acc, proba_val, proba_test, ypt, ypv, Z_test_c, y_test_c, Z_val_c, y_val_c, test_macro_f1, cm

def train_eval_xgboost_classifier_multiclass_old(
    Z_train, y_train, Z_val, y_val, Z_test, y_test,
    seed=9, K=1, show=True, class_names=None
):
    # Temporal concat
    Z_train_c, y_train_c = concat_temporal_embeddings(Z_train, y_train, K)
    Z_val_c,   y_val_c   = concat_temporal_embeddings(Z_val,   y_val,   K)
    Z_test_c,  y_test_c  = concat_temporal_embeddings(Z_test,  y_test,  K)

    print(f"After temporal concat (K={K}): train {Z_train_c.shape}, val {Z_val_c.shape}, test {Z_test_c.shape}")

    booster, val_auc, test_auc, val_acc, test_acc, proba_val, proba_test, ypv, ypt = \
        train_eval_xgb_train_api_multiclass(Z_train_c, y_train_c, Z_val_c, y_val_c, Z_test_c, y_test_c, seed=seed)

    if show:
        print(f"[XGB-MC] VAL : AUC(ovr,macro)={val_auc:.4f} | ACC={val_acc:.4f}")
        print(f"[XGB-MC] TEST: AUC(ovr,macro)={test_auc:.4f} | ACC={test_acc:.4f}")

        # classification report
        n_classes = proba_test.shape[1]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(n_classes)]
        print(classification_report(y_test_c, ypt, target_names=class_names))

        cm = confusion_matrix(y_test_c, ypt, labels=list(range(n_classes)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(xticks_rotation=45)
        plt.title("Confusion Matrix on Test Set")
        plt.tight_layout()
        plt.show()

        plot_multiclass_roc_ovr(proba_val, y_val_c, proba_test, y_test_c, class_names=class_names)
    return booster, val_auc, test_auc, val_acc, test_acc, proba_val, proba_test, y_test_c
    
