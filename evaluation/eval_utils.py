import numpy as np
import os
import sys

# Append the parent directory of the current file to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.formatstranslator import make_x_per_category
from evaluation.eval import calc_accumulated_predictions, plot_nice_roc_curve, generate_confusion_matrix_image,  evaluate_model, find_optimal_threshold, flatten_accumulated


'''
Helpers:
'''
  
def aggregate_per_category(config, predicted_per_category, K=25):
    """Aggregate per-chunk probs into K-chunk means per category, then flatten."""
    y_pred_reduced = [calc_accumulated_predictions_n(config, y_pred, K)
                      for y_pred in predicted_per_category]
    y_pred_reduced, y_true = flatten_accumulated(np.array(y_pred_reduced), config.binary_lables)
    return np.asarray(y_pred_reduced).squeeze(), np.asarray(y_true).squeeze()

def compute_val_threshold_from_aggregates(config, predicted_val, K=25):
    """Compute threshold on VALIDATION aggregated probs using Youden's J."""
    y_pred_val_agg, y_val_agg = aggregate_per_category(config, predicted_val, K)
    thr_val = find_optimal_threshold(y_val_agg, y_pred_val_agg)
    auc_val = roc_auc_score(y_val_agg, y_pred_val_agg)
    return thr_val, auc_val

def eval_accumulated_inner_th(config, predicted, x_test_per_category=None, num_of_chunks_to_aggregate=25,
                           fixed_threshold=None):
    # aggregate TEST
    y_pred_test_agg, y_test_agg = aggregate_per_category(config, predicted, num_of_chunks_to_aggregate)
    auc = roc_auc_score(y_test_agg, y_pred_test_agg)

    if fixed_threshold is None:
        threshold = find_optimal_threshold(y_test_agg, y_pred_test_agg)  # old behavior
    else:
        threshold = float(fixed_threshold)  # <-- use VAL-derived threshold

    if config.verbose:
        print(f' auc = {auc}, on {len(y_test_agg)} values from 2 categories')
        print(np.shape(y_pred_test_agg))
        print(np.shape(y_test_agg))
        print(f'Using threshold={threshold:.4f} ({"VAL" if fixed_threshold is not None else "TEST"}-derived)')

    results_df_a = evaluate_model(config, y_pred_test_agg, y_test_agg, threshold, filename='aggregated')
    y_pred_arr = np.array(y_pred_test_agg)
    generate_confusion_matrix_image(y_pred_arr, y_test_agg, threshold, show=False, save_path='confusion_matrix_agg.png')
    plot_nice_roc_curve(y_test_agg, y_pred_arr, show=True, save_path='roc_curve_agg.png')
    return results_df_a

'''
Interface:
'''
def eval_aggregated_test_set_th_on_val(Z_test, prob_test, y_test, prob_val, y_val,
                                       num_of_chunks_to_aggregate=25):
  # Build per-category lists for VAL and TEST
  predicted_val  = make_x_per_category(prob_val,  y_val,  class_order=(0, 1))
  predicted_test = make_x_per_category(prob_test, y_test, class_order=(0, 1))
  x_test_per_category = make_x_per_category(Z_test, y_test, class_order=(0, 1))  # still ok to pass/not used

  cfgT = Configuration_Minimal(1, total_number_of_splits = 10, 
                             model_name = "lv_mae_agg25", verbose = True)  # must provide cfg.binary_lables 
  # Threshold from VALIDATION after aggregation
  thr_val, auc_val_agg = compute_val_threshold_from_aggregates(cfgT, predicted_val, K=num_of_chunks_to_aggregate)
  # guard against degenerate perfect separation
  hi = np.max(predicted_val)
  if np.isclose(thr_val, hi) or thr_val >= hi:
    thr_val = hi - 1e-6

  print(f"[VAL-agg] AUC={auc_val_agg:.4f} | thr_val={thr_val:.4f}")
  print("VAL: min/max", float(np.min(prob_val)), float(np.max(prob_val)))
  print("TEST: min/max", float(np.min(prob_test)), float(np.max(prob_test)))

  # Evaluate TEST using that fixed validation threshold
  results_agg = eval_accumulated_inner_th(cfgT, predicted_test, x_test_per_category,
                                      num_of_chunks_to_aggregate=num_of_chunks_to_aggregate,
                                      fixed_threshold=thr_val)
  print(f"Aggregated (K={num_of_chunks_to_aggregate}) results (TEST, using VAL threshold):")
  print(results_agg)



def eval_aggregated_th_on_target_subj_firstK_chunks(
    Z_test, prob_test, y_test,
    K_thr=1500, # 1 min = number of chunks used to COMPUTE threshold (per class)
    num_of_chunks_to_aggregate=25 # number of chunks used to AGGREGATE at evaluation
):
    # --------------------------------
    # Build per-category lists (TEST)
    # --------------------------------
    predicted_test = make_x_per_category(prob_test, y_test, class_order=(0, 1))
    x_test_per_category = make_x_per_category(Z_test,  y_test, class_order=(0, 1))
    cfgT = Configuration_Minimal(1, total_number_of_splits = 10, 
                             model_name = "lv_mae_agg25", verbose = True)

    # Convert to arrays once (avoids repeated np.asarray calls)
    pred0 = np.asarray(predicted_test[0])
    pred1 = np.asarray(predicted_test[1])
    x0 = np.asarray(x_test_per_category[0])
    x1 = np.asarray(x_test_per_category[1])

    # --------------------------------
    # Threshold from TEST — FIRST K_thr chunks per class
    # --------------------------------
    if len(pred0) < K_thr or len(pred1) < K_thr:
        raise ValueError(
            f"Not enough chunks to compute threshold: "
            f"class0={len(pred0)}, class1={len(pred1)}, required K_thr={K_thr}"
        )

    predicted_test_firstK = [pred0[:K_thr], pred1[:K_thr]]

    thr_test, auc_test_agg_firstK = compute_val_threshold_from_aggregates(
        cfgT,
        predicted_test_firstK,
        K=K_thr
    )

    # Guard against degenerate perfect separation
    hi = float(np.max(np.concatenate(predicted_test_firstK)))
    if np.isclose(thr_test, hi) or thr_test >= hi:
        thr_test = hi - 1e-6

    print(f"[TEST-firstK] AUC={auc_test_agg_firstK:.4f} | thr_test={thr_test:.4f} | K_thr={K_thr}")
    print("TEST: min/max", float(np.min(prob_test)), float(np.max(prob_test)))

    # --------------------------------
    # Evaluate REMAINDER using FIXED threshold
    # (slice predictions AND x consistently)
    # --------------------------------
    predicted_test_remaining = [pred0[K_thr:], pred1[K_thr:]]
    x_test_remaining         = [x0[K_thr:],   x1[K_thr:]]

    # Optional guard: make sure we can actually aggregate
    rem0, rem1 = len(predicted_test_remaining[0]), len(predicted_test_remaining[1])
    if rem0 < num_of_chunks_to_aggregate or rem1 < num_of_chunks_to_aggregate:
        raise ValueError(
            f"Not enough remaining chunks to aggregate: "
            f"class0_rem={rem0}, class1_rem={rem1}, "
            f"required num_of_chunks_to_aggregate={num_of_chunks_to_aggregate}. "
            f"Reduce K_thr or reduce num_of_chunks_to_aggregate."
        )

    results_agg = eval_accumulated_inner_th(
        cfgT,
        predicted_test_remaining,
        x_test_remaining,
        num_of_chunks_to_aggregate=num_of_chunks_to_aggregate,
        fixed_threshold=thr_test,
    )

    print(
        f"Aggregated TEST results (evaluated on remaining chunks): "
        f"aggregation K={num_of_chunks_to_aggregate}, threshold from first K_thr={K_thr}"
    )
    print(results_agg)

    return thr_test, results_agg

