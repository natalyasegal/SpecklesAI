import numpy as np
import os
import sys

# Append the parent directory of the current file to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.formatstranslator import make_x_per_category

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
