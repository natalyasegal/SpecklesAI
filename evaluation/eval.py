import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, cohen_kappa_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
#from plot_metric.functions import BinaryClassification
from matplotlib import pyplot as plt
  
''' images and plots'''
def generate_confusion_matrix_image(y_test_predicted, y_test, threshold, show, save_path = 'confusion_matrix.png'):
  np.set_printoptions(precision=2)
  titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", "true"),]
  for title, normalize in titles_options:
      disp = ConfusionMatrixDisplay.from_predictions(
          y_test.flatten(), y_test_predicted.flatten() >threshold, cmap=plt.cm.Blues, normalize=normalize)
      disp.ax_.set_title(title)
      print(title)
      print(disp.confusion_matrix)
      plt.savefig(save_path, bbox_inches='tight')
      if show:
        plt.show()

def plot_nice_roc_curve(y_test, y_test_predicted, show, save_path = 'roc_curve.png'):
  # Visualisation with plot_metric
  #bc = BinaryClassification(y_test.flatten(), y_test_predicted.flatten(), labels=["Class 1", "Class 2"])
  #plt.figure(figsize=(5,5))
  #bc.plot_roc_curve()
  
  auc_roc = roc_auc_score(y_test, y_test_predicted)
  fpr, tpr, thresholds = roc_curve(y_test, y_test_predicted)
  plt.figure(figsize=(10, 6))
  plt.plot(fpr, tpr, color='blue', label=f'AUC-ROC (area = {auc_roc:.2f})')
  plt.plot([0, 1], [0, 1], color='red', linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc="lower right")
  plt.grid()

  plt.savefig(save_path, bbox_inches='tight')
  if show:
    plt.show()

''' metrics '''
def find_optimal_threshold_o(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters:
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations

    Returns:   
    list type, with optimal cutoff value  
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']) 

def find_optimal_threshold(target, predicted):
    """
    Find the optimal probability cutoff point for a classification model related to event rate.
    
    Parameters:
    target : np.ndarray or list : True binary labels (0s and 1s), where rows are observations.
    predicted : np.ndarray or list : Predicted probabilities for the positive class, where rows are observations.
    
    Returns:
    float: Optimal cutoff value for the probability threshold.
    """
    # Ensure inputs are numpy arrays
    target = np.asarray(target)
    predicted = np.asarray(predicted)
    
    # Flatten arrays if needed
    target = target.ravel()
    predicted = predicted.ravel()
    
    # Validate inputs
    if len(target) != len(predicted):
        raise ValueError(
            f"Inconsistent lengths: target has {len(target)} samples, "
            f"but predicted has {len(predicted)} samples."
        )
    if target.ndim != 1 or predicted.ndim != 1:
        raise ValueError(
            f"Inputs must be 1-dimensional. Got target.ndim={target.ndim}, predicted.ndim={predicted.ndim}."
        )
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("Target must contain binary labels (0 and 1).")
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(target, predicted)
    
    # Calculate the optimal threshold
    # Use Youden's J statistic: J = sensitivity + specificity - 1
    # sensitivity = tpr, specificity = 1 - fpr
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def evaluate_model(config, predictions, true_labels, th, filename = 'per_chunk'):
  # Calculate metrics
  auc = roc_auc_score(true_labels, predictions, multi_class='ovo', average='macro')
  if th is not None:
    accuracy = accuracy_score(true_labels, predictions > th)
    f1 = f1_score(true_labels, predictions > th) #use a default binary averaging option
    kappa = cohen_kappa_score(true_labels, predictions> th)

    model_name = f'{config.model_name}_split_{config.split_num}'
    # Create DataFrame
    data = {
              'Model Name': [model_name],
              'AUC': [auc],
              'Accuracy': [accuracy],
              'F1 Score': [f1],
              'Cohen Kappa': [kappa],
              'Threshold': [th]
    }
  else:
    data = {
              'Model Name': [model_name],
              'AUC': [auc]
    }
  results_df = pd.DataFrame(data)

  # Save to CSV
  results_df.to_csv(f'{config.model_name}_{filename}.csv', index=False)
  print(f'Results saved to {config.model_name}_{filename}.csv')
  return results_df

def evaluate_per_chunk(config, model, x_test, y_test, show = True):
  y_test_predicted = model.predict(x_test)
  print("y_test shape:", y_test.shape)
  
  y_test_predicted = np.array(y_test_predicted)
  print("y_test_predicted shape:", y_test_predicted.shape) 
  
  # Find optimal probability threshold
  threshold = find_optimal_threshold(y_test, y_test_predicted)
  print(f'The chosen(using AUC-ROC curve) optimal threshold is {threshold}')
  results_df = evaluate_model(config, y_test_predicted, y_test, threshold)
  generate_confusion_matrix_image(y_test_predicted, y_test, threshold, show, save_path = 'confusion_matrix.png')
  plot_nice_roc_curve(y_test, y_test_predicted, show, save_path = 'roc_curve.png')
  print(results_df)
  return results_df
  
def evaluate_per_chunk_e(config, models, x_test, y_test, show = True):
  assert(len(models) > 0)
  y_test_predicted_list = [np.array(model.predict(x_test)) for model in models]
  num_models = len(y_test_predicted_list)
  print(f"Evaluating {num_models} models...")

  # Initialize binary predictions for ensemble
  bin_predictions = np.zeros_like(y_test, dtype=float)
  print(f"y_test shape: {y_test.shape}")

  # Find optimal threshold and aggregate predictions
  for i, predictions in enumerate(y_test_predicted_list):
        th = find_optimal_threshold(y_test, predictions)
        if show:
            print(f"Model {i + 1}: Optimal threshold (using AUC-ROC curve) = {th:.4f}")
        bin_predictions += (predictions > th)

  # Average the predictions across all models
  bin_predictions /= num_models  # Ensemble averaging

  # Calculate evaluation metrics
  auc = roc_auc_score(y_test, bin_predictions, multi_class='ovo', average='macro')
  accuracy = accuracy_score(y_test, (bin_predictions > 0.5))
  f1 = f1_score(y_test, (bin_predictions > 0.5), average='binary')
  kappa = cohen_kappa_score(y_test, (bin_predictions > 0.5))

  model_name = f'{"ensemble"}'
  # Create DataFrame
  data = {
              'Model Name': [model_name],
              'AUC': [auc],
              'Accuracy': [accuracy],
              'F1 Score': [f1],
              'Cohen Kappa': [kappa],
              'Threshold': [th]
    }
  results_df = pd.DataFrame(data)

  # Save to CSV
  csv_name = f'{model_name}_of_{num_models}_models.csv'
  results_df.to_csv(csv_name, index=False)
  print(f'Results saved to {csv_name}')
  
  generate_confusion_matrix_image(bin_predictions, y_test, 0.5, show, save_path = 'confusion_matrix.png')
  plot_nice_roc_curve(y_test, bin_predictions, show, save_path = 'roc_curve.png')
  print(results_df)
  return results_df, bin_predictions, num_models

def flatten_accumulated(x, binary_labels):
    """
    Flattens the accumulated predictions, aligning them with the specified binary labels.

    Args:
    x (list): List of numpy arrays containing predictions for each category.
    binary_labels (list): List of binary label encodings for each category.

    Returns:
    tuple: A tuple containing:
           - x_out (numpy array): Concatenated predictions.
           - y_out (numpy array): Concatenated binary labels corresponding to predictions.
    """
    # Convert x elements to numpy arrays if they are not already
    x = [np.array(xi) for xi in x]
    
    # Create binary label arrays for each category
    y = [np.full((xi.shape[0], len(binary_labels[0])), binary_labels[i]) for i, xi in enumerate(x)]
    
    # Concatenate arrays across all categories
    x_out = np.concatenate(x, axis=0)
    y_out = np.concatenate(y, axis=0)
    return x_out, y_out

def calc_accumulated_predictions(config, y_test_predicted_per_lable, num_of_chunks_to_aggregate):
  assert(num_of_chunks_to_aggregate > 0)
  max_chunks_num = len(y_test_predicted_per_lable)
  max_iterations = int(round(max_chunks_num/num_of_chunks_to_aggregate))
  y_test_predicted_agg = [1.0*y_test_predicted_per_lable[i*num_of_chunks_to_aggregate:(i+1)*num_of_chunks_to_aggregate].sum()/num_of_chunks_to_aggregate for i in range(max_iterations)]
  if config.verbose:
    print(f'max_chunks_num ={max_chunks_num} max_iterations ={max_iterations}, len of accumulated predictions {len(y_test_predicted_agg)}')
  return y_test_predicted_agg

def eval_accumulated_inner(config, predicted, x_test_per_category, num_of_chunks_to_aggregate = 25):
  y_pred_reduced = [calc_accumulated_predictions(config, y_pred, 25) for y_pred in predicted]
  y_pred_reduced, y_true = flatten_accumulated(np.array(y_pred_reduced), config.binary_lables)
  y_true = np.array(y_true).squeeze()
  auc = roc_auc_score(y_true, y_pred_reduced)
  threshold = find_optimal_threshold(y_true, y_pred_reduced)
  if config.verbose:
      print(f' auc = {auc}, on {len(y_true)} values from 2 categories')
      print(np.shape(y_pred_reduced))
      print(np.shape(y_true))
      print(f'The chosen(using AUC-ROC curve) optimal threshold for aggregated chunks is {threshold}')
  results_df_a = evaluate_model(config, y_pred_reduced, y_true, threshold, filename = 'aggregated')
  y_pred_reduced = np.array(y_pred_reduced)
  generate_confusion_matrix_image(y_pred_reduced, y_true, threshold, show=False, save_path = 'confusion_matrix_agg.png')
  plot_nice_roc_curve(y_true, y_pred_reduced, show=True, save_path = 'roc_curve_agg.png')
  print('Accumulated metrics:')
  print(results_df_a)
  return results_df_a

def eval_accumulated(config, model, x_test_per_category, num_of_chunks_to_aggregate = 25):
    y_test_predicted = [model.predict(x_test) for x_test in x_test_per_category]
    eval_accumulated_inner(config, y_test_predicted, x_test_per_category, num_of_chunks_to_aggregate)

def eval_accumulated_e(config, models, x_test_per_category, num_of_chunks_to_aggregate=25):
    """
    Evaluates models on test data categorized into chunks and computes accumulated metrics.

    Parameters:
    - config: Configuration object (may contain settings for evaluation).
    - models: List of trained models to evaluate.
    - x_test_per_category: List of test data categories, where each category is a subset of test data.
    - num_of_chunks_to_aggregate: Number of chunks to aggregate for evaluation (default: 25).

    Returns:
    - results_df_a: DataFrame containing the evaluation results for accumulated metrics.
    """
    # Generate predictions for each model across all test categories
    y_test_predicted_per_category = [[model.predict(x_test) for model in models] for x_test in x_test_per_category]

    # Average predictions per category across models
    averaged_predictions_per_category = [np.mean(predictions, axis=0) for predictions in y_test_predicted_per_category]

    # Pass the averaged predictions to eval_accumulated_inner
    results_df_a = eval_accumulated_inner(
        config, averaged_predictions_per_category, x_test_per_category, num_of_chunks_to_aggregate
    )
    return results_df_a
