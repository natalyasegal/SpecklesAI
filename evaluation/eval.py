import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, cohen_kappa_score, roc_curve, confusion_matrix

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def evaluate_model(config, predictions, true_labels, th):
  # Calculate metrics
  auc = roc_auc_score(true_labels, predictions, multi_class='ovo', average='macro')
  if th is not None:
    accuracy = accuracy_score(true_labels, predictions > th)
    f1 = f1_score(true_labels, predictions > th, average='macro')
    kappa = cohen_kappa_score(true_labels, predictions> th)

    # Create DataFrame
    data = {
              'Model Name': [config.model_name],
              'AUC': [auc],
              'Accuracy': [accuracy],
              'F1 Score': [f1],
              'Cohen Kappa': [kappa]
    }
  else:
    data = {
              'Model Name': [config.model_name],
              'AUC': [auc]
    }
  results_df = pd.DataFrame(data)

  # Save to CSV
  results_df.to_csv(f'{config.model_name}.csv', index=False)
  print(f'Results saved to {config.model_name}.csv')
  return results_df

def evaluate_per_chunk(config, model, x_test, y_test, create_images = True):
  y_test_predicted = model.predict(x_test)

  # Find optimal probability threshold
  threshold = Find_Optimal_Cutoff(y_test, y_test_predicted)
  print(threshold)

  results_df = evaluate_model(config, y_test_predicted, y_test, threshold)
  print(results_df)
  return results_df
