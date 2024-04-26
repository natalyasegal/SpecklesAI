from sklearn.metrics import roc_auc_score

def flatten(x, binary_lables):
  y = []
  for i in range(config.number_of_classes):
        y.append([])
  for i in range(config.number_of_classes): #short loop, as number of categories
        x[i] = np.array(x[i])
        y[i] = np.full((np.shape(x[i])[0], len(binary_lables[0])), binary_lables[i])
  x, y = x, np.array(y)
  x_out = x[0]
  y_out = y[0]
  for i in range(1, len(binary_lables)): #short loop, as number of categories
    x_out = np.append(x_out, x[i], axis=0)
    y_out = np.append(y_out, y[i], axis=0)
  return x_out, y_out

def calc_accumulated(y_test_predicted_1, max_chunks_num = 2500, cum_sz = 1000, th = 0.5, debug = False):
  max_iterations = int(round(max_chunks_num/cum_sz))
  if debug:
    print(max_iterations)
  if th == None:
    y_test_predicted_1_clean = y_test_predicted_1
  else:
    y_test_predicted_1_clean = np.array([1 if i > th else 0 for i in y_test_predicted_1])
  y_test_predicted_1_clean_reduced = [1.0*y_test_predicted_1_clean[i*cum_sz:(i+1)*cum_sz].sum()/cum_sz for i in range(max_iterations)]
  if debug:
    print(len(y_test_predicted_1_clean_reduced))
  return y_test_predicted_1_clean_reduced

def search_params_no_th(config, cum_sz = 50, verbose = True):
  y_test_predicted_0 = model.predict(x_test_per_category[0])
  y_test_predicted_1 = model.predict(x_test_per_category[1])

  y_test_predicted_reduced_0 = calc_accumulated(y_test_predicted_0, max_chunks_num = 2500, cum_sz = cum_sz, th = None)
  y_test_predicted_reduced_1 = calc_accumulated(y_test_predicted_1, max_chunks_num = 2500, cum_sz = cum_sz, th = None)
  y_pred_reduced = []
  y_pred_reduced.append(y_test_predicted_reduced_0)
  y_pred_reduced.append(y_test_predicted_reduced_1)
  y_pred_reduced, y_true = flatten(y_pred_reduced, config.binary_lables)
  num_super_chunk = len(y_true)
  auc = roc_auc_score(y_true, y_pred_reduced)
  print(f' auc = {auc}')

def search_params(config, cum_sz = 50, verbose = True, thresholds_list = [0.0000001, 0.001, 0.01, 0.025, 0.05, 0.1, 0.185, 0.19, 0.195, 0.198, 0.2, 0.208, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]):
  y_test_predicted_0 = model.predict(x_test_per_category[0])
  y_test_predicted_1 = model.predict(x_test_per_category[1])
  num_super_chunk = 0
  best_auc = 0
  winning_th = 0
  for th in thresholds_list:
    y_test_predicted_reduced_0 = calc_accumulated(y_test_predicted_0, max_chunks_num = 2500, cum_sz = cum_sz, th = th)
    y_test_predicted_reduced_1 = calc_accumulated(y_test_predicted_1, max_chunks_num = 2500, cum_sz = cum_sz, th = th)
    #50 frames is 2sec = 50*40/1000 # 1000 is fps, 50 chunks, 40 frames in chunk
    y_pred_reduced = []
    y_pred_reduced.append(y_test_predicted_reduced_0)
    y_pred_reduced.append(y_test_predicted_reduced_1)
    y_pred_reduced, y_true = flatten(y_pred_reduced, config.binary_lables)
    num_super_chunk = len(y_true)
    auc = roc_auc_score(y_true, y_pred_reduced)
    acc_best = 0
    th_a_best = 0
    for th_a in thresholds_list:
      acc= sklearn.metrics.accuracy_score(y_true, y_pred_reduced >th)
      if acc > acc_best:
        acc_best = acc
        th_a_best = th_a
    if verbose:
      print(f'th={th} auc = {auc}, acc_best = {acc_best}, th_a_best={th_a_best}')
    if auc > best_auc:
      best_auc = auc
      winning_th = th
      winning_y_pred_reduced = y_pred_reduced
      winning_y_true = y_true
  return best_auc, winning_th, winning_y_pred_reduced, winning_y_true, num_super_chunk
  
