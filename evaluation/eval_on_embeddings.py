

def TestGen_ValHeldoutFromUnseen(
          train_x_list = [x1_n, x2_n, x3_n, x4_n, x5_n, x6_n, x7_n, x9_n, x10_n], 
          train_y_list = [y1_n, y2_n, y3_n, y4_n, y5_n, y6_n, y7_n, y9_n, y10_n],
          unseen = test_8_m_n_40ms, K_thr = 500, num_of_chunks_to_aggregate = 25):
  model, opt2, scaler2, start_ep = load_for_resume_and_infer(VideoMAE, "artifacts_lvmae_1/checkpoint.pt")
  X_train, y_train =  concatenate_train_or_val(train_x_list, train_y_list)
  val, test = split_by_chunks_v(unseen, val_n = K_thr)
  X_val, y_val = test2trainformat(val, need_to_shuffle_within_category = False)
  X_test, y_test = test2trainformat(test, need_to_shuffle_within_category = False)

  Z_train, y_train = extract_embeddings_wrapper_one(model, X_train, y_train)
  Z_val, y_val,  = extract_embeddings_wrapper_one(model, X_val, y_val)
  Z_test, y_test = extract_embeddings_wrapper_one(model, X_test, y_test)

  clf,_,_, _,prob_val,prob_test,y_c=train_eval_xgboost_classifier(Z_train,y_train,
                                                                  Z_val,y_val,
                                                                  Z_test,y_test,
                                                                  K = 1, show=True) # 128w and 128
  print(f'Aggregated k={num_of_chunks_to_aggregate}: =================================')
  eval_aggregated_test_set_th_on_val(Z_test, prob_test, y_test, prob_val, y_val,
                                       num_of_chunks_to_aggregate= num_of_chunks_to_aggregate)                              
