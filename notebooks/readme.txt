Notebooks with sample scenarios

Notebooks for BCI yes vs. no paper, green, see preprint here: https://www.researchgate.net/publication/400258992_Remote_Optical_Decoding_of_Inner_Speech_in_Broca's_Area_via_AI-based_Speckle_Pattern_Analysis

1. Table 1 (Forehead part) is created by: BCI__forehead_control_per_subj___BCI_paper_yes_vs_no.ipynb


Basic preprocessing:
To preprocess raw video files (resize to 32×32, convert to grayscale, segment into 40-frame temporal chunks) and export as .npy arrays, run (in your Colab):
!rm -r data
!mkdir exp3
!mkdir exp3/somedate_day1_1
!mkdir exp3/somedate_day1_1/SubjectOneName
!unzip 'your_files_location/video_files_from_experiment.zip'
# The following 3 lines may differ if you store the files in a different directory structure
!mv video_files_from_experiment/words exp3/somedate_day1_1/SubjectOneName/Broca
!mv exp3/somedate_day1_1/SubjectOneName/Broca/1yes exp3/somedate_day1_1/SubjectOneName/Broca/yes
!mv exp3/somedate_day1_1/SubjectOneName/Broca/2no exp3/somedate_day1_1/SubjectOneName/Broca/no
# create the npy files:
!python -u SpecklesAI/prepare_test_sets.py --split_num 1 --random_seed 9  --test_set_per_category_file test_per_category_split_morning_
# output in: test_per_category_split_morning__1.npy 

Preprocessing 2:
To apply min-max normalization with a gain factor of 10 to a preprocessed .npy array, run:
loaded_npy = load_dataset_x("name_of_your_npy_array.npy", with_normalization=False)

# n_chunks_per_clip: 1 = normalize per temporal chunk (40 frames); 5000 = normalize per clip
n_chunks_per_clip = 1
loaded_npy_with_norm_and_gain = normalize_per_fixedclip(loaded_npy, n_chunks_per_clip=n_chunks_per_clip, mode="minmax", gain=10.0)

# Output shape: (num_classes, num_chunks, temporal_chunk_size, 32, 32, color_channels)
# Example: (2, 4000, 40, 32, 32, 1)

To split into data and labels:
x, y = test2trainformat_binary_safe(loaded_npy_with_norm_and_gain, need_to_shuffle_within_category=False)



Recommended code imports:

# Visuals
!rm -rf SpeckleAI_Visuals/
!git clone https://github.com/danielrubinsteinishere/SpeckleAI_Visuals.git
import sys
sys.path.append('/content/SpeckleAI_Visuals')
from plots.bar_plots import plot_subject_metrics
from stats.eval import summarize
from cm.binary_cms import plot_binary_confusion_matrix_from_cm, create_image_with_multiple_binary_confusion_matrices, create_image_with_mean_binary_confusion_matrix
from cm.multiclass_cms import display_multiclass_cm_with_percents
from metrics.metrics import macro_F1_from_cm, macro_F1_accuracy_from_cm, calc_mean_acc_and_F1, create_mean_cm

# Models:
!rm -rf SpecklesAI/
#if you are clonning a public version, use:
!git clone https://github.com/natalyasegal/SpecklesAI.git
!cp SpecklesAI/config/config_compehension__N_S_M_P.py SpecklesAI/config/config.py # Pay attention to change to the config file, you need!

import sys
sys.path.append('/content/SpecklesAI')   # add package root to Python path
#from utils.swap import swap_categories
from config.config import binarize_lables, load_yaml, Configuration, Configuration_Minimal
from utils.norm import normalize_per_fixedclip
from utils.swap import *
from utils.stats import print_stats, print_test_stats
from utils.formatstranslator import make_x_per_category, test2trainformat, limit_rearrange_and_flatten_s
from utils.data import split_by_chunks, split_from_start, load_dataset_x, concatenate_train_or_val
from utils.utils import save_dataset_x, save_dataset
from utils.embeddings_utils import concat_temporal_embeddings_c_stride, make_clip_ids_from_fps, concat_temporal_embeddings_c, concat_temporal_embeddings # sliding window -> target size: N-k+1
from utils.utils_seed import set_seed_all
from preprocessing.preprocessing import unison_shuffled_copies
from pca.pca import visualize_embeddings_pca_3d, reduce_embeddings_pca_3d
from models.LvMAE_pt import VideoMAE
from models.LvMAE_pt import *
from models.binary_XGBoost import train_eval_xgboost_classifier_after_concatenation, train_eval_xgboost_classifier
from models.multiclass_XGBoost import train_eval_xgboost_classifier_multiclass, train_eval_xgb_train_api_multiclass, get_multiclass_cm_with_percents
from models.XGBoost_on_LV_MAE_embeddings import train_and_eval_classifier_on_embeddings_agg, train_and_eval_classifier_on_embeddings, train_and_eval_multiclass_classifier_on_embeddings
from models.clustering__on_LV_MAE_embeddings import calc_ARI_NMI_of_GMMcluster_c
from evaluation.eval import calc_accumulated_predictions, generate_confusion_matrix_image, plot_nice_roc_curve, find_optimal_threshold, evaluate_model, evaluate_per_chunk, flatten_accumulated
from evaluation.eval_utils import eval_aggregated_test_set_th_on_val, eval_aggregated_th_on_target_subj_firstK_chunks

set_seed_all(seed_for_init=1, random_seed=9, use_tf = False)

# place here you standard imports:
from sklearn.metrics import confusion_matrix
