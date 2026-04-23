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
