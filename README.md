Code for a scientific paper: "Remote Detection of Human Brain Reaction to Speech by AI Speckle Pattern Analysis"


In order to recreate per subjects results:
1. Clone the repository
3. Run: !python -u SpecklesAI/prepare_datasets_for_all_subjects_for_per_subjects_experiment.py --random_seed 9  --train_set_file train_set__per_subj_ --validation_set_file validation_set__per_subj_ --test_set_per_category_file test_per_category__per_subj_
4. Install the requirements
5. Train and evaluate the model by running:
   !cp gdrive/MyDrive/__MSc_2023/code_and_data/data_exp3_prep/per_subj_splits/*_1.npy .

   !python -u SpecklesAI/prep_and_train_tf.py --use_per_subj_config --batch_size 1000 --epochs 170 --num_of_chunks_to_aggregate 25 --random_seed 9  --train_set_file train_set__per_subj__1.npy --validation_set_file validation_set__per_subj__1.npy --test_set_per_category_file test_per_category__per_subj__1.npy --split_num 1  --read_stored_dataset
   Please adjust the numbers when you run for different subjects, example above is for subject 1.

In order to recreate the generalization results:
1. Clone the repository
2. Create your configuration file and copy it in place of the SpecklesAI/config/config_files
/subjects_and_dates.yaml, to recreate the results in paper, ask the authours for our configuration file
3. Create train, validation and test datasets, see example 2 for instructions on how to create those datasets for all the 6 gen splits from the paper.
4. Install the requirements
5. Train and evaluate the model. See example 3 on how to train and evaluate on the already prepared datasets
   

Example 1:

python -u SpecklesAI/prep_and_train.py --batch_size 1000 --epochs 10 --num_of_chunks_to_aggregate 25

Example 2:

python -u SpecklesAI/prep_and_train.py --batch_size 1000 --epochs 20 --num_of_chunks_to_aggregate 25 --random_seed 9  
  --train_set_file train_set_split6.npy --validationl_set_file validation_set_split6.npy --test_set_per_category_file test_per_category_split6.npy --split_num 6

Example 3:

Many of us prefer to separate preprocessing part that do not reqiore GPU, from training and inference.
To do that you can perform the preprocessing convering the video files to chunks of frames for each of the splits, designated in the configuration yaml files by running:

!python -u SpecklesAI/prepare_datasets_for_all_splits.py --random_seed 9  --train_set_file train_set_split_ --validation_set_file validation_set_split_ --test_set_per_category_file test_per_category_split_

License:
This code is made available under the GNU GPLv3 License and is available for non-commercial academic purposes.
