Code for a scientific paper: "Remote Detection of Human Brain Reaction to Speech by AI Speckle Pattern Analysis"

Example 1:

python -u SpecklesAI/prep_and_train.py --batch_size 1000 --epochs 10 --num_of_chunks_to_aggregate 25

Example 2:

python -u SpecklesAI/prep_and_train.py --batch_size 1000 --epochs 20 --num_of_chunks_to_aggregate 25 --random_seed 9  
  --train_set_file train_set_split6.npy --validationl_set_file validation_set_split6.npy --test_set_per_category_file test_per_category_split6.npy --split_num 6

License:
This code is made available under the GNU GPLv3 License and is available for non-commercial academic purposes.
