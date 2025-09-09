from sklearn.preprocessing import LabelBinarizer
from typing import Any, List
import yaml	
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from split import Logical_Split	

def binarize_lables(labels_orig: List[Any]) -> List[int]:
      """
      Converts a list of class labels into a one-hot encoded binary matrix.
      
      Args:
          labels_orig (List[Any]): List of original class labels.
      
      Returns:
          List[int]: A list containing the one-hot encoded binary matrix representation of the input labels.
      """
      lb = LabelBinarizer()
      return lb.fit_transform(labels_orig)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

class Configuration():
  def __init__(self, verbose = True):
    self.verbose = verbose  
    self.subjects_and_dates_config_file_name = 'SpecklesAI/config/config_files/subjects_and_dates.yaml'
    self.models_path='models'
    self.create_images = True
    self.be_consistent = True # initializze seeds for consistent results
    self.seed_for_init = 1 #seed for initialization

    self.MAX_CHUNKS_PER_CATEGORY = 100000
    self.chunk_size = 40 #64 #destination temporal chunk size
    self.frame_size_x = 32 #64 #128 # destination frames images size after resizing x
    self.frame_size_y = 32 #64 #128 # destination frames images size after resizing y
    #self.max_chunks_num = 2500 #calculated 10sec*10videos*1000fps / 40 frames in chunk, used for aggregted evaluation
    self.raw_data_path = 'exp3'
    self.data_path = 'data' #destination after preprocessing
    self.base_data_path_subdir = 'w'
    #self.sub_music = 'Wernike/music'
    self.sub_english = 'Broca/yes'
    self.sub_swedish = 'Broca/no'
    #self.sub_no = 'Wernike/no_sound'
    self.frames_subdirs_dict = {self.sub_english:"yes", self.sub_swedish:"no"}
    CLASSES = list(set(self.frames_subdirs_dict.values()))
    self.number_of_classes = len(CLASSES)
    self.lables_categories = [x for x in range(self.number_of_classes)]
    self.binary_lables = binarize_lables(self.lables_categories)
    if self.verbose:
      print(f' number_of_classes = {self.number_of_classes}\n binary_lables={self.binary_lables}')
  
  def get_number_of_subjects(self):
    with open(self.subjects_and_dates_config_file_name, 'r') as file:
        data = yaml.safe_load(file)
        subject_config = data.get('subjects', {})
    
    # Return the number of subjects
    return len(subject_config)
      
  def set_split(self, train_dates, train_subjects, val_dates, val_subjects, test_dates, test_subjects):
    self.train_dates = train_dates
    self.train_subjects = train_subjects
    self.val_dates = val_dates
    self.val_subjects = val_subjects
    self.test_dates = test_dates
    self.test_subjects = test_subjects
        
class Configuration_Gen(Configuration):
  def __init__(self, split_num, config_file_name = 'SpecklesAI/config/config_files/sample_gen_split.yaml', verbose = True):
    super().__init__(verbose)
    self.sample_splits_file_name = 'SpecklesAI/config/config_files/sample_gen_split.yaml'
    if len(config_file_name) > 0:
          self.sample_splits_file_name = config_file_name
    self.sample_splits = load_yaml(self.sample_splits_file_name)
    self.number_of_splits = len(self.sample_splits)
    if self.verbose:
          print("Splits list:")
          print(self.sample_splits)
    assert(split_num <= self.number_of_splits)
    self.split_num = split_num
    train_mix, val_mix, test_mix, model_name = self.sample_splits[self.split_num].values()
    if self.verbose:       
          print(f'The chosen split number is {self.split_num}, train mix is {train_mix}, validation: {val_mix}, test: {test_mix}, {model_name}')
    self.model_name = model_name
    self.set_split_by_subjects(train_mix, val_mix, test_mix)

  def get_number_of_splits(self):
    return self.number_of_splits

  def set_split_by_subjects(self, train_mix, val_mix, test_mix):
    ls = Logical_Split(self.subjects_and_dates_config_file_name, verbose = self.verbose)
    train_dates, train_subjects = ls.get_dates_and_subjects(train_mix, Logical_Split.Sample_time.MORNING_AND_MID_DAY)
    val_dates, val_subjects = ls.get_dates_and_subjects(val_mix, Logical_Split.Sample_time.MORNING_AND_MID_DAY)
    test_dates, test_subjects = ls.get_dates_and_subjects(test_mix, Logical_Split.Sample_time.MORNING_AND_MID_DAY)
    print(f' train {train_dates} {train_subjects}\n val {val_dates} {val_subjects}\n test {test_dates} {test_subjects}')
    self.set_split(train_dates, train_subjects, val_dates , val_subjects, test_dates, test_subjects)


class Configuration_PerSubjExperiment(Configuration):
      
  def __init__(self, split_num, verbose = True):
    super().__init__(verbose)
    self.subj_cofig_file_name = 'SpecklesAI/config/config_files/per_subj_experiment_val_split.yaml'
    self.split_num = split_num #in this experiment same as subject number
    if self.verbose:       
          print(f'The chosen split number (=subj number) is {self.split_num}')
    self.model_name = 'Subj_experiment_model'
    subj_cofig = load_yaml(self.subj_cofig_file_name)
    self.val_limit = subj_cofig[split_num]['val_limit'] #this recreates the division into train and validation sets per subject in the paper, feel free to use any other reasonable split in your experiments
    #2000 or 500 takes 500 to validation either from the beginning or from the end, for 1st subj it is 250 because of the difference in fps, subj 1 with 500fps and subj 2-6 with 1000 fps
    self.set_split_for_per_subj_experiment(self.split_num)

  def set_split_for_per_subj_experiment(self, subj_num):
    ls = Logical_Split(self.subjects_and_dates_config_file_name, verbose = self.verbose)
    train_dates, train_subjects = ls.get_dates_and_subjects([subj_num], Logical_Split.Sample_time.ONLY_MORNING)
    val_dates = [] #will be set later
    val_subjects = [subj_num] 
    test_dates, test_subjects = ls.get_dates_and_subjects([subj_num], Logical_Split.Sample_time.ONLY_MID_DAY)
    print(f' train {train_dates} {train_subjects}\n val {val_dates} {val_subjects}\n test {test_dates} {test_subjects}')
    self.set_split(train_dates, train_subjects, val_dates , val_subjects, test_dates, test_subjects)


# Configuration for creating test set separately
class Configuration_Test(Configuration):
  def __init__(self, split_num, verbose = True):
    super().__init__(verbose)
    self.sample_splits_file_name = 'SpecklesAI/config/config_files/sample_test_sets.yaml'
    self.sample_splits = load_yaml(self.sample_splits_file_name)
    self.number_of_splits = len(self.sample_splits)
    if self.verbose:
          print("Splits list:")
          print(self.sample_splits)
    assert(split_num <= self.number_of_splits)
    self.split_num = split_num
    test_mix, model_name = self.sample_splits[self.split_num].values()
    if self.verbose:       
          print(f'The chosen split number is {self.split_num}, test: {test_mix}, {model_name}')
    self.model_name = model_name
    
    ls = Logical_Split(self.subjects_and_dates_config_file_name, verbose = self.verbose)
    test_dates, test_subjects = ls.get_dates_and_subjects(test_mix, Logical_Split.Sample_time.ONLY_MID_DAY)
    print(f' test {test_dates} {test_subjects}')
    self.set_split([], [], [] , [], test_dates, test_subjects)
 
  def get_number_of_splits(self):
    return self.number_of_splits


# Configuration for creating pause set
class Configuration_Pause(Configuration_Test):
  def __init__(self, split_num, verbose = True):
    super().__init__(split_num, verbose)
    self.sub_nothing = 'Broca/nothing'
    self.frames_subdirs_dict = {self.sub_nothing:"pause"}
    CLASSES = list(set(self.frames_subdirs_dict.values()))
    self.number_of_classes = 1
    self.lables_categories = [x for x in range(self.number_of_classes)]
    self.binary_lables = binarize_lables(self.lables_categories)
    if self.verbose:
      print(f' number_of_classes = {self.number_of_classes}\n binary_lables={self.binary_lables}')
  
