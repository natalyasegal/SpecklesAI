from sklearn.preprocessing import LabelBinarizer
from typing import Any, List
import yaml	
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
    self.subjects_and_dates_config_file_name = 'config/subjects_and_dates.yaml'
    self.models_path='models'
    self.create_images = True
    self.be_consistent = True # initializze seeds for consistent results
    self.seed_for_init = 1 #seed for initialization

    self.MAX_CHUNKS_PER_CATEGORY = 100000
    self.chunk_size = 40 #64 #destination temporal chunk size
    self.frame_size_x = 32 #64 #128 # destination frames images size after resizing x
    self.frame_size_y = 32 #64 # destination frames images size after resizing y
    #self.max_chunks_num = 2500 #calculated 10sec*10videos*1000fps / 40 frames in chunk, used for aggregted evaluation
    self.raw_data_path = 'exp3'
    self.data_path = 'data' #destination after preprocessing
    self.base_data_path_subdir = 'w'
    self.sub_music = 'Wernike/music'
    self.sub_english = 'Wernike/english'
    self.sub_swedish = 'Wernike/swedish'
    self.sub_no = 'Wernike/no_sound'
    self.frames_subdirs_dict = {self.sub_english:"english", self.sub_swedish:"swedish"}
    CLASSES = list(set(self.frames_subdirs_dict.values()))
    self.number_of_classes = len(CLASSES)
    self.lables_categories = [x for x in range(self.number_of_classes)]
    self.binary_lables = binarize_lables(self.lables_categories)
    if self.verbose:
      print(f' number_of_classes = {self.number_of_classes}\n binary_lables={self.binary_lables}')

class Configuration_Gen(Configuration):
  def __init__(self, verbose = True):
    super().__init__(verbose)
    self.sample_splits_file_name = 'config/sample_gen_split.yaml'
    self.split_num = 6
    self.sample_splits = load_yaml(self.sample_splits_file_name)
    print(self.sample_splits)
    train_mix, val_mix, test_mix, model_name = self.sample_splits[self.split_num].values()
    print(f'{train_mix}, {val_mix}, {test_mix}, {model_name}')
    self.model_name = model_name
    self.set_split_by_subjects(train_mix, val_mix, test_mix)

  def set_split(self, train_dates, train_subjects, val_dates, val_subjects, test_dates, test_subjects):
    self.train_dates = train_dates
    self.train_subjects = train_subjects
    self.val_dates = val_dates
    self.val_subjects = val_subjects
    self.test_dates = test_dates
    self.test_subjects = test_subjects

  def set_split_by_subjects(self, train_mix, val_mix, test_mix):
    ls = Logical_Split(self.subjects_and_dates_config_file_name, verbose = self.verbose)
    train_dates, train_subjects = ls.get_dates_and_subjects(train_mix, Logical_Split.Sample_time.MORNING_AND_MID_DAY)
    val_dates, val_subjects = ls.get_dates_and_subjects(val_mix, Logical_Split.Sample_time.MORNING_AND_MID_DAY)
    test_dates, test_subjects = ls.get_dates_and_subjects(test_mix, Logical_Split.Sample_time.MORNING_AND_MID_DAY)
    print(f' train {train_dates} {train_subjects}\n val {val_dates} {val_subjects}\n test {test_dates} {test_subjects}')
    self.set_split(train_dates, train_subjects, val_dates , val_subjects, test_dates, test_subjects)
