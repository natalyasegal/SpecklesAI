from sklearn.preprocessing import LabelBinarizer
from typing import Any, List

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

class Configuration():
  def __init__(self, verbose = True):
    self.subjects_and_dates_config_file_name = 'config/subjects_and_dates.yml'
    self.models_path='models'
    self.raw_data_path = 'exp3'
    self.data_path = 'data' #destination after preprocessing
    self.MAX_CHUNKS_PER_CATEGORY = 100000
    self.chunk_size = 40 #64 #destination temporal chunk size
    self.frame_size_x = 32 #64 #128 # destination frames images size after resizing x
    self.frame_size_y = 32 #64 # destination frames images size after resizing y
    self.sub_music = 'Wernike/music'
    self.sub_english = 'Wernike/english'
    self.sub_swedish = 'Wernike/swedish'
    self.sub_no = 'Wernike/no_sound'
    self.frames_subdirs_dict = {self.sub_english:"english", self.sub_swedish:"swedish"}
    CLASSES = list(set(self.frames_subdirs_dict.values()))
    self.number_of_classes = len(CLASSES)
    self.lables_categories = [x for x in range(self.number_of_classes)]
    self.binary_lables = binarize_lables(self.lables_categories)
    if verbose:
      print(f' number_of_classes = {self.number_of_classes}\n binary_lables={self.binary_lables}')
