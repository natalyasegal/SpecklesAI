import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import sys
import os
import multiprocessing
from multiprocessing import Pool
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
#from config.config import Configuration

''' Suffle: '''
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p, ::], b[p]


def process_frame_bulk(args):
    """Process a bulk of frames in one process."""
    video_path, frames_path, frame_size_x, frame_size_y, frames_bulk = args
    for frame_id, image in frames_bulk:
        image = cv2.resize(image, (frame_size_x, frame_size_y))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filename = f"{frames_path}/{Path(video_path).stem}_frame_{frame_id}.jpg"
        cv2.imwrite(filename, image)

class Preprocessing():
  def __init__(self, config, verbose = True):
    self.config = config
    self.verbose = verbose
    self.__have_train_set_parsed = False 
    self.__have_val_set_parsed = False
    self.__have_test_set_parsed = False
      
  ''' Helper functions '''
  def log(self, message):
    if self.verbose:
        print(message)

  def __create_directory(self, directory):
    if not os.path.exists(directory):
      os.makedirs(directory)

  def __create_paths(self, out_paths):
    if len(out_paths) == 0:
      print('Error: Output paths list is empty.')
    else:
      for directory in out_paths:
        self.__create_directory(directory)
   
  def __are_all_directories_empty(self, directory_paths):
    '''
    non existant path will be considered empty
    '''
    for dir_path in directory_paths:
        # Check if the directory exists and is indeed a directory
        if os.path.isdir(dir_path):
            # os.listdir returns a list of entries in the directory
            if os.listdir(dir_path):
                # If list is not empty, directory is not empty
                return False
    return True

  def __contains_video_files(self, directory_paths):
    """
    Check if any provided directories contain video files.

    Args:
    directory_paths (list of str): A list containing directory paths to check.

    Returns:
    bool: True if at least one directory contains video files and is not empty, False otherwise.

    Note:
    Supports common video formats such as .avi, .mp4, .mkv, .flv, .mov, .wmv, .mpeg, and .mpg. It also warns if a path
    is not a directory or does not exist.
    """
    video_extensions = {'.avi', '.mp4', '.mkv', '.flv', '.mov', '.wmv', '.mpeg', '.mpg'}
    for dir_path in directory_paths:
        if os.path.isdir(dir_path):
            files = os.listdir(dir_path)
            if files:  # Check if directory is not empty
                for file in files:
                    if os.path.splitext(file)[1].lower() in video_extensions:
                        return True  # At least one video file found
        else:
            print(f"Warning: {dir_path} is not a directory or does not exist.")
    return False

  '''Video to frames helper function'''
  def __split_video_to_frames_mp(self, video_path, frames_path):
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            print(f"Error: Unable to open video {video_path}")
            return

        # Read all frames into a list
        frames = []
        frame_id = 0
        while True:
            success, image = vidcap.read()
            if not success:
                break
            frames.append((frame_id, image))
            frame_id += 1

        vidcap.release()

        # Split frames into chunks for multiprocessing
        num_processes = min(10, multiprocessing.cpu_count())  # Limit to 10 processes or CPU count
        chunk_size = len(frames) // num_processes + (len(frames) % num_processes > 0)
        frame_chunks = [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]

        # Create arguments for each process
        args = [
            (video_path, frames_path, self.config.frame_size_x, self.config.frame_size_y, chunk)
            for chunk in frame_chunks
        ]

        # Use multiprocessing Pool with spawn method
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(process_frame_bulk, args)
            
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
      
  def __split_video_to_frames(self, video_path, frames_path): # single process version
      vidcap = cv2.VideoCapture(video_path)
      self.log(f'--- {frames_path}   {video_path}')
      video_name = video_path.split(sep=os.sep)[-1].split('.')[-2]
      success, image = vidcap.read()
      count = 1
      while success:
          image = cv2.resize(image, (self.config.frame_size_x, self.config.frame_size_y))
          image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          new_path_and_filename = "{}/{}_frame_{}.jpg".format(frames_path, video_name, count)
          cv2.imwrite(new_path_and_filename, image)
          success, image = vidcap.read()
          count += 1

  def __get_in_out_paths(self, date, subj, out_subdir):
    in_video_subdirs = list(set(self.config.frames_subdirs_dict.keys()))

    base_raw_path = f"{self.config.raw_data_path}/{date}/{subj}"
    base_data_path = f"{self.config.data_path}/{out_subdir}/{subj}/{self.config.base_data_path_subdir}"

    in_paths = [f"{base_raw_path}/{sub_dir}" for sub_dir in in_video_subdirs]
    out_paths = [f"{base_data_path}/{self.config.frames_subdirs_dict[sub_dir]}" for sub_dir in in_video_subdirs]
    self.log(out_paths)
    return in_paths, out_paths

  ''' Rearrange: '''
  def __rearrange_input(self, x, need_to_shuffle_within_category):
      '''
      Rearranges the input data by randomly shuffling each category and then limits
      the data size to the configured maximum chunks per category.
      '''
      y = [[] for _ in range(self.config.number_of_classes)]
      self.log(f'max_chanks={self.config.MAX_CHUNKS_PER_CATEGORY}')
      for i in range(self.config.number_of_classes): #short loop, as number of categories
          if len(x[i]) == 0:
            print(f'no items in position {i}')
            continue
          self.log(f' x[{i}] shape is {np.shape(x[i])}')
          if need_to_shuffle_within_category:
            p = np.random.permutation(np.shape(x[i])[0])
            self.log(' Shaffling within the categories, permutation p shape is {np.shape(p)}')
            x[i] = np.array(x[i])
            x[i] = x[i][p, ::]
          real_max = min(np.shape(x[i])[0], self.config.MAX_CHUNKS_PER_CATEGORY)
          x[i] = x[i][0:real_max, ::]
          y[i] = np.full((np.shape(x[i])[0], len(self.config.binary_lables[0])), self.config.binary_lables[i])
      return x, np.array(y)
  
  ''' 
  public functions:
  '''

  def create_video_frames(self, dates, subjects_dirs, out_subdir):
      ''' 
      Processes a list of dates and subjects, extracting video frames and storing them
      in specified output directories. This creates necessary directories if they don't exist, 
      and processes each video file found in the specified input paths. This function uses the `tqdm` library 
      to display a progress bar for video processing tasks, enhancing visibility and tracking for large batch operations.

      Parameters:
      - dates (list): A list of date strings representing the folders that contain video files.
      - subjects_dirs (list): A list of subject directory names that correspond to subjects
        within each date directory.
      - out_subdir (str): The name of the output subdirectory where processed frames will be stored.

        returns at_least_one_non_empty_input
      Example usage:
        prep.create_video_frames(['20240101', '20240102'], ['subject1', 'subject2'], 'processed_frames')
      '''
      self.log(f"=======\nProcessing: {out_subdir}\n=======")
      at_least_one_non_empty_input = False
      for date in dates:
          for subj in subjects_dirs:
              in_paths, out_paths = self.__get_in_out_paths(date, subj, out_subdir)
              if self.__are_all_directories_empty(in_paths):
                  print(f'Empty in path for all paths in: {in_paths} set for subjects {subj} and date {date}')
              elif self.__contains_video_files(in_paths):
                  at_least_one_non_empty_input = True
                  self.__create_paths(out_paths)
                  for in_path, out_path in zip(in_paths, out_paths):
                      self.log(f"Input: {in_path}, Output: {out_path}")
                      for vid_filename in tqdm(glob.glob('{}/*'.format(in_path))):
                          if os.path.isfile(vid_filename):
                              self.__split_video_to_frames(vid_filename, out_path)
              else:
                  print(f'No video files found in {in_paths}, please copy your data or change the path in config.py file')
      return at_least_one_non_empty_input

  def create_test_data_set(self):     
      self.__have_test_set_parsed = self.create_video_frames(self.config.test_dates, self.config.test_subjects, 'test')
      print(f'test {self.__have_test_set_parsed} ')

  def create_data_set(self):     
      self.__have_train_set_parsed = self.create_video_frames(self.config.train_dates, self.config.train_subjects, 'train') 
      self.__have_val_set_parsed = self.create_video_frames(self.config.val_dates, self.config.val_subjects, 'validation')
      self.__have_test_set_parsed = self.create_video_frames(self.config.test_dates, self.config.test_subjects, 'test')
      print(f'train {self.__have_train_set_parsed}, validation {self.__have_val_set_parsed}, test {self.__have_test_set_parsed} ')

  ''' Frames to chunks of frames: '''
  def prep_frames_lists(self, dates, subjects, out_subdir, ABSOLUTE_MAX_FRAMES_PER_CATEGORY = 5000000):
      x = [[] for _ in range(self.config.number_of_classes)]
      for date in dates:
          for subj in subjects:
              in_paths, out_paths = self.__get_in_out_paths(date, subj, out_subdir)
              for out_path, index in zip(out_paths, self.config.lables_categories):
                  frames_list_chunk = []
                  for frame_filename in tqdm(glob.glob('{}/*'.format(out_path))):
                      if os.path.isfile(frame_filename):
                          image = cv2.imread(frame_filename)
                          luma = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)[::, ::, 0:1]
                          frames_list_chunk.append(luma)
                  if len(frames_list_chunk) > 0:
                      nitems = len(frames_list_chunk)
                      n_chunks = int(min(nitems, ABSOLUTE_MAX_FRAMES_PER_CATEGORY)/self.config.chunk_size)
                      x_sub = np.split(np.array(frames_list_chunk[0:(self.config.chunk_size*n_chunks)]), n_chunks)
                      if len(x[index]) == 0:
                        x[index] = x_sub
                      else:
                        x[index] = np.append(x[index], x_sub, axis=0)
                  print(f'index = {index} len = {len(x[index])}')
      min_length = min(len(inner) for inner in x) # Find the minimum length of the inner lists
      truncated_lists = [inner[:min_length] for inner in x] # Truncate each inner list to the minimum length
      x = np.array(truncated_lists)
      self.log(f'shape of the data {x.shape}')
      return x
  
  def limit_rearrange_and_flatten(self, input_data, need_to_shuffle_within_category):
    # Rearrange the input data and get the corresponding labels
    rearranged_data, labels = self.__rearrange_input(input_data, need_to_shuffle_within_category)
    
    # Concatenate the rearranged data and labels along the first axis
    output_data = np.concatenate(rearranged_data, axis=0)
    output_labels = np.concatenate(labels, axis=0)
    
    # Return the concatenated and rearranged data along with their labels
    return output_data, output_labels 

  def prepare_train_and_validation_data(self, need_to_shuffle_within_category):
    '''train set preprocessing'''
    assert(self.__have_train_set_parsed)
    x_train_per_category = self.prep_frames_lists(self.config.train_dates, self.config.train_subjects, 'train')
    x_train, y_train = self.limit_rearrange_and_flatten(x_train_per_category, need_to_shuffle_within_category)
    self.log(f'x_train shape is {np.shape(x_train)}, y_train shape is {np.shape(y_train)}')

    '''validation set preprocessing'''
    assert(self.__have_val_set_parsed)
    x_val_per_category = self.prep_frames_lists(self.config.val_dates, self.config.val_subjects, 'validation')
    x_val, y_val = self.limit_rearrange_and_flatten(x_val_per_category, need_to_shuffle_within_category)
    self.log(f'x_val shape is {np.shape(x_val)}, y_val shape is {np.shape(y_val)}')
  
    '''shaffle train and validation sets, data and lables are shaffled together'''
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_val, y_val = unison_shuffled_copies(x_val, y_val)
    return x_train, y_train, x_val, y_val

  def prepare_test_data(self):
    '''test set preprocessing'''
    assert(self.__have_test_set_parsed)
    x_test_per_category  = self.prep_frames_lists(self.config.test_dates, self.config.test_subjects, 'test')
    x_test, y_test = self.limit_rearrange_and_flatten(x_test_per_category, need_to_shuffle_within_category = False)
    self.log(f'x_test shape is {np.shape(x_test)}, y_test shape is {np.shape(y_test)}')
    return x_test, y_test, x_test_per_category
