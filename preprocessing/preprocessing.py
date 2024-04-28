import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
from ../config import Configuration

''' Suffle: '''
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p, ::], b[p]

class Preprocessing():
  def __init__(self, config, verbose = True):
    self.config = config
    self.verbose = verbose

  ''' Helper functions '''

  '''Video to frames felper function'''
  def __split_video_to_frames(self, video_path, frames_path):
      vidcap = cv2.VideoCapture(video_path)
      if self.verbose:
        print(f'--- {frames_path}   {video_path}')
      video_name = video_path.split(sep=os.sep)[-1].split('.')[-2]
      success, image = vidcap.read()
      count = 1
      while success:
          image = cv2.resize(image, (config.frame_size_x, config.frame_size_y))
          image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          new_path_and_filename = "{}/{}_frame_{}.jpg".format(frames_path, video_name, count)
          cv2.imwrite(new_path_and_filename, image)
          success, image = vidcap.read()
          count += 1

  def __get_in_out_paths(self, date, subj, out_subdir):
      in_video_subdirs = list(set(self.config.frames_subdirs_dict.keys()))
      in_paths = []
      out_paths = []
      for sub_dir in in_video_subdirs:
        in_paths.append('{}/{}/{}/{}'.format(self.config.raw_data_path, date, subj, sub_dir))
        out_paths.append('{}/{}/{}/w/{}'.format(self.config.data_path, out_subdir, subj, self.config.frames_subdirs_dict[sub_dir]))
      print(out_paths)
      return in_paths, out_paths

  ''' Rearrange: '''
  def __rearrange_input(self, x):
      y = []
      for i in range(self.config.number_of_classes):
          y.append([])
      if self.verbose:
        print(f'max_chanks={self.config.MAX_CHUNKS_PER_CATEGORY}')
      for i in range(self.config.number_of_classes): #short loop, as number of categories
          if len(x[i]) == 0:
            print(f'no items in position {i}')
            continue
          print(np.shape(x[i]))
          p = np.random.permutation(np.shape(x[i])[0])
          print(np.shape(p))
          print(type(x[i]))
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
      if self.verbose:
          print(f'=======\n{out_subdir}\n =======')

      for date in dates:
          for subj in subjects_dirs:
              in_paths, out_paths = self.__get_in_out_paths(date, subj, out_subdir)
              for d in out_paths:
                  if not os.path.exists(d):
                      os.makedirs(d)
              for p in zip(in_paths, out_paths):
                  print(p)
                  for vid_filename in tqdm(glob.glob('{}/*'.format(p[0]))):
                      if os.path.isfile(vid_filename):
                          self.__split_video_to_frames(vid_filename, p[1])

  def create_data_set(self):     
      self.create_video_frames(self.config.train_dates, self.config.train_subjects, 'train')
      self.create_video_frames(self.config.val_dates, self.config.val_subjects, 'validation')
      self.create_video_frames(self.config.test_dates, self.config.test_subjects, 'test')

  ''' Frames to chunks of frames: '''
  def prep_frames_lists(self, dates, samples, out_subdir, ABSOLUTE_MAX_FRAMES_PER_CATEGORY = 5000000):
      x = []
      for i in range (self.config.number_of_classes):
          x.append([])
      for date in dates:
          for sample in samples:
              in_paths, out_paths = self.__get_in_out_paths(date, sample, out_subdir)
              for p in zip(out_paths, self.config.lables_categories):
                  frames_list_chunk = []
                  for frame_filename in tqdm(glob.glob('{}/*'.format(p[0]))):
                      if os.path.isfile(frame_filename):
                          image = cv2.imread(frame_filename)
                          luma = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)[::, ::, 0:1]
                          frames_list_chunk.append(luma)
                  if len(frames_list_chunk) > 0:
                      nitems = len(frames_list_chunk)
                      n_chunks = int(min(nitems, ABSOLUTE_MAX_FRAMES_PER_CATEGORY)/self.config.chunk_size)
                      x_sub = np.split(np.array(frames_list_chunk[0:(self.config.chunk_size*n_chunks)]), n_chunks)
                      index = p[1]
                      if len(x[index]) == 0:
                        x[index] = x_sub
                      else:
                        x[index] = np.append(x[index], x_sub, axis=0)
      if self.verbose:
        print(f'shape of the data {np.shape(x)}')
      return np.array(x)

  def limit_rearrange_and_flatten(self, x):
    x, y = self.__rearrange_input(x)
    x_out = x[0]
    y_out = y[0]
    for i in range(1, len(self.config.binary_lables)): #short loop, as number of categories
      x_out = np.append(x_out, x[i], axis=0)
      y_out = np.append(y_out, y[i], axis=0)
    return x_out, y_out

  def prepare_train_and_validation_data(self):
    '''train set preprocessing'''
    x_train_per_category = self.prep_frames_lists(self.config.train_dates, self.config.train_subjects, 'train')
    x_train, y_train = self.limit_rearrange_and_flatten(x_train_per_category)
    if config.verbose:
      print(np.shape(x_train))
      print(np.shape(y_train))
    x_train, y_train = prep.limit_rearrange_and_flatten(x_train_per_category)
    if config.verbose:
      print(np.shape(x_train))
      print(np.shape(y_train))

    '''validation set preprocessing'''
    x_val_per_category = self.prep_frames_lists(self.config.val_dates, self.config.val_subjects, 'validation')
    x_val, y_val = self.limit_rearrange_and_flatten(x_val_per_category)
    if config.verbose:
      print(np.shape(x_val))

    '''shaffle train and validation sets'''
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_val, y_val = unison_shuffled_copies(x_val, y_val)
    return x_train, y_train, x_val, y_val

  def prepare_test_data(self):
    '''test set preprocessing'''
    x_test_per_category  = self.prep_frames_lists(self.config.test_dates, self.config.test_subjects, 'test')
    x_test, y_test = self.limit_rearrange_and_flatten(x_test_per_category)
    if self.verbose:
      print(np.shape(x_test))
    return x_test, y_test
