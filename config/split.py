import yaml
from typing import List, Tuple, Set
from enum import Enum
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

class Logical_Split():
  class Sample_time(Enum):
      ONLY_MORNING = 1
      ONLY_MID_DAY = 2
      MORNING_AND_MID_DAY = 3

  def __init__(self, config_file_name = 'config/subjects_and_dates.yml', verbose = True):
    """
    Initialize the Logical_Split object.
    Args:
      config_file_name: config file name.
      verbose: verbose mode.
    """
    self.config_file_name = config_file_name
    self.verbose = verbose

  ''' Translate subjects to dates and directories(name)'''
  def get_params_per_subject(self, subject, sample_time):
      """
      Retrieves dates and subject names for a given subject number based on sample time.
      
      Args:
      subject (int): Subject number.
      sample_time (Sample_time): Enum indicating the sample time preference.
      
      Returns:
      tuple: A tuple containing the list of dates and a list with a single subject name.
      
      Raises:
      ValueError: If the subject number is not found in the data.
      """
      # Load the configuration from the YAML file
      with open(self.config_file_name, 'r') as file:
          data = yaml.safe_load(file)
          subject_config = data['subjects']

      # Get the configuration for the given subject or raise an error if the subject is not defined
      config = subject_config.get(subject) 
      if config is None:
          raise ValueError("Invalid subject number")

      # Determine the dates based on the sample time
      if sample_time == self.Sample_time.ONLY_MORNING:
          dates = [config['dates'][0]]
      elif sample_time == self.Sample_time.ONLY_MID_DAY:
          dates = [config['dates'][1]]
      else:
          dates = config['dates']

      # Retrieve the name associated with the subject
      subjects = [config['name']]
      if self.verbose:
        print(f' subject number = {subject}, name = {subjects},  dates={dates} ')
      return dates, subjects

  def get_dates_and_subjects(self, subjects: List[int], sample_type: Sample_time = Sample_time.MORNING_AND_MID_DAY) -> Tuple[Set[str], Set[str]]:
      """
      Aggregate unique dates and subject directories from a list of subjects based on the specified sample time.

      Args:
          subjects (List[int]): List of subject identifiers.
          sample_type (Sample_time): Enum value that specifies the sample time preference.

      Returns:
          Tuple[Set[str], Set[str]]: A tuple of sets containing unique dates and subject directories.
      """
      unique_dates = set()
      unique_subject_dirs = set()

      for subject_id in subjects:
          dates, subject_names = self.get_params_per_subject(subject_id, sample_type)
          unique_dates.update(dates)
          unique_subject_dirs.update(subject_names)

      return unique_dates, unique_subject_dirs
