import numpy as np
np.random.seed(1337)  # for reproducibility
import pandas as pd
import os
from typing import List, Tuple, Generator, Iterator
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
from tabulate import tabulate
import csv
import time
from natsort import natsorted
import pickle
import multiprocessing as mp
from os import path

from sklearn import preprocessing
from sklearn import base
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler, OneHotEncoder, QuantileTransformer)
from sklearn.model_selection import train_test_split

import zipfile
import tarfile
import rarfile
from tqdm.auto import tqdm
TupleOrList = tuple([Tuple, List])

class CustomMerger(base.BaseEstimator, base.TransformerMixin):
  """Merge List of DataFrames"""

  def __init__(self):
    pass

  def fit(self, X: pd.DataFrame, y=None):
    return self

  def transform(self, X: pd.DataFrame, y='deprecated', copy=True):
    if isinstance(X, TupleOrList):
        return pd.concat(X, ignore_index=True).reset_index(drop=True)

    return X
  
class CustomEncoder(base.BaseEstimator, base.TransformerMixin):
  """Custom encoder data"""
  def __init__(self):
      self.mapping = {}
      self.inverse_mapping = {}
      self.next_label = 0
  
  def fit(self, data):
      self._check_for_null(data)
      for value in data:
          if value not in self.mapping:
              self.mapping[value] = self.next_label
              self.inverse_mapping[self.next_label] = value
              self.next_label += 1
  
  def transform(self, data, copy=True):
      self._check_for_null(data)
      encoded_data = []
      for value in data:
          encoded_value = self.mapping.get(value, -1)  # Return -1 for unseen values
          encoded_data.append(encoded_value)

      if isinstance(encoded_data, TupleOrList):
        return pd.concat(encoded_data, ignore_index=True).reset_index(drop=True)
      return encoded_data
  
  def _check_for_null(self, data):
      if any(value is None or (isinstance(value, float) and math.isnan(value)) for value in data):
          raise ValueError("Input data contains null or NaN values.")
  
class CustomScaler(base.BaseEstimator, base.TransformerMixin):
  """Standardize custom features"""

  def __init__(self):
    self.mean_ = None
    self.std_ = None

  def fit(self, X, y=None):
    self.mean_ = sum(X) / len(X)
    self.std_ = (sum((x - self.mean_) ** 2 for x in X) / len(X)) ** 0.5
    return self

  def transform(self, X, y='deprecated', copy=True):
    if self.mean_ is None or self.std_ is None:
        raise ValueError("Scaler has not been fitted yet.")
    scaled_X = []
    for x in X:
        scaled_x = (x - self.mean_) / self.std_
        scaled_X.append(scaled_x)

    if isinstance(scaled_X, TupleOrList):
        return pd.concat(scaled_X, ignore_index=True).reset_index(drop=True)
    return scaled_X
  
def ExtractFile(file_path, extract_to='.'):
    """
    Extracts a compressed file to the specified directory.
    
    Args:
        file_path (str): The path to the compressed file.
        extract_to (str): The directory to extract the files to. Defaults to the current directory.
    """
    total_size = 0
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    try:
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                total_size = sum([info.file_size for info in zip_ref.infolist()]) / (1024 * 1024)
                print(f"Attention !!! Your chosen dataset will take {total_size:.2f} MB in local storage. Use Ctrl+C to abort before the process start.")
                for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), desc="Extracting ZIP"):
                    zip_ref.extract(member=file, path=extract_to)

        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                total_size = sum([member.size for member in tar_ref.getmembers()]) / (1024 * 1024)
                print(f"Attention !!! Your chosen dataset will take {total_size:.2f} MB in local storage. Use Ctrl+C to abort before the process start.")
                for member in tqdm(iterable=tar_ref.getmembers(), total=len(tar_ref.getmembers()), desc="Extracting TAR.GZ"):
                    tar_ref.extract(member=member, path=extract_to)
        
        elif file_path.endswith('.rar'):
            with rarfile.RarFile(file_path, 'r') as rar_ref:
                total_size = sum([info.file_size for info in rar_ref.infolist()]) / (1024 * 1024)
                print(f"Attention !!! Your chosen dataset will take {total_size:.2f} MB in local storage. Use Ctrl+C to abort before the process start.")
                for file in tqdm(iterable=rar_ref.infolist(), total=len(rar_ref.infolist()), desc="Extracting RAR"):
                    rar_ref.extract(member=file, path=extract_to)
        else:
            print(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"Failed to extract file: {file_path}")
        print(e)
    return total_size
