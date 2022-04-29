"""sumary_line"""
import os
import cv2
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

import math

class DataSet:
    def __init__(
        self,
        save_dir: str = "./data/",
        raw_dir: str = "./raw/",
        train_data_file_name: str = "train.csv",
        test_data_file_name: str = "test.csv",
        sample_submission: str = "sample_submission.csv",
    ) -> None:
        self.save_dir = save_dir
        self.raw_dir = raw_dir
        self.data = pd.read_csv(os.path.join(self.raw_dir, train_data_file_name))
        self.test_data = pd.read_csv(os.path.join(self.raw_dir, test_data_file_name))
        self.sample_submission = pd.read_csv(os.path.join(self.raw_dir, sample_submission))

    
