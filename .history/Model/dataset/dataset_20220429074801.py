"""sumary_line"""
import os
import cv2
import pandas as pd
import numpy as np
import torch


class DataSet:
    def __init__(self, save_dir, raw_dir) -> None:
        self.save_dir = save_dir
        self.raw_dir = raw_dir
        self.data = pd.read_csv(os.path.join(self.save_dir, 'data.csv'))
