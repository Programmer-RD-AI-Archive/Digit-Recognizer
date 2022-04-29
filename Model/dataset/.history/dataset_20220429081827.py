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
        self.data = self.data.sample(frac=1.0)
        self.test_data = pd.read_csv(os.path.join(self.raw_dir, test_data_file_name))
        self.sample_submission = pd.read_csv(os.path.join(self.raw_dir, sample_submission))

    def data_to_X_and_y(self):
        images = self.data.drop("label", axis=1)
        labels = self.data["label"].values
        img_size = math.sqrt(np.array(images.shape[1]))
        images = images.values.reshape(-1, 1, img_size, img_size)
        labels = np.array(labels)
        images = np.array(images)
        return images, labels, self.data["label"].value_counts().value

    def get_labels(self, y):
        idx = 0
        labels = {}
        labels_r = {}
        for y_iter in y:
            pass
