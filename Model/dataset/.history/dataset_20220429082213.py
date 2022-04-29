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
        images = np.array(images) / 255.0
        return images, labels, self.data["label"].value_counts().value

    def get_labels(self, y):
        idx = 0
        labels = {}
        labels_r = {}
        for y_iter in y:
            idx += 1
            labels[y_iter] = idx
            labels_r[idx] = y_iter
        return labels, idx, labels_r

    @staticmethod
    def create_np_eye_list_with_label(idx: int, class_name: any, labels: dict) -> np.array:
        current_idx = labels[class_name]
        max_idx = idx
        np_eye = np.eye(current_idx, max_idx)
        np_eye = np_eye[-1]
        return np_eye

    def load_data(self):
        X, y, classes = self.data_to_X_and_y()
        labels, idx, labels_r = self.get_labels(y)
        new_y = []
        for y_iter in y:
            new_y.append(self.create_np_eye_list_with_label(idx, y_iter, labels))
        
