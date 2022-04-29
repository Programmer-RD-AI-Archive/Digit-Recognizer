"""sumary_line"""
import pandas as pd
import numpy as np
from preproccessing import PreProccessing
import torch
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DataSet:
    def __init__(
        self,
        save_dir: str = "./data/",
        raw_dir: str = "./raw/",
        train_data_file_name: str = "train.csv",
        test_data_file_name: str = "test.csv",
        sample_submission: str = "sample_submission.csv",
        preproccessing: PreProccessing = PreProccessing(),
    ) -> None:
        """sumary_line"""
        self.save_dir = save_dir
        self.raw_dir = raw_dir
        self.data = pd.read_csv(f"{raw_dir}{train_data_file_name}")
        self.data = self.data.sample(frac=1.0)
        self.test_data = pd.read_csv(f"{raw_dir}{test_data_file_name}")
        self.sample_submission = pd.read_csv(f"{raw_dir}{sample_submission}")
        self.preproccessing = preproccessing

    # Analytics

    def analytics(self):
        """sumary_line"""
        chart_info = self.data["label"].value_counts().to_dict()
        classes = chart_info.keys()
        num_of_imgs = chart_info.values()
        plt.figure(figsize=(10, 6))
        plt.bar(classes, num_of_imgs)
        plt.xlabel("Class")
        plt.ylabel("Amount of Imgs")
        plt.title("Class in Relation to Amount of Imgs")
        plt.savefig("./data/class_in_relation_to_amount_of_imgs.png")
        plt.close()
        return chart_info, classes, num_of_imgs

    # Load Data

    def data_to_X_and_y(self):
        """sumary_line"""
        images = self.data.drop("label", axis=1)
        labels = self.data["label"]
        img_size = int(math.sqrt(np.array(images.shape[1])))
        images = np.array(images).reshape(-1, img_size, img_size)
        labels = np.array(labels)
        new_images = []
        for img in images:
            img = self.preproccessing.preproccess(img)
            new_images.append(img)
        images = np.array(new_images.copy())
        return images, labels, list(dict(self.data["label"].value_counts()).keys())

    def get_labels(self, y):
        """sumary_line"""
        idx = 0
        labels = {}
        labels_r = {}
        for y_iter in tqdm(y):
            idx += 1
            labels[y_iter] = idx
            labels_r[idx] = y_iter
        return labels, idx, labels_r

    @staticmethod
    def create_np_eye_list_with_label(idx: int, class_name: any, labels: dict) -> np.array:
        """sumary_line"""
        current_idx = labels[class_name]
        max_idx = idx
        np_eye = np.eye(current_idx, max_idx)
        np_eye = np_eye[-1]
        return np_eye

    def X_and_y_to_X_train_y_train_X_test_y_test(
        self, X: list, y: list, test_size: float = 0.25, shuffle: bool = True
    ) -> tuple:
        """sumary_line"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle
        )
        X_train = torch.from_numpy(np.array(X_train))
        y_train = torch.from_numpy(np.array(y_train))
        X_test = torch.from_numpy(np.array(X_test))
        y_test = torch.from_numpy(np.array(y_test))
        torch.save(X_train, self.save_dir + "X_train.pt")
        torch.save(X_train, self.save_dir + "X_train.pth")
        torch.save(X_test, self.save_dir + "X_test.pt")
        torch.save(X_test, self.save_dir + "X_test.pth")
        torch.save(y_train, self.save_dir + "y_train.pt")
        torch.save(y_train, self.save_dir + "y_train.pth")
        torch.save(y_test, self.save_dir + "y_test.pt")
        torch.save(y_test, self.save_dir + "y_test.pth")
        return (X_train, X_test, y_train, y_test)

    def load_data(self, matrix_type_y: bool = True):
        """sumary_line"""
        print("Loading Data")
        X, y, classes = self.data_to_X_and_y()
        print("Data Loaded")
        print("Creating Labels from the Classes")
        labels, idx, labels_r = self.get_labels(classes)
        print("Created Labels from the Classes")
        new_y = []
        for y_iter in y:
            if matrix_type_y:
                new_y.append(self.create_np_eye_list_with_label(idx, y_iter, labels))
            else:
                new_y.append(labels[y_iter])
        y = np.array(new_y)
        print("Converting Data -> X,y + train,test")
        X_train, X_test, y_train, y_test = self.X_and_y_to_X_train_y_train_X_test_y_test(
            list(X), list(y)
        )
        return X, y, classes, labels, idx, labels_r, X_train, y_train, X_test, y_test


Testing
ds = DataSet()
X, y, classes, labels, idx, labels_r, X_train, y_train, X_test, y_test = ds.load_data()
print(len(X_train), len(X_test), len(y_train), len(y_test))
