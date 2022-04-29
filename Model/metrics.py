"""sumary_line"""
import os

import cv2
import matplotlib.pyplot as plt
import torch

from Model import *


class Metrics:
    def loss(self, model, X, y, criterion) -> float:
        preds = model(X)
        loss = criterion(preds, y)
        return loss.item()

    def accuracy(self, model, X, y) -> float:
        correct = 0
        total = 0
        preds = model(X)
        for pred, y_iter in tqdm(zip(preds, y)):
            pred = torch.argmax(pred)
            y_iter = torch.argmax(y_iter)
            if pred == y_iter:
                correct += 1
            total += 1
        return round(correct / total, 3)

    def test_images(self, model, labels_r, device):
        model.eval()
        with torch.no_grad():
            for file_path in os.listdir("./Model/tests/"):
                img = cv2.imread(
                    f"./Model/tests/{file_path}", cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (28, 28))
                img = img / 255.0
                pred = model(torch.tensor(
                    img).view(-1, 1, 28, 28).float().to(device))
                print("*" * 50)
                print(file_path)
                print(pred)
                pred = torch.argmax(pred)
                print(pred)
                print("*" * 50)
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                plt.title(f"Prediction: {labels_r[int(pred)]}")
                plt.savefig(f"./Model/preds/{file_path}")
                plt.close()
            preds_files = []
            for file_path in os.listdir("./Model/preds/"):
                preds_files.append(
                    [file_path, cv2.imread(f"./Model/preds/{file_path}")]
                )
        model.train()
        return preds_files
