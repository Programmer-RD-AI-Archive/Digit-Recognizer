"""sumary_line"""
from Model import *


class Metrics:
    def loss(self,model,X,y,criterion):
        preds = model(X)
        loss = criterion(preds,y)
        return loss.item()

    def accuracy(self,model,X,y):
        preds = model(X)
        for pred,y_iter in tqdm(zip(preds,y)):
            pred = torch.argmax(pred)
            y_iter = torch.argmax(y_iter)
