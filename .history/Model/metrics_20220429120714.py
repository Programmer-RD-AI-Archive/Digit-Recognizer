"""sumary_line"""
from Model import *


class Metrics:
    def loss(self,model,X,y,criterion):
        preds = model(X)
        loss = criterion(preds,y)
        

    def accuracy(self):
        pass
