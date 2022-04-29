"""sumary_line"""
from Model import *


class CLF:
    pass


class CNN(Module):
    def __init__(self,activation=ReLU(),idx_of_classes:int=0):
        super().__init__()
        self.max_pool2d = MaxPool2d((2,2))
        self.activation = activation
        self.conv1 = Conv2d(1,16,(3,3))
        self.conv2batchnorm = BatchNorm2d(16)
        self.conv3 = Conv2d(16,32,(3,3))
        self.conv4batchnorm = BatchNorm2d(32)
        self.conv5 = Conv2d(32,64,(3,3))
        self.linear1 = Linear(64*3*3,128)
        self.linear2batchnorm = BatchNorm1d(128)
        self.linear3 = Linear(128,256)
        self.linear4batchnorm = Linear(256,512)
        self.linear5 = Linear(512,256)
        self.output = Linear(256,idx_of_classes)
        


class TL_Model:
    pass
