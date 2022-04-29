"""sumary_line"""
from Model import *


class CLF:
    pass


class CNN(Module):
    def __init__(self,activation=ReLU):
        super().__init__()
        self.max_pool2d = MaxPool2d((2,2))
        self.activation = activation
        self.conv1 = Conv2d(1,16,(3,3))
        self.conv2batchnorm = BatchNorm2d(16)
        self.conv3 = Conv2d(16,32,(3,3))
        self.conv4batchnorm = BatchNorm2d(32)
        self.conv5 = Conv2d(32,64,(3,3))
        


class TL_Model:
    pass
