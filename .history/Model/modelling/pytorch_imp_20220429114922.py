"""sumary_line"""
from Model import *


class CLF:
    pass


class CNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1,16,(3,3))
        self.conv2batchnorm = BatchNorm2d()


class TL_Model:
    pass
