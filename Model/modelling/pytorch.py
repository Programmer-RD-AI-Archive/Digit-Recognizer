from Model import *


class Clf(Module):

    def __init__(self, classes: list):
        self.activation = ReLU()
        self.linear1 = Linear(28 * 28, 128)
        self.linear2 = Linear(128, 256)
        self.batchnorm = BatchNorm1d(256)
        self.linear3 = Linear(256, 512)
        self.linear4 = Linear(512, 256)
        self.output = Linear(256, len(classes))

    def forward(self, X):
        pass
