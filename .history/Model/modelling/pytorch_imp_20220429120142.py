"""sumary_line"""
from Model import *


class CLF(Module):
    def __init__(self, img_size: int = 1 * 28 * 28, idx_of_classes: int = 0) -> None:
        super().__init__()
        self.activation = ReLU()
        self.linear1 = Linear(img_size, 256)
        self.linear2 = Linear(256, 512)
        self.linear3batchnorm = BatchNorm1d(512)
        self.linear4 = Linear(512, 1024)
        self.linear5 = Linear(1024, 512)
        self.output = Linear(512, idx_of_classes)

    def forward(self, X) -> torch.tensor():
        preds = self.activation(self.linear1(X))
        preds = self.activation(self.linear2(preds))
        preds = self.linear3batchnorm(preds)
        preds = self.activation(self.linear4(preds))
        preds = self.activation(self.linear5(preds))
        preds = self.output(preds)
        return preds


class CNN(Module):
    def __init__(self, activation=ReLU(), idx_of_classes: int = 0) -> None:
        """sumary_line"""
        super().__init__()
        self.max_pool2d = MaxPool2d((2, 2))
        self.activation = activation
        self.conv1 = Conv2d(1, 16, (3, 3))
        self.conv2batchnorm = BatchNorm2d(16)
        self.conv3 = Conv2d(16, 32, (3, 3))
        self.conv4batchnorm = BatchNorm2d(32)
        self.conv5 = Conv2d(32, 64, (3, 3))
        self.linear1 = Linear(64 * 3 * 3, 128)
        self.linear2batchnorm = BatchNorm1d(128)
        self.linear3 = Linear(128, 256)
        self.linear4batchnorm = Linear(256, 512)
        self.linear5 = Linear(512, 256)
        self.output = Linear(256, idx_of_classes)

    def forward(self, X) -> torch.tensor:
        """sumary_line"""
        preds = self.max_pool2d(self.activation(self.conv1(X)))
        preds = self.max_pool2d(self.activation(self.conv2batchnorm(preds)))
        preds = self.max_pool2d(self.activation(self.conv3(preds)))
        preds = self.max_pool2d(self.activation(self.conv4batchnorm(preds)))
        preds = self.conv5(preds)
        print(preds.shape)
        preds = preds.view(-1, 64 * 3 * 3)
        preds = self.activation(self.linear1(preds))
        preds = self.activation(self.linear2batchnorm(preds))
        preds = self.activation(self.linear3(preds))
        preds = self.activation(self.linear4batchnorm(preds))
        preds = self.activation(self.linear5(preds))
        preds = self.output(preds)
        return preds


class TL_Model(Module):
    def __init__(self,tl_model):
        super().__init__()
        self.tl_model = 
