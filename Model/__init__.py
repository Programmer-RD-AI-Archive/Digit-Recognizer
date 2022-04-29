"""sumary_line"""
import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import *
from torch.optim import *
from torchvision.transforms import *
from tqdm import tqdm
import wandb
from torchvision.models import *

SAVE_DIR = "./dataset/data/"
RAW_DIR = "./dataset/raw/"
TEST_SIZE = 0.25
EPOCHS = 100
BATCH_SIZE = 32
DEVICE = "cuda"
PROJECT_NAME = "Digit-Recognizer-Competition"

from Model.dataset.preproccessing import *
from Model.dataset.dataset import *
from Model.help_funcs import *
from Model.modelling.pytorch_imp import *
