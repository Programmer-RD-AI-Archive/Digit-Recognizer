import math
import os
import threading

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import wandb
from sklearn.model_selection import train_test_split
from torch.nn import *
from torch.optim import *
from torchvision.models import *

from Model.dataset import *

try:
    from tqdm import tqdm
except Exception as e:
    raise ImportError(f"""
        Cannot Import Tqdm try installing it using 
        `pip3 install tqdm` 
        or 
        `conda install tqdm`.
        \n 
        {e}""")
try:
    import ray
    from ray import tune
except Exception as e:
    tune = None
device = "cuda"
PROJECT_NAME = "Digit-Recognizer"
