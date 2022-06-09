import os
import cv2
import torch
import torchvision
import wandb
import threading
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision.models import *
from sklearn.model_selection import train_test_split
from torch.nn import *
from torch.optim import *

try:
    from tqdm import tqdm
except Exception as e:
    raise ImportError(
        f"""
        Cannot Import Tqdm try installing it using 
        `pip3 install tqdm` 
        or 
        `conda install tqdm`.
        \n 
        {e}"""
    )
try:
    import ray
    from ray import tune
except Exception as e:
    tune = None
device = "cuda"
PROJECT_NAME = "Digit-Recognizer"
from Model.dataset import *
