"""sumary_line"""
import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dataset.dataset import *
from dataset.preprocessing import *
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import *
from tqdm import tqdm

SAVE_DIR = "./dataset/data/"
RAW_DIR = "./dataset/raw/"
TEST_SIZE = 0.25
EPOCHS = 100
BATCH_SIZE = 32
