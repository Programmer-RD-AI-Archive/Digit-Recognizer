"""sumary_line"""
from torchvision.transforms import *
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd
import torch
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SAVE_DIR = "./dataset/data/"
RAW_DIR = "./dataset/raw/"
TEST_SIZE = 0.25
EPOCHS = 100
BATCH_SIZE = 32

from dataset.preprocessing import *
from dataset.dataset import *
