# libraries.py – Common file for all libraries

# System libraries
import os
import sys
import time
import random
import warnings

# Numerical calculations and data processing
import numpy as np
import pandas as pd
import cupy as cp
import csv
import random as rd
from pathlib import Path

# Visualization
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import seaborn as sns
from PIL import Image
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import spectral

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import joblib

# Deep Learning (PyTorch, Keras)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
from torch.cuda import amp
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Weight Adjusters
from collections import Counter

# Optimizers
from ranger_adabelief import RangerAdaBelief

# HYPSO Package
import hypso
from hypso import Hypso2
from hypso.load import load_l1a_nc_cube # Raw
from hypso.load import load_l1b_nc_cube # Radiance
from hypso.load import load_l1c_nc_cube # Reflectance
from hypso.load import load_l1d_nc_cube # Reflectance

# Progress Bar
from tqdm import tqdm

# Cache
sys.dont_write_bytecode = True
torch.cuda.empty_cache()

# Logger
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
from termcolor import colored
warnings.filterwarnings("ignore")
import mlflow
import mlflow.pytorch

print("Libraries are loaded!")