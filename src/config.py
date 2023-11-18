import os

# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 3
LEARNING_RATE = 0.001

BATCH_SIZE = 2
# BATCH_SIZE = 64
NUM_EPOCHS = 3

RESIZE_SIZE = {
    'efficientnet_b0': (256, 256, 3), 
    'efficientnet_b1': (256, 256, 3), 
    'efficientnet_b2': (280, 280, 3), 
    'efficientnet_b3': (320, 320, 3), 
    'efficientnet_b4': (384, 384, 3), 
    'efficientnet_v2_s': (224, 224, 3), 
    'efficientnet_v2_m': (224, 224, 3), 
}

CROP_SIZE = {
    'efficientnet_b0': (224, 224, 3), 
    'efficientnet_b1': (240, 240, 3), 
    'efficientnet_b2': (280, 280, 3), 
    'efficientnet_b3': (224, 224, 3), 
    'efficientnet_b4': (224, 224, 3), 
    'efficientnet_v2_s': (384, 384, 3), 
    'efficientnet_v2_m': (480, 480, 3), 
}

# Dataset
ROOT_DIR = os.getcwd()
# ROOT_DIR = os.path.abspath('..')
# DATA_DIR = "../data/raw/"
DATA_DIR = "D:/Projects/Skin Disease Detection/Dataset/LatestDataset"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16