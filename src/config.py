import os
import torch
from torchvision.models import (efficientnet_b0, EfficientNet_B0_Weights,
                                efficientnet_b1, EfficientNet_B1_Weights, 
                                efficientnet_b2, EfficientNet_B2_Weights, 
                                efficientnet_b3, EfficientNet_B3_Weights, 
                                efficientnet_b4, EfficientNet_B4_Weights, 
                                efficientnet_v2_m, EfficientNet_V2_M_Weights, 
                                efficientnet_v2_s, EfficientNet_V2_S_Weights)

# Hyperparameters
NUM_CLASSES = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 5
# BATCH_SIZE = 32
NUM_EPOCHS = 3

# Paths
# ROOT_DIR = os.path.abspath('..')
# DATA_DIR = "../data/raw/"
ROOT_DIR = os.getcwd()
DATA_DIR = 'D:/Projects/Skin Disease Detection/Dataset/LatestDataset'
TB_LOG_DIR = '../logs/tensorboard/'

# Compute related
NUM_WORKERS = 4
ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
DEVICES = [0]
PRECISION = 'bf16-mixed' #16

#TODO: figure out what this does
PERCENT_VALID_EXAMPLES = 0.2

# For base model search
RESIZE_SIZE = {
    'efficientnet_b0': (256, 256, 3), 
    'efficientnet_b1': (255, 255, 3), 
    'efficientnet_b2': (288, 288, 3), 
    'efficientnet_b3': (320, 320, 3), 
    'efficientnet_b4': (384, 384, 3), 
    'efficientnet_v2_s': (384, 384, 3), 
    'efficientnet_v2_m': (480, 480, 3), 
}

CROP_SIZE = {
    'efficientnet_b0': (224, 224, 3), 
    'efficientnet_b1': (240, 240, 3), 
    'efficientnet_b2': (288, 288, 3), 
    'efficientnet_b3': (300, 300, 3), 
    'efficientnet_b4': (380, 380, 3), 
    'efficientnet_v2_s': (384, 384, 3), 
    'efficientnet_v2_m': (480, 480, 3), 
}

def get_base_model(base_model_name):
    assert base_model_name in [
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
        'efficientnet_b4', 'efficientnet_v2_s', 'efficientnet_v2_m'
    ], f'Invalid base model name: {base_model_name}'

    return (
        efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT) if base_model_name == 'efficientnet_b0' else
        efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT) if base_model_name == 'efficientnet_b1' else
        efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT) if base_model_name == 'efficientnet_b2' else
        efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT) if base_model_name == 'efficientnet_b3' else
        efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT) if base_model_name == 'efficientnet_b4' else
        efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT) if base_model_name == 'efficientnet_v2_s' else
        efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    )
