import os
import tensorflow as tf
from model import build_simple_cnn
from utils import get_callbacks, plot_history
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    print("Loading datasets...")
    BASE_DIR = "C:\\My folder\\CS417 (Neural Networks)\\MyProjects\\AI_Project\\Code\\PlantVillage_split"
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    VAL_DIR = os.path.join(BASE_DIR, "val")

    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 32
    SEED = 42