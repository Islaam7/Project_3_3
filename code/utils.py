import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from textwrap import wrap


def make_dirs(path):
    os.makedirs(path, exist_ok=True)
