import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from textwrap import wrap


def make_dirs(path):
    os.makedirs(path, exist_ok=True)
def get_callbacks(output_dir: str):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(output_dir + 'best_model.h5', monitor='val_loss', save_best_only=True)
    ]
    return callbacks
