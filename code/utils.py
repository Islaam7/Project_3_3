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


def plot_history(history, output_dir: str = None):
    if isinstance(history, dict):
        h = history
    else:
        h = history.history

    make_dirs(output_dir) if output_dir else None

    # Loss
    plt.figure()
    plt.plot(h.get("loss", []), label="train_loss")
    plt.plot(h.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if output_dir:
        p = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(p)
    plt.close()

    # Accuracy (supports 'accuracy' or 'categorical_accuracy')
    acc_key = "accuracy" if "accuracy" in h else "categorical_accuracy" if "categorical_accuracy" in h else None
    if acc_key:
        plt.figure()
        plt.plot(h.get(acc_key, []), label="train_acc")
        plt.plot(h.get("val_" + acc_key, []), label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        if output_dir:
            p = os.path.join(output_dir, "accuracy_curve.png")
            plt.savefig(p)
        plt.close()