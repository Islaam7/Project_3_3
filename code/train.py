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

train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2)
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=SEED
    )

    class_indices = train_gen.class_indices
    print("Class indices (name -> index):")
    print(class_indices)

    model = build_simple_cnn(input_shape=(128, 128, 3), num_classes=len(class_indices))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = get_callbacks('C:\\My folder\\CS417 (Neural Networks)\\MyProjects\\AI_Project\\saved_model\\')

    print(model.summary())

    EPOCHS = 25
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    final_model_path = os.path.join("C:\\My folder\\CS417 (Neural Networks)\\MyProjects\\AI_Project\\saved_model\\",
                                    "best_model.h5")
    model.save(final_model_path)
    plot_history(history, "C:\\My folder\\CS417 (Neural Networks)\\MyProjects\\AI_Project\\results\\")