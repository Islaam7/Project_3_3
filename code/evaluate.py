import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import plot_confusion , plot_sample_predictions

def main():
    BASE_DIR = r"C:\\Users\\MS\\Desktop\\mohamed\\417_Ai\\project_1_3\\code\\PlantVillage_split"
    TEST_DIR = os.path.join(BASE_DIR, "test")

    # generator for test (no augment, only rescale)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,   # مهم: لا shuffle للتوافق بين filenames و labels
        seed=42
    )

    # load model
    model = tf.keras.models.load_model(
        r"C:\\Users\\MS\\Desktop\\mohamed\\417_Ai\\project_1_3\\saved_model\\best_model.h5"
    )

    # predict on generator -> returns (n_samples, n_classes)
    y_prob = model.predict(test_gen, verbose=1)          # numpy array
    y_pred = np.argmax(y_prob, axis=1)                   # convert to class indices (1D)

    # true labels from generator
    # DirectoryIterator has attribute 'classes' which is a 1D array of labels (length = n_samples)
    y_true = test_gen.classes

    # Sanity check lengths
    print("y_true.shape =", y_true.shape)
    print("y_pred.shape =", y_pred.shape)
    assert len(y_true) == len(y_pred), "Number of predictions and true labels must match!"

    # class names (order matches indices)
    class_names = list(test_gen.class_indices.keys())

    # classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)


    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # use your plot_confusion from utils; it should accept (cm, class_names, out_path)
    out_cm = r"C:\\Users\\MS\\Desktop\\mohamed\\417_Ai\\project_1_3\\results\\confusion_matrix.png"
    # plot_confusion(cm, class_names, out_cm)
    plot_confusion(cm, class_names, out_cm, normalize=True, figsize=(18,18), dpi=200)
    sample_path = r"C:\\Users\\MS\\Desktop\\mohamed\\417_Ai\\project_1_3\\results\\sample_predictions.png"
    plot_sample_predictions(model, test_gen, class_names, sample_path, n_samples=16)

if __name__ == "__main__":
    main()
