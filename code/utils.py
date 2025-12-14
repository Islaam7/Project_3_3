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

        def plot_confusion(cm, class_names, out_path,
                           normalize=True,
                           figsize=(20, 20),
                           cmap=plt.cm.Blues,
                           dpi=200,
                           annotate=True,
                           annotate_threshold=40,
                           max_label_len=25):
            """
            Improved confusion matrix plot for many classes.

            - normalize: show proportions (True) or raw counts (False)
            - annotate: whether to write numbers inside cells (disabled automatically if too many classes)
            - annotate_threshold: if n_classes > threshold then disable annotations to avoid clutter
            - max_label_len: truncate/word-wrap long class names to make labels shorter
            """

            cm = np.array(cm)
            n_classes = len(class_names)

            # Possibly normalize
            if normalize:
                with np.errstate(all='ignore'):
                    row_sums = cm.sum(axis=1, keepdims=True)
                    cm_norm = cm.astype('float') / (row_sums + 1e-12)
                disp_cm = cm_norm
                fmt = '.2f'
            else:
                disp_cm = cm
                fmt = 'd'

            # Shorten or wrap class names to reduce clutter
            short_names = []
            for name in class_names:
                name = name.replace('_', ' ')
                if len(name) > max_label_len:
                    # wrap long names into multiple lines
                    name = '\n'.join(wrap(name, max_label_len))
                short_names.append(name)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            im = ax.imshow(disp_cm, interpolation='nearest', cmap=cmap)
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Proportion' if normalize else 'Count', rotation=-90, va="bottom")

            # Tick marks and labels
            tick_marks = np.arange(n_classes)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(short_names, fontsize=10)

            ax.set_ylabel('True label', fontsize=12)
            ax.set_xlabel('Predicted label', fontsize=12)
            ax.set_title('Confusion matrix', fontsize=14)

            # Decide whether to annotate numbers inside cells
            do_annotate = annotate and (n_classes <= annotate_threshold)
            if do_annotate:
                # choose text color depending on cell brightness
                thresh = disp_cm.max() / 2.0
                for i in range(disp_cm.shape[0]):
                    for j in range(disp_cm.shape[1]):
                        value = disp_cm[i, j]
                        if normalize:
                            txt = format(value, fmt)
                        else:
                            txt = format(int(value), fmt)
                        color = "white" if value > thresh else "black"
                        ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

            # Tight layout to avoid cutoff
            plt.tight_layout()

            # Save with high resolution
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
            plt.close(fig)

            # If annotations were disabled because too many classes, save a small text file to help interpret
            if (not do_annotate):
                mapping_path = os.path.splitext(out_path)[0] + "_class_index.txt"
                with open(mapping_path, "w", encoding="utf-8") as f:
                    f.write("Index -> Class name (wrapped/shortened)\n")
                    for idx, name in enumerate(short_names):
                        f.write(f"{idx}\t{name}\n")
def plot_sample_predictions(model, test_gen, class_names, out_path, n_samples=39):
    """
    Generates a grid of sample test images with predicted + true labels.
    Saves output to sample_predictions.png
    """

    # Fetch a full batch from generator (no shuffle)
    images, labels = next(test_gen)  # labels are one-hot encoded
    y_true = np.argmax(labels, axis=1)

    # Predict
    y_prob = model.predict(images)
    y_pred = np.argmax(y_prob, axis=1)

    # Plot
    cols = 4
    rows = n_samples // cols
    plt.figure(figsize=(16, 16))

    for i in range(n_samples):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i])
        plt.axis('off')

        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]

        color = "green" if y_true[i] == y_pred[i] else "red"
        title = f"T: {true_label}\nP: {pred_label}"

        plt.title(title, color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()