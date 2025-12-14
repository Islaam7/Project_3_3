import os
import shutil
import random
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
# constants
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def split_dataset(
    data_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(seed)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)


    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        random.shuffle(images)

        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        train_imgs = images[:train_count]
        val_imgs = images[train_count: train_count + val_count]
        test_imgs = images[train_count + val_count:]

        print(f"\nClass: {cls}")
        print(f"Total = {total} | Train = {len(train_imgs)} | Val = {len(val_imgs)} | Test = {len(test_imgs)}")

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        for img in tqdm(train_imgs, desc=f"{cls} → Train"):
            shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))

        for img in tqdm(val_imgs, desc=f"{cls} → Val"):
            shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

        for img in tqdm(test_imgs, desc=f"{cls} → Test"):
            shutil.copy(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))

    print("\n🎉 Done! Dataset split successfully.")

    if __name__ == "__main__":
        split_dataset(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=SEED
        )
