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