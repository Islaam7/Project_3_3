import os
import tensorflow as tf
from model import build_simple_cnn
from utils import get_callbacks, plot_history
from tensorflow.keras.preprocessing.image import ImageDataGenerator