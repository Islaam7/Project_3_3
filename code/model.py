from tensorflow.keras import layers, models


def build_simple_cnn(input_shape=(128, 128, 3), num_classes=38):
    model = models.Sequential([
        layers.Input(shape=input_shape),
    ])
    return model
