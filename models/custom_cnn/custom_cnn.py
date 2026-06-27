from __future__ import annotations

from tensorflow import keras

from shared import config


def build_model(num_classes: int, augmentation: keras.Sequential) -> keras.Model:

    inputs = keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))

    x = augmentation(inputs)                  
    x = keras.layers.Rescaling(1.0 / 255)(x)  
    # Convolutional Block 1
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)

    # Convolutional Block 2
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)

    # Convolutional Block 3
    x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)

    # Classification head
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="custom_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
