from __future__ import annotations

from tensorflow import keras

from shared import config


def build_model(num_classes: int, augmentation: keras.Sequential) -> keras.Model:
    inputs = keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))

    x = augmentation(inputs)
    x = keras.layers.Rescaling(1./255)(x)

    # Convolutional Block 1
    x = keras.layers.Conv2D(256,(3,3), activation='relu', strides=(1,1), kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)

    # Convolutional Block 2
    x = keras.layers.Conv2D(128,(3,3), activation='relu', strides=(1,1), kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)

    # Convolutional Block 3
    x = keras.layers.Conv2D(128,(3,3), activation='relu', strides=(1,1), kernel_regularizer=keras.regularizers.l2(0.0001))(x)

    # Convolutional Block 4
    x = keras.layers.Conv2D(128,(3,3), activation='relu', strides=(1,1), kernel_regularizer=keras.regularizers.l2(0.0001))(x)

    # Convolutional Block 5
    x = keras.layers.Conv2D(64,(3,3), activation='relu', strides=(1,1), kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='valid')(x)

    # Fully Connected Layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="custom_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00025),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks() -> list[keras.callbacks.Callback]:

    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.25, patience=4, min_lr=1e-6, verbose=1
        )
    ]
