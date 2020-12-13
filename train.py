import keras
import math
import os
import numpy as np
from keras import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from src.model import (
    get_model_artifacts,
    get_combined_soft_targets,
    get_data_generator_with_soft_targets,
    MODELS_CATALOG,
)
from src.paths import get_logs_path, get_model_path
import tensorflow as tf


ALIAS = "baseline-20201213"
T = 1.2535  # Temperature calculated to match distribution
validation_dataset_path = "/mnt/FLASH/datasets/imagenet/validation_256"

transfer_dataset_path = "/mnt/FLASH/datasets/imagenet/train_256"
transfer_dataset_alias = "train"
unlabeled = False

batch_size = 35
prob_norm = 0.35
base_model_name = "mobilenetv2"


def main():
    # Load base model
    get_model, preprocess_input, size = get_model_artifacts("mobilenetv2")
    model = get_model(classifier_activation="linear", weights="imagenet")

    # Warm last model layer
    input_ = model.input
    output = keras.backend.softmax(model.output / T)
    model = Model(input_, output)

    # Build soft-targets
    soft_targets = get_combined_soft_targets(
        MODELS_CATALOG, transfer_dataset_alias, prob_norm
    )

    # Build train data generator
    idg_train = ImageDataGenerator(preprocessing_function=preprocess_input)
    flow_train = idg_train.flow_from_directory(
        transfer_dataset_path,
        shuffle=True,
        target_size=(size, size),
        batch_size=batch_size,
    )
    gen_train = get_data_generator_with_soft_targets(
        flow_train, soft_targets, unlabeled
    )

    # Build validation data generator
    idg_val = ImageDataGenerator(preprocessing_function=preprocess_input)
    flow_val = idg_val.flow_from_directory(
        validation_dataset_path,
        shuffle=True,
        target_size=(size, size),
        batch_size=batch_size,
    )

    model.compile(
        Adam(learning_rate=1e-4), "categorical_crossentropy", metrics=["accuracy"]
    )

    print("Running initial validation...")
    model.evaluate(flow_val)
    print("Running train...")
    tb_callback = tf.keras.callbacks.TensorBoard(get_logs_path(ALIAS), update_freq=100)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            get_model_path(ALIAS), "model.{epoch:02d}-{val_loss:.2f}.h5"
        ),
        period=10,  # Save every 10 epochs
        verbose=1,
        save_weights_only=True,
    )
    model.fit(
        gen_train,
        steps_per_epoch=math.ceil(flow_train.samples / flow_train.batch_size),
        validation_data=flow_val,
        epochs=100,
        callbacks=[tb_callback, cp_callback],
    )


if __name__ == "__main__":
    main()