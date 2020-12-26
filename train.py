import keras
import math
import os
import argparse
from keras import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from src.model import (
    get_model_artifacts,
    get_combined_soft_targets,
    get_data_generator_with_soft_targets,
    MODELS_CATALOG,
)
from src.paths import get_logs_path, get_model_path, get_dataset_path
import tensorflow as tf


BATCH_SIZE = 35
PROB_NORM = 0.35
base_model_name = "mobilenetv2"


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-b",
        "--base-model",
        required=True,
        help="Name of the base model. Allowed names: {MODEL_CATALOG}.",
    )
    argparser.add_argument(
        "-t",
        "--teachers-combination-method",
        required=True,
        help="Method used to combine the teachers. Allowed values: "
        "'mean', 'median', 'random'.",
    )
    argparser.add_argument(
        "-r",
        "--random-seed",
        required=True,
        help="Random seed used for the random processes involved.",
    )
    return argparser.parse_args()


def train_model(base_model_name, teachers_combination_method, random_seed):
    # Load base model
    get_model, preprocess_input, size, T = get_model_artifacts("mobilenetv2")
    model = get_model(classifier_activation="linear", weights="imagenet")

    # Build input data paths
    data_path = get_dataset_path()
    validation_dataset_path = os.path.join(data_path, f"validation_{size}")
    transfer_dataset_path = os.path.join(data_path, f"test_{size}")
    transfer_dataset_alias = os.path.split(transfer_dataset_path)[1].split("_")[0]
    unlabeled = True if transfer_dataset_alias == "test" else False

    # Alias name generation
    model_alias = f"{base_model_name}-{teachers_combination_method}-{random_seed}"

    # Warm last model layer
    input_ = model.input
    output = keras.backend.softmax(model.output / T)
    model = Model(input_, output)

    # Build soft-targets
    soft_targets = get_combined_soft_targets(
        models=MODELS_CATALOG, dataset=transfer_dataset_alias, prob1=PROB_NORM
    )

    # Build train data generator
    idg_train = ImageDataGenerator(preprocessing_function=preprocess_input)
    flow_train = idg_train.flow_from_directory(
        transfer_dataset_path,
        shuffle=True,
        target_size=(size, size),
        BATCH_SIZE=BATCH_SIZE,
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
        BATCH_SIZE=BATCH_SIZE,
    )

    model.compile(
        Adam(learning_rate=1e-5), "categorical_crossentropy", metrics=["accuracy"]
    )

    print("Running initial validation...")
    model.evaluate(flow_val)
    print("Running train...")
    tb_callback = tf.keras.callbacks.TensorBoard(
        get_logs_path(model_alias), update_freq=100
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            get_model_path(model_alias), "model.{epoch:02d}-{val_loss:.2f}.h5"
        ),
        period=10,  # Save every 10 epochs
        verbose=1,
        save_weights_only=True,
    )
    model.fit(
        gen_train,
        steps_per_epoch=math.ceil(flow_train.samples / flow_train.BATCH_SIZE),
        validation_data=flow_val,
        epochs=1000,
        callbacks=[tb_callback, cp_callback],
    )


if __name__ == "__main__":
    args = parse_args()
    train_model(
        base_model_name=args.base_model,
        teachers_combination_method=args.teachers_combination_method,
        random_seed=args.random_seed,
    )
