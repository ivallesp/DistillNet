import keras
import math
import os
import argparse
from keras import Model
from functools import partial
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from src.model import (
    get_model_artifacts,
    get_combined_soft_targets,
    get_data_generator_with_soft_targets,
    MODELS_CATALOG,
)
from src.paths import get_logs_path, get_model_path, get_dataset_path
from src.utils import set_seed
import tensorflow as tf

BATCH_SIZE = 35
PROB_NORM = 0.35


class TBCallbackMod(tf.keras.callbacks.TensorBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_started = False

    def on_train_end(self, logs=None):
        # Do not close the writers when train ends
        pass

    def on_train_begin(self, logs=None):
        # Do instantiate the writer and global step only the first time
        if not self.train_started:
            self._global_train_batch = 0
            self._push_writer(self._train_writer, self._train_step)
            self.train_started = True


def get_training_generator(
    transfer_dataset_path,
    preprocess_input,
    size,
    teachers_combination_method,
    batch_size,
):
    transfer_dataset_alias = os.path.split(transfer_dataset_path)[1].split("_")[0]
    unlabeled = True if transfer_dataset_alias == "test" else False

    # Build soft-targets
    soft_targets = get_combined_soft_targets(
        models=MODELS_CATALOG,
        dataset=transfer_dataset_alias,
        prob1=PROB_NORM,
        method=teachers_combination_method,
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

    steps_per_epoch = math.ceil(flow_train.samples / flow_train.batch_size)
    return gen_train, steps_per_epoch


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-b",
        "--base-model",
        required=True,
        type=str,
        help="Name of the base model. Allowed names: {MODEL_CATALOG}.",
    )
    argparser.add_argument(
        "-t",
        "--teachers-combination-method",
        required=True,
        type=str,
        help="Method used to combine the teachers. Allowed values: "
        "'mean', 'median', 'random'.",
    )
    argparser.add_argument(
        "-r",
        "--random-seed",
        required=True,
        type=int,
        help="Random seed used for the random processes involved.",
    )
    return argparser.parse_args()


def train_model(base_model_name, teachers_combination_method, random_seed):
    # Set the random seed
    set_seed(random_seed)

    # Load base model
    get_model, preprocess_input, size, T = get_model_artifacts(base_model_name)
    model = get_model(classifier_activation="linear", weights="imagenet")

    # Build input data paths
    data_path = get_dataset_path()
    validation_dataset_path = os.path.join(data_path, f"validation_{size}")
    transfer_dataset_path = os.path.join(data_path, f"test_{size}")

    # Alias name generation
    model_alias = f"{base_model_name}-{teachers_combination_method}-{random_seed}"

    # Warm last model layer
    input_ = model.input
    output = keras.backend.softmax(model.output / T)
    model = Model(input_, output)

    # Define top5 accuracy metric
    top5_acc = partial(tf.keras.metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = "top5_accuracy"

    # Compile the model
    model.compile(
        Adam(learning_rate=1e-6),
        "categorical_crossentropy",
        metrics=["accuracy", top5_acc],
    )

    # Build callbacks
    if teachers_combination_method == "random":
        TBCallback = TBCallbackMod
    else:
        TBCallback = tf.keras.callbacks.TensorBoard
    tb_callback = TBCallback(
        get_logs_path(model_alias),
        update_freq=100,
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            get_model_path(model_alias), "model.{epoch:02d}-{val_loss:.2f}.h5"
        ),
        period=10,  # Save every 10 epochs
        verbose=1,
        save_weights_only=True,
    )

    # Build validation data generator
    idg_val = ImageDataGenerator(preprocessing_function=preprocess_input)
    flow_val = idg_val.flow_from_directory(
        validation_dataset_path,
        shuffle=True,
        target_size=(size, size),
        batch_size=BATCH_SIZE,
    )
    # Running baseline evaluation against validation data
    print("Running initial validation...")
    model.evaluate(flow_val)

    if teachers_combination_method == "random":
        # Update the random teachers every epoch
        for epoch in range(100):
            gen_train, steps_per_epoch = get_training_generator(
                transfer_dataset_path=transfer_dataset_path,
                preprocess_input=preprocess_input,
                size=size,
                teachers_combination_method=teachers_combination_method,
                batch_size=BATCH_SIZE,
            )
            model.fit(
                gen_train,
                steps_per_epoch=steps_per_epoch,
                validation_data=flow_val,
                epochs=epoch + 1,
                initial_epoch=epoch,
                callbacks=[tb_callback, cp_callback],
            )
        # Manually close the writer
        tb_callback._pop_writer()
        if tb_callback._is_tracing:
            tb_callback._stop_trace()

        tb_callback._close_writers()
        tb_callback._delete_tmp_write_dir()
    else:
        gen_train, steps_per_epoch = get_training_generator(
            transfer_dataset_path=transfer_dataset_path,
            preprocess_input=preprocess_input,
            size=size,
            teachers_combination_method=teachers_combination_method,
            batch_size=BATCH_SIZE,
        )

        model.fit(
            gen_train,
            steps_per_epoch=steps_per_epoch,
            validation_data=flow_val,
            epochs=100,
            callbacks=[tb_callback, cp_callback],
        )


if __name__ == "__main__":
    args = parse_args()
    train_model(
        base_model_name=args.base_model,
        teachers_combination_method=args.teachers_combination_method,
        random_seed=args.random_seed,
    )
