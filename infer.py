import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
from tqdm import tqdm
from src.model import get_model_artifacts
from src.paths import get_dataset_path
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from functools import partial
from keras.optimizers import Adam


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-m",
        "--model-path",
        required=True,
        type=str,
        help="",
    )
    return argparser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path
    model_name = os.path.split(os.path.split(model_path)[0])[-1]
    base_model_name = model_name.split("-")[0]
    output_path = os.path.splitext(model_path)[0] + ".json"
    get_model, preprocess_input, size, T = get_model_artifacts(base_model_name)
    model = get_model(weights=None)
    model.load_weights(model_path)

    # Build input data paths
    data_path = get_dataset_path()
    validation_dataset_path = os.path.join(data_path, f"validation_{size}")

    top5_acc = partial(tf.keras.metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = "top5_accuracy"
    model.compile(
        Adam(learning_rate=1e-6),
        "categorical_crossentropy",
        metrics=["accuracy", top5_acc],
    )

    idg = ImageDataGenerator(preprocessing_function=preprocess_input)
    loss, top1, top5 = model.evaluate(
        idg.flow_from_directory(
            validation_dataset_path,
            target_size=(size, size),
            batch_size=256,
        ),
        verbose=0,
    )

    output = dict(loss=loss, top1=top1, top5=top5)
    with open(output_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
