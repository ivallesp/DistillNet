import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import math
import numpy as np
import glob
from tqdm import tqdm
from PIL import Image
import argparse
import keras
from keras.preprocessing import image
from keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import time
import tensorflow as tf

tf.get_logger().setLevel("INFO")


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i",
        "--dataset-path",
        required=True,
        help="Path of the folder containing the input validation data",
    )
    argparser.add_argument(
        "-m",
        "--model-name",
        required=True,
        help="Name of the model to loadz",
    )

    argparser.add_argument("-t", "--type", default="jpeg", help="'jpeg' or 'npz'")
    return argparser.parse_args()


def get_model_artifacts(model_name):
    if model_name == "mobilenet":
        # loss: 1.1488 - accuracy: 0.7172 - OK
        from keras.applications.mobilenet import MobileNet as Model
        from keras.applications.mobilenet import preprocess_input
        size = 256
    elif model_name == "mobilenetv2":
        # loss: 1.3987 - accuracy: 0.7298 - OK
        from keras.applications.mobilenet_v2 import MobileNetV2 as Model
        from keras.applications.mobilenet_v2 import preprocess_input
        size = 256
    elif model_name == "densenet121":
        # loss: 0.9733 - accuracy: 0.7544 OK
        from keras.applications.densenet import DenseNet121 as Model
        from keras.applications.densenet import preprocess_input
        size = 256
    elif model_name == "densenet169":
        # loss: 0.9287 - accuracy: 0.7650 OK
        from keras.applications.densenet import DenseNet169 as Model
        from keras.applications.densenet import preprocess_input
        size = 256
    elif model_name == "densenet201":
        # loss: 0.8892 - accuracy: 0.7779 OK
        from keras.applications.densenet import DenseNet201 as Model
        from keras.applications.densenet import preprocess_input
        size = 256
    elif model_name == "resnet50":
        # loss: 0.9786 - accuracy: 0.7555 OK
        from keras.applications.resnet50 import ResNet50 as Model
        from keras.applications.resnet50 import preprocess_input
        size = 256
    elif model_name == "resnet152v2":
        # loss: 1.1993 - accuracy: 0.7502 NOPE
        # resnet v2 does not work well
        from keras.applications.resnet_v2 import ResNet152V2 as Model
        from keras.applications.resnet_v2 import preprocess_input
        size = 256
    elif model_name == "inceptionresnetv2":
        # loss: 1.1852 - accuracy: 0.7844 @ 256
        # loss: 0.8362 - accuracy: 0.8044 @ 299 OK
        from keras.applications.inception_resnet_v2 import InceptionResNetV2 as Model
        from keras.applications.inception_resnet_v2 import preprocess_input
        size = 299
    elif model_name == "nasnetlarge":
        # loss: 0.7973 - accuracy: 0.8244 OK
        from keras.applications.nasnet import NASNetLarge as Model
        from keras.applications.nasnet import preprocess_input
        size = 331
    elif model_name == "xception":
        # loss: 0.9050 - accuracy: 0.7892 OK
        from keras.applications.xception import Xception as Model
        from keras.applications.xception import preprocess_input
        size = 299
    elif model_name == "efficientnetb7":
        # loss: 0.9824 - accuracy: 0.7788 OK
        from tensorflow.keras.applications.efficientnet import EfficientNetB7 as Model
        from tensorflow.keras.applications.efficientnet import preprocess_input
        size = 256
    elif model_name == "efficientnetb0":
        # accuracy: 0.75172 - OK
        from tensorflow.keras.applications.efficientnet import EfficientNetB0 as Model
        from tensorflow.keras.applications.efficientnet import preprocess_input

        size = 256
    else:
        raise ValueError(f"Model name '{model_name}' not recognized as a valid name")
    return Model, preprocess_input, size


def evaluate():
    args = parse_args()
    model_name = args.model_name
    dataset_path = args.dataset_path
    Model, preprocess_input, size = get_model_artifacts(model_name)
    idg = ImageDataGenerator(preprocessing_function=preprocess_input)
    model = Model(weights="imagenet")
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    out = model.evaluate(
        idg.flow_from_directory(dataset_path, target_size=(size, size))
    )
    print(out)


def to_int(x):
    try:
        return int(x)
    except ValueError:
        return -1


def predict():
    args = parse_args()
    model_name = args.model_name
    dataset_path = args.dataset_path
    batch_size = 32
    # Load model and get logits
    Model, preprocess_input, size = get_model_artifacts(model_name)
    idg = ImageDataGenerator(preprocessing_function=preprocess_input)
    model = Model(weights="imagenet", classifier_activation=None)
    flow = idg.flow_from_directory(
        dataset_path, shuffle=False, target_size=(size, size), batch_size=batch_size
    )
    # Generate predictions
    predictions = []
    n_batches = 0
    for x, y in tqdm(flow):
        preds = model.predict(x, verbose=0)
        predictions.append(preds)
        n_batches += 1
        if n_batches >= math.ceil(len(flow.filenames) / batch_size):
            # Manually break the loop as the generator loops indefinitely
            break
    predictions = np.concatenate(predictions)[: len(flow.filenames)]

    # Calculate accuracy for double checking
    targets = np.array(list(map(lambda x: to_int(x.split("/")[0]), flow.filenames)))
    y = predictions.argmax(axis=1)
    print(f"Accuracy = {np.mean(targets==y)}")

    # Save paths and predictions into an npz file
    dataset_dir_name = os.path.split(dataset_path)[1]
    output_filename = f"soft_targets-{model_name}-{dataset_dir_name}.npz"
    output_path = os.path.join(os.path.split(dataset_path)[0], output_filename)
    np.savez(output_path, paths=flow.filenames, preds=predictions)
    print(f"Results saved in '{output_path}'")


if __name__ == "__main__":
    predict()
