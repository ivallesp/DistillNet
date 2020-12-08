import pandas as pd
import argparse
import os
import glob
import requests
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat


TF_META_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i",
        "--input-path",
        required=True,
        help="Path of the folder containing the input training data",
    )
    argparser.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Path of the folder that will contain the ordered images",
    )
    return argparser.parse_args()


def get_imagenet_tf_mapping():
    tf_json = requests.get(TF_META_URL).json()
    tf_json = {v[0]: int(k) for k, v in tf_json.items()}
    return tf_json


def generate_training_data_structure():
    args = parse_args()

    # Map imagenet indices to tensorflow class id
    index = get_imagenet_tf_mapping()
    os.makedirs(args.output_path, exist_ok=True)
    input_paths = glob.glob(os.path.join(args.input_path, "*", "*.JPEG"))
    for input_path in tqdm(input_paths):  # Number of validation images at ImageNet

        imagenet_id = os.path.split(os.path.split(input_path)[0])[1]
        class_name = f"{int(index[imagenet_id]):04d}"
        filename = os.path.split(input_path)[1]
        output_path = os.path.join(args.output_path, class_name, filename)
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        os.symlink(input_path, output_path)


if __name__ == "__main__":
    generate_training_data_structure()