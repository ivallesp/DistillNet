import pandas as pd
import argparse
import os
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
        help="Path of the folder containing the input validation data",
    )
    argparser.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Path of the folder that will contain the ordered images",
    )
    argparser.add_argument(
        "-x",
        "--index-path",
        required=True,
        help="Path of the file containing the validation data index",
    )
    argparser.add_argument(
        "-m",
        "--meta-path",
        required=True,
        help="Meta info path (with .mat extension) provided with Imagenet.",
    )

    return argparser.parse_args()


def get_imagenet_tf_mapping(meta_path):
    tf_json = requests.get(TF_META_URL).json()
    imagenet_meta = imagenet_meta = loadmat(meta_path)
    ids = np.array(list(map(lambda x: x[0][0], imagenet_meta["synsets"]))).squeeze()
    codes = np.array(list(map(lambda x: x[0][1], imagenet_meta["synsets"]))).squeeze()
    imagenet_json = dict(zip(codes, ids))

    mapping = {}
    for key_tf, val_tf in tf_json.items():
        tf_code, tf_desc = val_tf
        imagenet_id = imagenet_json[tf_code]
        mapping[int(imagenet_id)] = int(key_tf)
    return mapping


def generate_validation_data_structure():
    args = parse_args()

    # Load the index file, that maps from image number to class number
    index = map(lambda x: int(x), open(args.index_path, "r").readlines())

    # Map imagenet indices to tensorflow class id
    tf_mapping = get_imagenet_tf_mapping(args.meta_path)
    index = list(map(tf_mapping.get, index))
    filename_template = "ILSVRC2012_val_{:08d}.JPEG"
    os.makedirs(args.output_path, exist_ok=True)
    for i in tqdm(range(50000)):  # Number of validation images at ImageNet
        filename = filename_template.format(i + 1)
        input_path = os.path.join(args.input_path, filename)
        class_name = f"{int(index[i]):04d}"
        output_path = os.path.join(args.output_path, class_name, filename)
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        os.symlink(input_path, output_path)


if __name__ == "__main__":
    generate_validation_data_structure()