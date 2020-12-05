import pandas as pd
import argparse
import os
from tqdm import tqdm


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
    return argparser.parse_args()


def generate_validation_data_structure():
    args = parse_args()

    # Load the index file, that maps from image number to class number
    index = open(args.index_path, "r").readlines()

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