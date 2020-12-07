import numpy as np
import os
import math
import glob
from tqdm import tqdm
from PIL import Image
import argparse


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i",
        "--input-path",
        required=True,
        help="Path of the folder containing the input validation data",
    )
    argparser.add_argument(
        "-s",
        "--size",
        required=True,
        help="Size of the output images",
    )
    argparser.add_argument("-t", "--type", default="jpeg", help="'jpeg' or 'npz'")
    return argparser.parse_args()


def preprocess(img, size):
    w, h = img.size

    # Resize min dim to target size
    new_w, new_h = (
        (size, round(h * (size / w))) if w <= h else (round(w * (size / h)), size)
    )
    img = img.resize((new_w, new_h), Image.ANTIALIAS)
    # Center crop to target size
    left = math.ceil((new_w - size) / 2)
    top = math.ceil((new_h - size) / 2)
    right = math.ceil((new_w + size) / 2)
    bottom = math.ceil((new_h + size) / 2)
    img = img.crop((left, top, right, bottom))
    return img


def process_images():
    args = parse_args()
    type_ = args.type
    size = int(args.size)
    input_paths = glob.glob(os.path.join(args.input_path, "*", "*.JPEG"))
    output_dir = f"{os.path.normpath(args.input_path)}_{args.size}"
    os.makedirs(output_dir, exist_ok=True)
    for input_path in tqdm(input_paths):

        img = Image.open(input_path)

        img = preprocess(img, size)
        assert img.size == (size, size)

        class_name = os.path.split(os.path.split(input_path)[0])[1]
        parent_folder = os.path.join(output_dir, class_name)
        os.makedirs(parent_folder, exist_ok=True)

        if type_ == "jpeg":
            file_name = os.path.splitext(os.path.split(input_path)[1])[0] + ".JPEG"
            output_path = os.path.join(parent_folder, file_name)
            img.save(output_path, quality=100, subsampling=0)
        elif type_ == "npz":  # Convert to numpy and save to npz
            img_np = np.array(img)
            file_name = os.path.splitext(os.path.split(input_path)[1])[0] + ".npz"
            output_path = os.path.join(parent_folder, file_name)
            np.savez(output_path, img=img_np)
        else:
            raise ValueError(
                f"Type '{type_}' not recognized as a valid type (jpeg or npz)."
            )


if __name__ == "__main__":
    process_images()
