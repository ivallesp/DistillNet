import numpy as np
import os
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
    return argparser.parse_args()


def preprocess(img, size):
    w, h = img.size

    # Resize max dim to target size
    new_w, new_h = (
        (size, round(h * (size / w))) if w >= h else (round(w * (size / h)), size)
    )
    img = img.resize((new_w, new_h))

    # Center crop to target size
    left = (new_w - size) / 2
    top = (new_h - size) / 2
    right = (new_w + size) / 2
    bottom = (new_h + size) / 2
    img = img.crop((left, top, right, bottom))
    return img


def process_images():
    args = parse_args()
    size = int(args.size)
    input_paths = glob.glob(os.path.join(args.input_path, "*", "*.JPEG"))
    output_dir = f"{os.path.normpath(args.input_path)}_{args.size}"
    os.makedirs(output_dir, exist_ok=True)
    for input_path in tqdm(input_paths):

        img = Image.open(input_path)

        img = preprocess(img, size)
        assert img.size == (size, size)
        
        # Convert to numpy and save to npz
        img_np = np.array(img)
        class_name = os.path.split(os.path.split(input_path)[0])[1]
        file_name = os.path.splitext(os.path.split(input_path)[1])[0] + ".npz"
        parent_folder = os.path.join(output_dir, class_name)
        os.makedirs(parent_folder, exist_ok=True)
        output_path = os.path.join(parent_folder, file_name)
        np.savez(output_path, img=img_np)


if __name__ == "__main__":
    process_images()
