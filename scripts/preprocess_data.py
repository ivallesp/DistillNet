import numpy as np
import os
import math
import glob
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
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
    argparser.add_argument("-j", "--n-jobs", default=-1, help="Number of threads")
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

def load_process_and_save(input_path, output_dir, size):
    input_dir = os.path.split(os.path.split(input_path)[0])[0]

    img = Image.open(input_path)

    img = preprocess(img, size)
    assert img.size == (size, size)

    class_name = os.path.split(os.path.split(input_path)[0])[1]
    parent_folder = os.path.join(output_dir, class_name)
    os.makedirs(parent_folder, exist_ok=True)

    file_name = os.path.splitext(os.path.split(input_path)[1])[0] + ".JPEG"
    output_path = os.path.join(parent_folder, file_name)
    try:
        img.save(output_path, quality=100, subsampling=0)
    except OSError:
        img = img.convert("RGB")
        img.save(output_path, quality=100, subsampling=0)

def process_images():
    args = parse_args()
    size = int(args.size)
    n_jobs = int(args.n_jobs) if int(args.n_jobs) != -1 else cpu_count()

    input_paths = glob.glob(os.path.join(args.input_path, "*", "*.JPEG"))
    output_dir = f"{os.path.normpath(args.input_path)}_{args.size}"
    os.makedirs(output_dir, exist_ok=True)

    pool = Pool(n_jobs)
    p=pool.imap(lambda x: load_process_and_save(x, output_dir, size), input_paths)
    for _ in tqdm(p, total=len(input_paths)):
        pass
    pool.close()
    pool.join()

if __name__ == "__main__":
    process_images()
