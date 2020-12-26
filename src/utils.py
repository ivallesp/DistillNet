import numpy as np
import tensorflow as tf
import random


def to_int(x):
    try:
        return int(x)
    except ValueError:
        return -1


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)
