import numpy as np
from glob import glob
from src.paths import get_dataset_path
from scipy.optimize import bisect
import os
from src.utils import to_int


MODELS_CATALOG = [
    "mobilenet",
    "mobilenetv2",
    "densenet121",
    "densenet169",
    "densenet201",
    "resnet50",
    "inceptionresnetv2",
    "nasnetlarge",
    "xception",
    "efficientnetb7",
]


def get_model_artifacts(model_name):
    """
    Warning, the temperature harcoded in this function corresponds to the normalization
    of 0.35. If the normalization constant changes, the temperature must be recomputed.
    """
    if model_name == "mobilenet":
        # loss: 1.1488 - accuracy: 0.7172 - OK
        from keras.applications.mobilenet import MobileNet as Model
        from keras.applications.mobilenet import preprocess_input
        temperature = 2.1864
        size = 256
    elif model_name == "mobilenetv2":
        # loss: 1.3987 - accuracy: 0.7298 - OK
        from keras.applications.mobilenet_v2 import MobileNetV2 as Model
        from keras.applications.mobilenet_v2 import preprocess_input
        temperature = 1.1956
        size = 256
    elif model_name == "densenet121":
        # loss: 0.9733 - accuracy: 0.7544 OK
        from keras.applications.densenet import DenseNet121 as Model
        from keras.applications.densenet import preprocess_input
        temperature = 1.9816
        size = 256
    elif model_name == "densenet169":
        # loss: 0.9287 - accuracy: 0.7650 OK
        from keras.applications.densenet import DenseNet169 as Model
        from keras.applications.densenet import preprocess_input
        temperature = 2.0492
        size = 256
    elif model_name == "densenet201":
        # loss: 0.8892 - accuracy: 0.7779 OK
        from keras.applications.densenet import DenseNet201 as Model
        from keras.applications.densenet import preprocess_input
        temperature = 1.9561
        size = 256
    elif model_name == "resnet50":
        # loss: 0.9786 - accuracy: 0.7555 OK
        from keras.applications.resnet50 import ResNet50 as Model
        from keras.applications.resnet50 import preprocess_input
        temperature = 2.2800
        size = 256
    elif model_name == "inceptionresnetv2":
        # loss: 1.1852 - accuracy: 0.7844 @ 256
        # loss: 0.8362 - accuracy: 0.8044 @ 299 OK
        from keras.applications.inception_resnet_v2 import InceptionResNetV2 as Model
        from keras.applications.inception_resnet_v2 import preprocess_input
        temperature = 1.4743
        size = 299
    elif model_name == "nasnetlarge":
        # loss: 0.7973 - accuracy: 0.8244 OK
        from keras.applications.nasnet import NASNetLarge as Model
        from keras.applications.nasnet import preprocess_input
        temperature = 1.4179
        size = 331
    elif model_name == "xception":
        # loss: 0.9050 - accuracy: 0.7892 OK
        from keras.applications.xception import Xception as Model
        from keras.applications.xception import preprocess_input
        temperature = 1.4850
        size = 299
    elif model_name == "efficientnetb7":
        # loss: 0.9824 - accuracy: 0.7788 OK
        from tensorflow.keras.applications.efficientnet import EfficientNetB7 as Model
        from tensorflow.keras.applications.efficientnet import preprocess_input
        temperature = 1.3969
        size = 256
    else:
        raise ValueError(f"Model name '{model_name}' not recognized as a valid name")
    return Model, preprocess_input, size, temperature


def get_soft_target_path(model_name, dataset):
    base_dir = get_dataset_path()
    path = glob(os.path.join(base_dir, f"soft_targets-{model_name}-{dataset}_*.npz"))
    if len(path) == 0:
        raise ValueError(
            f"No soft targets were found at {base_dir} for model "
            f"{model_name} and dataset {dataset}"
        )
    elif len(path) == 1:
        path = path[0]
    else:
        raise ValueError(
            f"Multiple soft targets were found at {base_dir} for model "
            f"{model_name} and dataset {dataset}: {path}"
        )
    return path


def softmax(x, T=1):
    exps = np.exp(x / T)
    p = exps / exps.sum(axis=1, keepdims=True)
    return p


def print_soft_targets_accuracy(soft_targets, filepaths):
    hard_targets = np.array([to_int(os.path.split(fp)[0]) for fp in filepaths])
    soft_targets_argmax = soft_targets.argmax(axis=1)
    accuracy = (soft_targets_argmax == hard_targets).mean()
    print(f"Soft-targets accuracy={accuracy}")
    return accuracy


def get_soft_targets(dataset, model_name, p):
    print(f"Generating soft-targets for '{model_name}-{dataset}'")
    # Load the npz file
    path = get_soft_target_path(model_name, dataset)
    soft_target_npz = np.load(path)
    # Get the contents of the npz file
    preds, filepaths = soft_target_npz["preds"], soft_target_npz["paths"]
    if p is not None:
        preds, temperature = normalize_soft_targets_by_p1(preds, p)
    else:
        temperature = 1
        preds = softmax(preds, temperature)
    # Check the accuracy
    accuracy = print_soft_targets_accuracy(preds, filepaths)
    # Compile data and metadata in a dictionary and yield
    data = {
        "model_name": model_name,
        "dataset": dataset,
        "temperature": temperature,
        "accuracy": accuracy,
        "filepaths": filepaths,
        "soft_targets": preds,
    }
    return data


def normalize_soft_targets_by_p1(logits, p, sample_size=50000):
    # Calculate soft-targets from logits using softmax w/ temperature
    print("Calculating T using bisection method...")
    if logits.shape[0] >= sample_size:  # If array is bigger than sample
        # Draw a sample
        _logits = logits[np.random.choice(logits.shape[0], 1000, replace=False), :]
    else:
        # Take the full population
        _logits = logits
    _logits = np.flip(np.sort(_logits, axis=1), axis=1)
    temperature = bisect(lambda T: softmax(_logits, T)[:, 0].mean() - p, 1, 6)
    print(f"Found T={temperature}")
    probs = softmax(logits, T=temperature)
    return probs, temperature


def normalize_soft_targets_by_top5(logits, p, sample_size=50000):
    # Calculate soft-targets from logits using softmax w/ temperature
    print("Calculating T using bisection method...")
    if logits.shape[0] >= sample_size:  # If array is bigger than sample
        # Draw a sample
        _logits = logits[np.random.choice(logits.shape[0], 1000, replace=False), :]
    else:
        # Take the full population
        _logits = logits
    _logits = np.flip(np.sort(_logits, axis=1), axis=1)
    temperature = bisect(
        lambda T: softmax(_logits, T)[:, :5].sum(axis=1).mean() - p, 1, 6
    )
    print(f"Found T={temperature}")
    probs = softmax(logits, T=temperature)
    return probs, temperature


def get_combined_soft_targets(models, dataset, prob1):
    preds_comb = 0
    for model in models:
        soft_targets = get_soft_targets(dataset, model, prob1)
        preds_comb += soft_targets["soft_targets"]

    print(f"Combining {len(models)} models...")

    preds_comb /= len(models)
    filepaths = soft_targets["filepaths"]
    accuracy = print_soft_targets_accuracy(preds_comb, filepaths)
    data = {
        "model_name": "combined_soft_targets",
        "dataset": soft_targets["dataset"],
        "temperature": None,
        "accuracy": accuracy,
        "filepaths": filepaths,
        "soft_targets": preds_comb,
    }
    return data


def get_indices_from_keras_generator(gen, batch_size):
    """
    Given a keras data generator, it returns the indices and the filepaths
    corresponding the current batch.
    :param gen: keras generator.
    :param batch_size: size of the last batch generated.
    :return: tuple with indices and filenames
    """

    idx_left = (gen.batch_index - 1) * batch_size
    idx_right = idx_left + gen.batch_size if idx_left >= 0 else None
    indices = gen.index_array[idx_left:idx_right]
    filenames = [gen.filenames[i] for i in indices]
    return indices, filenames


def get_data_generator_with_soft_targets(gen, soft_targets, unlabeled):
    """
    Given a keras data generator, switches the hard targets by soft-targets.
    :param gen: keras generator.
    :param soft_targets: soft_targets dict to use for replacing the targets
    :param unlabeled: if set to true, no cross checks with the paths are performed.
    :return: generator
    """
    # Check if the filenames in the generator are aligned with the ones in the soft
    # targets struct. This is crucial, otherwise we mess up the training.
    assert soft_targets["filepaths"].tolist() == gen.filenames
    for x, y in gen:
        indices, filepaths = get_indices_from_keras_generator(gen, x.shape[0])
        soft_targets_batch = soft_targets["soft_targets"][indices]
        if not unlabeled:
            # Assure labels are in line with indices, through filepaths.
            labels = np.argmax(y, axis=1)
            labels_fn = [to_int(x.split("/")[0]) for x in filepaths]
            assert labels.tolist() == labels_fn
        # Return y if needed
        yield x, soft_targets_batch
