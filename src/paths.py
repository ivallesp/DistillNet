import os


def get_dataset_path():
    return os.path.join("/mnt", "FLASH", "datasets", "imagenet")


def get_model_path(alias):
    path = os.path.join(get_models_path(), alias)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_logs_path(alias):
    path = os.path.join(get_model_path(alias), "logs")
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_models_path():
    return "./models"