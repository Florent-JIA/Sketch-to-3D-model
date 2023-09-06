import importlib


def get_dataset(alias):
    dataset_module = importlib.import_module('Reconstruction_3D.datasets.' + alias.lower())
    return dataset_module.Dataset
