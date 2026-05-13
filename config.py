import json
import os
from copy import deepcopy
from pathlib import Path
from easydict import EasyDict as edict


def _deep_update(base, override):
    """Recursively merge override into base and return a new dict."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_model_config(config_all, model_name):
    """Resolve a model config, allowing variants to inherit from baseModel."""
    if model_name not in config_all:
        supported = [key for key in config_all.keys() if key != 'datasetCommonParams']
        raise ValueError(f"Model {model_name} not found in config. Supported models: {supported}")

    model_config = deepcopy(config_all[model_name])
    base_model = model_config.pop('baseModel', None)
    if base_model is None:
        return model_config

    base_config = _resolve_model_config(config_all, base_model)
    return _deep_update(base_config, model_config)


def get_config_regression(model_name, dataset_name, config_file=""):
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        config_file (str): Path to config file, if given an empty string, will use default config file.
        model_name (str): Name of model.
        dataset_name (str): Name of dataset.

    Returns:
        config (dict): config of the given dataset and model
    """
    model_name = model_name.upper()
    dataset_name = dataset_name.lower()
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_config = _resolve_model_config(config_all, model_name)
    model_common_args = model_config['commonParams']
    model_dataset_args = model_config['datasetParams'][dataset_name]
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    config = edict(config)

    return config
