import os

import yaml


def read_config(file, project):
    if 'yaml' not in file:
        file += '.yaml'
    config_folder = os.path.normpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'config', project))
    return yaml.load(open(os.path.join(config_folder, file), 'r'), yaml.SafeLoader)