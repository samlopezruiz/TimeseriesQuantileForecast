import os

import yaml

from src.models.attn.data_formatter.base import DataTypes, InputTypes


def get_cfg_folder(project):
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'config', project))


def read_config(file, project, subfolder=None):
    if 'yaml' not in file:
        file += '.yaml'
    config_folder = get_cfg_folder(project)
    file_path = os.path.join(config_folder, '' if subfolder is None else subfolder, file)
    return yaml.load(open(file_path, 'r'), yaml.SafeLoader)


def get_variable_definitions(project):
    cfgs_filename = os.listdir(os.path.join(get_cfg_folder(project), 'vars_definition'))

    var_cfgs = [read_config(cfg, project, subfolder='vars_definition') for cfg in cfgs_filename]

    variable_definitions = {}
    for cfg_name, cfg in zip(cfgs_filename, var_cfgs):
        if 'includeColumnDefinition' not in cfg:

            col_def = []
            for var in cfg['columnDefinition']:
                col_def.append((var['varName'], DataTypes[var['dataType']], InputTypes[var['inputType']]))

            for add_cfg in cfg['additionalDefinitions']:
                for var in read_config(add_cfg['filename'],
                                       project,
                                       subfolder='vars_definition')['includeColumnDefinition']:
                    col_def.append((var['varName'], DataTypes[var['dataType']], InputTypes[var['inputType']]))

            variable_definitions[cfg_name.split('.')[0]] = col_def

    return variable_definitions
