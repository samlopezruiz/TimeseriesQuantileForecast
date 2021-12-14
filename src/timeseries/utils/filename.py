import os

import numpy as np


def get_date_text(cfg):
    if cfg.get('trim_data_from', None) is not None and cfg.get('trim_data_from', None) is not None:
        date_text = '_' + cfg['trim_data_to'] + '_to_' + cfg['trim_data_to']
    else:
        date_text = ''
    return date_text


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


def update_trim_range(df, cfg):
    cfg['trim_data_from'] = str(df.index[0].year) + '_' + str(df.index[0].month)
    cfg['trim_data_to'] = str(df.index[-1].year) + '_' + str(df.index[-1].month)

def hmm_filename(hmm_cfg):
    return 'regime_{}{}'.format('_'.join(hmm_cfg['vars']).replace("^", ""), get_date_text(hmm_cfg))


def res_mkt_filename(data_cfg, training_cfg, model_cfg):
    data_name = data_cfg['filename'][find_nth(data_cfg['filename'], '_', 1):find_nth(data_cfg['filename'], '_', 3)]
    return 'res' + data_name + '_y' + training_cfg['y_true_var'] + '_m' + model_cfg['name'] + '_reg' + str(
        model_cfg['use_regimes'])


def split_filename(cfg):
    data_cfg, split_cfg = cfg['data_cfg'], cfg['split_cfg']
    d_text = dwn_smple_text(data_cfg)
    return 'split_' + data_cfg['inst'] + '_' + data_cfg['subfolder'] + d_text + '_' + data_cfg[
        'trim_data_from'] + '_to_' + \
           data_cfg['trim_data_to'] + '_g' + str(split_cfg['groups_of']) + split_cfg['group'] + \
           '_r' + str(split_cfg['test_ratio'])[-2:]


def subset_filename(data_cfg):
    d_text = dwn_smple_text(data_cfg)
    return 'subset_' + data_cfg['inst'] + '_' + data_cfg['subfolder'] + d_text + '_' + data_cfg[
        'trim_data_from'] + '_to_' + \
           data_cfg['trim_data_to']


def dwn_smple_text(data_cfg):
    if data_cfg.get('downsample', False):
        d_text = '_' + data_cfg['downsample_p'] + '_dwn_smpl'
    else:
        d_text = ''
    return d_text


def get_output_folder():
    root_folder = os.path.normpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'outputs'))
    return root_folder


def get_result_folder(cfg, project=''):
    return os.path.join(get_output_folder(),
                        'results',
                        project,
                        cfg.get('experiment_name', ''))


def quantiles_name(quantiles):
    return ''.join((np.array(quantiles) * 10).astype(int).astype(str))


def termination_name(termination):
    return '{}{}'.format('g' if termination[0] == 'n_gen' else 'e',
                         termination[1])


def risk_name(risk):
    text = ''
    for key, value in risk.items():
        text += '{}{}_'.format(key, str(value)[2:])
    if len(text) > 0:
        return text[:-1]
    else:
        return text

# def subsets_and_test_filename(data_cfg, split_cfg):
#     return 'split_' + data_cfg['inst'] + '_' + data_cfg['sampling'] + '_' + data_cfg['data_from'] + '_to_' + \
#             data_cfg['data_to'] + '_g' + str(split_cfg['groups_of']) + split_cfg['group'] + \
#             '_r' + str(split_cfg['test_ratio'])
