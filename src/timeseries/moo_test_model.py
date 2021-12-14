import os

import joblib
import numpy as np
import seaborn as sns
import tensorflow as tf

from src.timeseries.plot.moo import get_ixs_risk, plot_2D_moo_dual_results
from src.timeseries.utils.filename import get_result_folder, risk_name, quantiles_name, termination_name
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.harness import get_model_data_config, load_predict_model
from src.timeseries.utils.moo import get_selected_ix, get_new_weights, compute_moo_q_loss
from src.timeseries.utils.results import post_process_results

sns.set_theme('poster')

if __name__ == "__main__":
    # %%
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    general_cfg = {'save_forecast': True,
                   'save_plot': True,
                   'plot_title': False,
                   'use_all_data': True,
                   'use_moo_weights': True,
                   'plot_tolerance': True,
                   'manual_selection': True}

    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_NSGA2_g100_p100_s0_c1_eq0_dual_wmoo'
                   }

    moo_results = joblib.load(os.path.join(get_result_folder(results_cfg, project), 'moo', results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(project,
                                                            moo_results['lq']['experiment_cfg'],
                                                            moo_results['lq']['model_params'],
                                                            moo_results['lq']['fixed_params'])

    experiment_cfg = moo_results['lq']['experiment_cfg']

    # %%
    add_risk = 0.05
    xaxis_limit = 1.

    risk_selected = {'qcrl': 0.4,
                     'qcru': 0.15}

    limits_ixs = [get_ixs_risk(moo_result['F'], add_risk) for _, moo_result in moo_results.items()]

    manual_selected_ixs = {'lq': limits_ixs[0][1],
                           'uq': limits_ixs[1][0]}
    # manual_selected_ixs = {'lq': 68,
    #                        'uq': 76}

    labels, quantiles_losses, original_ixs, selected_ixs = [], [], [], []
    selected_weights, original_weights = {}, {}
    original_losses = []
    for bound, moo_result in moo_results.items():
        weights, quantiles_loss, eq_quantiles_loss = moo_result['X'], moo_result['F'], moo_result['eq_F']
        original_ix = np.argmin(np.sum(np.abs(quantiles_loss - moo_result['original_losses']), axis=1))
        original_losses.append(moo_result['original_losses'])
        if general_cfg['manual_selection']:
            selected_ix = manual_selected_ixs[bound]
        else:
            selected_ix = get_selected_ix(quantiles_loss, risk_selected, upper=bound == 'uq')

        moo_results[bound]['original_ix'] = original_ix
        moo_results[bound]['selected_ix'] = selected_ix
        labels.append('upper quantile' if bound == 'uq' else 'lower quantile')
        quantiles_losses.append(quantiles_loss)
        original_ixs.append(original_ix)
        selected_ixs.append(selected_ix)
        selected_weights[bound] = weights[selected_ix, :]
        original_weights[bound] = weights[original_ix, :]

    # Overwrite selected ixs if needed
    risk_lbl = ''
    if general_cfg['use_moo_weights']:
        if general_cfg['manual_selection']:
            risk_lbl = 'lix{}_uix{}'.format(selected_ixs[0], selected_ixs[1])
        else:
            risk_lbl = risk_name(risk_selected)
    basename = '{}_{}_q{}_{}_{}_p{}_s{}_{}_tol{}_'.format(experiment_cfg['architecture'],
                                                          experiment_cfg['vars_definition_cfg'],
                                                          quantiles_name(moo_results['lq']['quantiles']),
                                                          moo_results['lq']['moo_method'],
                                                          termination_name(
                                                              moo_results['lq']['algo_cfg']['termination']),
                                                          moo_results['lq']['algo_cfg']['pop_size'],
                                                          int(moo_results['lq']['algo_cfg']['use_sampling']),
                                                          risk_lbl,
                                                          int(add_risk * 100)
                                                          )

    img_path = os.path.join(config.results_folder,
                            experiment_cfg['experiment_name'],
                            'img',
                            'moo',
                            '{}pf'.format(basename))

    plot_2D_moo_dual_results(quantiles_losses,
                             selected_ixs=selected_ixs if general_cfg['use_moo_weights'] else None,
                             save=general_cfg['save_plot'],
                             file_path=img_path,
                             original_losses=original_losses,
                             # original_ixs=original_ixs,
                             figsize=(20, 15),
                             xaxis_limit=xaxis_limit,
                             col_titles=labels,
                             legend_labels=None,
                             add_risk=add_risk if general_cfg['plot_tolerance'] else None,
                             markersize=8,
                             plot_title=general_cfg['plot_title'],
                             title='MOO using {} for quantiles: {}'.format(moo_result['moo_method'],
                                                                           moo_result['quantiles']))

    # %%
    if general_cfg['save_forecast']:
        new_weights = get_new_weights(moo_results['lq']['original_weights'], selected_weights) if \
            general_cfg['use_moo_weights'] else None

        results, data = load_predict_model(use_gpu=True,
                                           architecture=experiment_cfg['architecture'],
                                           model_folder=model_folder,
                                           data_config=config.data_config,
                                           data_formatter=formatter,
                                           use_all_data=general_cfg['use_all_data'],
                                           last_layer_weights=new_weights)

        post_process_results(results, formatter, experiment_cfg, plot_=False)

        q_losses = compute_moo_q_loss(results['quantiles'], results['forecasts'])
        print('lower quantile risk: {} \nupper quantile risk: {}'.format(q_losses[0, :], q_losses[2, :]))

        results['data'] = data
        results['objective_space'] = q_losses

        save_vars(results, os.path.join(config.results_folder,
                                        experiment_cfg['experiment_name'],
                                        'moo',
                                        'selec_sols',
                                        '{}{}pred'.format(basename,
                                                          'all_' if general_cfg['use_all_data'] else '')))
