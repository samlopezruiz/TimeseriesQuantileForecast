import os

import joblib
import pandas as pd
import tensorflow as tf

from src.timeseries.plot.ts import plotly_color_1st_row
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.harness import get_model_data_config, load_predict_model
from src.timeseries.utils.results import post_process_results

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    general_cfg = {'save_forecast': True,
                   'save_plot': True,
                   'use_all_data': True,
                   'plot_title': False
                   }

    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q357',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'
                   }

    model_results = joblib.load(os.path.join(get_result_folder(results_cfg, project), results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(project,
                                                            model_results['experiment_cfg'],
                                                            model_results['model_params'],
                                                            model_results['fixed_params'])
    experiment_cfg = model_results['experiment_cfg']

    results, data = load_predict_model(use_gpu=True,
                                       architecture=experiment_cfg['architecture'],
                                       model_folder=model_folder,
                                       data_config=config.data_config,
                                       data_formatter=formatter,
                                       use_all_data=general_cfg['use_all_data'])

    post_process_results(results, formatter, experiment_cfg)

    results['data'] = data


    # %%
    subsets_lbls = {0: 'train', 1: 'test', 2: 'validation'}
    weighted_errors = []
    for q_error_lbl, weighted_error in results['weighted_errors'].items():
        df = weighted_error.copy()
        df.set_index(df['forecast_time'], inplace=True)
        df.drop(['forecast_time', 'identifier'], axis=1, inplace=True)
        weighted_errors.append(df.mean(axis=1).to_frame(name='{} error'.format(q_error_lbl)))
    weighted_errors = pd.concat(weighted_errors, axis=1)

    data = results['data'].loc[:, ['ESc', 'test', 'datetime']].copy()
    data.set_index(['datetime'], inplace=True, drop=True)
    data = pd.concat([data, weighted_errors], axis=1, join='inner')
    mean_e = {}
    cum_mean_e = {}
    for ss, df_ss in data.groupby(by='test'):
        cumm_df = df_ss.loc[:, list(weighted_errors.columns)].expanding().mean()
        cumm_df = cumm_df.mean(axis=1).to_frame(name='{} error'.format(subsets_lbls[ss]))
        mean_e[ss] = df_ss.loc[:, list(weighted_errors.columns)].mean(axis=0)
        cum_mean_e[ss] = cumm_df

    cummulative_mean_error = pd.concat([m for _, m in cum_mean_e.items()], axis=1).sort_index()
    data = results['data'].loc[:, ['ESc', 'test', 'datetime']].copy()
    data.set_index(['datetime'], inplace=True, drop=True)
    data = pd.concat([data, cummulative_mean_error], axis=1, join='inner')
    data.rename(columns={'test': 'subset'}, inplace=True)
    data.fillna(method='ffill', inplace=True)


    if general_cfg['save_forecast']:
        results['cummulative_mean_error'] = data
        results['weighted_errors'] = weighted_errors
        save_vars(results, os.path.join(config.results_folder,
                                        experiment_cfg['experiment_name'],
                                        '{}{}_pred'.format('all_' if general_cfg['use_all_data'] else '',
                                                           experiment_cfg['vars_definition_cfg'])))

    #%%
    # plotly_time_series(data, rows=[0, 1, 2, 2, 2])

    plotly_color_1st_row(data,
                         color_col='subset',
                         first_row_feats=['ESc'],
                         rows=[1, 1, 1],
                         save_png=True,
                         label_scale=1.5,
                         size=(1980*2//3, 1080*2//3),
                         save=general_cfg['save_plot'],
                         file_path=os.path.join(config.results_folder,
                                                experiment_cfg['experiment_name'],
                                                'img',
                                                '{}_subset_errors'.format(experiment_cfg['vars_definition_cfg'])),
                         other_feats=list(cummulative_mean_error.columns))
