import os

import pandas as pd
import tensorflow as tf

from src.timeseries.plot.ts import plotly_time_series
from src.timeseries.utils.config import read_config
from src.timeseries.utils.harness import get_model_data_config, train_test_model

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    general_cfg = {'save_forecast': False}

    project = 'snp'
    experiment_cfg = {'experiment_name': '60t_ema_q357',
                     'model_cfg': 'q159_i48_o5_h4_e100',
                     'preprocess_cfg': 'ES_60t_regime_2015_1_to_2021_6_grp_w8_ema_r',
                     'vars_definition_cfg': 'ES_ema_r',
                     'architecture': 'TFTModel'
                     }

    model_cfg = read_config(experiment_cfg['model_cfg'], project, subfolder='model')
    config, data_formatter, model_folder = get_model_data_config(project,
                                                                 experiment_cfg,
                                                                 model_cfg['model_params'],
                                                                 model_cfg['fixed_params'])

    n_repeat = 2
    results = []
    for _ in range(n_repeat):
        results.append(train_test_model(use_gpu=True,
                                        architecture=experiment_cfg['architecture'],
                                        prefetch_data=False,
                                        model_folder=model_folder,
                                        data_config=config.data_config,
                                        data_formatter=data_formatter,
                                        use_testing_mode=True,
                                        predict_eval=True,
                                        tb_callback=False,
                                        use_best_params=False,
                                        indicators_use_time_subset=False
                                        ))

    # %%
    histories = [res['history'] for res in results]

    df = pd.DataFrame(histories)
    plotly_time_series(df,
                       title='Loss History',
                       save=False,
                       legend=True,
                       rows=[1, 1],
                       file_path=os.path.join(config.results_folder,
                                              'img',
                                              '{}_{}_loss_hist'.format(experiment_cfg['architecture'],
                                                                       experiment_cfg['vars_definition'])),
                       size=(1980, 1080),
                       color_col=None,
                       markers='lines+markers',
                       xaxis_title="epoch",
                       markersize=5,
                       plot_title=True,
                       label_scale=1,
                       plot_ytitles=False)

    # post_process_results(results, formatter, experiment_cfg)
    #
    # if general_cfg['save_forecast']:
    #     save_vars(results, os.path.join(config.results_folder,
    #                                     experiment_cfg['experiment_name'],
    #                                     '{}_{}_forecasts'.format(experiment_cfg['architecture'],
    #                                                              experiment_cfg['vars_definition'])))
    #
    # print(results['hit_rates']['global_hit_rate'][1])
