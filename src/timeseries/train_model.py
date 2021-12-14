import os
import time
import telegram_send
import tensorflow as tf
from src.timeseries.utils.config import read_config
from src.timeseries.utils.filename import quantiles_name
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.harness import get_model_data_config, train_test_model
from src.timeseries.utils.results import post_process_results

# tensorboard --logdir src/timeseries/experiments/market/outputs/saved_models/snp/5t_ema_q258/logs/fit
if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    project = 'snp'
    general_cfg = {'save_results': True,
                   'send_notifications': True}

    experiment_cfg = {'experiment_name': '60t_ema_q159',
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

    t0 = time.time()
    results = train_test_model(use_gpu=True,
                               architecture=experiment_cfg['architecture'],
                               prefetch_data=False,
                               model_folder=model_folder,
                               data_config=config.data_config,
                               data_formatter=data_formatter,
                               use_testing_mode=False,
                               predict_eval=True,
                               tb_callback=True,
                               use_best_params=True,
                               indicators_use_time_subset=True
                               )

    filename = '{}_{}_q{}_lr{}_pred'.format(experiment_cfg['architecture'],
                                            experiment_cfg['vars_definition_cfg'],
                                            quantiles_name(results['quantiles']),
                                            str(results['learning_rate'])[2:],
                                            )

    if general_cfg['send_notifications']:
        try:
            mins = round((time.time() - t0) / 60, 0)
            gens = 'in {} epochs'.format(len(results['fit_history']['loss']) if results['fit_history'] is not None else '')
            telegram_send.send(messages=["training for {} completed in {} mins {}".format(filename, mins, gens)])
        except Exception as e:
            pass

    post_process_results(results, data_formatter, experiment_cfg)

    if general_cfg['save_results']:
        results['model_params'] = model_cfg['model_params']
        results['fixed_params'] = model_cfg['fixed_params']
        results['experiment_cfg'] = experiment_cfg
        save_vars(results, os.path.join(config.results_folder,
                                        experiment_cfg['experiment_name'],
                                        filename))

    print(results['hit_rates']['global_hit_rate'][1])
