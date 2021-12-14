import os
import time

import joblib
import seaborn as sns
import telegram_send

from src.timeseries.moo.dual_problem_def import DualQuantileWeights
from src.timeseries.utils.filename import get_result_folder, quantiles_name, termination_name
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.harness import get_model_data_config
from src.timeseries.utils.moo_harness import run_dual_moo_weights
from src.timeseries.utils.parallel import repeat

sns.set_theme('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'save_results': True,
                   'save_history': True,
                   'send_notifications': True}

    prob_cfg = {}
    project = 'snp'
    algo_cfg = {'termination': ('n_gen', 100),
                'pop_size': 100,
                'use_sampling': False,
                'optimize_eq_weights': True,
                'use_constraints': True,
                'constraints': [1., 2.5],
                }

    experiment_names = [
        '60t_ema_q159',
        '60t_ema_q258',
        '60t_ema_q357',
    ]
    results_names = [
        'TFTModel_ES_ema_r_q159_lr01_pred',
        'TFTModel_ES_ema_r_q258_lr01_pred',
        'TFTModel_ES_ema_r_q357_lr01_pred',
    ]

    moo_methods = ['NSGA2']#, 'NSGA3', 'MOEAD']
    n_repeats = 3

    for experiment_name, results_name in zip(experiment_names, results_names):


        results_cfg = {'formatter': 'snp',
                       'experiment_name': experiment_name,
                       'results': results_name
                       }
        model_results = joblib.load(
            os.path.join(get_result_folder(results_cfg, project), results_cfg['results'] + '.z'))

        config, data_formatter, model_folder = get_model_data_config(project,
                                                                     model_results['experiment_cfg'],
                                                                     model_results['model_params'],
                                                                     model_results['fixed_params'])
        experiment_cfg = model_results['experiment_cfg']

        dual_q_problem = DualQuantileWeights(architecture=experiment_cfg['architecture'],
                                             model_folder=model_folder,
                                             data_formatter=data_formatter,
                                             data_config=config.data_config,
                                             use_gpu=True,
                                             parallelize_pop=True,
                                             constraints_limits=algo_cfg['constraints'] if algo_cfg[
                                                 'use_constraints'] else None,
                                             optimize_eq_weights=algo_cfg['optimize_eq_weights'])

        lower_q_problem, upper_q_problem = dual_q_problem.get_problems()

        for moo_method in moo_methods:
            lower_q_problem.parallelize_pop = False if moo_method == 'MOEAD' else True
            upper_q_problem.parallelize_pop = False if moo_method == 'MOEAD' else True
            filename = '{}_{}_q{}_{}_{}_p{}_s{}_c{}_eq{}_dual_wmoo'.format(experiment_cfg['architecture'],
                                                                           experiment_cfg['vars_definition_cfg'],
                                                                           quantiles_name(dual_q_problem.quantiles),
                                                                           moo_method,
                                                                           termination_name(algo_cfg['termination']),
                                                                           algo_cfg['pop_size'],
                                                                           int(algo_cfg['use_sampling']),
                                                                           int(algo_cfg['use_constraints']),
                                                                           int(algo_cfg['optimize_eq_weights']),
                                                                           )

            args = (moo_method,
                    algo_cfg,
                    general_cfg,
                    prob_cfg,
                    lower_q_problem,
                    upper_q_problem,
                    dual_q_problem,
                    model_results)

            t0 = time.time()
            results_repeat = repeat(run_dual_moo_weights, args, n_repeat=n_repeats, parallel=False)
            t1 = round((time.time() - t0) / 60, 0)
            print(time.time() - t0)

            if general_cfg['send_notifications']:
                try:
                    telegram_send.send(messages=["{} repeats moo for {} completed in {} mins".format(n_repeats,
                                                                                                     filename,
                                                                                                     t1)])
                except Exception as e:
                    print(e)

            results_repeat = [res['results'] for res in results_repeat]
            if general_cfg['save_results']:
                save_vars(results_repeat,
                          os.path.join(config.results_folder,
                                       experiment_cfg['experiment_name'],
                                       'moo',
                                       '{}_repeat{}'.format(filename, n_repeats)))
