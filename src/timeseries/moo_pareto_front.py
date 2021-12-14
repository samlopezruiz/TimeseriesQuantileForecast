import os

import joblib
import seaborn as sns
import telegram_send

from src.timeseries.moo.dual_problem_def import DualQuantileWeights
from src.timeseries.utils.filename import get_result_folder, quantiles_name, termination_name
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.harness import get_model_data_config
from src.timeseries.utils.moo_harness import run_dual_moo_weights

sns.set_theme('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'save_results': True,
                   'save_history': False,
                   'send_notifications': True}

    prob_cfg = {}
    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'
                   }

    algo_cfg = {'termination': ('n_gen', 100),
                'pop_size': 100,
                'use_sampling': False,
                'optimize_eq_weights': False,
                'use_constraints': True,
                'constraints': [1., 1.],
                }

    moo_method = 'NSGA2'

    model_results = joblib.load(os.path.join(get_result_folder(results_cfg, project), results_cfg['results'] + '.z'))

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
                                         parallelize_pop=False if moo_method == 'MOEAD' else True,
                                         constraints_limits=algo_cfg['constraints'] if algo_cfg[
                                             'use_constraints'] else None,
                                         optimize_eq_weights=algo_cfg['optimize_eq_weights'])

    lower_q_problem, upper_q_problem = dual_q_problem.get_problems()

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

    res = run_dual_moo_weights(moo_method,
                               algo_cfg,
                               general_cfg,
                               prob_cfg,
                               lower_q_problem,
                               upper_q_problem,
                               dual_q_problem,
                               model_results,
                               verbose=2)

    if general_cfg['send_notifications']:
        try:
            telegram_send.send(messages=["moo for {} completed in {} mins, tot: {}".format(filename,
                                                                                           res['times'],
                                                                                           sum(res['times']))])
        except Exception as e:
            print(e)

    if general_cfg['save_results']:
        save_vars(res['results'],
                  os.path.join(config.results_folder,
                               experiment_cfg['experiment_name'],
                               'moo',
                               filename))
