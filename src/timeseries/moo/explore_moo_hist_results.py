import os

import joblib
import numpy as np
import seaborn as sns

from src.models.compare.winners import Winners
from src.timeseries.plot.moo import plot_runs, plot_boxplot
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.results import compile_multiple_results, get_hv_results_from_runs
from src.timeseries.utils.util import write_latex_from_scores, write_text_file, latex_table

sns.set_theme('poster')
sns.set_style("dark")

if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': False,
                   'save_results': False,
                   'show_title': False,
                   'comparison_name': 'moo_methods_ES_ema_r_q258_g100_p100_s0_c1_eq0',
                   }

    project = 'snp'

    weights_files = [
        ('60t_ema_q258', 'TFTModel_ES_ema_r_q258_NSGA2_g100_p100_s0_c1_eq0_dual_wmoo_repeat20'),
        ('60t_ema_q258', 'TFTModel_ES_ema_r_q258_NSGA3_g100_p100_s0_c1_eq0_dual_wmoo_repeat20'),
    ]

    results_folder = os.path.join(project, 'compare', general_cfg['comparison_name'])
    moo_results = [joblib.load(os.path.join(get_result_folder({}, project), file[0], 'moo', file[1]) + '.z')
                   for file in weights_files]

    # experiment_labels = [
    #     'q {}'.format('-'.join((np.array(experiment[0]['lq']['quantiles']) * 10).astype(int).astype(str))) for
    #     experiment
    #     in moo_results]
    experiment_labels = [experiment[0]['lq']['moo_method'] for experiment in moo_results]
    results = compile_multiple_results(moo_results, experiment_labels, hv_ref=[10] * 2)

    # %% HV results
    q_exp_hvs, hvs_df = get_hv_results_from_runs(results, experiment_labels)

    # %% Winners
    metric = np.negative(np.mean(q_exp_hvs, axis=2))
    winners = Winners(metric, experiment_labels)
    scores = winners.score(q_exp_hvs, alternative='greater')

    # %% Individual runs per quantile

    for q_lbl, q_res in results.items():
        for exp_lbl, exp_res in q_res['hv_hist'].items():
            y_runs = np.array(exp_res)[:, 1:] #remove first generation in case of constraints

            filename = '{}_{}_runs'.format(exp_lbl, q_lbl.replace(" ", "_"))
            plot_runs(y_runs,
                      mean_run=np.mean(y_runs, axis=0),
                      x_label='Generation',
                      y_label='Hypervolume',
                      title='{} HV history for {}'.format(exp_lbl, q_lbl),
                      size=(15, 9),
                      file_path=os.path.join(results_folder, 'img', filename),
                      save=general_cfg['save_plot'],
                      legend_labels=None,
                      show_grid=True,
                      show_title=general_cfg['show_title'])

    # %% Grouped runs per quantile
    q_exp_hist, q_exp_mean, all_plot_labels = [], [], []
    for q_lbl, q_res in results.items():
        y_runs = []
        for exp_lbl, exp_res in q_res['hv_hist'].items():
            exp_res_mod = np.array(exp_res)[:, 1:]
            q_exp_mean.append(np.mean(exp_res_mod, axis=0))
            y_runs.append(np.mean(exp_res_mod, axis=0))
            all_plot_labels.append('{} {}'.format(exp_lbl, 'lq' if q_lbl == 'lower quantile' else 'uq'))

        filename = '{}_{}_comparison'.format(general_cfg['comparison_name'], q_lbl.replace(" ", "_"))
        exp_runs = np.array(y_runs)
        q_exp_hist.append(exp_runs)

    q_exp_mean = np.array(q_exp_mean)

    # plot all
    plot_runs(q_exp_hist,
              mean_run=None,
              x_label='Generation',
              y_label='Hypervolume',
              title='HV history',
              size=(15, 9),
              file_path=os.path.join(results_folder, 'img', filename),
              save=general_cfg['save_plot'],
              legend_labels=all_plot_labels,
              show_grid=True,
              show_title=general_cfg['show_title'])

    plot_boxplot(q_exp_mean,
                 all_plot_labels,
                 x_label='Algorithm',
                 y_label='Hypervolume',
                 title='Hypervolume for quantiles',
                 size=(15, 9),
                 # ylim=(96, 99),
                 file_path=os.path.join(results_folder, 'img', filename),
                 save=general_cfg['save_plot'],
                 show_grid=True,
                 show_title=general_cfg['show_title'])

    # %%
    if general_cfg['save_results']:
        results['weights_files'] = weights_files
        save_vars(results, os.path.join(results_folder,
                                        '{}'.format(general_cfg['comparison_name'])))

        write_latex_from_scores(scores,
                                os.path.join(results_folder,
                                             'txt',
                                             '{}_scores'.format(general_cfg['comparison_name'])))

        write_text_file(os.path.join(results_folder,
                                     'txt',
                                     '{}'.format(general_cfg['comparison_name'])),
                        latex_table('Hypervolume for quantiles', hvs_df.to_latex()))
