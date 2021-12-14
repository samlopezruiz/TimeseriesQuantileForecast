import os
from copy import copy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.models.attn.utils import get_col_mapping
from src.timeseries.plot.ts import plotly_time_series, plotly_time_series_bars_hist, plotly_multiple
from src.timeseries.plot.utils import group_forecasts
from src.timeseries.utils.moo import get_hypervolume
from src.timeseries.utils.preprocessing import reconstruct_forecasts
from src.timeseries.utils.util import array_from_lists, mean_std_from_array


def rename_ensemble(df):
    as_list = df.index.tolist()
    ixs = np.flatnonzero(np.core.defchararray.find(as_list, '&') != -1)
    for ix in ixs:
        as_list[ix] = 'ENSEMBLE'
    df.index = as_list
    return df


def load_results(cfg, back_folders=4, result_folder='results', suffix=None):
    preffix, series, steps = cfg['preffix'], cfg['series'], cfg['steps']
    date, model, stage = cfg['date'], cfg['model'], cfg['stage']
    if suffix is not None:
        file_name = preffix + '_' + series + str(steps) + '_' + date + '_' + suffix + '.z'
    else:
        file_name = preffix + '_' + series + str(steps) + '_' + date + '.z'
    model_name = preffix + '_' + series + str(steps)
    back = ['..'] * back_folders

    path = os.path.join(*back, result_folder, model, stage, file_name)
    return joblib.load(path), model_name


def get_col_and_rename(d, e, var, comparison, comp_value):
    d = d.loc[:, [var[0]]]
    if var[1] is None:
        e['none'] = 0
        e = e.loc[:, ['none']]
    else:
        e = e.loc[:, [var[1]]]
    d.columns = [col + ' ' + comparison.upper() + '=' + str(comp_value).upper() for col in d.columns]
    e.columns = [col + ' ' + comparison.upper() + '=' + str(comp_value).upper() for col in e.columns]
    return d, e


def concat_sort_results(dat, err):
    data = pd.concat(dat, axis=1)
    errors = pd.concat(err, axis=1)
    data.sort_values([data.columns[0]], ascending=False, inplace=True)
    errors = errors.loc[data.index, :]
    return data, errors


def subset_results(metrics, metric_names=['rmse', 'minmax']):
    results = np.zeros((len(metrics), len(metric_names)))
    for i, metric in enumerate(metrics):
        for m, name in enumerate(metric_names):
            results[i][m] = metrics[i][name]

    return pd.DataFrame(results, columns=metric_names)


def get_results(metrics, model_cfg, test_x_pp, metric_names=['rmse', 'minmax'], plot_=True, print_=True):
    results = subset_results(metrics, metric_names=metric_names)
    if model_cfg['use_regimes']:
        results['regime'] = [np.mean(np.argmax(prob.to_numpy(), axis=1)) for x, prob in test_x_pp]
    if print_:
        for m in metric_names:
            print('Test {}: {} +-({})'.format(m, round(np.mean(results[m]), 2),
                                              round(np.std(results[m]), 4)))

    if plot_:
        rows = list(range(3 if model_cfg['use_regimes'] else 2))
        type_plot = ['bar' for _ in range(3 if model_cfg['use_regimes'] else 2)]
        plotly_time_series(results, rows=rows, xaxis_title='test subset', type_plot=type_plot, plot_ytitles=True)
        # plot_corr_df(results)
    if model_cfg['use_regimes']:
        results['reg_round'] = np.round(results['regime'])
    return results


def results_by_state(all_forecast_df):
    groupby_state = all_forecast_df.groupby('state')
    score_states = pd.DataFrame()
    score_states['count'] = groupby_state.count()['rse']
    score_states['mean'] = groupby_state.mean()['rse']
    score_states['std'] = groupby_state.std()['rse']
    score_states['perc'] = score_states['count'] / sum(score_states['count'])
    score_states.index.name = 'state'
    return score_states


def plot_multiple_results_forecast(all_forecast_df, forecast_dfs, use_regimes, results, max_subplots=15, n_plots=2,
                                   save=False, file_path=None):
    if use_regimes:
        file_path0 = copy(file_path)
        file_path0[-1] = file_path0[-1] + '_hist'
        plotly_time_series_bars_hist(all_forecast_df, features=['data', 'forecast', 'rse', 'state'],
                                     color_col='state', bars_cols=['rse'], save=save, file_path=file_path0)
        results_state = results_by_state(all_forecast_df).round(decimals=3)

        n_states = len(pd.unique(all_forecast_df['state']))
        subsets_state = []
        for i in range(n_states):
            subsets_state.append(list(results.loc[results['reg_round'] == i, 'reg_round'].index))

        file_path[-1] = file_path[-1] + '_plt' + str(0)
        for r, subsets in enumerate(subsets_state):
            chosen = sorted(np.random.choice(subsets, size=min(max_subplots, len(subsets))))
            dfs = [forecast_dfs[i] for i in chosen]
            title = 'Regime {}: {}'.format(r, str(results_state.iloc[r, :].to_dict()))
            file_path[-1] = file_path[-1][:-1] + str(r)
            plotly_multiple(dfs, features=['data', 'forecast'], title=title, save=save, file_path=file_path)
    else:
        file_path[-1] = file_path[-1] + '_plt' + str(0)
        chosen = sorted(np.random.choice(list(range(len(results))), size=min(max_subplots * n_plots, len(results))))
        for i in range(n_plots):
            chosen_plot = chosen[i * max_subplots:i * max_subplots + max_subplots]
            dfs = [forecast_dfs[i] for i in chosen_plot]
            title = 'Randomly selected forecasts'
            file_path[-1] = file_path[-1][:-1] + str(i)
            plotly_multiple(dfs, features=['data', 'forecast'], title=title, save=save, file_path=file_path)


def confusion_mat(y_true, y_pred, plot_=True, self_change=False):
    hit_rate = pd.DataFrame()
    hit_rate['up_down'] = y_true > y_true.shift(1)
    if not self_change:
        hit_rate['up_down_pred'] = y_pred > y_true.shift(1)
    else:
        hit_rate['up_down_pred'] = y_pred > y_pred.shift(1)
    hit_rate['hit_rate'] = hit_rate['up_down'] == hit_rate['up_down_pred']
    cm = confusion_matrix(hit_rate['up_down'], hit_rate['up_down_pred'], normalize='all')

    tn, fp, fn, tp = cm.ravel()
    if plot_:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)  # , display_labels = clf.classes_)
        disp.plot()
        plt.show()

    return cm, {'tn': round(tn, 4), 'fp': round(fp, 4), 'fn': round(fn, 4), 'tp': round(tp, 4)}


def hit_rate_from_forecast(results, n_output_steps, plot_=True):
    target_col = results['target']

    forecasts = results['reconstructed_forecasts'] if results['target'] else results['forecasts']
    if target_col:
        label = '{} t+{}'.format(target_col, 1)
    else:
        label = 't+{}'.format(1)

    try:
        cm, cm_metrics = confusion_mat(y_true=forecasts['targets'][label],
                                       y_pred=forecasts['p50'][label],
                                       plot_=plot_)
    except:
        cm, cm_metrics = None, None

    grouped = group_forecasts(forecasts, n_output_steps, target_col)

    confusion_mats = []
    for identifier, ss_df in grouped['targets'].items():
        try:
            confusion_mats.append(confusion_mat(y_true=ss_df[label],
                                                y_pred=grouped['p50'][identifier][label],
                                                plot_=False)[0])
            confusion_mats = np.stack(confusion_mats)
        except:
            confusion_mats = None

    return {
        'global_hit_rate': (cm, cm_metrics),
        'grouped_by_id_hit_rate': confusion_mats
    }


def post_process_results(results, formatter, experiment_cfg, plot_=True):
    model_params = formatter.get_default_model_params()
    n_output_steps = model_params['total_time_steps'] - model_params['num_encoder_steps']

    print('Reconstructing forecasts...')
    if results['target']:
        results['reconstructed_forecasts'] = reconstruct_forecasts(formatter, results['forecasts'], n_output_steps)
    results['hit_rates'] = hit_rate_from_forecast(results, n_output_steps, plot_=plot_)
    results['experiment_cfg'] = experiment_cfg


def compile_multiple_results(moo_results, experiment_labels, hv_ref=None):
    if hv_ref is None:
        hv_ref = [10] * 2

    results = {}
    for q_lbl, bound in zip(['lower quantile', 'upper quantile'], ['lq', 'uq']):
        results[q_lbl] = {}
        results[q_lbl]['risks'] = {}
        results[q_lbl]['eq_risks'] = {}
        results[q_lbl]['history'] = {}
        results[q_lbl]['hv_hist'] = {}
        results[q_lbl]['hv'] = {}
        for experiment, exp_lbl in zip(moo_results, experiment_labels):
            results[q_lbl]['risks'][exp_lbl] = [e[bound]['F'] for e in experiment]
            results[q_lbl]['eq_risks'][exp_lbl] = [e[bound]['eq_F'] for e in experiment]
            results[q_lbl]['history'][exp_lbl] = [e[bound]['pop_hist'] for e in experiment]
            results[q_lbl]['hv_hist'][exp_lbl] = [[get_hypervolume(F, hv_ref) for F in hist] for hist
                                                  in [e[bound]['pop_hist'] for e in experiment] if hist is not None]
            results[q_lbl]['hv'][exp_lbl] = [get_hypervolume(hist[-1], hv_ref) for hist
                                             in [e[bound]['pop_hist'] for e in experiment] if hist is not None]

    return results


def compile_multiple_results_q(moo_results, q_items=None, experiment_labels=None):
    if experiment_labels is None:
        experiment_labels = list(range(len(moo_results)))

    if q_items is None:
        q_items = {'lq': 'lower quantile', 'uq': 'upper quantile'}

    results = {}
    for q_lbl, q_name in q_items.items():
        exps = {}
        for i, moo_result in enumerate(moo_results):
            exps[experiment_labels[i]] = moo_result[q_lbl]
        results[q_name] = exps

    return results


def get_hv_results_from_runs(results, experiment_labels):
    hvs, q_exp_hvs = [], []
    for q_lbl, q_res in results.items():
        y_runs = [exp_res for exp_lbl, exp_res in q_res['hv'].items()]
        exp_runs = array_from_lists(y_runs)
        mean_std_df = mean_std_from_array(exp_runs, labels=experiment_labels)
        text = ['{:.3f} ({:.3f})'.format(s['mean'], s['std']) for _, s in mean_std_df.iterrows()]
        hvs.append(pd.DataFrame(text, columns=['Hv {}'.format(q_lbl)], index=experiment_labels))
        q_exp_hvs.append(exp_runs)

    q_exp_hvs = np.stack(q_exp_hvs)
    hvs_df = pd.concat(hvs, axis=1)
    return q_exp_hvs, hvs_df


def process_self_attention(attentions, params, taus=[1, 3, 5]):
    self_attentions = []
    # Plot attention for each head
    for i, head_self_attn in enumerate(attentions['decoder_self_attn']):
        self_attn_sample_avg = np.mean(head_self_attn, axis=0)
        n_ts, pred_steps = params['total_time_steps'], params['total_time_steps'] - params['num_encoder_steps']

        self_attn_taus = [pd.Series(self_attn_sample_avg[n_ts - (pred_steps - tau) - 1, :n_ts - (pred_steps - tau)],
                                    name='self_attn t={}'.format(tau)) for tau in taus]
        self_attns = pd.concat(self_attn_taus, axis=1)
        self_attns.index = np.array(self_attns.index) - params['num_encoder_steps']
        self_attentions.append(self_attns)
    return self_attentions


def process_historical_vars_attention(attentions, params):
    col_mapping = get_col_mapping(params['column_definition'])
    historical_attn = pd.DataFrame(np.mean(attentions['historical_flags'], axis=0),
                                   columns=col_mapping['historical_inputs'])
    historical_attn.index = np.array(historical_attn.index) - params['num_encoder_steps']

    mean_hist_attn = historical_attn.mean(axis=0).sort_values(ascending=False).to_frame(name='mean attn')
    sorted_hist_attn = historical_attn.loc[:, mean_hist_attn.index]
    return sorted_hist_attn, mean_hist_attn
