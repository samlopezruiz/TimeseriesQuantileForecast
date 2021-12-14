import os

import numpy as np
import pandas as pd
import plotly.io as pio
import seaborn as sns
from matplotlib.patches import Rectangle, Patch
from pymoo.core.result import Result

from src.timeseries.utils.files import create_dir, get_new_file_path
from src.timeseries.utils.moo import get_pymoo_pops_obj, get_deap_pops_obj

pio.renderers.default = "browser"

colors = [
    '#1f77b4',  # muted blue
    '#2ca02c',  # cooked asparagus green
    '#ff7f0e',  # safety orange
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
    'rgba(0, 0, 0, 0)'  # transparent ix=
]

marker_symbols = ['circle', 'x', 'cross', 'circle-open']

import pymoo
from matplotlib import pyplot as plt
from pymoo.factory import get_performance_indicator

sns_colors = sns.color_palette()


def plot_hv(hypervols, title='', save=False):
    plt.plot(hypervols)
    plt.xlabel('Iterations (t)')
    plt.ylabel('Hypervolume')
    plt.title(title)
    plt.show()


def plot_hist_hv(hist, save=False):
    if isinstance(hist, pymoo.core.result.Result):
        pops_obj, ref = get_pymoo_pops_obj(hist)
    else:
        pops_obj, ref = get_deap_pops_obj(hist)

    hv = get_performance_indicator("hv", ref_point=ref)
    hypervols = [hv.do(pop_obj) for pop_obj in pops_obj]
    plot_hv(hypervols, title='Hypervolume History', save=save)
    return hypervols

def plot_runs(y_runs,
              mean_run=None,
              x_label=None,
              y_label=None,
              title=None,
              size=(15, 9),
              file_path=None,
              save=False,
              legend_labels=None,
              show_grid=True,
              use_date=False,
              show_title=True,
              linewidth=5):
    fig, ax = plt.subplots(figsize=size)

    if isinstance(y_runs, list):
        xs = [np.arange(y_run.shape[1]).astype(int) + 1 for y_run in y_runs]
        for j, (x, y_run) in enumerate(zip(xs, y_runs)):
            for i, y in enumerate(y_run):
                ax.plot(x, y, '--' if j == 1 else '-',
                        color=sns_colors[i],
                        alpha=1,
                        linewidth=linewidth,
                        )

        plt.xlim([0, max([len(x) for x in xs])])

    else:
        x = np.arange(y_runs.shape[1]).astype(int) + 1

        for y in y_runs:
            sns.lineplot(x=x, y=y, ax=ax,
                         color='gray' if mean_run is not None else None,
                         alpha=0.8,
                         linewidth=linewidth)
        if mean_run is not None:
            sns.lineplot(x=x, y=mean_run, ax=ax,
                         color='blue',
                         linewidth=int(linewidth * 1.2))

        plt.xlim([0, len(x)])

    ax.set(xlabel='x' if x_label is None else x_label,
           ylabel='x' if y_label is None else y_label)
    if show_title:
        ax.set_title('' if title is None else title)

    if legend_labels is not None:
        plt.legend(labels=legend_labels)
    if show_grid:
        plt.grid()
    plt.show()

    if save:
        save_fig(fig, file_path, use_date)


def plot_boxplot(Y,
                 x_labels,
                 x_label=None,
                 y_label=None,
                 title=None,
                 size=(15, 9),
                 file_path=None,
                 save=False,
                 show_grid=True,
                 use_date=False,
                 ylim=None,
                 show_title=True):
    Y = pd.DataFrame(Y.T, columns=x_labels)
    df = Y.melt()
    df.columns = ['variable' if x_label is None else x_label,
                  'value' if y_label is None else y_label]
    fig, ax = plt.subplots(figsize=size)
    if show_grid:
        plt.grid()
    # for i, y in enumerate(Y):
    sns.boxplot(data=df, x=x_label, y=y_label)
    # sns.barplot(data=df, x=x_label, y=y_label, capsize=.2)
    if show_title:
        ax.set_title('' if title is None else title)
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.show()

    if save:
        save_fig(fig, file_path, use_date)


def save_fig(fig, file_path, use_date):
    if file_path is not None:
        create_dir(file_path)
        file_path = get_new_file_path(file_path, '.png', use_date)
        print('Saving image:\n{}\n'.format(file_path))
        fig.savefig(os.path.join(file_path))



def get_ixs_risk(F, add_risk):
    tot_risk = np.sum(F, axis=1)
    min_ix = np.argmin(tot_risk)
    min_risk = min(tot_risk)
    max_risk = min_risk * (1 + add_risk)
    l_ix = np.argmin(np.abs(tot_risk[:min_ix] - max_risk))
    r_ix = np.argmin(np.abs(tot_risk[min_ix:] - max_risk)) + min_ix
    return l_ix, r_ix


def get_rect_risk(F, add_risk, color_ix=None, external_F=None):
    if external_F is None:
        external_F = F

    color = 'g' if color_ix is None else sns_colors[color_ix]

    l_ix, r_ix = get_ixs_risk(F, add_risk)

    tot_risk = np.sum(F, axis=1)
    min_risk = min(tot_risk[l_ix:r_ix])
    max_risk = min_risk * (1 + add_risk)
    vertex = (external_F[l_ix, 0], external_F[r_ix, 1])
    width = external_F[r_ix, 0] - external_F[l_ix, 0]
    height = external_F[l_ix, 1] - external_F[r_ix, 1]
    rect_pareto = Rectangle(vertex, width, height, linewidth=3, edgecolor=color, facecolor='none', linestyle='--')

    # print('\nqcr: {}-{}, {}%'.format(np.round(vertex[0], 2),
    #                                  np.round(vertex[0] + width, 2),
    #                                  np.round(width / vertex[0] * 100, 1)))
    # print('qce: {}-{}, {}%'.format(np.round(vertex[1], 2),
    #                                np.round(vertex[1] + height, 2),
    #                                np.round(height / vertex[1] * 100, 1)))

    min_tot_risk = min(np.sum(external_F, axis=1)[l_ix:r_ix])
    max_tot_risk = max(np.sum(external_F, axis=1)[l_ix:r_ix])
    vertex = (external_F[l_ix, 0], min_tot_risk)
    width = external_F[r_ix, 0] - external_F[l_ix, 0]
    height = max_tot_risk - min_tot_risk
    rect_total = Rectangle(vertex, width, height, linewidth=3, edgecolor=color, facecolor='none', linestyle='--')

    return rect_pareto, rect_total

def plot_2D_trace_moo(ax,
                      col,
                      F,
                      x_mask,
                      color_row0,
                      color_row1,
                      marker='-o',
                      markersize=8,
                      label=None,
                      edgecolor=(False, 0)
                      ):
    ax[0, col].plot(F[x_mask, 0], F[x_mask, 1],
                    marker,
                    markersize=markersize,
                    markeredgecolor=sns_colors[edgecolor[1]] if edgecolor[0] else color_row0,
                    markeredgewidth=2 if edgecolor[0] else 1,
                    color=color_row0,
                    label=label)
    ax[1, col].plot(F[x_mask, 0], np.sum(F[x_mask, :], axis=1),
                    marker,
                    markersize=markersize,
                    markeredgecolor=sns_colors[edgecolor[1]] if edgecolor[0] else color_row1,
                    markeredgewidth=2 if edgecolor[0] else 1,
                    color=color_row1,
                    label=label
                    )

def get_x_mask(F_input, xaxis_limit):
    x_mask = []
    for Fs in F_input:
        if xaxis_limit is not None:
            if isinstance(Fs, list):
                Fs_x_plot_masks = [F[:, 0] < xaxis_limit for F in Fs]
            else:
                Fs_x_plot_masks = Fs[:, 0] < xaxis_limit
        else:
            if isinstance(Fs, list):
                Fs_x_plot_masks = [np.ones((F.shape[0],)).astype(bool) for F in Fs]
            else:
                Fs_x_plot_masks = np.ones((Fs.shape[0],)).astype(bool)

        x_mask.append(Fs_x_plot_masks)
    return x_mask

def plot_2D_moo_dual_results(Fs,
                             save=False,
                             file_path=None,
                             selected_ixs=None,
                             original_ixs=None,
                             original_losses=None,
                             figsize=(15, 15),
                             use_date=False,
                             title='Multi objective optimization',
                             col_titles=None,
                             legend_labels=None,
                             add_risk=None,
                             xaxis_limit=None,
                             markersize=5,
                             plot_title=True):
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    x_mask = get_x_mask(Fs, xaxis_limit)

    for i, F in enumerate(Fs):
        if isinstance(F, list):
            for j, f in enumerate(F):
                plot_2D_trace_moo(ax, i, f, x_mask[i][j], sns_colors[j], sns_colors[j],
                                  markersize=markersize,
                                  label=legend_labels[j])

            for j, f in enumerate(F):
                if add_risk is not None:
                    rect_pareto, rect_total = get_rect_risk(f, add_risk, j)
                    ax[0, i].add_patch(rect_pareto)
                    ax[1, i].add_patch(rect_total)

            for j, f in enumerate(F):
                if original_losses is not None:
                    # ix_x_mask = np.arange(f.shape[0]) == original_ixs[i][j]
                    original_loss = np.array(original_losses[i][j]).reshape((1, -1))
                    plot_2D_trace_moo(ax, i, original_loss, [0], 'black', 'black', marker='*',
                                      markersize=26,
                                      label='Original' if j == 0 else None,
                                      edgecolor=(True, j))

                if original_ixs is not None:
                    ix_x_mask = np.arange(f.shape[0]) == original_ixs[i][j]
                    plot_2D_trace_moo(ax, i, f, ix_x_mask, 'black', 'black', marker='*',
                                      markersize=26,
                                      label='Original' if j == 0 else None,
                                      edgecolor=(True, j))
                if selected_ixs is not None:
                    ix_x_mask = np.arange(f.shape[0]) == selected_ixs[i][j]
                    plot_2D_trace_moo(ax, i, f, ix_x_mask, 'red', 'red',
                                      marker='*',
                                      markersize=24,
                                      label='Selected' if j == 0 else None)
        else:
            plot_2D_trace_moo(ax, i, F, x_mask[i], sns_colors[0], 'gray',
                              markersize=markersize)

            if original_ixs is not None:
                ix_x_mask = np.arange(F.shape[0]) == original_ixs[i]
                plot_2D_trace_moo(ax, i, F, ix_x_mask, 'black', 'black', marker='*',
                                  markersize=24,
                                  label='Original')

            if original_losses is not None:
                original_loss = np.array(original_losses[i]).reshape((1, -1))
                plot_2D_trace_moo(ax, i, original_loss, [0], 'black', 'black', marker='*',
                                  markersize=26,
                                  label='Original')

            if selected_ixs is not None:
                ix_x_mask = np.arange(F.shape[0]) == selected_ixs[i]
                plot_2D_trace_moo(ax, i, F, ix_x_mask, 'red', 'red',
                                  marker='*',
                                  markersize=24,
                                  label='Selected')

            if add_risk is not None:
                rect_pareto, rect_total = get_rect_risk(F, add_risk)
                ax[0, i].add_patch(rect_pareto)
                ax[1, i].add_patch(rect_total)

    if col_titles is None:
        col_titles = ['quantile A', 'quantile B']

    for i, label in enumerate(col_titles):
        ax[0, i].set_title('Pareto front: {}'.format(label), fontweight="bold")
        ax[0, i].set_xlabel('Quantile coverage risk')
        ax[0, i].set_ylabel('Quantile estimation risk')

        ax[1, i].set_title('Total risk: {}'.format(label), fontweight="bold")
        ax[1, i].set_xlabel('Quantile coverage risk')
        ax[1, i].set_ylabel('Risk')

        ax[0, i].legend()
        ax[1, i].legend()

    if add_risk is not None:
        for i in range(2):
            for j in range(2):
                handles, labels = ax[i, j].get_legend_handles_labels()
                ax[i, j].legend(
                    handles=handles + [Patch(facecolor='None', edgecolor='g', label='Tolerance',
                                             linestyle='--')],
                )
    if plot_title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    if save:
        save_fig(fig, file_path, use_date)

