from src.timeseries.plot.ts import plotly_ts_regime_hist_vars
from src.timeseries.utils.hmm import count_regimes

import numpy as np
from scipy import stats as ss
import seaborn as sns
import matplotlib.pyplot as plt


def plotDistribution(Q, mus, sigmas, P, filename):
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
    one_eigval = np.argmin(np.abs(eigenvals - 1))
    pi = eigenvecs[:, one_eigval] / np.sum(eigenvecs[:, one_eigval])

    x_0 = np.linspace(mus[0] - 4 * sigmas[0], mus[0] + 4 * sigmas[0], 10000)
    fx_0 = pi[0] * ss.norm.pdf(x_0, mus[0], sigmas[0])

    x_1 = np.linspace(mus[1] - 4 * sigmas[1], mus[1] + 4 * sigmas[1], 10000)
    fx_1 = pi[1] * ss.norm.pdf(x_1, mus[1], sigmas[1])

    x = np.linspace(mus[0] - 4 * sigmas[0], mus[1] + 4 * sigmas[1], 10000)
    fx = pi[0] * ss.norm.pdf(x, mus[0], sigmas[0]) + \
         pi[1] * ss.norm.pdf(x, mus[1], sigmas[1])

    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(Q, color='k', alpha=0.5, density=True)
    l1, = ax.plot(x_0, fx_0, c='r', linewidth=2, label='Dry State Distn')
    l2, = ax.plot(x_1, fx_1, c='b', linewidth=2, label='Wet State Distn')
    l3, = ax.plot(x, fx, c='k', linewidth=2, label='Combined State Distn')

    fig.subplots_adjust(bottom=0.15)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True)
    fig.savefig(filename)
    plt.show()
    fig.clf()

    return None


def plotTimeSeries(Q, hidden_states, ylabel, filename):
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs = np.arange(len(Q)) + 1909
    masks = hidden_states == 0
    ax.scatter(xs[masks], Q[masks], c='r', label='State 0')
    masks = hidden_states == 1
    ax.scatter(xs[masks], Q[masks], c='b', label='State 1')
    ax.plot(xs, Q, c='k')

    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
    fig.savefig(filename)
    plt.show()
    fig.clf()

    return None


def plot_hmm(df_reg,
             price_col,
             hmm_vars,
             n_states,
             in_cfg,
             regime_col='state',
             resample=False,
             n_regimes=None,
             label_scale=1):

    save_folder = in_cfg['image_folder']
    save_plots = in_cfg['save_results']
    df_plot = df_reg.resample('90T').last() if resample else df_reg
    if n_regimes is not None:
        title = 'Regime Changes: {}, n_states: {}, vars: {}'.format(count_regimes(n_regimes), n_states, str(hmm_vars))
    else:
        title = 'n_states: {}, vars: {}'.format(count_regimes(n_regimes), n_states, str(hmm_vars))
    name = 'hmm_{}'.format('_' + '_'.join(hmm_vars))
    plotly_ts_regime_hist_vars(df_plot,
                               price_col,
                               regime_col,
                               features=hmm_vars,
                               adjust_height=(True, 0.6),
                               markersize=4,
                               save_png=True,
                               save=save_plots,
                               file_path=[save_folder, name],
                               size=(1980, 1080),
                               plot_title=in_cfg['plot_title'],
                               title=title, label_scale=label_scale)
