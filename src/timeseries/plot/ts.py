import math
import time
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt

from plotly import graph_objects as go
from plotly.subplots import make_subplots

import plotly.io as pio

from src.timeseries.plot.utils import plotly_save, plotly_params_check
from src.timeseries.utils.dataframe import check_cols_exists, check_col_exists

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
]

gray_colors = [
    'lightgray', 'gray', 'darkgray', 'lightslategray',
]


def plotly_multiple(dfs,
                    title=None,
                    save=False,
                    features=None,
                    file_path=None,
                    size=(1980, 1080),
                    markers='lines+markers',
                    markersize=5,
                    plot_title=True,
                    label_scale=1,
                    **kwargs):
    n_rows = math.ceil(len(dfs) / 2)
    n_cols = 2 if len(dfs) > 2 else 1

    rows, cols = [], []
    j = -1
    for i in range(len(dfs)):
        if i % 2 == 0:
            j += 1
        cols.append(i % 2)
        rows.append(j)

    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False)
    for j, df in enumerate(dfs):
        features = df.columns if features is None else features
        for i in range(len(features)):
            df_ss = df[features[i]].dropna()
            fig.append_trace(
                go.Scatter(
                    x=df_ss.index,
                    y=df_ss,
                    visible=True,
                    showlegend=False if j < len(dfs) - 1 else True,
                    name=features[i],
                    mode=markers,
                    line=dict(color=colors[i]),
                    marker=dict(size=markersize,
                                color=colors[i]),
                ),
                row=rows[j] + 1,
                col=cols[j] + 1
            )

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)


def plotly_time_series(df,
                       title=None,
                       save=False,
                       legend=True,
                       file_path=None,
                       size=(1980, 1080),
                       color_col=None,
                       markers='lines+markers',
                       xaxis_title="time",
                       markersize=5,
                       plot_title=True,
                       label_scale=1,
                       adjust_height=(False, 0.6),
                       plot_ytitles=False,
                       save_png=False,
                       **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, alphas = params
    n_rows = len(set(rows))
    n_cols = 1  # len(set(cols))
    if not params_ok:
        return

    f = len(features)
    if adjust_height[0]:
        heights = [(1 - adjust_height[1]) / (n_rows - 1) for _ in range(n_rows - 1)]
        heights.insert(0, adjust_height[1])
    else:
        heights = [1 / n_rows for _ in range(n_rows)]

    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True, row_heights=heights)
    n_colors = len(df[color_col].unique()) if color_col is not None else 2
    for i in range(f):
        df_ss = df[features[i]].dropna()
        fig.append_trace(
            go.Bar(
                x=df_ss.index,
                y=df_ss,
                orientation="v",
                visible=True,
                showlegend=legend,
                name=features[i],
                opacity=alphas[i]
            ) if type_plot[i] == 'bar' else
            go.Scatter(
                x=df_ss.index,
                y=df_ss,
                visible=True,
                showlegend=legend,
                name=features[i],
                mode=markers,
                line=dict(color='lightgray' if color_col is not None else None),
                marker=dict(size=markersize,
                            color=None if color_col is None else df[color_col].values,
                            colorscale=colors[:n_colors]),  # "Bluered_r"),
                opacity=alphas[i]
            ),
            row=rows[i] + 1,
            col=cols[i] + 1
        )
        if i == n_rows - 1:
            fig['layout']['xaxis' + str(i + 1)]['title'] = xaxis_title

        if plot_ytitles:
            for i in range(n_rows):
                fig['layout']['yaxis' + str(i + 1)]['title'] = features[i]

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)
    # return fig


def plotly_color_1st_row(df,
                         color_col,
                         first_row_feats,
                         other_feats,
                         rows=None,
                         title=None,
                         save=False,
                         legend=True,
                         file_path=None,
                         size=(1980, 1080),
                         xaxis_title="time",
                         markersize=5,
                         plot_title=True,
                         label_scale=1,
                         adjust_height=(False, 0.6),
                         plot_ytitles=False,
                         save_png=False,
                         **kwargs):
    # params_ok, params = plotly_params_check(df, **kwargs)
    # features, rows, cols, type_plot, alphas = params
    if rows is None:
        rows = list(range(len(other_feats)))
    n_rows = len(set(rows)) + 1
    n_cols = 1  # len(set(cols))
    # if not params_ok:
    #     return

    f = len(other_feats) + 1
    if adjust_height[0]:
        heights = [(1 - adjust_height[1]) / (n_rows - 1) for _ in range(n_rows - 1)]
        heights.insert(0, adjust_height[1])
    else:
        heights = [1 / n_rows for _ in range(n_rows)]

    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True, row_heights=heights)

    for i, (color_lbl, df_ss) in enumerate(df.groupby(by=color_col)):
        for f in range(len(first_row_feats)):
            df_ssf = df_ss[first_row_feats[f]]  # .dropna()
            fig.append_trace(
                go.Scatter(
                    x=df_ssf.index,
                    y=df_ssf,
                    visible=True,
                    showlegend=True if i == 0 else False,
                    name=first_row_feats[f],
                    mode='markers',
                    line=dict(color='lightgray' if color_col is not None else None),
                    marker=dict(size=markersize,
                                color=colors[i]),
                ),
                row=1,
                col=1
            )

    for i in range(len(other_feats)):
        df_ss = df[other_feats[i]]  # .dropna()
        fig.append_trace(
            go.Scatter(
                x=df_ss.index,
                y=df_ss,
                visible=True,
                showlegend=legend,
                name=other_feats[i],
                mode='lines',
                # line=dict(color='lightgray' if color_col is not None else None),
                marker=dict(size=markersize,
                            color=colors[i]),
            ),
            row=rows[i] + 1,
            col=1
        )

        if i == n_rows - 1:
            fig['layout']['xaxis' + str(i + 1)]['title'] = xaxis_title

        # if plot_ytitles:
        #     for i in range(n_rows):
        #         fig['layout']['yaxis' + str(i + 1)]['title'] = features[i]

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)


def plotly_ts_regime(df,
                     title=None,
                     save=False,
                     legend=False,
                     file_path=None,
                     size=(1980, 1080),
                     regime_col=None,
                     regimes=None,
                     markers='lines+markers',
                     xaxis_title="time",
                     markersize=5,
                     plot_title=True,
                     template='plotly_white',
                     adjust_height=(False, 0.6),
                     label_scale=1,
                     use_date_suffix=False,
                     save_png=False,
                     legend_labels=None,
                     **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, alphas = params
    n_rows = len(set(rows))
    n_cols = 1  # len(set(cols))
    if not params_ok:
        return

    f = len(features)
    if adjust_height[0]:
        heights = [(1 - adjust_height[1]) / (n_rows - 1) for _ in range(n_rows - 1)]
        heights.insert(0, adjust_height[1])
    else:
        heights = [1 / n_rows for _ in range(n_rows)]
    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True, row_heights=heights)
    reg_colors = ['lightblue', 'dodgerblue', 'darkblue', 'black']

    if regime_col is not None:
        color_scale = colors[:int(max(df[regime_col].dropna().values)) + 1]

    for i in range(f):

        if regime_col is not None and len(features) == 1:

            for j, (val, df_g) in enumerate(df.groupby(by=regime_col)):
                df_ss = df_g[features[i]].dropna()

                fig.append_trace(
                    go.Scatter(
                        x=df_ss.index,
                        y=df_ss,
                        visible=True,
                        showlegend=False,
                        name=val if legend_labels is None else legend_labels[j],
                        mode=markers,
                        line=dict(color='lightgray' if regime_col is not None else None),
                        marker=dict(size=markersize,
                                    color=color_scale[j],
                                    colorscale=color_scale),
                    ),
                    row=rows[i] + 1,
                    col=cols[i] + 1
                )
                # only for legend markers
                fig.append_trace(
                    go.Scatter(
                        x=df_ss[:1] * np.nan,
                        y=df_ss[:1] * np.nan,
                        visible=True,
                        showlegend=legend,
                        name=val if legend_labels is None else legend_labels[j],
                        mode=markers,
                        marker=dict(size=markersize * 3,
                                    color=color_scale[j],
                                    colorscale=color_scale),
                    ),
                    row=rows[i] + 1,
                    col=cols[i] + 1
                )

        else:
            df_ss = df[features[i]].dropna()
            fig.append_trace(
                go.Scatter(
                    x=df_ss.index,
                    y=df_ss,
                    visible=True,
                    showlegend=legend,
                    name=features[i],
                    mode=markers,
                    line=dict(color='lightgray' if regime_col is not None else None),
                    marker=dict(size=markersize,
                                color=None if regime_col is None else df[regime_col].dropna().values,
                                colorscale=None if regime_col is None else color_scale),
                    # ["red", "green", "blue"]),  # Bluered_r, Inferno
                    opacity=alphas[i]
                ),
                row=rows[i] + 1,
                col=cols[i] + 1
            )
        for i in range(n_rows):
            fig['layout']['yaxis' + str(i + 1)]['title'] = features[i]

        if regimes is not None:
            for r, regime in enumerate(regimes):
                for i in range(0, len(regime), 2):
                    fig.add_vrect(
                        x0=regime[i], x1=regime[i + 1],
                        fillcolor=reg_colors[r], opacity=0.3,
                        layer="below", line_width=0,
                    )
        if i == n_rows - 1:
            fig['layout']['xaxis' + str(i + 1)]['title'] = xaxis_title

    # 'lightcyan',
    # colors = ['beige', 'palegoldenrod', 'burlywood', 'orange', 'dodgerblue', 'teal']

    fig.update_layout(legend={'itemsizing': 'trace'})
    fig.update_layout(template=template, xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(color='black', tickfont=dict(size=14 * label_scale),
                     title_font=dict(size=18 * label_scale),
                     showgrid=True, gridwidth=1, gridcolor='gray')
    fig.update_yaxes(color='black', tickfont=dict(size=14 * label_scale),
                     title_font=dict(size=18 * label_scale),
                     showgrid=True, gridwidth=1, gridcolor='gray')

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png, use_date_suffix)


def plotly_one_series(s, title=None, save=False, legend=True, file_path=None, size=(1980, 1080),
                      markers='lines+markers', xaxis_title="time", label_scale=1, **kwargs):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.append_trace(
        go.Scatter(
            x=s.index,
            y=np.array(s),
            orientation="v",
            visible=True,
            showlegend=legend,
            name=s.model_name,
        ),
        row=1,
        col=1
    )
    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title, legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plotly_phase_plots(df, title=None, save=False, file_path=None, size=(1980, 1080), label_scale=1, legend=True,
                       **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, alphas = params

    if not params_ok:
        return

    f = len(features)
    fig = make_subplots(rows=1, cols=f)
    comb = combinations([0, 1, 2], 2)
    for i, c in enumerate(comb):
        fig.append_trace(
            go.Scatter(
                x=df[features[c[0]]],
                y=df[features[c[1]]],
                visible=True,
                showlegend=False,
            ),
            row=1,
            col=i + 1
        )
        fig['layout']['xaxis' + str(i + 1)]['title'] = features[c[0]]
        fig['layout']['yaxis' + str(i + 1)]['title'] = features[c[1]]

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, title=title)
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    # plotly(fig)
    fig.show()
    time.sleep(1)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plotly_3d(df, title=None, save=False, file_path=None, size=(1980, 1080), legend=True, label_scale=1, **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, alphas = params

    if not params_ok:
        return

    f = len(features)
    if f >= 3:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=df[features[0]],
                    y=df[features[1]],
                    z=df[features[2]],
                    visible=True,
                    showlegend=False,
                    mode='lines',
                )]
        )
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1, y=-1, z=1.25)
        )
        fig.update_layout(template="plotly_white",
                          scene=dict(
                              xaxis_title='x',
                              yaxis_title='y',
                              zaxis_title='z'),
                          title=title, scene_camera=camera
                          )
        fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
        fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

        # plotly(fig)
        fig.show()
        time.sleep(2)

        if file_path is not None and save is True:
            plotly_save(fig, file_path, size)


def plotly_acf_pacf(df_acf, df_pacf, save=False, legend=True, file_path=None, size=(1980, 1080),
                    label_scale=1, title_bool=True):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Bar(
            x=np.arange(len(df_acf)),
            y=df_acf,
            name='ACF',
            orientation="v",
            showlegend=legend,
        ),
        row=1,
        col=1
    )
    fig['layout']['yaxis' + str(1)]['title'] = "Autocorrelation"
    fig.add_trace(
        go.Bar(
            x=np.arange(len(df_pacf)),
            y=df_pacf,
            name='PACF',
            orientation="v",
            showlegend=legend,
        ),
        row=2,
        col=1
    )
    fig['layout']['xaxis' + str(2)]['title'] = 'Lags'
    fig['layout']['yaxis' + str(2)]['title'] = "Partial Autocorrelation"
    fig.update_layout(xaxis_rangeslider_visible=False, title="ACF and PACF" if title_bool else None,
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plot_history(history, title=None, save=False, legend=True, file_path=None, size=(1980, 1080),
                 markers='lines+markers', label_scale=1, plot_title=True):
    fig, ax = plt.subplots()
    x = list(range(len(history.history['loss'])))
    y = history.history['loss']
    plt.plot(x, y)
    plt.show()
    """
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.append_trace(
        go.Scatter(
            x=list(range(len(history.history['loss']))),
            y=history.history['loss'],
            orientation="v",
            visible=True,
            showlegend=legend,
            name='loss',
            mode=markers,
        ),
        row=1,
        col=1
    )
    fig['layout']['xaxis' + str(1)]['title'] = 'loss'
    fig['layout']['yaxis' + str(1)]['title'] = 'epoch'
    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig
    """


def plot_scores(scores, score_type, title=None, save=False, file_path=None, size=(1980, 1080),
                label_scale=1, plot_title=True):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.append_trace(
        go.Box(y=scores),
        row=1,
        col=1
    )
    fig['layout']['yaxis' + str(1)]['title'] = score_type
    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plot_multiple_scores(scores, score_type, names, title=None, save=False, file_path=None, size=(1980, 1080),
                         label_scale=1, plot_title=True):
    fig = make_subplots(rows=1, cols=len(scores), shared_xaxes=True)
    for i, s in enumerate(scores):
        fig.append_trace(
            go.Box(y=s,
                   name=names[i]),
            row=1,
            col=1
        )
    fig['layout']['yaxis' + str(1)]['title'] = score_type
    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plot_bar_summary(df, errors, title=None, save=False, file_path=None, size=(1980, 1080), shared_yaxes=False,
                     label_scale=1, plot_title=True, n_cols_adj_range=None, showlegend=False):
    if n_cols_adj_range is None:
        n_cols_adj_range = df.shape[1]
    bars = []
    fig = make_subplots(rows=1, cols=df.shape[1], shared_xaxes=True,
                        shared_yaxes=shared_yaxes, subplot_titles=df.columns)
    for i, col in enumerate(df.columns):
        bars.append(
            px.bar(df, x=df.index, y=col, color=df.index,
                   error_y=errors.iloc[:, i] if i < errors.shape[1] else None))

    for i, bar in enumerate(bars):
        for trace in bar.data_map:
            fig.add_trace(trace, 1, 1 + i)

    if not shared_yaxes:
        for i, col in enumerate(df.columns[:n_cols_adj_range]):
            p = max((max(df[col]) - min(df[col])) / 10, (max(errors.iloc[:, i]) if i < errors.shape[1] else 0))
            f = max(min(df[col]) - p * 1.1, 0)
            c = max(df[col]) + p * 1.1
            fig.update_yaxes(range=[f, c], row=1, col=1 + i)
    else:
        p = max((df.max().max() - df.min().min()) / 10, errors.max().max())
        f = df.min().min() - p * 1.1
        c = df.max().max() + p * 1.1
        fig.update_yaxes(range=[f, c])

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, showlegend=showlegend, barmode="stack")
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_annotations(font_size=14 * label_scale)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plot_bar_summary_2rows(df, errors, df2, errors2, title=None, save=False, file_path=None, size=(1980, 1080),
                           shared_yaxes=False, label_scale=1, plot_title=True, n_cols_adj_range=None, showlegend=False):
    if n_cols_adj_range is None:
        n_cols_adj_range = df.shape[1]

    fig = make_subplots(rows=2, cols=df.shape[1], shared_xaxes=True,
                        shared_yaxes=shared_yaxes, subplot_titles=list(df.columns) + list(df2.columns))

    bars = []
    for i, col in enumerate(df.columns):
        bars.append(
            px.bar(df, x=df.index, y=col, color=df.index,
                   error_y=errors.iloc[:, i] if i < errors.shape[1] else None))

    for i, bar in enumerate(bars):
        for trace in bar.data_map:
            fig.add_trace(trace, 1, 1 + i)

    bars = []
    for i, col in enumerate(df2.columns):
        bars.append(
            px.bar(df2, x=df2.index, y=col, color=df2.index,
                   error_y=errors2.iloc[:, i] if i < errors2.shape[1] else None))

    for i, bar in enumerate(bars):
        for trace in bar.data_map:
            fig.add_trace(trace, 2, 1 + i)

    if not shared_yaxes:
        for i, col in enumerate(df.columns[:n_cols_adj_range]):
            p = max((max(df[col]) - min(df[col])) / 10, (max(errors.iloc[:, i]) if i < errors.shape[1] else 0))
            f = max(min(df[col]) - p * 1.1, 0)
            c = max(df[col]) + p * 1.1
            fig.update_yaxes(range=[f, c], row=1, col=1 + i)

    else:
        p = max((df.max().max() - df.min().min()) / 10, errors.max().max())
        f = df.min().min() - p * 1.1
        c = df.max().max() + p * 1.1
        fig.update_yaxes(range=[f, c])

    p = max((df2.max().max() - df2.min().min()) / 10, errors.max().max())
    f = df2.min().min() - p * 1.1
    c = df2.max().max() + p * 1.1
    fig.update_yaxes(range=[f, c], row=2)

    # ytitles = ['minmax', 'n_params']
    # for i in range(2):
    #     fig['layout']['yaxis' + str(i * df.shape[1] + 1)]['title'] = ytitles[i]

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, showlegend=showlegend, barmode="stack")
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_annotations(font_size=14 * label_scale)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)
    return fig


def plot_dc_clusters(dc_df, labels, n_clusters, plot_title=True, title=None, save=False,
                     file_path=None, size=(1980, 1080), label_scale=1, markersize=5):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for c in range(n_clusters):
        fig.append_trace(
            go.Scatter(
                x=dc_df['t'].dropna()[labels == c],
                y=dc_df['tmv'].dropna()[labels == c],
                visible=True,
                showlegend=True,
                mode='markers',
                name='cluster:' + str(c),
                marker=dict(size=markersize),
            ),
            row=1,
            col=1
        )
    fig['layout']['xaxis1']['title'] = 't'
    fig['layout']['yaxis1']['title'] = 'tmv'

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)

    return fig


# def plot_gs_results(input_cfg, gs_cfg, model_cfg, in_cfg, data, errors):
#     file_name = '_'.join(gs_cfg.keys()) + '_' + get_suffix(input_cfg, in_cfg['steps'])
#     plot_bar_summary(data, errors, title="SERIES: " + str(input_cfg) + '<br>' + 'MODEL: ' + str(model_cfg),
#                      file_path=[in_cfg['image_folder'], file_name], plot_title=in_cfg['plot_title'],
#                      save=in_cfg['save_results'], n_cols_adj_range=1)


def drop_rows(df, rows):
    if rows is not None:
        for row in rows:
            if row in df.index:
                df.drop(row, axis=0, inplace=True)


# def plot_multiple_results(res_cfg, compare, var=None, size=(1980, 1080), label_scale=1, drop=None):
#     data, errors, input_cfg = load_data_err(res_cfg, compare)
#     drop_rows(data, drop)
#     drop_rows(errors, drop)
#     models_name = res_cfg['preffix']
#     image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(res_cfg)
#
#     if var is not None:
#         data2, errors2, input_cfg = load_data_err(res_cfg, compare, var)
#         data2 = data2.loc[data.index, :]
#         errors2 = errors2.loc[errors.index, :]
#         plot_bar_summary_2rows(data, errors, data2, errors2, title="SERIES: " + str(input_cfg), plot_title=plot_title,
#                                file_path=[image_folder, models_name], showlegend=False, shared_yaxes=True,
#                                save=save_results, n_cols_adj_range=data.shape[1], size=size, label_scale=label_scale)
#     else:
#         plot_bar_summary(data, errors, title="SERIES: " + str(input_cfg), plot_title=plot_title,
#                          file_path=[image_folder, models_name], showlegend=False, shared_yaxes=True,
#                          save=save_results, n_cols_adj_range=data.shape[1], size=size, label_scale=label_scale)
#     return data, errors


def plotly_ts_candles(df, instrument, title=None, save=False, legend=True, file_path=None, size=(1980, 1080),
                      color_col=None,
                      markers='lines+markers', xaxis_title="time", markersize=5, plot_title=True, label_scale=1,
                      template='plotly_white', adjust_height=(False, 0.6), **kwargs):
    params_ok, params = plotly_params_check(df, instrument, **kwargs)
    features, rows, cols, type_plot, alphas = params
    if not params_ok:
        return

    n_rows = len(set(rows)) + 1
    n_cols = 1  # len(set(cols))

    if instrument + 'c' in features: features.remove(instrument + 'o')
    if instrument + 'h' in features: features.remove(instrument + 'h')
    if instrument + 'l' in features: features.remove(instrument + 'l')
    if instrument + 'c' in features: features.remove(instrument + 'c')
    f = len(features)

    ts_height = adjust_height[1]
    hist_height = 1 - adjust_height[1]
    if adjust_height[0]:
        heights = [ts_height] + [hist_height / (f) for _ in range(f)]
    else:
        heights = [1 / (f + 1) for _ in range(f + 1)]

    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True, row_heights=heights)

    candls = instrument is not None
    if candls:
        fig.append_trace(
            go.Candlestick(
                x=df.index,
                open=df[instrument + 'o'],
                high=df[instrument + 'h'],
                low=df[instrument + 'l'],
                close=df[instrument + 'c'],
                visible=True,
                showlegend=False
            ),
            row=1,
            col=1
        )

    for i in range(f):
        df_ss = df[features[i]].dropna()
        fig.append_trace(
            go.Bar(
                x=df_ss.index,
                y=df_ss,
                orientation="v",
                visible=True,
                showlegend=legend,
                name=features[i],
                opacity=alphas[i]
            ) if type_plot[i] == 'bar' else
            go.Scatter(
                x=df_ss.index,
                y=df_ss,
                visible=True,
                showlegend=legend,
                name=features[i],
                mode=markers,
                marker=dict(size=markersize,
                            color=None if color_col is None else df[color_col].values,
                            colorscale="Bluered_r"),
                opacity=alphas[i]
            ),
            row=rows[i] + 2,
            col=cols[i] + 1
        )
        if i == n_rows - 1:
            fig['layout']['xaxis' + str(i + 1)]['title'] = xaxis_title

    fig.update_layout(template=template, xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)


def plotly_cand_vol(df_ss, instrument, vol_ss, volp_ss, plot_title=True, title=None, save=False,
                    file_path=None, size=(1980, 1080), label_scale=1, markersize=5):
    fig = go.Figure(
        data=[
            go.Bar(
                x=vol_ss,
                y=volp_ss,
                orientation="h",
                xaxis="x",
                yaxis="y",
                visible=True,
                showlegend=False
            ),
            go.Candlestick(
                x=df_ss.index,
                open=df_ss[instrument + 'o'],
                high=df_ss[instrument + 'h'],
                low=df_ss[instrument + 'l'],
                close=df_ss[instrument + 'c'],
                name='Price',
                xaxis="x2",
                yaxis="y2",
                visible=True,
                showlegend=False)
        ],
        layout=go.Layout(
            title=go.layout.Title(text="Candlestick With Volume Profile"),
            xaxis=go.layout.XAxis(
                side="top",
                rangeslider=go.layout.xaxis.Rangeslider(visible=False),
                showticklabels=False
            ),
            yaxis=go.layout.YAxis(
                side="right",
                showticklabels=False
            ),
            xaxis2=go.layout.XAxis(
                side="bottom",
                title="Date",
                rangeslider=go.layout.xaxis.Rangeslider(visible=False),
                overlaying="x"
            ),
            yaxis2=go.layout.YAxis(
                side="right",
                title="Price",
                overlaying="y"
            )
        )
    )
    fig['layout']['xaxis2']['showgrid'] = False
    fig['layout']['yaxis2']['showgrid'] = False
    template = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "none"]
    fig.update_layout(template=template[2])

    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)


def plotly_merge(df_original, df, inst, ix_range=50):
    ini_ix = df_original.index[-ix_range]
    end_ix = df.index[df_original.shape[0] + ix_range]
    df_ss = df.loc[ini_ix:end_ix, :]
    plotly_ts_candles(df_ss, instrument=inst, rows=[i for i in range(df_ss.shape[1] - 4)])


def plotly_histogram_regimes(df, color_col, n_components, title=None, save=False, file_path=None,
                             size=(1980, 1080), plot_title=True, label_scale=1, n_bins=200, **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, alphas = params
    if not params_ok:
        return
    n_rows = len(features)
    n_cols = 2
    masks = [df.loc[:, color_col] == i for i in range(n_components)]
    subtitles = [item for sublist in [[f] * 2 for f in features] for item in sublist]
    fig = make_subplots(rows=len(features), cols=n_cols, shared_xaxes=False, subplot_titles=subtitles)
    for r, var in enumerate(features):
        max_val = max(df.loc[:, var])
        min_val = min(df.loc[:, var])
        n_bins = n_bins
        bin_size = (max_val - min_val) / n_bins
        for i, mask in enumerate(masks):
            fig.append_trace(
                go.Histogram(x=df.loc[mask, var],
                             marker_color=colors[i],
                             xbins=dict(start=min_val, end=max_val, size=bin_size),
                             showlegend=False),
                row=r + 1,
                col=1
            )
            fig.append_trace(
                go.Box(
                    x=df.loc[masks[i], var],
                    marker_color=colors[i],
                    showlegend=False,
                    notched=True,
                    name=str(i)
                ),
                row=r + 1,
                col=2
            )

    for i in range(n_rows * n_cols):
        if i % n_cols == 0:
            fig['layout']['yaxis' + str(i + 1)]['title'] = 'count'
        else:
            fig['layout']['yaxis' + str(i + 1)]['title'] = color_col
    # barmode='overlay', opacity=0.75,

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, barmode='overlay',
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))
    fig.update_traces(opacity=0.75)
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)


def plotly_ts_regime_hist_vars(df, price_col, regime_col, n_bins=200, title=None, save=False, legend=False,
                               file_path=None, size=(1980, 1080), markers='lines+markers', markersize=5,
                               plot_title=True, save_png=False,
                               adjust_height=(False, 0.5), label_scale=1, **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, alphas = params
    n_rows = len(features) + 2
    n_cols = 2
    if not params_ok:
        return

    ts_cols = [price_col, regime_col]
    f = len(features)
    ts_height = adjust_height[1]
    hist_height = 1 - adjust_height[1]
    if adjust_height[0]:
        heights = [0.9 * ts_height, 0.1 * ts_height] + [hist_height / (f) for _ in range(f)]
    else:
        heights = [1 / (f + 2) for _ in range(f + 2)]
    subtitles = ['Regime Change', None] + [item for sublist in [[f] * 2 for f in features] for item in sublist]
    for i in range(len(subtitles)):
        if subtitles[i] is not None:
            subtitles[i] = '<b>' + subtitles[i] + '</b>'

    fig = make_subplots(rows=n_rows,
                        cols=n_cols,
                        specs=[[{"colspan": 2}, None], [{"colspan": 2}, None]] + [[{}, {}]] * f,
                        shared_xaxes=False,
                        row_heights=heights,
                        subplot_titles=subtitles)

    n_states = int(max(df[regime_col].dropna().values)) + 1

    for i in range(len(ts_cols)):
        df_ss = df[ts_cols[i]].dropna()
        fig.append_trace(
            go.Scatter(
                x=df_ss.index,
                y=df_ss,
                visible=True,
                showlegend=legend,
                name=ts_cols[i],
                mode=markers,
                line=dict(color='lightgray' if regime_col is not None else None),
                marker=dict(size=markersize,
                            color=None if regime_col is None else df[regime_col].dropna().values,
                            colorscale=None if regime_col is None else colors[:n_states]),
                opacity=alphas[i]
            ),
            row=i + 1,
            col=1
        )
        masks = [df.loc[:, regime_col] == i for i in range(n_states)]
        for r, var in enumerate(features):
            max_val = max(df.loc[:, var])
            min_val = min(df.loc[:, var])
            n_bins = n_bins
            bin_size = (max_val - min_val) / n_bins
            for i, mask in enumerate(masks):
                fig.append_trace(
                    go.Histogram(x=df.loc[mask, var],
                                 marker_color=colors[i],
                                 xbins=dict(start=min_val, end=max_val, size=bin_size),
                                 showlegend=False),
                    row=r + 3,
                    col=1
                )
                fig.append_trace(
                    go.Box(
                        x=df.loc[masks[i], var],
                        marker_color=colors[i],
                        showlegend=False,
                        notched=True,
                        name=str(i)
                    ),
                    row=r + 3,
                    col=2
                )

    for i, name in enumerate(ts_cols):
        fig['layout']['yaxis' + str(i + 1)]['title'] = name

    for i in range((n_rows - 2) * n_cols):
        if i % n_cols == 0:
            fig['layout']['yaxis' + str(i + 3)]['title'] = 'count'
        else:
            fig['layout']['yaxis' + str(i + 3)]['title'] = regime_col

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, barmode='overlay',
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))
    # fig.update_traces(opacity=0.9)
    fig.update_xaxes(matches='x')
    fig.update_xaxes(color='black', tickfont=dict(size=14 * label_scale),
                     title_font=dict(size=18 * label_scale),
                     showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(color='black', tickfont=dict(size=14 * label_scale),
                     title_font=dict(size=14 * label_scale),
                     showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_annotations(font_size=14 * label_scale)
    fig.update_annotations(font=dict(
        size=14 * label_scale,
        color="black"
    ))

    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)


def plotly_ts_regime2():
    return None


def plotly_time_series_bars_hist(df, color_col, bars_cols, features=None, title=None, save=False, legend=True,
                                 n_bins=200, file_path=None, size=(1980, 1080), markers='lines+markers', markersize=5,
                                 plot_title=True, label_scale=1):
    features = df.columns if features is None else features
    col_exist = [check_cols_exists(df, features), check_cols_exists(df, bars_cols), check_col_exists(df, color_col)]
    if not np.all(col_exist):
        return

    features.remove(color_col)
    [features.remove(bars_col) for bars_col in bars_cols]
    n_rows = len(bars_cols) + 1
    n_cols = 2
    subtitles = ['time series']
    subtitles += [item for sublist in [[f + ' histogram', f + ' box'] for f in bars_cols] for item in sublist]
    for i in range(len(subtitles)):
        if subtitles[i] is not None:
            subtitles[i] = '<b>' + subtitles[i] + '</b>'

    fig = make_subplots(rows=n_rows,
                        cols=n_cols,
                        specs=[[{"colspan": 2}, None]] + [[{}, {}]] * len(bars_cols),
                        shared_xaxes=False,
                        shared_yaxes=False,
                        subplot_titles=subtitles)
    colors_unique = df[color_col].unique()
    masks = [df.loc[:, color_col] == i for i in colors_unique]

    for i in range(len(features)):
        df_ss = df[features[i]].dropna()
        fig.append_trace(
            go.Scatter(
                x=df_ss.index,
                y=df_ss,
                visible=True,
                showlegend=legend,
                name=features[i],
                mode=markers,
                line=dict(color=gray_colors[i] if color_col is not None else None),
                marker=dict(size=markersize,
                            color=None if color_col is None else df[color_col].values,
                            colorscale=colors[:len(colors_unique)]),
            ),
            row=1,
            col=1
        )

    for r, var in enumerate(bars_cols):
        max_val = max(df.loc[:, var])
        min_val = min(df.loc[:, var])
        n_bins = n_bins
        bin_size = (max_val - min_val) / n_bins
        for i, mask in enumerate(masks):
            fig.append_trace(
                go.Histogram(x=df.loc[mask, var],
                             marker_color=colors[i],
                             xbins=dict(start=min_val, end=max_val, size=bin_size),
                             showlegend=False),
                row=r + 2,
                col=1
            )
            fig.append_trace(
                go.Box(
                    x=df.loc[masks[i], var],
                    marker_color=colors[i],
                    showlegend=False,
                    notched=True,
                    name=str(i)
                ),
                row=r + 2,
                col=2
            )

    fig['layout']['yaxis' + str(1)]['title'] = 'features'
    for i in range((n_rows - 1) * n_cols):
        if i % n_cols == 0:
            fig['layout']['yaxis' + str(i + 2)]['title'] = 'count'
        else:
            fig['layout']['yaxis' + str(i + 2)]['title'] = color_col

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, barmode='overlay',
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))
    fig.update_traces(opacity=0.9)
    # fig.update_xaxes(matches='x')
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=14 * label_scale))

    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)


def add_3d_scatter_trace(data, name=None, color_ix=0, markersize=5, marker_symbol='circle',
                         mode='markers', legend=True, opacity=1, line_width=None, transparent_fill=False):
    return go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        visible=True,
        showlegend=legend,
        name=name,
        mode=mode,
        opacity=opacity,
        marker_symbol=marker_symbol,
        marker=dict(size=markersize,
                    color='rgba(0, 0, 0, 0)' if transparent_fill else colors[color_ix % 10],
                    line=dict(
                        color=colors[color_ix % 10],
                        width=line_width
                    ) if line_width is not None else None)
    )


def plot_4D(F,
            color_col,
            ranked_F=None,
            original_point=None,
            selected_point=None,
            save=False,
            file_path=None,
            label_scale=1,
            size=(1980, 1080),
            save_png=False,
            title='',
            markersize=5,
            axis_labels=None,
            camera_position=None):
    color_scale = F[:, color_col]  # / max(F[:, color_col])
    data = np.delete(F, color_col, axis=1)
    # best_point, worst_point, extreme_points, intercepts = calc_geo_points(data)
    cmin, cmax = min(color_scale), max(color_scale)
    traces = []
    colorbar_label = axis_labels[color_col] if axis_labels is not None else \
        ['f1(x)', 'f2(x)', 'f3(x)', 'f4(x)'][color_col]

    if ranked_F is not None:
        for i, ranked_f in enumerate(ranked_F):
            color_scale = ranked_f[:, color_col]
            data = np.delete(ranked_f, color_col, axis=1)
            traces.append(
                go.Scatter3d(
                    x=data[:, 0],
                    y=data[:, 1],
                    z=data[:, 2],
                    visible=True,
                    showlegend=True if i == 0 else False,
                    name='Solutions',
                    mode='markers',
                    opacity=(len(ranked_F) - i) / len(ranked_F),
                    marker_symbol='circle',
                    marker=dict(size=markersize,
                                color=color_scale,
                                colorbar=dict(title=colorbar_label) if i == 0 else None,
                                showscale=True if i == 0 else False,
                                cmin=cmin,
                                cmax=cmax,
                                colorscale=[[0, "blue"],
                                            [1, 'green']],
                                )
                ))
    else:
        # fig = make_subplots(rows=1, cols=1)
        # Population
        traces.append(
            go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                visible=True,
                showlegend=True,
                name='Solutions',
                mode='markers',
                # opacity=opacity,
                marker_symbol='circle',
                marker=dict(size=markersize,
                            color=color_scale,
                            showscale=True
                            )
            )
        )

    if original_point is not None:
        data = np.delete(original_point, color_col, axis=0)
        traces.append(
            go.Scatter3d(
                x=[data[0]],
                y=[data[1]],
                z=[data[2]],
                visible=True,
                showlegend=True,
                name='Original solution',
                mode='markers',
                marker_symbol='cross',
                marker=dict(size=markersize * 2,
                            color='black'
                            )
            )
        )

    if selected_point is not None:
        data = np.delete(selected_point, color_col, axis=0)
        traces.append(
            go.Scatter3d(
                x=[data[0]],
                y=[data[1]],
                z=[data[2]],
                visible=True,
                showlegend=True,
                name='Selected solution',
                mode='markers',
                marker_symbol='cross',
                marker=dict(size=markersize * 2,
                            color='red'
                            )
            )
        )

    # Ideal and worst point
    # traces.append(
    #     add_3d_scatter_trace(best_point.reshape(1, -1), name='ideal_point', color_ix=2, markersize=10,
    #                          marker_symbol='cross'))
    # traces.append(
    #     add_3d_scatter_trace(worst_point.reshape(1, -1), name='ndir_point', color_ix=5, markersize=10,
    #                          marker_symbol='cross'))

    fig = go.Figure(data=traces)
    if axis_labels is not None:
        axis_labels = np.delete(axis_labels, color_col)
    else:
        axis_labels = ['f1(x)', 'f2(x)', 'f3(x)']

    fig.update_layout(scene=dict(
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1],
        zaxis_title=axis_labels[2]))

    if camera_position is not None:
        fig.update_layout(scene_camera=dict(eye=dict(x=camera_position[0],
                                                     y=camera_position[1],
                                                     z=camera_position[2])))

    fig.update_layout(legend_orientation="h")
    fig.update_layout(title=title,
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()

    if file_path is not None and save is True:
        time.sleep(1.5)
        plotly_save(fig, file_path, size, save_png)


def plot_forecast_intervals(forecasts_grouped,
                            n_output_steps,
                            id,
                            additional_vars=[],
                            additional_rows=[],
                            additional_data=None,
                            markersize=3,
                            fill_max_opacity=0.1,
                            label_scale=1,
                            title='',
                            mode='light',
                            save=False,
                            file_path=None,
                            size=(1980, 1080),
                            save_png=False,
                            x_range=None,
                            y_range=None,
                            ):
    steps = list(forecasts_grouped[list(forecasts_grouped.keys())[0]][id].columns)
    opacities = np.array([n_output_steps / (i + 1) for i in range(n_output_steps)]) / n_output_steps
    fill_opacities = (np.array([n_output_steps / (i + 1) for i in range(n_output_steps)])
                      / n_output_steps) * fill_max_opacity

    prob_forecasts = sorted(forecasts_grouped.keys())
    if 'p50' in prob_forecasts:
        prob_forecasts.remove('p50')
    if 'targets' in prob_forecasts:
        prob_forecasts.remove('targets')

    pairs = [(prob_forecasts[i], prob_forecasts[len(prob_forecasts) - 1 - i])
             for i in range(len(prob_forecasts) // 2)]

    # plot only one pair of bounds
    for p, pair in enumerate(pairs):
        lower_bound = forecasts_grouped[pair[0]][id]
        upper_bound = forecasts_grouped[pair[1]][id]

        if len(additional_rows) > 0 and max(additional_rows) > 0:
            fig = make_subplots(rows=max(additional_rows) + 1, cols=1, shared_xaxes=True)
        else:
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

        if len(additional_vars) > 0 and additional_data is not None:
            ix = forecasts_grouped['targets'][id].index
            add_data = additional_data.loc[ix, :]
            for i, var in enumerate(additional_vars):
                fig.add_trace(
                    go.Scatter(
                        x=add_data.index,
                        y=add_data[var],
                        visible=True,
                        showlegend=True,
                        name=var,
                        mode='lines+markers',
                        opacity=1,
                        line=dict(color=colors[4 + i], width=2),
                        marker=dict(size=markersize,
                                    color=colors[4 + i]),
                    ),
                    row=additional_rows[i] + 1,
                    col=1
                )

        for i, step in enumerate(steps[::-1]):
            if mode == 'light':
                fillcolor = 'rgba(0, 0, 0, {})'.format(fill_opacities[::-1][i])
            else:
                fillcolor = 'rgba(255, 255, 255, {})'.format(fill_opacities[::-1][i])

            for b, bound in enumerate([lower_bound, upper_bound]):
                fig.add_trace(
                    go.Scatter(
                        x=bound.index,
                        y=bound[step],
                        visible=True,
                        showlegend=False,
                        mode='lines',
                        fill='tonexty' if b == 1 else None,
                        fillcolor=fillcolor,
                        line=dict(color=fillcolor),
                        marker=dict(size=markersize, color=fillcolor),
                    ),
                    row=1,
                    col=1
                )
        if 'p50' in forecasts_grouped.keys():
            for i, step in enumerate(steps):
                fig.add_trace(
                    go.Scatter(
                        x=forecasts_grouped['p50'][id][step].index,
                        y=forecasts_grouped['p50'][id][step],
                        visible=True,
                        showlegend=True,
                        name='{} pred in t{}'.format(step[:-3], str(-(i + 1))),
                        mode='lines+markers',
                        opacity=opacities[i],
                        line=dict(color=colors[0], width=2),
                        marker=dict(size=markersize,
                                    color=colors[0]),
                    ),
                    row=1,
                    col=1
                )
        if 'targets' in forecasts_grouped.keys():
            fig.add_trace(
                go.Scatter(
                    x=forecasts_grouped['targets'][id][steps[0]].index,
                    y=forecasts_grouped['targets'][id][steps[0]],
                    visible=True,
                    showlegend=True,
                    name='target',
                    mode='lines+markers',
                    opacity=1,
                    line=dict(color=colors[3], width=3),
                    marker=dict(size=markersize,
                                color=colors[3]),
                ),
                row=1,
                col=1
            )

        fig.update_layout(template="plotly_white" if mode == 'light' else 'plotly_dark',
                          xaxis_rangeslider_visible=False, title=title,
                          legend=dict(font=dict(size=18 * label_scale)))

        if y_range is not None:
            fig.update_layout(yaxis=dict(range=y_range))

        if x_range is not None:
            fig.update_layout(xaxis=dict(range=x_range))

        fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
        fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
        fig.show()
        time.sleep(1.5)

        if file_path is not None and save is True:
            plotly_save(fig, file_path, size, save_png=save_png)


def append_scatter_trace(fig, df_ss, feature, opacity, markersize=5, color_ix=None, name=None):
    name = feature if name is None else name
    fig.append_trace(
        go.Scatter(
            x=df_ss.index,
            y=df_ss[feature],
            visible=True,
            showlegend=True,
            name=name,
            opacity=opacity,
            mode='lines+markers',
            line=dict(color=None if color_ix is None else colors[color_ix]),
            marker=dict(size=markersize,
                        color=None if color_ix is None else colors[color_ix]),
        ),
        row=1,
        col=1
    )


def get_plot_segments(plot_identifiers, forecasts_grouped):
    plot_segments = []
    for i, id in enumerate(plot_identifiers):
        # df = forecasts_grouped['targets'][id]
        dt_index = forecasts_grouped['targets'][id].index
        date_ix = (pd.Series(forecasts_grouped['targets'][id].index.dayofweek) > 4).astype(int)
        step_up = date_ix.diff(1).fillna(0).astype(int)
        step_down = date_ix.diff(-1).fillna(0).astype(int)
        # df['step_up'] = step_up.values
        # df['step_down'] = step_down.values
        # df['weekend'] = date_ix.values
        ini_ixs = np.where(step_up == np.max(step_up))[0]
        ini_ixs_2dary = np.where(step_up == np.min(step_up))[0]
        # plotly_time_series(df, features=['ESc_e11 t+1', 'step_up', 'step_down', 'weekend'], rows=[0, 1, 1, 2])

        if sum(step_down < 0) > 3 or sum(step_up > 0) > 3:
            x_range = None
        else:
            if len(ini_ixs) == 1 and len(ini_ixs_2dary) > 0 and ini_ixs_2dary[0] < ini_ixs[0]:
                x_range = [dt_index[ini_ixs_2dary[0]], dt_index[ini_ixs[0] - 1]]
            else:
                x_range = [dt_index[ini_ixs[0]], dt_index[ini_ixs[1] - 1]]

        plot_segments.append({'id': id,
                              'y_range': None,
                              'x_range': x_range})
    return plot_segments
