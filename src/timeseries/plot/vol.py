import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import time

from src.timeseries.plot.utils import plotly_save
from src.timeseries.utils.volprofile_process import max_min_mask, get_full_vol_profile

pio.renderers.default = "browser"


def plotly_overlap(dfs, modes=None, fills=None):
    date_title = dfs[0].name.strftime("%m/%d/%Y")
    fig = make_subplots(rows=1, cols=1)
    if modes is None:
        modes = ['lines' for _ in range(len(dfs))]

    if fills is None:
        fills = [None for _ in range(len(dfs))]

    ymax = max(dfs[0].index)
    ymin = min(dfs[0].index)
    for i, last_vp in enumerate(dfs):
        vol = np.array(last_vp)[::-1]
        volp = np.array(last_vp.index)[::-1]
        ymax = max(ymax, max(volp))
        ymin = min(ymin, min(volp))
        fig.append_trace(
            go.Scatter(
                x=vol,
                y=volp,
                orientation="v",
                visible=True,
                showlegend=False,
                opacity=0.9,
                mode=modes[i],
                fill=fills[i],
            ),
            row=1,
            col=1
        )

    fig['layout']['yaxis']['title'] = "Price"
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                      title="Volume Profile " + date_title,
                      yaxis_range=[ymin, ymax])
    fig.show()
    time.sleep(1)


def plot_min_max_vp(df, df2):
    max_mask, min_mask = max_min_mask(df)
    df_max = df[max_mask].copy()
    df_min = df[min_mask].copy()

    plotly_overlap([df2, df_max, df_min],
                   modes=['lines', 'markers', 'markers'],
                   fills=['tozeroy', None, None])


def plotly_vol_profile_levels(df, file_path=None, save_html=False, label_scale=1, size=(1980, 1080),
                              save_png=False, title='', ):
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        hoverongaps=False,
        colorscale=[[0, "green"],
                    [0.5, 'yellow'],
                    [1, "blue"]],

        hovertemplate='<b>Date: %{x}</b><br>Price: %{y:.2f} <br>Norm Vol: %{z:.2f} <extra></extra>',
    ),
    )
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, )
    fig.update_layout(title=title, legend_itemsizing='constant',
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()

    time.sleep(1.5)
    if file_path is not None and save_html is True:
        plotly_save(fig, file_path, size, save_png)


def plotly_cols_in_df(df,
                      modes=None,
                      fills=None,
                      file_path=None,
                      save_html=False,
                      label_scale=1,
                      size=(1980, 1080),
                      save_png=False,
                      title='',
                      float_y_axis=True,
                      swap_xy=False,
                      markersize=5):

    if modes is None:
        modes = ['markers' for _ in range(len(df.columns if swap_xy else df.index))]

    if fills is None:
        fills = [None for _ in range(len(df.columns if swap_xy else df.index))]

    legends = df.columns if swap_xy else df.index.to_numpy()
    print(legends)
    fig = make_subplots(rows=1, cols=1)

    for i, feature in enumerate(df.columns):
        serie = df[feature]
        fig.append_trace(
            go.Scatter(
                x=serie.values if not swap_xy else \
                    serie.index.to_numpy().astype(float) if float_y_axis else serie.index.to_numpy(),
                y=serie.index.to_numpy().astype(float) if float_y_axis else \
                    serie.index.to_numpy() if not swap_xy else serie.values,
                orientation="v",
                visible=True,
                showlegend=True,
                opacity=0.9,
                mode=modes[i],
                fill=fills[i],
                name=legends[i],
                marker=dict(size=markersize)
            ),
            row=1,
            col=1
        )

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, )
    fig.update_layout(title=title, legend_itemsizing='constant',
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()

    time.sleep(1.5)
    if file_path is not None and save_html is True:
        plotly_save(fig, file_path, size, save_png)



def plotly_years_vol_profile(df, inst, years):
    fig = make_subplots(rows=1, cols=len(years), shared_yaxes=True,
                        subplot_titles=years)

    for i, year in enumerate(years):
        last_vp = get_full_vol_profile(df, str(year))
        vol = np.array(last_vp) #[::-1]
        volp = np.array(last_vp.index).astype(float) #[::-1]
        fig.append_trace(
            go.Scatter(
                x=vol,
                y=volp,
                orientation="h",
                visible=True,
                showlegend=False,
                fill='tozeroy',
            ),
            row=1,
            col=i + 1
        )

    fig['layout']['yaxis']['title'] = "Price"
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                      title=inst + " Volume Profile", yaxis_range=[min(volp), max(volp)])
    fig.show()
    time.sleep(.5)


def plotly_vol_profile(last_vp):
    date_title = last_vp.name.strftime("%m/%d/%Y")
    vol = np.array(last_vp) #[::-1]
    volp = np.array(last_vp.index).astype(float) #[::-1]
    fig = go.Figure(
        data=[go.Scatter(
            x=vol,
            y=volp,
            orientation='h',
            visible=True,
            showlegend=False,
            fill='tozeroy',
        )]
    )

    fig['layout']['yaxis']['title'] = "Price"
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                      title="Volume Profile " + date_title,
                      yaxis_range=[min(volp), max(volp)])
    fig.show()
    time.sleep(.5)