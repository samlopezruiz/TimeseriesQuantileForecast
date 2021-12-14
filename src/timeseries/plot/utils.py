import numpy as np
import pandas as pd
import os
from datetime import date
import datetime

from src.timeseries.utils.files import create_dir, get_new_file_path

template = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "none"]

def group_forecasts(forecasts, n_output_steps, target_col):
    if target_col:
        features = ['{} t+{}'.format(target_col, i + 1) for i in range(n_output_steps)]
    else:
        features = ['t+{}'.format(i + 1) for i in range(n_output_steps)]

    forecasts_grouped = {}
    for key, df in forecasts.items():

        identifiers_forecasts = {}
        for id, df_grouped in df.groupby('identifier'):
            df_grouped = df_grouped.set_index('forecast_time', inplace=False)
            shifted = {}
            for i, feature in enumerate(features):
                shifted[feature] = df_grouped[feature].shift(i)
            identifiers_forecasts[id] = pd.DataFrame(shifted)

        forecasts_grouped[key] = identifiers_forecasts

    return forecasts_grouped

def plotly_params_check(df, instrument=None, **kwargs):
    if not isinstance(df, pd.DataFrame):
        print("ERROR: First parameter is not pd.DataFrame")
        return False, None

    params_ok = True
    features = kwargs.get('features') if kwargs.get('features', None) is not None else df.columns
    f = len(features)
    rows = kwargs.get('rows') if kwargs.get('rows', None) is not None else [0 for _ in range(f)]
    cols = kwargs.get('cols') if kwargs.get('cols', None) is not None else [0 for _ in range(f)]
    alphas = kwargs.get('alphas') if kwargs.get('alphas', None) is not None else [1 for _ in range(f)]
    type_plot = kwargs.get('type_plot') if kwargs.get('type_plot', None) is not None else ["line" for _ in range(f)]

    for feature in features:
        if feature not in df.columns:
            print("ERROR: feature ", feature, "not found")
            params_ok = False

    if len(rows) != f:
        print("ERROR: len(rows) != features")
        params_ok = False

    if len(cols) != f:
        print("ERROR: len(cols) != features")
        params_ok = False

    if len(type_plot) != f:
        print("ERROR: len(type_plot) != features")
        params_ok = False

    return params_ok, (list(features), rows, cols, type_plot, alphas)


def set_y_labels(f, features, fig):
    for i in range(f):
        fig['layout']['yaxis' + str(i + 1)]['title'] = features[i]


def plotly_save(fig, file_path, size, save_png=False, use_date_suffix=False):
    print("Saving image:")
    create_dir(file_path)
    image_path = get_new_file_path(file_path, '.png', use_date_suffix)
    html_path = get_new_file_path(file_path, '.html', use_date_suffix)
    if size is None:
        size = (1980, 1080)

    if save_png:
        print(image_path)
        fig.write_image(image_path, width=size[0], height=size[1], engine='orca')

    print(html_path)
    fig.write_html(html_path)


def find_extreme_points(fitnesses, best_point):
    'Finds the individuals with extreme values for each objective function.'

    # Translate objectives
    ft = fitnesses - best_point

    # Find achievement scalarizing function (asf)
    asf = np.eye(best_point.shape[0])
    asf[asf == 0] = 1e6
    asf = np.max(ft * asf[:, np.newaxis, :], axis=2)

    # Extreme point are the fitnesses with minimal asf
    min_asf_idx = np.argmin(asf, axis=1)
    return fitnesses[min_asf_idx, :]


def find_intercepts(extreme_points, best_point, current_worst, front_worst):
    """Find intercepts between the hyperplane and each axis with
    the ideal point as origin."""
    # Construct hyperplane sum(f_i^n) = 1
    b = np.ones(extreme_points.shape[1])
    A = extreme_points - best_point
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        intercepts = current_worst
    else:
        intercepts = 1 / x

        if (not np.allclose(np.dot(A, x), b) or
                np.any(intercepts <= 1e-6) or
                np.any((intercepts + best_point) > current_worst)):
            intercepts = front_worst

    return intercepts


def calc_geo_points(pop):
    best_point = np.min(pop, axis=0)
    worst_point = np.max(pop, axis=0)

    extreme_points = find_extreme_points(pop, best_point)
    intercepts = find_intercepts(extreme_points, best_point, worst_point, worst_point)

    return best_point, worst_point, extreme_points, intercepts

