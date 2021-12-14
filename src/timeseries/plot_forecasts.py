import os

import joblib
import numpy as np

from src.timeseries.data_formatter.configs import ExperimentConfig
from src.timeseries.plot.ts import plot_forecast_intervals
from src.timeseries.plot.utils import group_forecasts
from src.timeseries.utils.filename import get_result_folder

if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': True,
                   'plot_title': False}

    project = 'snp'
    forecast_cfg = {'experiment_name': '60t_ema_q159',
                    'forecast': 'TFTModel_ES_ema_r_q159_lr01_pred',
                    'subfolders': []  # ['moo', 'selec_sols']
                    }

    additional_vars = ['ESc']

    base_path = os.path.join(get_result_folder(forecast_cfg, project),
                             *forecast_cfg['subfolders'],
                             forecast_cfg['forecast'])
    suffix = ''

    results = joblib.load(base_path + suffix + '.z')

    if len(additional_vars) > 0:
        config = ExperimentConfig(project, results['experiment_cfg'])
        formatter = config.make_data_formatter()
        mkt_data, _ = formatter.load_data(config.data_config)

    forecasts = results['reconstructed_forecasts'] if 'reconstructed_forecasts' in results else results['forecasts']

    identifiers = forecasts['targets']['identifier'].unique()
    target_col = results.get('target', 'ESc')

    n_output_steps = results['model_params']['total_time_steps'] - results['model_params']['num_encoder_steps']
    forecasts_grouped = group_forecasts(forecasts, n_output_steps, target_col)

    if results['target']:
        steps = ['{} t+{}'.format(target_col, i + 1) for i in range(n_output_steps)]
    else:
        steps = ['t+{}'.format(i + 1) for i in range(n_output_steps)]

    img_path = os.path.join(get_result_folder(forecast_cfg, project),
                            'img',
                            *forecast_cfg['subfolders'])

    filename = forecast_cfg['forecast']

    # %%
    # for 5t
    # plot_identifiers = identifiers[:5]
    # plot_segments = get_plot_segments(plot_identifiers, forecasts_grouped)

    # for 60T
    # plot segments can be specified with x and y ranges
    # plot_segments.append({'id': 2,
    #                       'y_range': [1960, 2040],
    #                       'x_range': ['2015-03-15T19:00', '2015-03-20T16:00']})
    plot_segments = []
    plot_segments.append({'id': 2, 'y_range': None, 'x_range': None})
    plot_segments.append({'id': 11, 'y_range': None, 'x_range': None})
    plot_segments.append({'id': 31, 'y_range': None, 'x_range': None})

    # %%

    for plot_segment in plot_segments:
        id = plot_segment['id']
        title = 'Filename: {} <br>Model: {}, Vars Definition: {},' \
                '<br>Dataset: {}, <br>Quantiles: {}, Group Id: {}'.format(forecast_cfg['forecast'],
                                                                          results['experiment_cfg']['architecture'],
                                                                          results['experiment_cfg']['vars_definition_cfg'],
                                                                          results['experiment_cfg']['preprocess_cfg'],
                                                                          results['quantiles'],
                                                                          id)
        if 'objective_space' in results:
            obj = np.round(results['objective_space'], 3)
            title += '<br>Objective space: {}'.format(obj)

        plot_forecast_intervals(forecasts_grouped, n_output_steps, id,
                                markersize=3, mode='light',
                                fill_max_opacity=0.2,
                                additional_vars=['ESc'],
                                additional_rows=[0],
                                additional_data=mkt_data,
                                title=title if general_cfg['plot_title'] else None,
                                save=general_cfg['save_plot'],
                                file_path=os.path.join(img_path, filename + '_id{}'.format(id)),
                                y_range=plot_segment['y_range'],
                                x_range=plot_segment['x_range'],
                                save_png=True,
                                label_scale=1.2,
                                size=(1980, 1080 * 2 // 3),
                                )
