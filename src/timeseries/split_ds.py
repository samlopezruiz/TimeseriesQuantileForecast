import os

import numpy as np
import pandas as pd

from src.timeseries.plot.ts import plotly_ts_candles, plotly_ts_regime
from src.timeseries.utils.config import read_config
from src.timeseries.utils.dataframe import resample_dfs
from src.timeseries.utils.dataset import load_dataset, get_data_root, load_file
from src.timeseries.utils.filename import split_filename, update_trim_range
from src.timeseries.utils.preprocessing import downsample_df, append_timediff_subsets, add_date_known_inputs
from src.timeseries.utils.save import save_subsets_and_test
from src.timeseries.utils.split import time_subset, set_subsets_and_test

np.random.seed(42)

if __name__ == '__main__':
    # %%
    in_cfg = {'save_results': True,
              'save_plot': True,
              'verbose': 1,
              'plot_title': False}

    project = 'snp'
    cfg = read_config('split_dataset_60T_8w', project)

    df = load_dataset(cfg['data_cfg'], project)
    df = time_subset(df, cfg['data_cfg'])

    if cfg['data_cfg']['append_datasets'] is not None and len(cfg['data_cfg']['append_datasets']) > 0:
        for ds in cfg['data_cfg']['append_datasets']:
            file_path = os.path.join(get_data_root(project), *ds['path'], ds['filename'])
            if os.path.exists(file_path):
                df_add = load_file(file_path)
                df_resampled = resample_dfs(df, df_add)
                df = pd.concat([df, df_resampled], axis=1)
            else:
                print('File {} not found'.format(file_path))


    if cfg['data_cfg']['downsample']:
        df = downsample_df(df,
                           cfg['data_cfg']['downsample_p'],
                           ohlc_features=False,
                           inst=cfg['data_cfg']['inst'])

    # Plot sample
    plotly_ts_candles(df.iloc[-5000:, :],
                      features=df.columns[4:8],
                      instrument='ES',
                      adjust_height=(True, 0.6),
                      template='plotly_dark',
                      rows=[i for i in range(len(df.columns[4:8]))])

    # %%
    df_subsets = set_subsets_and_test(df, cfg['split_cfg'])
    append_timediff_subsets(df_subsets, cfg['split_cfg']['time_thold'])

    # Add known inputs
    add_date_known_inputs(df_subsets)

    #%%
    update_trim_range(df_subsets, cfg['data_cfg'])
    img_path = os.path.join(get_data_root(project), 'split', 'img', split_filename(cfg))

    plotly_ts_regime(df_subsets.iloc[-40000:-10000, :],
                     features=['ESc', 'test_train_subset', 'week_of_year'],
                     rows=[0, 1, 2],
                     resample=False,
                     regime_col='test',
                     period='90T',
                     markers='markers',
                     markersize=5,
                     plot_title=in_cfg['plot_title'],
                     template='plotly_white',
                     save=in_cfg['save_plot'],
                     file_path=img_path,
                     save_png=True,
                     legend=True,
                     label_scale=1,
                     title='SPLIT CFG: {}'.format(str(cfg['split_cfg'])),
                     legend_labels=['train', 'test', 'val'])

    df_subsets['symbol'] = cfg['data_cfg']['inst']
    result = {
        'data': df_subsets,
        'split_cfg': cfg['split_cfg'],
        'data_cfg': cfg['data_cfg']
    }
    if in_cfg['save_results']:
        save_subsets_and_test(result, project, cfg)

