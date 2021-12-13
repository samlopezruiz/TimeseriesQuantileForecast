import os
import time

import pandas as pd

from src.timeseries.plot.vol import plotly_cols_in_df
from src.timeseries.utils.config import read_config
from src.timeseries.utils.dataset import load_dataset, get_data_root
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.volprofile_process import get_vol_profiles_history_levels, create_vp_levels_indicators

if __name__ == '__main__':
    project = 'snp'
    vp_cfg = read_config('volume_profile', project)

    market = load_dataset(vp_cfg['mkt_ds_cfg'], project)
    vol = load_dataset(vp_cfg['vol_ds_cfg'], project)

    price_history = market[vp_cfg['indicator_cfg']['price']]

    print('Filtering min max points in volume profile...')
    t0 = time.time()
    vol_profile_levels, vol_profile_is_max = get_vol_profiles_history_levels(vol,
                                                                             normalize_profile=True,
                                                                             filter_thold=3,
                                                                             windows_size=51,
                                                                             poly_order=4)

    print('Volume level processing time: {}s'.format(round(time.time() - t0, 2)))

    distance_df, inverse_distance_df, norm_vol_df, is_max_df = create_vp_levels_indicators(price_history,
                                                                                           vol_profile_levels,
                                                                                           vol_profile_is_max,
                                                                                           vp_cfg['indicator_cfg'])

    # %% Save as compressed file
    volume_profile_levels_complete = pd.concat([inverse_distance_df,
                                                norm_vol_df,
                                                is_max_df], axis=1)

    save_vars(volume_profile_levels_complete,
              file_path=[os.path.join(get_data_root(project), 'vol_profile'),
                         'Vol_{}levels_{}_{}'.format(vp_cfg['indicator_cfg']['n_levels'],
                                                     vp_cfg['indicator_cfg']['price'],
                                                     vp_cfg['vol_ds_cfg']['filename'].split('.')[0])])

    # %% PLOT
    n_plot = 3000
    test_ix = 200000

    plot_df = inverse_distance_df.iloc[test_ix - n_plot:test_ix, :]

    plotly_cols_in_df(plot_df,
                      float_y_axis=False,
                      swap_xy=True,
                      modes=['markers' for _ in range(len(plot_df.columns))],
                      markersize=3)

    plot_df = distance_df.iloc[test_ix - n_plot:test_ix, :]

    plotly_cols_in_df(plot_df,
                      float_y_axis=False,
                      swap_xy=True,
                      modes=['markers' for _ in range(len(plot_df.columns))],
                      markersize=3)
