import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.timeseries.plot.vol import plotly_years_vol_profile, plotly_vol_profile, plotly_overlap, plot_min_max_vp
from src.timeseries.utils.dataset import load_dataset
from src.timeseries.utils.volprofile_process import get_full_vol_profile, get_max_min, vol_vp_filter, price_vp_filtering

if __name__ == '__main__':
    # %% CONSTANTS
    project = 'snp'
    data_cfg = {'inst': 'ES',
                'subfolder': "vol",
                'filename': 'ES_vol_2021-2021_6.csv'}

    df = load_dataset(data_cfg, project)


    # %% PLOT YEARLY VOLUME PROFILES
    years = list(range(2013, 2023))
    plotly_years_vol_profile(df, data_cfg['inst'], years)


    # %% PLOT VOLUME PROFILES
    date_input = '2014'
    last_vp = get_full_vol_profile(df, date_input)
    plotly_vol_profile(last_vp)

    # %% LOG AND SMOOTH
    vp_log = np.log(last_vp)
    vp_log_hat = savgol_filter(vp_log, 51, 4)  # window size 51, polynomial order 3
    vp_log_hat = pd.Series(data=vp_log_hat, name=vp_log.name, index=vp_log.index)

    plotly_overlap([vp_log, vp_log_hat])
    plotly_vol_profile(vp_log_hat)

    # %% GET MAX AND MIN
    vp_log_hat_min_max = get_max_min(vp_log_hat)
    plot_min_max_vp(vp_log_hat_min_max, vp_log_hat)

    # %% VOL FILTERING
    vp_log_hat_min_max_filtered = vol_vp_filter(vp_log_hat_min_max, thold=.05)
    plot_min_max_vp(vp_log_hat_min_max_filtered, vp_log_hat)

    # %% PRICE FILTERING
    vp_log_hat_min_max_filtered = price_vp_filtering(vp_log_hat_min_max, thold=3)
    plot_min_max_vp(vp_log_hat_min_max_filtered, vp_log_hat)
