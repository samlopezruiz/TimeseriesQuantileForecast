import datetime
import os

from src.timeseries.utils.dataset import get_data_root
from src.timeseries.utils.filename import hmm_filename, split_filename, subset_filename
from src.timeseries.utils.files import save_vars


def save_hmm(var, project, hmm_cfg, folder='regime', use_date_suffix=False):
    file_path = os.path.join(get_data_root(project), folder, hmm_filename(hmm_cfg))
    save_vars(var, file_path, use_date_suffix=use_date_suffix)


def save_subsets_and_test(var, project, cfg, folder='split', use_date_suffix=False):
    file_path = os.path.join(get_data_root(project), folder, split_filename(cfg))
    save_vars(var, file_path, use_date_suffix=use_date_suffix)


def save_market_data(var, project, data_cfg, folder='mkt', use_date_suffix=False):
    file_path = os.path.join(get_data_root(project), folder, subset_filename(data_cfg))
    save_vars(var, file_path, use_date_suffix=use_date_suffix)


def save_forecasts(config, experiment_name, results):
    save_folder = os.path.join(config.results_folder, experiment_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")[2:]
    results['targets'].to_csv(os.path.join(save_folder, 'targets' + time_stamp + '.csv'))
    results['p10_forecast'].to_csv(os.path.join(save_folder, 'p10_forecast' + time_stamp + '.csv'))
    results['p50_forecast'].to_csv(os.path.join(save_folder, 'p50_forecast' + time_stamp + '.csv'))
    results['p90_forecast'].to_csv(os.path.join(save_folder, 'p90_forecast' + time_stamp + '.csv'))


