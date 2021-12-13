import os

from src.timeseries.utils.dataset import get_data_root
from src.timeseries.utils.filename import hmm_filename, subsets_and_test_filename, subset_filename
from src.timeseries.utils.files import save_vars


def save_hmm(var, project, hmm_cfg, folder='regime', use_date_suffix=False):
    file_path = os.path.join(get_data_root(project), folder, hmm_filename(hmm_cfg))
    save_vars(var, file_path, use_date_suffix=use_date_suffix)


def save_subsets_and_test(var, project, cfg, folder='split', use_date_suffix=False):
    file_path = os.path.join(get_data_root(project), folder, subsets_and_test_filename(cfg))
    save_vars(var, file_path, use_date_suffix=use_date_suffix)


def save_market_data(var, project, data_cfg, folder='mkt', use_date_suffix=False):
    file_path = os.path.join(get_data_root(project), folder, subset_filename(data_cfg))
    save_vars(var, file_path, use_date_suffix=use_date_suffix)


