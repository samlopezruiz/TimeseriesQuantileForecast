import os
import shutil
import zipfile
from os import listdir

import joblib
import pandas as pd

from src.timeseries.utils.dataframe import renamer
from src.timeseries.utils.mega import Mega


def download_datasets(dataset_cfg, project):

    save_path = os.path.join('outputs', 'data', project, 'raw')
    os.makedirs(save_path, exist_ok=True)

    for key, files in dataset_cfg.items():
        for file_cfg in files:
            os.makedirs(os.path.join(save_path, key), exist_ok=True)
            file_path = os.path.join(save_path, key, file_cfg['file_name'])
            if os.path.exists(file_path):
                print('\nFile already downloaded: \n{}'.format(file_path))

            else:
                print('\nDownloading: {}'.format(file_cfg['file_name']))
                if 'mega' in file_cfg['url']:
                    mega = Mega()
                    mega.download_url(file_cfg['url'],
                                      dest_path=os.path.join(save_path, key),
                                      dest_filename=file_cfg['file_name'])
                else:
                    raise Exception('method not implemented to download from url: \n{}'.format(file_cfg['url']))

                if os.path.exists(file_path):
                    print('\nFile successfully saved in: \n{}'.format(file_path))

                    if 'zip' in file_path:
                        print('decompressing...')
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            tmb_folder = os.path.join(os.path.dirname(file_path), 'tmp')
                            zip_ref.extractall(tmb_folder)
                            files = os.listdir(tmb_folder)

                            if len(files) == 1:
                                _, file_extension = os.path.splitext(files[0])
                                file_name, _ = os.path.splitext(file_cfg['file_name'])
                                shutil.move(os.path.join(tmb_folder, files[0]),
                                            os.path.join(os.path.dirname(file_path), file_name + file_extension))
                            else:
                                for file in files:
                                    shutil.move(os.path.join(tmb_folder, file),
                                                os.path.join(os.path.dirname(file_path), file))

                        os.rmdir(tmb_folder)


# DATA_ROOT = 'D:\\MEGA\\Proyectos\\Trading\\Algo_Trading\\Historical_Data'
# PROJECT_ROOT = 'D:\\MEGA\\CienciasDeLaComputacion\\Tesis\\CodeProjectTimeSeries\\src\\timeseries'
#
#
def get_data_root(project):
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'outputs', 'data', project))


#
#
# def get_project_root():
#     return PROJECT_ROOT

#
# def load_files(data_cfg, subfolder, last_folder='src_folder', end=".z"):
#     path = get_model_market_path(data_cfg, subfolder=subfolder, last_folder=last_folder)
#     filename = data_cfg.get('filename', None)
#     if filename is not None:
#         filename = filename + end
#         print("\nLoading", filename)
#         if filename.endswith(".csv") or filename.endswith(".txt"):
#             data = read_csv(os.path.join(path, filename))
#         else:
#             data = joblib.load(os.path.join(path, filename))
#         return data
#     else:
#         raise Exception('filename not found in data_cfg')


# def get_model_market_path(data_cfg, subfolder='split', last_folder='src_folder'):
#     src_folder = data_cfg.get(last_folder, 'res')
#     return os.path.join(PROJECT_ROOT, 'experiments', 'market', subfolder, src_folder)
#
#
# def get_market_path(data_cfg, last_folder='src_folder'):
#     sampling, src_folder = data_cfg['sampling'], data_cfg[last_folder]
#     inst, market = data_cfg['inst'], data_cfg['market']
#     return os.path.join(DATA_ROOT, market, sampling, inst, src_folder)


# def list_files(data_cfg, suffix=".txt", last_folder='src_folder', include_substring=''):
#     path = get_market_path(data_cfg, last_folder=last_folder)
#     files = find_filenames(path, suffix=suffix, include_substring=include_substring)
#     return files


def find_filenames(path_to_dir, suffix=".txt", include_substring=''):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix) and include_substring in filename]


def load_dataset(data_cfg, project, src_folder='raw'):
    file_path = os.path.join(get_data_root(project),
                             src_folder,
                             data_cfg['subfolder'],
                             data_cfg['filename'])

    dataset = load_file(file_path)
    return dataset


def load_multiple_markets(data_cfgs, project, resampling='D', ffill=True):
    data = [load_dataset(cfg, project) for cfg in data_cfgs]
    data_resampled = [df.resample(resampling).last() for df in data]
    df = pd.concat(data_resampled, axis=1)
    df.rename(columns=renamer(), inplace=True)
    if ffill:
        df.ffill(axis=0, inplace=True)
        df.dropna(inplace=True)
    return df



# def save_df(df, data_cfg, timestamp=True, last_folder='src_folder', end='.csv', suffix=''):
#     filename = data_cfg.get('filename', None)
#     if filename is None:
#         inst = data_cfg['inst']
#         if timestamp:
#             ini_date = str(df.index[0].year) + '_' + str(df.index[0].month)
#             end_date = str(df.index[-1].year) + '_' + str(df.index[-1].month)
#         sufx = ('_' + suffix) if len(suffix) > 1 else ''
#         filename = inst + "_" + ini_date + "-" + end_date + sufx + end
#     path = get_market_path(data_cfg, last_folder=last_folder)
#     df.to_csv(os.path.join(path, filename), index=True)
#     print("File {} saved".format(filename))


def describe(df):
    print('Initial Date : ' + str(df.index[0]))
    print('Final Date   : ' + str(df.index[-1]))
    print('Dataset Shape: ' + str(df.shape))


def read_csv(file_path, datetime_col='datetime'):
    return pd.read_csv(file_path, header=0, infer_datetime_format=True,
                       parse_dates=[datetime_col], index_col=[datetime_col],
                       converters={'vol': eval, 'volp': eval})


def load_file(file_path):
    path = os.path.normpath(file_path)
    if os.path.exists(path):
        print("\nLoading file: \n{}".format(file_path))
        if file_path.endswith(".csv") or file_path.endswith(".txt"):
            data = read_csv(path)
        else:
            data = joblib.load(path)
        return data
    else:
        raise Exception("file {} doesn't exist".format(path))


def get_inst_ohlc_names(inst):
    if inst.endswith('_r'):
        return [inst[:-3] + 'o' + inst[-3 + 1:], inst[:-3] + 'h' + inst[-3 + 1:],
                inst[:-3] + 'l' + inst[-3 + 1:], inst[:-3] + 'c' + inst[-3 + 1:]]
    else:
        return [inst + 'o', inst + 'h', inst + 'l', inst + 'c']