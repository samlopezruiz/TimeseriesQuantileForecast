import os
from os.path import isfile

import numpy as np
import pandas as pd

from src.timeseries.utils.files import create_dir, get_new_file_path


def files_with_substring(file_path, substring):
    path = os.path.join(*file_path)
    files = [f for f in os.listdir(path) if (isfile(os.path.join(path, f)) and substring in f)]
    return files


def latex_table(title, tabbular_text):
    table_str = '\\begin{table}[h] \n\\begin{center}\n'
    table_str += '\\caption{{{0}}}\\label{{tbl:{1}}}\n'.format(title.replace('_', ' '),
                                                               title.lower().replace(' ', '_'))
    table_str += tabbular_text
    table_str += '\\end{center} \n\\end{table}\n'
    return table_str


def write_text_file(file_path, text, extension='.txt', use_date=False):
    create_dir(file_path)
    path = get_new_file_path(file_path, extension, use_date_suffix=use_date)
    print('Saving text file to:\n{}\n'.format(path))
    with open(os.path.join(path), "w") as text_file:
        text_file.write(text)


def write_latex_from_scores(scores, out_file_path):
    output_text = ''
    for key in scores:
        if isinstance(scores[key], pd.DataFrame):
            table_text = latex_table(key, scores[key].to_latex())
            output_text += table_text + '\n\n'

    write_text_file(out_file_path, output_text)


def get_type_str(obj):
    return str(type(obj)).split("'")[1].split('.')[-1]


def array_from_lists(lists):
    max_shape = max([len(a) for a in lists])
    arr = np.zeros((len(lists), max_shape))
    arr.fill(np.nan)
    for i, a in enumerate(lists):
        arr[i, :len(a)] = a
    return arr


def mean_std_text_df(df, label, round=3):
    mean = df.mean(axis=0).round(round).astype(str)
    std = df.std(axis=0).round(round).astype(str)
    text = (mean + ' (' + std + ')').to_frame(name=label).T
    return text


def mean_std_from_array(arr, labels):
    df = pd.DataFrame()
    df['mean'] = np.mean(arr, axis=1)
    df['std'] = np.std(arr, axis=1)
    df.index = labels
    return df
