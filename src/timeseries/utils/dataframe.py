import numpy as np
import pandas as pd


def append_to_df(df, new_x_values, col_name):
    # must have same indices
    df[col_name] = new_x_values
    df[col_name].fillna(method='ffill', inplace=True)
    # mask = ~df[col_name].isna()
    # return df.iloc[mask.values, :].copy()


def trim_min_len(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


class renamer():
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])


def relabel(labels, map):
    if map is not None:
        for i in range(len(labels)):
            labels[i] = map[int(labels[i])]


def relabel_col(df, col, map):
    labels = df[col].to_numpy()
    relabel(labels, map=map)
    df[col] = labels


def check_col_exists(df, col):
    if col not in list(df.columns):
        print('{} column not found'.format(col))
        return False
    return True


def check_cols_exists(df, cols):
    return np.all([check_col_exists(df, col) for col in cols])


def resample_dfs(df_keep_index, df_external):
    resampled_df = pd.concat([df_keep_index.iloc[:, 0], df_external], axis=1)
    return resampled_df.loc[:, df_external.columns].ffill().loc[df_keep_index.index, :].fillna(0)

