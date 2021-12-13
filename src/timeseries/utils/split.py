import numpy as np
import pandas as pd

from src.timeseries.utils.dataset import describe, get_inst_ohlc_names


def s_threshold(time_diff_cfg):
    hour_s = 60 * 60
    day_s = hour_s * 24
    hour_thold = 0 if time_diff_cfg.get('hours', 0) is None else time_diff_cfg.get('hours', 0)
    day_thold = 0 if time_diff_cfg.get('days', 0) is None else time_diff_cfg.get('days', 0)
    second_thold = 0 if time_diff_cfg.get('seconds', 0) is None else time_diff_cfg.get('seconds', 0)
    s_thold = hour_thold * hour_s + day_thold * day_s + second_thold
    return s_thold

def set_subsets_and_test(df, split_cfg):
    df_subsets = group_by(df, split_cfg)
    df_subsets = group_by(df_subsets, split_cfg, new_col_name='test_train_subset')
    if split_cfg['random'] and split_cfg['groups_of'] == 1 and split_cfg['group'] == 'day':
        print('random test split')
        random_test_split(df_subsets, split_cfg)
    else:
        print('simple test split')
        simple_test_split(df_subsets, test_ratio=split_cfg['test_ratio'], valid_ratio=split_cfg['valid_ratio'])

    if split_cfg['time_delta_split']:
        s_thold = s_threshold(split_cfg['time_thold'])
        update_subset_time_delta(df_subsets, time_thold=s_thold)
    df_subsets.drop(['year', 'week', 'day', 'hour'], axis=1, inplace=True)
    return df_subsets


def shift_subsets(df):
    # mask_ini = (df_subsets['test'].shift(periods=-1, fill_value=0) - df_subsets['test']).eq(1)
    # ix_ini = df_subsets.loc[mask_ini, :].index
    mask_end = (df['test'].shift(periods=1, fill_value=0) - df['test']).eq(1)
    ix_end = df.loc[mask_end, :].index
    df['subset'] = 0
    for i in range(len(ix_end) - 1):
        df.loc[ix_end[i]:ix_end[i + 1], 'subset'] = i + 1
    df.loc[ix_end[i + 1]:, 'subset'] = i + 2


def random_test_split(df, split_cfg):
    test_time_ini, test_time_end = split_cfg['test_time_start'], split_cfg['test_time_end']
    test_ratio = split_cfg['test_ratio']
    valid_ratio = split_cfg['valid_ratio']
    df['test'] = 0
    grp = df.groupby('subset')
    for i, (group_cols, subset) in enumerate(grp):
        t = subset.index[0]
        t_ini = t.replace(hour=test_time_ini[0], minute=test_time_ini[1], second=0)
        t_end = t.replace(hour=test_time_end[0], minute=test_time_end[1], second=0)
        test_len = int(round(subset.shape[0] * test_ratio, 0))
        low = subset.index.searchsorted(t_ini)
        high = subset.index.searchsorted(t_end)
        if high - test_len > low:
            ix_start = np.random.randint(low, high - test_len)
            df.loc[subset.index[ix_start:ix_start + test_len], 'test'] = 1

    shift_subsets(df)


def time_subset(df, cfg, describe_=True):
    df_ss = df.loc[cfg.get('trim_data_from', None):cfg.get('trim_data_to', None)].copy()
    if describe_:
        describe(df_ss)
    return df_ss


def group_by(df_orig, cfg, new_col_name='subset'):
    group, groups_of = cfg['group'], cfg['groups_of']
    df = df_orig.copy()
    df['year'] = df.index.isocalendar().year
    df['week'] = df.index.isocalendar().week
    df['day'] = df.index.isocalendar().day
    df['hour'] = df.index.hour
    df[new_col_name] = 0

    group_periods = ['year', 'week', 'day', 'hour']
    if group == 'hour':
        grp = df.groupby(group_periods)
    elif group == 'day':
        grp = df.groupby(group_periods[:-1])
    elif group == 'week':
        grp = df.groupby(group_periods[:-2])
    elif group == 'year':
        grp = df.groupby(group_periods[:-3])
    else:
        raise Exception('Group has to be one of: [year, week, day, hour]')

    # dates, dfs, = [], []
    # for group_cols, data in grp:
    #     dates.append(group_cols)
    #     dfs.append(data)

    # dfs_groups = []
    # new_df = pd.DataFrame()
    # group_df = pd.DataFrame()
    g = 0
    for i, (group_cols, data) in enumerate(grp):
        if i % groups_of == 0:
            g += 1
        df.loc[data.index, new_col_name] = g
    #     group_df = pd.concat([group_df, df], axis=1)
    #
    #     if i % groups_of == 0:
    #         dfs_groups.append(new_df)
    #         new_df = pd.DataFrame()
    #     new_df = pd.concat([new_df, df], axis=0)
    # dfs_groups.append(new_df)
    # del dfs_groups[0]

    return df  # dfs_groups


def simple_test_split(df, valid_ratio=0.15, test_ratio=0.15):
    df['test'] = 0
    grp = df.groupby('subset')
    for i, (group_cols, data) in enumerate(grp):
        test_len = int(data.shape[0] * test_ratio)
        valid_len = int(data.shape[0] * valid_ratio)
        df.loc[data.index[-test_len:], 'test'] = 1
        df.loc[data.index[-(test_len+valid_len):-test_len], 'test'] = 2
    #
    # dfs_train, dfs_test = [], []
    # for data in dfs_groups:
    #     test_len = int(round(data.shape[0] * test_ratio, 0))
    #     dfs_train.append(data.iloc[:-test_len, :])
    #     dfs_test.append(data.iloc[-test_len:, :])
    # return dfs_train, dfs_test


def merge_train_test_groups(dfs_train, dfs_test):
    df_merged = pd.DataFrame()
    for i, (train, test) in enumerate(zip(dfs_train, dfs_test)):
        train_df, test_df = train.copy(), test.copy()
        train_df['test'], test_df['test'] = 0, 1
        train_df['group'], test_df['group'] = i, i
        df_merged = pd.concat([df_merged, train_df, test_df], axis=0)
    return df_merged


def update_subset_time_delta(df, time_thold=1000, subset_col='subset'):
    if subset_col in df.columns:
        df['diff_s'] = (pd.Series(df.index).shift(periods=1, fill_value=np.nan) - pd.Series(
            df.index)).dt.total_seconds().values
        time_mask = df['diff_s'] < -time_thold
        subset_mask = (df[subset_col].shift(periods=1, fill_value=0) - df[subset_col]).eq(-1)
        mask = pd.concat([time_mask, subset_mask], axis=1)
        mask['step'] = mask['diff_s'] | mask[subset_col]
        ix_end = df.loc[mask['step'], :].index

        df[subset_col] = 0
        i = 0
        for i in range(len(ix_end) - 1):
            df.loc[ix_end[i]:ix_end[i + 1], subset_col] = i + 1
        df.loc[ix_end[i + 1]:, subset_col] = i + 2
        df.drop('diff_s', axis=1, inplace=True)
    else:
        print('subset column: {} not found'.format(subset_col))


def get_subsets(df_pp, n_states=None, features=None):
    if 'subset' not in df_pp.columns or 'test' not in df_pp.columns:
        print('subset or test column not found')
        return None
    else:
        grp = df_pp.groupby(['subset', 'test'])
        subsets = []
        train_lens, test_lens = [], []
        for i, ((n_ss, test), subset) in enumerate(grp):
            if test == 0:
                train_lens.append(subset.shape[0])
            else:
                test_lens.append(subset.shape[0])
            drop_cols = ['subset', 'test'] + ([] if n_states is None else ['p_'+str(i) for i in range(n_states)])
            data = subset.drop(drop_cols, axis=1)
            if features is not None:
                if 'time_subset' in df_pp.columns and 'time_subset' not in features:
                    features.append('time_subset')
                data = data.loc[:, features]
            prob = subset.loc[:, ['p_'+str(i) for i in range(n_states)]] if n_states is not None else None
            subsets.append({'subset': n_ss, 'test': test, 'data': data, 'prob': prob})

        print('Total subsets: train={}, test={:.0f}'.format(len(train_lens), len(test_lens)))
        print('Average lengths: train={:.0f}, test={:.0f}'.format(np.mean(train_lens),
                                                                  np.mean(test_lens)))
        return subsets


def get_xy_from_subsets(subsets, min_dim, look_back=0):
    train_X, test_X = [], []
    split_timediff = []
    for i, subset in enumerate(subsets):
        # if len(test_X) > 46:
        #     a = 1
        # if i == 247:
        #     a = 1
        test, df_ss, df_p = subset['test'], subset['data'], subset['prob']
        # delete time_subset column
        if 'time_subset' in df_ss.columns:
            if i + 1 < len(subsets):
                t_subset = subsets[i + 1]['data']['time_subset'][0]
            else:
                t_subset = subsets[i]['data']['time_subset'][0]
            # print(np.mean(df_ss['time_subset']), df_ss['time_subset'][0])
            split_timediff.append(abs(t_subset - np.mean(df_ss['time_subset'])) < 0.0001)
            df_ss.drop('time_subset', inplace=True, axis=1)
        else:
            split_timediff.append(True)

        if df_ss.shape[0] > min_dim and test == 0:
            train_X.append((df_ss, df_p if df_p is not None else None))
        elif test == 1:
            # append look_back from last test subset
            if look_back > 0 and i > 0:
                subs_1 = subsets[i - 1]
                test_1, df_ss_1, df_p_1 = subs_1['test'], subs_1['data'], subs_1['prob']
                # check if there is no time-diff subset change in previous subset
                if test_1 == 0 and split_timediff[-2]:
                    df_ss = pd.concat([df_ss_1.iloc[-look_back:, :], df_ss], axis=0)
                    if df_p is not None:
                        df_p = pd.concat([df_p_1.iloc[-look_back:, :], df_p], axis=0)
            if df_ss.shape[0] > min_dim:
                test_X.append((df_ss, df_p if df_p is not None else None))

    # to check continuous time comment line:
    # df_ss.drop('time_subset', inplace=True, axis=1)
    # import numpy as np
    # for x, p in test_X:
    #     print(np.mean(x[:, -1]))
    return train_X, test_X


def get_xy(subsets, training_cfg, lookback=0, dim_f=1):
    '''
    :param subsets:
    :param training_cfg: specify y_var and features
    :param lookback: if 0, no data is appended to test data
    :param dim_f: multiplier for lookback
    :return: list of train and test subsets
    '''

    features = get_train_features(training_cfg)
    append_train_to_test = training_cfg.get('append_train_to_test', False)
    # n_seq, n_steps_in, n_steps_out = model_cfg.get('n_seq', 1), model_cfg['n_steps_in'], model_cfg['n_steps_out']
    dim_limit = int(lookback * dim_f) #model_func['lookback'](model_cfg) * dim_f

    look_back = dim_limit if append_train_to_test else 0
    train_X, test_X = get_xy_from_subsets(subsets, dim_limit, look_back)
    return train_X, test_X, features


def get_train_features(training_cfg):
    inst, y_var = training_cfg.get('inst', None), training_cfg.get('y_train_var', None)
    if training_cfg.get('include_ohlc', False) and inst is not None:
        if y_var is not None:
            return get_inst_ohlc_names(inst) + training_cfg['features'] + [y_var]
        else:
            return get_inst_ohlc_names(inst) + training_cfg['features']
    else:
        if y_var is not None:
            return training_cfg['features'] + [y_var]
        else:
            return training_cfg['features']
    return features

def append_subset_cols(df, orig_df, timediff=False):
    if 'subset' in orig_df.columns and 'test' in orig_df.columns:
        df['subset'] = orig_df['subset']
        df['test'] = orig_df['test']
    if timediff and 'time_subset' in orig_df.columns:
        df['time_subset'] = orig_df['time_subset']
