import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def get_diff_mask(diff_df, thold=0.05):
    diff_mask = diff_df > thold
    diff_mask[0] = True

    in_mask = False
    for i, diff in enumerate(diff_df):
        if diff < thold and not in_mask:
            in_mask = True
            c = 2
            if i > 0:
                diff_mask[i - 1] = False
        elif diff >= thold and in_mask:
            in_mask = False
            if c % 2 == 1:
                diff_mask[i - 1 - c // 2] = True
        elif in_mask:
            c += 1
    return diff_mask


def max_min_mask(ser, verbose=True):
    if len(ser) > 2:
        pos_switch = []
        for i in range(1, len(ser)):
            if ser[i] > ser[i - 1]:
                pos_switch.append(True)
            else:
                pos_switch.append(False)
        pos_switch.append(not pos_switch[-1])

    else:
        pos_switch = np.ones((len(ser),)).astype(bool)

    min_mask = pd.Series(pos_switch, name=ser.name, index=ser.index)

    if verbose:
        if sum(min_mask.astype(int).diff().fillna(0).ne(0).astype(int)) > 1:
            print("Max-Min not alternating")

    return ~min_mask, min_mask


def get_max_min(df):
    vp_diff1 = df.diff().fillna(0)

    diff_mask = np.sign(vp_diff1).diff().fillna(0).ne(0)
    return df[diff_mask].copy()


def vol_vp_filter(df, thold=.05):
    v_diff = np.abs(df.diff().fillna(0))
    v_diff_mask = get_diff_mask(v_diff, thold=thold)
    return df[v_diff_mask].copy()


def price_vp_filtering(df, thold=3):
    ix_diff = pd.Series(df.index.astype(float)).diff().fillna(0)
    ix_diff.index = df.index
    ix_diff_mask = get_diff_mask(ix_diff, thold=thold)
    return df[ix_diff_mask].copy()


def min_max_from_vp(volume_profile,
                    filter_thold=3,
                    windows_size=51,
                    poly_order=4):
    # Smooth
    vp_log_hat = savgol_filter(volume_profile, windows_size, poly_order)  # window size 51, polynomial order 3
    vp_log_hat = pd.Series(data=vp_log_hat, name=volume_profile.name, index=volume_profile.index)

    # Get minimums and maximums
    vp_log_hat_min_max = get_max_min(vp_log_hat)

    # Filter out near min/max points
    vp_log_hat_min_max_filtered = price_vp_filtering(vp_log_hat_min_max, thold=filter_thold)

    return vp_log_hat_min_max_filtered


def get_full_vol_profile(df, date_input):
    ix = df.index.searchsorted(date_input) - 1
    df_vp = df.iloc[ix, :].dropna()
    return df_vp.sort_index()


def get_levels_and_vol_profiles(date_input,
                                volume_profile,
                                vol_profile_levels,
                                vol_profile_is_max):
    last_vp = get_full_vol_profile(volume_profile, date_input)
    max_min_points = vol_profile_levels.loc[:, last_vp.name]
    max_points = vol_profile_is_max.loc[:, last_vp.name]
    vp_log = np.log(last_vp)
    vp_log_norm = vp_log / vp_log.max()

    df_merge = pd.concat([vp_log_norm,
                          max_min_points[max_points == True],
                          max_min_points[max_points == False]], axis=1)
    labels = ['vol profile norm', 'max', 'min']
    df_merge.columns = ['{} {}'.format(col, labels[i]) for i, col in enumerate(df_merge.columns)]

    return df_merge


def get_vol_profiles_history_levels(df,
                                    normalize_profile,
                                    filter_thold=3,
                                    windows_size=51,
                                    poly_order=4):
    vp_min_max, min_max = [], []
    for ix, vol_profile in df.iterrows():

        vp = vol_profile.dropna().sort_index()
        vp_log = np.log(vp)

        # round to closest odd number that is smaller than the series length
        win_size = int(np.floor(min(len(vp_log), windows_size)) // 2 * 2 - 1)

        vp_log_hat_min_max_filtered = min_max_from_vp(vp_log,
                                                      filter_thold=filter_thold,
                                                      windows_size=win_size,
                                                      poly_order=poly_order)
        if normalize_profile:
            vp_log_hat_min_max_filtered = vp_log_hat_min_max_filtered / vp_log_hat_min_max_filtered.max()

        max_mask, _ = max_min_mask(vp_log_hat_min_max_filtered, verbose=False)
        vp_min_max.append(vp_log_hat_min_max_filtered)
        min_max.append(max_mask)

    vol_profile_levels = pd.concat(vp_min_max, axis=1)
    vol_profile_levels.sort_index(inplace=True)
    vol_profile_is_max = pd.concat(min_max, axis=1)
    vol_profile_is_max.sort_index(inplace=True)

    return vol_profile_levels, vol_profile_is_max


def create_vp_levels_indicators(price_history,
                                vol_profile_levels,
                                vol_profile_is_max,
                                indicator_cfg):
    # trim price history until there is volume data
    if price_history.index[0] < vol_profile_levels.columns[0]:
        ix = price_history.index.searchsorted(vol_profile_levels.columns[0]) + 1
        print('Trimming {} timesteps from price history'.format(ix))
        price_history = price_history[ix:]

    distance = np.ones((len(price_history), indicator_cfg['n_levels'] * 2)) * np.nan
    norm_vol = np.ones((len(price_history), indicator_cfg['n_levels'] * 2)) * np.nan
    is_max = np.ones((len(price_history), indicator_cfg['n_levels'] * 2)) * np.nan

    n_levels = indicator_cfg['n_levels']
    vol_profile_col_ix = 0
    current_vol_profile = vol_profile_levels.iloc[:, vol_profile_col_ix].dropna()
    current_is_max = vol_profile_is_max.iloc[:, vol_profile_col_ix].dropna()
    price_index = current_vol_profile.index.to_numpy().astype(float)

    for i, (timestamp, price) in enumerate(price_history.iteritems()):

        if timestamp > vol_profile_levels.columns[vol_profile_col_ix + 1]:
            # necessary change of profile
            vol_profile_col_ix += 1
            current_vol_profile = vol_profile_levels.iloc[:, vol_profile_col_ix].dropna()
            current_is_max = vol_profile_is_max.iloc[:, vol_profile_col_ix].dropna()
            vol_profile_current_timestamp = current_vol_profile.name
            price_index = current_vol_profile.index.to_numpy().astype(float)

        # price_index[i-1] < price <= price_index[i]
        ix_price = np.searchsorted(price_index, price, side='left')
        if i % 10000 == 0:
            print('\rIndicator creation progress: {} %'.format(round(100 * (i / len(price_history)), 1)), end='')

        # iterate higher levels
        for n in range(n_levels):
            if ix_price + n < len(current_vol_profile):
                distance[i, n_levels + n] = price_index[ix_price + n] - price
                norm_vol[i, n_levels + n] = current_vol_profile[ix_price + n]
                is_max[i, n_levels + n] = current_is_max[ix_price + n]

        # iterate lower levels
        for n in range(-1, -n_levels - 1, -1):
            if ix_price + n >= 0:
                distance[i, n_levels + n] = price_index[ix_price + n] - price
                norm_vol[i, n_levels + n] = current_vol_profile[ix_price + n]
                is_max[i, n_levels + n] = current_is_max[ix_price + n]

    labels_levels = np.arange(-n_levels, n_levels) + (np.arange(-n_levels, n_levels) >= 0).astype(int)
    distance_df = pd.DataFrame(distance,
                               columns=['p_dist_{}'.format(i) for i in labels_levels],
                               index=price_history.index)
    norm_vol_df = pd.DataFrame(norm_vol,
                               columns=['norm_vol_{}'.format(i) for i in labels_levels],
                               index=price_history.index)
    is_max_df = pd.DataFrame(is_max,
                             columns=['is_max_{}'.format(i) for i in labels_levels],
                             index=price_history.index)

    return prep_vp_level_indicator(distance_df, norm_vol_df, is_max_df)


def prep_vp_level_indicator(distance_df, norm_vol_df, is_max_df):
    # compute inverse distance in order to set nans to zero
    max_abs_dist = distance_df.abs().max().max()

    inverse_distance_df = distance_df.copy()
    inverse_distance_df[distance_df >= 0] = max_abs_dist - distance_df[distance_df >= 0]
    inverse_distance_df[distance_df < 0] = -max_abs_dist - distance_df[distance_df < 0]

    # fill nans
    inverse_distance_df.fillna(0, inplace=True)
    norm_vol_df.fillna(0, inplace=True)
    is_max_df.fillna(-1, inplace=True)

    return distance_df, inverse_distance_df, norm_vol_df, is_max_df
