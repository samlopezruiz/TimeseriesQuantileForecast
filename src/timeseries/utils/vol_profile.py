import os
import pandas as pd
import numpy as np
import joblib
from copy import copy


def merge_vol(data_cfg, vp_cumm):
    if vp_cumm is not None:
        vol_cfg = copy(data_cfg)
        vol_cfg['suffix'] = vol_cfg['vol_suffix']
        vol_profile, _ = load_market(vol_cfg, end=".csv", last_folder='vol_folder')
        vol_profile.columns = vol_profile.columns.astype(float)

        dfs = []
        for ix, vp in vp_cumm:
            dfs.append(pd.DataFrame(vp, index=[ix]).astype(int))

        vp_new_df = pd.concat(dfs)
        complete_vp = pd.concat([vol_profile, vp_new_df], axis=0)
        complete_vp.index.names = ['datetime']
        save_df(complete_vp, data_cfg, last_folder='vol_folder', suffix='vol')
        return complete_vp
    else:
        return None


def calc_cumm_vp(data_cfg):
    new_vol = load_new_vol(data_cfg)
    if new_vol is not None:
        current_vp = load_current_vol(data_cfg)
        dt = list(new_vol.index)
        session_ends = get_sessions_ix(dt)
        vp_cumm = get_cumm_vp(new_vol, session_ends, current_vp, dt)
        save_vp_cumm(vp_cumm, data_cfg)
        return vp_cumm
    else:
        return None


def load_new_vol(data_cfg):
    z_files = list_files(data_cfg, suffix=".z", include_substring='vol')
    if len(z_files) > 0:
        path_new_vol = get_market_path(data_cfg, last_folder='src_folder')
        print("Loading New Vol File: {}".format(z_files[0]))
        new_vol = joblib.load(os.path.join(path_new_vol, z_files[0]))
        return new_vol
    else:
        print('No new volume file found')
        return None


def load_current_vol(data_cfg):
    z_files = sorted(list_files(data_cfg, suffix=".z", last_folder='split_folder', include_substring='vol'))
    path_current_vol = get_market_path(data_cfg, last_folder='split_folder')
    print("Loading Current Vol File: {}".format(z_files[-1]))
    current_cumm_vp = joblib.load(os.path.join(path_current_vol, z_files[-1]))
    timestamp, volume_profile = current_cumm_vp[-1]
    return volume_profile


def get_sessions_ix(dt, rth_start=8.5, rth_end=15.25):
    session_ends = []
    for i in range(1, len(dt)):
        t1 = dt[i - 1].hour + dt[i - 1].minute / 60
        t0 = dt[i].hour + dt[i].minute / 60
        if t1 < rth_start <= t0:
            session_ends.append(i - 1)
        elif t1 < rth_end <= t0:
            session_ends.append(i - 1)
    return session_ends


def get_cumm_vp(new_vol_df, session_ends, volume_profile, dt):
    vp_cumm = []
    for sess in session_ends:
        vol = new_vol_df.iloc[sess, 0]
        volp = new_vol_df.iloc[sess, 1]
        for v, price in enumerate(volp):
            if price not in volume_profile:
                volume_profile[price] = vol[v]
            else:
                volume_profile[price] += vol[v]
        vp_cumm.append((dt[sess], copy(volume_profile)))
    return vp_cumm


def split_data_vol(df, vol_cols=None):
    if vol_cols is None:
        vol_cols = ['vol', 'volp']

    if vol_cols[0] in df.columns:
        vol1 = df[vol_cols]
        df.drop(vol_cols, axis=1, inplace=True)
        return df, vol1
    else:
        df.drop(vol_cols, axis=1, inplace=True)
        return df, None


def save_vp_cumm(vp_cumm, data_cfg, end='.z'):
    ini_date = str(vp_cumm[0][0].year) + '_' + str(vp_cumm[0][0].month)
    end_date = str(vp_cumm[-1][0].year) + '_' + str(vp_cumm[-1][0].month)
    filename = data_cfg['inst'] + "_" + ini_date + "-" + end_date + '_vol' + end
    path_save_vol = get_market_path(data_cfg, last_folder='split_folder')
    joblib.dump(vp_cumm, os.path.join(path_save_vol, filename))
    print("File {} saved".format(filename))





