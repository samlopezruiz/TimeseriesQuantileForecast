import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.timeseries.utils.dataset import get_inst_ohlc_names
from src.timeseries.utils.indicators import ln_returns, ema, ismv, rsi, macd
from src.timeseries.utils.split import s_threshold, get_train_features


def preprocess_x(x, detrend=None, scale=True, standard_scaler=None, ema_period=0):
    if len(detrend) == 2:
        detrend, period = detrend  # ('ema_diff', 14)
    elif detrend == 'ema_diff':
        print('ERROR: specify period if detrend is emma_diff')
        return

    if detrend == 'ln_return':
        if ema_period > 0:
            x = ln_returns(ema(x, ema_period))
        else:
            x = ln_returns(x)
    if detrend == 'ema_diff':
        x = np.array(x) - ema(x, period)
    if detrend == 'diff':
        x = np.diff(np.array(x), axis=0)
    if scale:
        if standard_scaler is None:
            standard_scaler = StandardScaler()
            x = standard_scaler.fit_transform(x)
        else:
            x = standard_scaler.transform(x)

    return x, standard_scaler


def reconstruct_x(forecast, y_col, steps=1, test=None, standscaler=None, a_1=None, detrend='ln_return'):
    if a_1 is None and test is None:
        raise Exception('test or a_1 have to be specified')
    elif test is not None:
        a_1 = test[0]
        test_ = test[1:]
        forecast = forecast[1:]
    if len(detrend) == 2:
        detrend, period = detrend  # ('ema_diff', 14)
    elif detrend == 'ema_diff':
        print('ERROR: specify period if detrend is emma_diff')
        return

    all_var_unscaled = inverse_scaler(forecast, standscaler)
    if ismv(all_var_unscaled):
        forecast_unscaled = all_var_unscaled[:, y_col]
    else:
        forecast_unscaled = all_var_unscaled

    if detrend == 'ln_return':
        pred = reconstruct_from_ln_r(forecast_unscaled, a_1, steps, test_)
    elif detrend == 'ema_diff':
        pred = reconstruct_from_ema_diff(forecast_unscaled, a_1, steps, test_, period)
    elif detrend == 'diff':
        pred = reconstruct_from_diff(forecast_unscaled, a_1, steps, test_)
    else:
        pred = forecast_unscaled
    return np.array(pred).astype(float).reshape(-1, )


def series_from_ln_r(p_1, ln_r):
    p = []
    if isinstance(p_1, np.ndarray):
        p_1 = p_1[0]
    for r in ln_r:
        p_1 = np.exp(np.log(p_1) + r)
        p.append(p_1)
    return np.array(p)


def series_from_diff(p_1, diff):
    p = []
    for d in diff:
        p_1 = p_1 + d
        p.append(p_1)
    return np.array(p)


def series_from_ema_diff(ema_1, ema_diff, period):
    p = []
    c1 = 2 / (1 + period)
    c2 = 1 - (2 / (1 + period))
    ema_x = np.zeros((len(ema_diff) + 1,))
    ema_x[0] = ema_1
    for i, e_diff in enumerate(ema_diff):
        ema_x[i + 1] = (ema_x[i] + e_diff) * c1 + c2 * ema_x[i]
        p.append(e_diff + ema_x[i + 1])

    return np.array(p)


def reconstruct_from_diff(diff, p_1, steps, test):
    if test is not None:
        pred = []
        actual = np.hstack((p_1, test))  # actual data starts with p(t-1)
        for i in range(0, len(test), steps):
            end = min(i + steps, len(diff))
            # returns have a 1 delay, therefore P(t) = exp(r(t)+P(t-1))
            y_pred = series_from_diff(actual[i], diff[i:end])
            [pred.append(y) for y in y_pred]
    else:
        pred = series_from_diff(p_1, diff)
        pred = np.hstack((p_1, pred))
    return pred


def reconstruct_from_ema_diff(ema_diff, ema_1, steps, test, period):
    if test is not None:
        pred = []
        ema_ref = ema(test, period, last_ema=ema_1)

        for i in range(0, len(test), steps):
            end = min(i + steps, len(ema_diff))
            y_pred = series_from_ema_diff(ema_ref[i], ema_diff[i:end], period)
            [pred.append(y) for y in y_pred]
    else:
        pred = series_from_ema_diff(ema_1, ema_diff, period)
    return pred


def reconstruct_from_ln_r(ln_r, p_1, steps, test):
    if test is not None:
        test = test.reshape(-1, 1)
        pred = [p_1]
        actual = np.vstack((p_1, test))  # actual data starts with p(t-1)
        for i in range(0, len(test), steps):
            end = min(i + steps, len(ln_r))
            # returns have a 1 delay, therefore P(t) = exp(r(t)+P(t-1))
            y_pred = series_from_ln_r(actual[i], ln_r[i:end])
            [pred.append(y) for y in y_pred]
    else:
        pred = series_from_ln_r(p_1, ln_r)
        pred = np.hstack((p_1, pred))
    return pred


def inverse_scaler(ln_r, standscaler):
    if standscaler is not None:
        s = ln_r.shape[1] if len(ln_r.shape) > 1 else 1
        if standscaler.n_features_in_ > s:
            n = standscaler.n_features_in_ - s + 1
            ln_r = standscaler.inverse_transform(np.array([ln_r] * n).T)
        else:
            ln_r = standscaler.inverse_transform(ln_r)
    return ln_r


def preprocess(input_cfg, train, test, reg_prob_train=None, reg_prob_test=None, ss=None):
    if input_cfg.get('preprocess', False):
        detrend = input_cfg.get('detrend', 'ln_return')
        if len(train.shape) == 1:
            train = train.reshape(-1, 1)
            test = test.reshape(-1, 1)
        ema_period = input_cfg.get('ema_period', 0)
        train_pp, ss = preprocess_x(train, detrend=detrend, ema_period=ema_period, standard_scaler=ss)
        test_pp, _ = preprocess_x(test, detrend=detrend, ema_period=ema_period, standard_scaler=ss)

        reg_prob_train = reg_prob_train[1:] if reg_prob_train is not None else None
        reg_prob_test = reg_prob_test[1:] if reg_prob_test is not None else None
        return train_pp, test_pp, reg_prob_train, reg_prob_test, ss
    else:
        return train, test, reg_prob_train, reg_prob_test, ss


def reconstruct_pred(scaled_pred_y, model_n_steps_out, unscaled_y=None, ss=None,
                     preprocess=True):
    if preprocess:
        scaled_pred_y = prep_forecast(scaled_pred_y)
        unscaled_y = prep_forecast(unscaled_y)
        assert len(unscaled_y) == len(scaled_pred_y)

        y_col = ss.n_features_in_ - 1 if ss is not None else None
        return reconstruct_x(scaled_pred_y, y_col, steps=model_n_steps_out,
                             detrend='ln_return', standscaler=ss, test=unscaled_y)
    else:
        return scaled_pred_y


def prep_forecast(forecast):
    forecast = np.array(forecast)
    if len(forecast.shape) == 2:
        if forecast.shape[1] == 1:
            # case of an array of arrays
            forecast = forecast.ravel()
    return forecast


def downsample_df(df, resample_period, ohlc_features=False, inst=None):
    df_new = pd.DataFrame()
    df_resampled = df.resample(resample_period)

    if inst is not None:
        inst_cols = get_inst_ohlc_names(inst)
        if np.all([col in list(df.columns) for col in inst_cols]):
            df_new[inst + 'c'] = df_resampled[inst + 'c'].last()
            df_new[inst + 'o'] = df_resampled[inst + 'o'].first()
            df_new[inst + 'l'] = df_resampled[inst + 'l'].min()
            df_new[inst + 'h'] = df_resampled[inst + 'h'].max()
            other_features = [col for col in df.columns if col not in inst_cols]
        else:
            other_features = list(df.columns)
    else:
        other_features = list(df.columns)

    for feature in other_features:
        if ohlc_features:
            df_new[feature + 'c'] = df_resampled[feature].last()
            df_new[feature + 'o'] = df_resampled[feature].first()
            df_new[feature + 'l'] = df_resampled[feature].min()
            df_new[feature + 'h'] = df_resampled[feature].max()
        else:
            df_new[feature] = df_resampled[feature].last()

    # in trading the bar is created with past data
    # whereas pandas original index suggests it is created with future data
    df_new.index = df_new.index.shift(1)
    df_new.dropna(inplace=True)
    return df_new


def add_date_known_inputs(df):
    df['week_of_year'] = df.index.isocalendar().week
    df['day_of_week'] = df.index.dayofweek
    df['hour_of_day'] = df.index.hour
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['days_from_start'] = list(range(df.shape[0]))


def reconstruct_forecasts(formatter, results, n_output_steps):
    true_target = formatter.test_true_y
    true_target_col = formatter.test_true_y.columns[0]
    return_forecast_cols = ['t+{}'.format(i+1) for i in range(n_output_steps)]
    target_forecast_cols = ['{} t+{}'.format(true_target_col, i + 1) for i in range(n_output_steps)]

    reconstructed_forecast = {}
    for key, forecast in results.items():
        df = forecast.copy()

        # append true target to dataframe
        df.index = df['forecast_time']
        df = pd.concat([df, true_target], axis=1, join='inner')

        # calculate forecasts from return forecasts
        P_1 = df.loc[:, true_target_col].values.reshape(-1, 1)
        Ln_r = df.loc[:, return_forecast_cols].values
        # arr = np.array([series_from_ln_r(p_1, ln_r) for p_1, ln_r in zip(P_1, Ln_r)])
        df.loc[:, target_forecast_cols] = np.array([series_from_ln_r(p_1, ln_r) for p_1, ln_r in zip(P_1, Ln_r)])

        reconstructed_forecast[key] = df

    return reconstructed_forecast


def append_timediff_subsets(df, time_diff_cfg, new_col='time_subset', plot_=False):
    diff = pd.Series(df.index).diff()
    diff_s = diff.dt.total_seconds().fillna(0)
    if plot_:
        sns.histplot(diff_s / 1000, bins=100)
        plt.show()

    s_thold = s_threshold(time_diff_cfg)
    keep = (diff_s <= s_thold).astype(int)

    print('Good intervals: {}/{}  {}% of data'.format(sum(keep), diff_s.shape[0],
                                                      round(sum(keep) * 100 / diff_s.shape[0], 2)))

    df[new_col] = 0
    if sum(keep) != diff_s.shape[0]:
        df_time = pd.DataFrame(keep.values, index=df.index, columns=['time_switch'])
        time_subsets_ix = df_time.loc[df_time['time_switch'] == 0, ['time_switch']].index
        i = 0
        for i in range(1, len(time_subsets_ix)):
            df.loc[time_subsets_ix[i - 1]:time_subsets_ix[i], new_col] = i
        df.loc[time_subsets_ix[i]:, new_col] = i + 1


def add_features(df,
                 macds=None,
                 rsis=None,
                 returns=None,
                 use_time_subset=True,
                 p0s=[12],
                 p1s=[26],
                 returns_from_ema=(3, False)):

    if 'time_subset' in df.columns and use_time_subset:
        df_grp = df.groupby('time_subset')
        for i, (group_cols, df_subset) in enumerate(df_grp):
            if returns is not None:
                for var in returns:
                    if returns_from_ema[1]:
                        p = int(returns_from_ema[0])
                        df.loc[df_subset.index, '{}_e{}'.format(var, p)] = ema(df_subset[var], period=p)
                        df.loc[df_subset.index, '{}_e{}_r'.format(var, p)] = ln_returns(df.loc[df_subset.index, '{}_e{}'.format(var, p)])

                    df.loc[df_subset.index, var + '_r'] = ln_returns(df_subset[var])

            if macds is not None:
                for var in macds:
                    for p0, p1 in zip(p0s, p1s):
                        df.loc[df_subset.index, '{}_macd_{}_{}'.format(var, p0, p1)] = macd(df_subset[var], p0=p0, p1=p1)

            if rsis is not None:
                for var in rsis:
                    df.loc[df_subset.index, '{}_rsi'.format(var)] = rsi(df_subset[var], periods=14, ema=True)
    else:
        if returns is not None:
            for var in returns:
                if returns_from_ema[1]:
                    p = int(returns_from_ema[0])
                    df['{}_e{}'.format(var, p)] = ema(df[var], period=p)
                    df['{}_e{}_r'.format(var, p)] = ln_returns(df['{}_e{}'.format(var, p)])

                df[var + '_r'] = ln_returns(df[var])

        if macds is not None:
            for var in macds:
                for p0, p1 in zip(p0s, p1s):
                    df['{}_macd_{}_{}'.format(var, p0, p1)] = macd(df[var], p0=p0, p1=p1)

        if rsis is not None:
            for var in rsis:
                df['{}_rsi'.format(var)] = rsi(df[var], periods=14, ema=True)

    if returns is not None:
        for var in returns:
            df.loc[:, var + '_r'].fillna(0, inplace=True)

            if returns_from_ema[1]:
                p = int(returns_from_ema[0])
                df.loc[:, '{}_e{}_r'.format(var, p)].fillna(0, inplace=True)


def scale_df(df, training_cfg):
    train_features = get_train_features(training_cfg)
    if training_cfg['scale']:
        if 'test' in df.columns:
            ss = StandardScaler()
            ss.fit(df.loc[df.loc[:, 'test'] == 0, train_features])
            df_scaled = pd.DataFrame(ss.transform(df.loc[:, train_features]),
                                     columns=train_features, index=df.index)
            return df_scaled, ss, train_features
        else:
            ss = StandardScaler()
            ss.fit(df.loc[:, train_features])
            df_scaled = pd.DataFrame(ss.transform(df.loc[:, train_features]),
                                     columns=train_features, index=df.index)
            return df_scaled, ss, train_features
    else:
        return df.loc[:, train_features], None, train_features
