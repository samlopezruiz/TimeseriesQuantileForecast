import numpy as np
import pandas as pd

def ismv(train):
    if len(train.shape) > 1:
        is_mv = True if train.shape[1] > 1 else False
    else:
        is_mv = False
    return is_mv

def ln_returns(x):
    ln_r = np.log(x) - np.log(np.roll(x, 1, axis=0))

    # first row has no returns
    if isinstance(ln_r, pd.Series):
        return ln_r.iloc[1:]
    else:
        return ln_r[1:]

def ema(x, period, last_ema=None):
    c1 = 2 / (1 + period)
    c2 = 1 - (2 / (1 + period))
    x = np.array(x)
    if last_ema is None:
        ema_x = np.array(x)
        for i in range(1, ema_x.shape[0]):
            ema_x[i] = x[i] * c1 + c2 * ema_x[i - 1]
    else:
        ema_x = np.zeros((len(x) + 1,))
        ema_x[0] = last_ema
        for i in range(1, ema_x.shape[0]):
            ema_x[i] = x[i] * c1 + c2 * ema_x[i - 1]
        ema_x = ema_x[1:]
    return ema_x

def rsi(data, periods=14, ema=True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = data.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema == True:
        # Use exponential moving average
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window=periods, adjust=False).mean()
        ma_down = down.rolling(window=periods, adjust=False).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi.fillna(50)


def atr_bad(df, inst, n=14):
    high_low = df[inst + 'h'] - df[inst + 'l']
    high_close = np.abs(df[inst + 'h'] - df[inst + 'c'].shift())
    low_close = np.abs(df[inst + 'l'] - df[inst + 'c'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    atr = true_range.rolling(n).sum() / n
    return atr


def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1 / n, adjust=False).mean()


def atr(df, inst, n=14):
    data = df.copy()
    high = df[inst + 'h']
    low = df[inst + 'l']
    close = df[inst + 'c']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    return atr


def macd(x, p0=12, p1=26):
    # x = np.array(x)
    ema0 = ema(x, p0)
    ema1 = ema(x, p1)
    return ema0 - ema1