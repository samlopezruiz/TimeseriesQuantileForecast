from src.timeseries.utils.config import read_config
from src.timeseries.utils.dataset import load_dataset
from src.timeseries.utils.filename import update_trim_range
from src.timeseries.utils.preprocessing import downsample_df
from src.timeseries.utils.save import save_market_data
from src.timeseries.utils.split import time_subset

if __name__ == '__main__':
    # %%
    in_cfg = {'save_results': True,
              }

    project = 'snp'
    cfg = read_config('additional_ds', project)

    for data_cfg in cfg['datasets']:
        df = load_dataset(data_cfg, project)
        df = time_subset(df, data_cfg)

        if data_cfg['downsample']:
            df = downsample_df(df, data_cfg['downsample_p'], ohlc_features=False, inst=data_cfg['inst'])
            # plot_mkt_candles(df.iloc[-30000:, :], data_cfg['inst'], resample=False, period='90T', template='plotly_dark')

        if in_cfg['save_results']:
            result = {
                'data': df
            }
            update_trim_range(df, data_cfg)
            save_market_data(result, project, data_cfg)

