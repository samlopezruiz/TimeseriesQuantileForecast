import os

from src.timeseries.plot.ts import plotly_ts_regime_hist_vars
from src.timeseries.utils.config import read_config
from src.timeseries.utils.dataset import load_multiple_markets, get_data_root
from src.timeseries.utils.filename import hmm_filename, update_trim_range
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.hmm import append_hmm_states, count_regimes
from src.timeseries.utils.preprocessing import add_features
from src.timeseries.utils.split import time_subset

if __name__ == '__main__':
    # %%
    in_cfg = {'save_results': True,
              'save_plot': True,
              'plot_title': False}

    project = 'snp'
    cfg = read_config('hidden_markov_model', project)
    df = load_multiple_markets(cfg['datasets'],
                               project,
                               resampling=cfg['hmm_cfg']['resampling'],
                               ffill=True)

    df_ss = time_subset(df, cfg['hmm_cfg'])
    add_features(df_ss,
                 macds=cfg['hmm_cfg']['macd_vars'],
                 returns=cfg['hmm_cfg']['return_vars'])

    plot_features = cfg['hmm_cfg']['vars'] + ['volume', 'atr', 'DGS10']

    # %%
    df_reg, n_regimes, df_proba = append_hmm_states(df_ss,
                                                    cfg['hmm_cfg']['vars'],
                                                    cfg['hmm_cfg']['number_of_states'],
                                                    regime_col='state')

    if in_cfg['save_results']:
        result = {
            'data': df_reg,
            'n_regimes': n_regimes,
            'df_proba': df_proba,
            'hmm_cfg': cfg['hmm_cfg'],
            'data_cfgs': cfg['datasets']
        }

        update_trim_range(df_reg, cfg['hmm_cfg'])
        file_path = os.path.join(get_data_root(project), 'regime', hmm_filename(cfg['hmm_cfg']))
        save_vars(result, file_path, use_date_suffix=False)

    img_path = os.path.join(get_data_root(project), 'regime', 'img', hmm_filename(cfg['hmm_cfg']))

    title = 'Regime Changes: {}, n_states: {}, vars: {}'.format(count_regimes(n_regimes),
                                                                cfg['hmm_cfg']['number_of_states'],
                                                                str(cfg['hmm_cfg']['vars']))
    plotly_ts_regime_hist_vars(df_reg,
                               price_col='ESc',
                               regime_col='state',
                               features=cfg['hmm_cfg']['vars'],
                               adjust_height=(True, 0.6),
                               markersize=4,
                               save_png=True,
                               save=in_cfg['save_plot'],
                               file_path=img_path,
                               size=(1980, 1080),
                               plot_title=in_cfg['plot_title'],
                               title=title,
                               label_scale=1)
