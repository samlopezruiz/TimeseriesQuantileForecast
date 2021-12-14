import os

import joblib as joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.timeseries.plot.moo import save_fig
from src.timeseries.plot.ts import plotly_time_series
from src.timeseries.utils.filename import get_result_folder

sns.set_theme('poster')
if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': True,
                   }

    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q159',
                   'results': 'attention_processed'
                   }

    results = joblib.load(os.path.join(get_result_folder(results_cfg, project), results_cfg['results'] + '.z'))

    # %%
    self_attentions = results['self_attentions']
    features_attentions = results['features_attentions']
    mean_features_attentions = results['mean_features_attentions']

    for i, self_attn in enumerate(self_attentions):
        plotly_time_series(self_attn,
                           xaxis_title='Position Index (n)',
                           title='Self Attention Head {}'.format(i),
                           label_scale=1,
                           save=general_cfg['save_plot'],
                           file_path=os.path.join(get_result_folder(results_cfg, project),
                                                  'img',
                                                  '{}_head_{}'.format(results_cfg['experiment_name'], i)),
                           save_png=True,
                           size=(1980 * 2 // 3, 1080 * 2 // 3))

    # %%
    n_features_plot = 3
    plotly_time_series(features_attentions,
                       features=features_attentions.columns[:n_features_plot],
                       rows=list(range(n_features_plot)),
                       xaxis_title='Position Index (n)',
                       plot_ytitles=True,
                       save=general_cfg['save_plot'],
                       file_path=os.path.join(get_result_folder(results_cfg, project),
                                              'img',
                                              '{}_feat_attn_pos'.format(results_cfg['experiment_name'])),
                       label_scale=1.5,
                       save_png=True,
                       size=(1980 * 2 // 3, 1080 * 2 // 3))

    # %%
    df = mean_features_attentions.copy()
    df['feature'] = df.index
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.barplot(data=df, x='mean attn', ax=ax, y='feature', orient='h')
    plt.tight_layout()
    plt.show()

    if general_cfg['save_plot']:
        save_fig(fig,
                 file_path=os.path.join(get_result_folder(results_cfg, project),
                                        'img',
                                        '{}_feat_attn'.format(results_cfg['experiment_name'])),
                 use_date=False)
