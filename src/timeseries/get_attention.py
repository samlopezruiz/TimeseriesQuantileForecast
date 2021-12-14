import os
import joblib
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.files import save_vars
from src.timeseries.utils.harness import get_model_data_config, get_attention_model
from src.timeseries.utils.results import process_self_attention, process_historical_vars_attention

if __name__ == "__main__":

    general_cfg = {'save_results': True,
                   }

    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'
                   }

    model_results = joblib.load(os.path.join(get_result_folder(results_cfg, project), results_cfg['results'] + '.z'))

    config, data_formatter, model_folder = get_model_data_config(project,
                                                                 model_results['experiment_cfg'],
                                                                 model_results['model_params'],
                                                                 model_results['fixed_params'])
    experiment_cfg = model_results['experiment_cfg']

    results = get_attention_model(use_gpu=False,
                                  architecture=experiment_cfg['architecture'],
                                  model_folder=model_folder,
                                  data_config=config.data_config,
                                  data_formatter=data_formatter,
                                  get_attentions=True,
                                  samples=None)

    self_attentions = process_self_attention(results['attentions'],
                                             results['params'],
                                             taus=[1, 3, 5])

    features_attn, mean_hist_attn = process_historical_vars_attention(results['attentions'],
                                                                      results['params'])

    results_processed = {
        'self_attentions': self_attentions,
        'features_attentions': features_attn,
        'mean_features_attentions': mean_hist_attn
    }

    if general_cfg['save_results']:
        print('Saving File')
        save_vars(results_processed,
                  file_path=os.path.join(get_result_folder(results_cfg, project), 'attention_processed'))
        save_vars(results,
                  file_path=os.path.join(get_result_folder(results_cfg, project), 'attention_valid'))
    print('Done')
