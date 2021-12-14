import copy
import os

from src.timeseries.data_formatter.snp import SnPFormatter
from src.timeseries.utils.config import get_cfg_folder, read_config
from src.timeseries.utils.dataset import get_inst_ohlc_names, get_data_root
from src.timeseries.utils.filename import get_output_folder


def prepare_dataset_cfg(project, cfg):
    for list_var in ['macd_vars', 'rsi_vars', 'returns_vars', 'macd_periods']:
        if list_var not in cfg:
            cfg[list_var] = []

    for list_var in ['prefix_col', 'true_target']:
        if list_var not in cfg:
            cfg[list_var] = None

    if 'returns_from_ema' not in cfg:
        cfg['returns_from_ema'] = (5, False)
    else:
        cfg['returns_from_ema'] = (cfg['ema_period'], cfg['returns_from_ema'])

    cfg['file_path'] = os.path.join(get_data_root(project), *cfg['subfolder'], cfg['filename'] + '.z')
    return cfg


class ExperimentConfig(object):
    """Defines experiment configs and paths to outputs.

  Attributes:
    root_folder: Root folder to contain all experimental outputs.
    experiment: Name of experiment to run.
    data_folder: Folder to store data for experiment.
    model_folder: Folder to store serialised models.
    results_folder: Folder to store results.
    data_csv_path: Path to primary data csv file used in experiment.
    hyperparam_iterations: Default number of random search iterations for
      experiment.
  """

    default_experiments = ['snp']

    def __init__(self,
                 project,
                 cfg,
                 ):
        """Creates configs based on default experiment chosen.

    Args:
      experiment: Name of experiment.
      root_folder: Root folder to save all outputs of training.
    """

        if project not in self.default_experiments:
            raise ValueError('Unrecognised experiment={}'.format(project))

        root_folder = cfg.get('root_folder', None)


        # Defines all relevant paths
        if root_folder is None:
            root_folder = get_output_folder()
            print('Using root folder {}'.format(root_folder))

        self.architecture = cfg.get('architecture', 'TFTModel')
        self.project = project
        self.root_folder = root_folder
        self.dataset_config = cfg.get('preprocess_cfg', None)
        self.vars_definition = cfg['vars_definition_cfg']
        self.data_folder = get_data_root(project)
        self.model_folder = os.path.join(root_folder, 'saved_models', project)
        self.results_folder = os.path.join(root_folder, 'results', project)

        dataset_cfg = self.get_dataset_cfg()

        # dataset configurations of timeseries
        self.target_dataset = prepare_dataset_cfg(project, dataset_cfg['target_dataset'])
        if dataset_cfg['additional_datasets'] is not None:
            self.add_datasets = [prepare_dataset_cfg(project, cfg) for cfg in dataset_cfg['additional_datasets']]
        else:
            self.add_datasets = None

        # Creates folders if they don't exist
        for relevant_directory in [
            self.root_folder, self.data_folder, self.model_folder,
            self.results_folder
        ]:
            if not os.path.exists(relevant_directory):
                os.makedirs(relevant_directory)

    @property
    def data_config(self):

        ans = {
            'target_dataset': self.target_dataset,
            'additional_datasets': self.add_datasets,
        }

        return ans

    @property
    def hyperparam_iterations(self):

        return 240

    def make_data_formatter(self):
        """Gets a data formatter object for experiment.

    Returns:
      Default DataFormatter per experiment.
    """

        data_formatter_class = {
            'snp': SnPFormatter,
        }

        return data_formatter_class[self.project](self.project, self.vars_definition, self.architecture)

    def get_dataset_cfg(self):
        cfg_folder = get_cfg_folder(self.project)
        dataset_class = {}
        for yaml_file in os.listdir(os.path.join(cfg_folder, 'preprocess')):
            dataset_class[yaml_file.split('.')[0]] = read_config(yaml_file,
                                                                 self.project,
                                                                 subfolder='preprocess')

        if self.dataset_config not in dataset_class:
            raise Exception('{} not found in dataset configurations. '
                            '\nOptions are: \n{}'.format(self.dataset_config, dataset_class.keys()))

        return dataset_class[self.dataset_config]
