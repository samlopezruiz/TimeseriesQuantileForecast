# Lint as: python3
"""Custom formatting functions for Volatility dataset.

Defines dataset specific column definitions and data transformations.
"""
import joblib
import numpy as np
import pandas as pd
import sklearn.preprocessing

from src.models.attn import utils
from src.models.attn.data_formatter.base import GenericDataFormatter, InputTypes, DataTypes
# from src.timeseries.expt_settings.definitions import variable_definitions
from src.timeseries.utils.config import get_variable_definitions
from src.timeseries.utils.dataframe import resample_dfs, new_cols_names
from src.timeseries.utils.preprocessing import add_features


class SnPFormatter(GenericDataFormatter):
    """Defines and formats data for the volatility dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

    _column_definition = []

    fixed_params = {
        'quantiles': [0.1, 0.5, 0.9],
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 12,
    }

    model_params = {
        'total_time_steps': 48 + 5,
        'num_encoder_steps': 48,
        'dropout_rate': 0.3,
        'hidden_layer_size': 16,
        'learning_rate': 0.01,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
    }

    def __init__(self, project, vars_definition, architecture):
        """Initialises formatter."""

        variable_definitions = get_variable_definitions(project)
        if vars_definition not in variable_definitions.keys():
            raise Exception('vars_definition: {} not found in {}'.format(vars_definition,
                                                                         variable_definitions.keys()))

        self._column_definition = variable_definitions[vars_definition]
        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self.n_states = None
        self.valid_true_y = None
        self.test_true_y = None
        self.architecture = architecture
        self._append_model_params()

    # def split_data(self, df, valid_boundary=2016, test_boundary=2018):
    #     pass

    def split_data(self,
                   data_config,
                   scale=True,
                   indicators_use_time_subset=True,
                   ):

        """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      data_config: Source config to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
        :param scale:
    """
        mkt_data, add_data = self.load_data(data_config)

        mkt_data = self.preprocess_data(mkt_data, add_data, data_config, indicators_use_time_subset)
        train, test = mkt_data.loc[mkt_data.loc[:, 'test'] == 0, :], mkt_data.loc[mkt_data.loc[:, 'test'] == 1, :]
        valid = mkt_data.loc[mkt_data.loc[:, 'test'] == 2, :]

        if data_config['target_dataset']['true_target'] is not None:
            self.set_true_target(data_config['target_dataset']['true_target'], valid, test)

        self.set_scalers(train)

        if scale:
            return (self.transform_inputs(data) for data in [train, valid, test])
        else:
            return train, valid, test

    def process_data(self,
                     data_config,
                     scale=True,
                     indicators_use_time_subset=True,
                     ):

        """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      data_config: Source config to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
        :param scale:
    """
        mkt_data, add_data = self.load_data(data_config)

        mkt_data = self.preprocess_data(mkt_data, add_data, data_config, indicators_use_time_subset)

        if data_config['target_dataset']['true_target'] is not None:
            self.set_true_target(data_config['target_dataset']['true_target'], mkt_data, mkt_data)

        self.set_scalers(mkt_data)

        if scale:
            return self.transform_inputs(mkt_data)
        else:
            return mkt_data

    def preprocess_data(self, mkt_data, add_data, data_config, indicators_use_time_subset):

        print('\nPreprocessing market data...')
        # Add processed features
        add_features(mkt_data,
                     macds=data_config['target_dataset']['macd_vars'],
                     returns=data_config['target_dataset']['returns_vars'],
                     returns_from_ema=data_config['target_dataset']['returns_from_ema'],
                     use_time_subset=indicators_use_time_subset,
                     rsis=data_config['target_dataset']['rsi_vars'],
                     p0s=data_config['target_dataset']['macd_periods'],
                     p1s=np.array(data_config['target_dataset']['macd_periods']) * 2)

        add_resampled = []
        if add_data is not None:
            for pp_cfg, add_df in zip(data_config['additional_datasets'], add_data):
                additional_resampled = resample_dfs(mkt_data, add_df)
                additional_resampled.columns = new_cols_names(additional_resampled, pp_cfg['prefix_col'])

                # append additional features
                add_features(additional_resampled,
                             macds=pp_cfg['macd_vars'],
                             returns=pp_cfg['returns_vars'],
                             returns_from_ema=pp_cfg['returns_from_ema'],
                             use_time_subset=indicators_use_time_subset,
                             rsis=pp_cfg['rsi_vars'],
                             p0s=pp_cfg['macd_periods'],
                             p1s=np.array(pp_cfg['macd_periods']) * 2)

                if 'use_only_vars' in pp_cfg:
                    additional_resampled = additional_resampled.loc[:, pp_cfg['use_only_vars']]

                add_resampled.append(additional_resampled)
                # append additional features
                # add_features(additional_resampled,
                #              macds=data_config['add_macd_vars'],
                #              returns=data_config['add_returns_vars'])

            # discard all but relevant features ?
            # additional_resampled = additional_resampled.loc[:, [col[0] for col in _add_column_definition]]
            mkt_data = pd.concat([mkt_data] + add_resampled, axis=1)

        # if reg_data is not None:
        #     reg_resampled = resample_dfs(mkt_data, reg_data)
        #     mkt_data['regime'] = reg_resampled['state']

        print('\n{} Available Features: {}'.format(mkt_data.shape[1], list(mkt_data.columns)))
        mkt_data['datetime'] = mkt_data.index
        mkt_data.reset_index(drop=True, inplace=True)

        return mkt_data

    def load_data(self, data_config):
        print('\nLoading market ds: \n{}'.format(data_config['target_dataset']['file_path']))
        split_data = joblib.load(data_config['target_dataset']['file_path'])

        if data_config['additional_datasets']:
            additional_data = []
            for add_ds in data_config['additional_datasets']:
                print('\nLoading additional ds: \n{}'.format(add_ds['file_path']))
                additional_data.append(joblib.load(add_ds['file_path']))
        else:
            additional_data = None

        # if data_config['regime_file']:
        #     print('Loading Regime Data: {}'.format(data_config['regime_file']))
        #     regime_data = joblib.load(data_config['regime_file'])
            # self.n_states = regime_data['n_regimes']
        # else:
        #     regime_data = None
        mkt_data = split_data['data']
        add_data = [add_ds.get('data', None) for add_ds in additional_data]
        # reg_data = regime_data.get('data', None)

        return mkt_data, add_data #, reg_data

    def set_true_target(self, true_target, valid, test):

        print('Setting target vars for reconstruction...')
        column_definitions = self.get_column_definition()
        time_column = utils.get_single_col_by_input_type(InputTypes.TIME, column_definitions)
        self.valid_true_y = pd.Series(valid[true_target].values, index=valid[time_column], name=true_target).to_frame()
        self.test_true_y = pd.Series(test[true_target].values, index=test[time_column], name=true_target).to_frame()



    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()

        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values)  # used for predictions

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(predictions[col])

        return output

    # Default params
    def update_model_params(self, new_cfg):
        for key, val in new_cfg.items():
            self.model_params[key] = val

    def update_fixed_params(self, new_cfg):
        for key, val in new_cfg.items():
            self.fixed_params[key] = val

    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        return self.fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        return self.model_params

    def _append_model_params(self):
        if self.architecture == 'TFTModel':
            default_values = {
                'num_heads': 4,
                'stack_size': 1
            }
        elif self.architecture == 'LSTMModel':
            default_values = {
            }
        elif self.architecture == 'DCNNModel':
            default_values = {
                'n_layers': 4,
                'reg': 'L2',
                'n_kernel': 3,
            }
        self.update_model_params(default_values)

