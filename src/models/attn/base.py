# Lint as: python3
"""Temporal Fusion Transformer Model.
Based on: https://github.com/google-research/google-research/tree/master/tft

Contains the full TFT architecture and associated components. Defines functions
for training, evaluation and prediction using simple Pandas Dataframe inputs.
"""

import datetime
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import src.models.attn.utils as utils
# Layer definitions.
from src.models.attn.data_formatter.base import InputTypes
from src.models.attn.mo_keras_model import MOModel
from src.models.attn.nn_funcs import DataCache, QuantileLossCalculator

concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Input = tf.keras.Input
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization
Dense = tf.keras.layers.Dense
Multiply = tf.keras.layers.Multiply
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Lambda = tf.keras.layers.Lambda
Attention = tf.keras.layers.Attention


# TFT model definitions.
class TSModel(object):
    """Defines Temporal Fusion Transformer.

  Attributes:
    name: Name of model
    time_steps: Total number of input time steps per forecast date (i.e. Width
      of Temporal fusion decoder N)
    input_size: Total number of inputs
    output_size: Total number of outputs
    category_counts: Number of categories per categorical variable
    n_multiprocessing_workers: Number of workers to use for parallel
      computations
    column_definition: List of tuples of (string, DataType, InputType) that
      define each column
    quantiles: Quantiles to forecast for TFT
    use_cudnn: Whether to use Keras CuDNNLSTM or standard LSTM layers
    hidden_layer_size: Internal state size of TFT
    dropout_rate: Dropout discard rate
    max_gradient_norm: Maximum norm for gradient clipping
    learning_rate: Initial learning rate of ADAM optimizer
    minibatch_size: Size of minibatches for training
    num_epochs: Maximum number of epochs for training
    early_stopping_patience: Maximum number of iterations of non-improvement
      before early stopping kicks in
    num_encoder_steps: Size of LSTM encoder -- i.e. number of past time steps
      before forecast date to use
    num_stacks: Number of self-attention layers to apply (default is 1 for basic
      TFT)
    num_heads: Number of heads for interpretable mulit-head attention
    model: Keras model for TFT
  """

    def __init__(self, raw_params, verbose=2, tb_callback=True):
        """Builds TFT from parameters.

    Args:
      raw_params: Parameters to define TFT
      use_cudnn: Whether to use CUDNN GPU optimised LSTM
    """

        self.fit_history = None
        self.name = self.__class__.__name__

        params = dict(raw_params)  # copy locally

        # Data parameters
        self.time_steps = int(params['total_time_steps'])
        self.input_size = int(params['input_size'])
        self.output_size = int(params['output_size'])
        self.category_counts = json.loads(str(params['category_counts']))
        self.n_multiprocessing_workers = int(params['multiprocessing_workers'])
        self.model_folder = params['model_folder']

        # Relevant indices
        self._input_obs_loc = json.loads(str(params['input_obs_loc']))
        self._static_input_loc = json.loads(str(params['static_input_loc']))
        self._known_regular_input_idx = json.loads(str(params['known_regular_inputs']))
        self._known_categorical_input_idx = json.loads(str(params['known_categorical_inputs']))

        self.column_definition = params['column_definition']

        # Network params
        self.tb_callback = tb_callback
        self.quantiles = params.get('quantiles', [0.1, 0.5, 0.9])
        # self.use_cudnn = use_cudnn
        self.learning_rate = float(params['learning_rate'])
        self.num_epochs = int(params['num_epochs'])
        self.early_stopping_patience = int(params['early_stopping_patience'])
        self.minibatch_size = int(params['minibatch_size'])
        self.num_encoder_steps = int(params['num_encoder_steps'])
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.max_gradient_norm = float(params['max_gradient_norm'])

        # Serialisation options
        self._temp_folder = os.path.join(params['model_folder'], 'tmp')
        self.reset_temp_folder()

        # Extra components to store Tensorflow nodes for attention computations
        self._input_placeholder = None
        self._prediction_parts = None
        self.output_labels = None

        self.verbose = verbose

        if self.verbose > 1:
            print('\n*** {} params ***'.format(self.name))
            for k in params:
                print('# {} = {}'.format(k, params[k]))

        # Build model
        self.model = self.build_model()

    def get_embeddings(self, all_inputs):
        """Transforms raw inputs to embeddings.

    Applies linear transformation onto continuous variables and uses embeddings
    for categorical variables.

    Args:
      all_inputs: Inputs to transform

    Returns:
      Tensors for transformed inputs.
    """

        time_steps = self.time_steps

        # Sanity checks
        for i in self._known_regular_input_idx:
            if i in self._input_obs_loc:
                raise ValueError('Observation cannot be known a priori!')
        for i in self._input_obs_loc:
            if i in self._static_input_loc:
                raise ValueError('Observation cannot be static!')

        if all_inputs.get_shape().as_list()[-1] != self.input_size:
            raise ValueError(
                'Illegal number of inputs! Inputs observed={}, expected={}'.format(
                    all_inputs.get_shape().as_list()[-1], self.input_size))

        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_sizes = [
            self.hidden_layer_size for i, size in enumerate(self.category_counts)
        ]

        embeddings = []
        for i in range(num_categorical_variables):
            embedding = tf.keras.Sequential([
                tf.keras.layers.InputLayer([time_steps]),
                tf.keras.layers.Embedding(
                    self.category_counts[i],
                    embedding_sizes[i],
                    input_length=time_steps,
                    dtype=tf.float32)
            ])
            embeddings.append(embedding)

        regular_inputs, categorical_inputs \
            = all_inputs[:, :, :num_regular_variables], \
              all_inputs[:, :, num_regular_variables:]

        embedded_inputs = [
            embeddings[i](categorical_inputs[Ellipsis, i])
            for i in range(num_categorical_variables)
        ]

        # Static inputs
        if self._static_input_loc:
            static_inputs = [
                                tf.keras.layers.Dense(self.hidden_layer_size)(regular_inputs[:, 0, i:i + 1])
                                for i in range(num_regular_variables) if i in self._static_input_loc
                            ] + \
                            [
                                embedded_inputs[i][:, 0, :]
                                for i in range(num_categorical_variables)
                                if i + num_regular_variables in self._static_input_loc
                            ]
            static_inputs = tf.keras.backend.stack(static_inputs, axis=1)

        else:
            static_inputs = None

        def convert_real_to_embedding(x):
            """Applies linear transformation for time-varying inputs."""
            return tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.hidden_layer_size))(x)

        # Targets
        obs_inputs = tf.keras.backend.stack([
            convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1])
            for i in self._input_obs_loc
        ], axis=-1)

        # Observed (a prioir unknown) inputs
        wired_embeddings = []
        for i in range(num_categorical_variables):
            if i not in self._known_categorical_input_idx and i + num_regular_variables not in self._input_obs_loc:
                e = embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
            if i not in self._known_regular_input_idx and i not in self._input_obs_loc:
                e = convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1])
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            unknown_inputs = tf.keras.backend.stack(unknown_inputs + wired_embeddings, axis=-1)
        else:
            unknown_inputs = None

        # A priori known inputs
        known_regular_inputs = [
            convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1])
            for i in self._known_regular_input_idx
            if i not in self._static_input_loc
        ]
        known_categorical_inputs = [
            embedded_inputs[i]
            for i in self._known_categorical_input_idx
            if i + num_regular_variables not in self._static_input_loc
        ]

        known_combined_layer = tf.keras.backend.stack(known_regular_inputs + known_categorical_inputs, axis=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def _get_single_col_by_type(self, input_type):
        """Returns name of single column for input type."""

        return utils.get_single_col_by_input_type(input_type, self.column_definition)

    def training_data_cached(self):
        """Returns boolean indicating if training data has been cached."""

        return DataCache.contains('train') and DataCache.contains('valid')

    def cache_batched_data(self, data, cache_key, num_samples=-1):
        """Batches and caches data once for using during training.

    Args:
      data: Data to batch and cache
      cache_key: Key used for cache
      num_samples: Maximum number of samples to extract (-1 to use all data)
    """

        if num_samples > 0:
            DataCache.update(self._batch_sampled_data(data, max_samples=num_samples), cache_key)
        else:
            DataCache.update(self._batch_data(data), cache_key)

        if self.verbose > 0:
            print('Cached data "{}" updated'.format(cache_key))

    def _batch_sampled_data(self, data, max_samples):
        """Samples segments into a compatible format.

    Args:
      data: Sources data to sample and batch
      max_samples: Maximum number of samples in batch

    Returns:
      Dictionary of batched data with the maximum samples specified.
    """

        if max_samples < 1:
            raise ValueError('Illegal number of samples specified! samples={}'.format(max_samples))

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)

        data.sort_values(by=[id_col, time_col], inplace=True)

        print('Getting valid sampling locations.')
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_col):
            # print('Getting locations for {}'.format(identifier))
            num_entries = len(df)
            if num_entries >= self.time_steps:
                valid_sampling_locations += [
                    (identifier, self.time_steps + i)
                    for i in range(num_entries - self.time_steps + 1)
                ]
            split_data_map[identifier] = df

        inputs = np.zeros((max_samples, self.time_steps, self.input_size))
        outputs = np.zeros((max_samples, self.time_steps, self.output_size))
        time = np.empty((max_samples, self.time_steps, 1), dtype=object)
        identifiers = np.empty((max_samples, self.time_steps, 1), dtype=object)

        if max_samples > 0 and len(valid_sampling_locations) > max_samples:
            print('Extracting {} samples...'.format(max_samples))
            ranges = [
                valid_sampling_locations[i] for i in np.random.choice(
                    len(valid_sampling_locations), max_samples, replace=False)
            ]
        else:
            print('Max samples={} exceeds # available segments={}'.format(
                max_samples, len(valid_sampling_locations)))
            ranges = valid_sampling_locations

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        for i, tup in enumerate(ranges):
            if (i + 1 % 1000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx -
                                                     self.time_steps:start_idx]
            inputs[i, :, :] = sliced[input_cols]
            outputs[i, :, :] = sliced[[target_col]]
            time[i, :, 0] = sliced[time_col]
            identifiers[i, :, 0] = sliced[id_col]

        sampled_data = {
            'inputs': inputs,
            'outputs': outputs[:, self.num_encoder_steps:, :],
            'active_entries': np.ones_like(outputs[:, self.num_encoder_steps:, :]),
            'time': time,
            'identifier': identifiers
        }

        return sampled_data

    def _batch_data(self, data):
        """Batches data for training.

    Converts raw dataframe from a 2-D tabular format to a batched 3-D array
    to feed into Keras model.

    Args:
      data: DataFrame to batch

    Returns:
      Batched Numpy array with shape=(?, self.time_steps, self.input_size)
    """

        # Functions.
        def _batch_single_entity(input_data):
            time_steps = len(input_data)
            lags = self.time_steps
            x = input_data.values
            if time_steps >= lags:
                return np.stack(
                    [x[i:time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1)

            else:
                return None

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        data_map = {}
        for _, sliced in data.groupby(id_col):

            col_mappings = {
                'identifier': [id_col],
                'time': [time_col],
                'outputs': [target_col],
                'inputs': input_cols
            }

            for k in col_mappings:
                cols = col_mappings[k]
                arr = _batch_single_entity(sliced[cols].copy())

                if k not in data_map:
                    data_map[k] = [arr]
                else:
                    data_map[k].append(arr)

        # Combine all data
        for k in data_map:
            non_empty_map = [batch for batch in data_map[k] if batch is not None]
            data_map[k] = np.concatenate(non_empty_map, axis=0)

        # Shorten target so we only get decoder steps
        data_map['outputs'] = data_map['outputs'][:, self.num_encoder_steps:, :]

        active_entries = np.ones_like(data_map['outputs'])
        if 'active_entries' not in data_map:
            data_map['active_entries'] = active_entries
        else:
            data_map['active_entries'].append(active_entries)

        return data_map

    def _get_active_locations(self, x):
        """Formats sample weights for Keras training."""
        return (np.sum(x, axis=-1) > 0.0) * 1.0

    def _build_base_graph(self):
        """Returns graph defining layers of the Model.
        return extraction_output, attention_components (if any)
        """
        pass

    def get_input_shape(self):
        return (self.time_steps, self.input_size)

    def build_model(self):
        """Build model and defines training losses.

        Returns:
          Fully defined Keras model.
        """

        # Inputs.
        all_inputs = tf.keras.layers.Input(shape=self.get_input_shape())
        self._input_placeholder = all_inputs

        extraction_output, attention_components = self._build_base_graph(all_inputs)
        quantile_dense = tf.keras.layers.Dense(self.output_size * len(self.quantiles), name="dense_quantiles")
        outputs = tf.keras.layers.TimeDistributed(quantile_dense, name="td_quantiles")(extraction_output)

        self._attention_components = attention_components

        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=self.max_gradient_norm)

        attn_outputs = [attention_components[key] for key in attention_components]
        model = MOModel(inputs=all_inputs,
                        outputs=[outputs] + attn_outputs + [extraction_output])

        # print(model.summary())
        self.output_labels = ['quantiles'] + [key for key in attention_components] + ['transformer_output']

        quantile_loss = QuantileLossCalculator(self.quantiles, self.output_size).quantile_loss
        model.compile(loss=quantile_loss, optimizer=adam, sample_weight_mode='temporal')

        return model

    def fit(self, prefetch_data=True, train_df=None, valid_df=None):
        """Fits deep neural network for given training and validation data.

    Args:
      train_df: DataFrame for training data
      valid_df: DataFrame for validation data
      :param prefetch_data: prefetch training and validation data
    """

        print('\n*** Fitting {} ***'.format(self.name))
        log_dir = os.path.join(self.model_folder,
                               "logs/fit/" + self.name+'-' + datetime.datetime.now().strftime("%Y%m%d-%H%M")[2:])

        # Add relevant callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                min_delta=1e-4),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.get_keras_saved_path(self._temp_folder),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        if self.tb_callback:
            callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=0))

        print('Getting batched_data')
        if train_df is None:
            print('Using cached training data')
            train_data = DataCache.get('train')
        else:
            train_data = self._batch_data(train_df)

        if valid_df is None:
            print('Using cached validation data')
            valid_data = DataCache.get('valid')
        else:
            valid_data = self._batch_data(valid_df)

        print('Using keras standard fit')

        def _unpack(data):
            return data['inputs'], data['outputs'], \
                   self._get_active_locations(data['active_entries'])

        # Unpack without sample weights
        data, labels, active_flags = _unpack(train_data)
        val_data, val_labels, val_flags = _unpack(valid_data)

        all_callbacks = callbacks

        t0 = time.time()
        if prefetch_data:
            train_dataset = tf.data.Dataset.from_tensor_slices((data,
                                                                np.concatenate([labels, labels, labels], axis=-1))
                                                               ).prefetch(tf.data.AUTOTUNE)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data,
                                                              np.concatenate([val_labels, val_labels, val_labels],
                                                                             axis=-1),
                                                              val_flags)
                                                             ).prefetch(tf.data.AUTOTUNE)
            history = self.model.fit(
                x=train_dataset.shuffle(data.shape[0]).batch(self.minibatch_size),
                epochs=self.num_epochs,
                validation_data=val_dataset.shuffle(val_data.shape[0]).batch(self.minibatch_size),
                callbacks=all_callbacks,
                use_multiprocessing=True)
        else:
            history = self.model.fit(
                x=data,
                y=np.concatenate([labels, labels, labels], axis=-1),
                sample_weight=active_flags,
                epochs=self.num_epochs,
                batch_size=self.minibatch_size,
                validation_data=(val_data,
                                 np.concatenate([val_labels, val_labels, val_labels], axis=-1),
                                 val_flags),
                callbacks=all_callbacks,
                shuffle=True,
                use_multiprocessing=True,
                workers=self.n_multiprocessing_workers)

        self.fit_history = history.history
        print('Training finished in {}s'.format(round(time.time() - t0, 0)))

        # Load best checkpoint again
        tmp_checkpont = self.get_keras_saved_path(self._temp_folder)
        print(tmp_checkpont, os.path.exists(tmp_checkpont))
        if os.path.exists('{}.index'.format(tmp_checkpont)):
            self.load(
                self._temp_folder,
                use_keras_loadings=True)

        else:
            print('Cannot load from {}, skipping ...'.format(self._temp_folder))

    def evaluate(self, data=None, eval_metric='loss'):
        """Applies evaluation metric to the training data.

    Args:
      data: Dataframe for evaluation
      eval_metric: Evaluation metic to return, based on model definition.

    Returns:
      Computed evaluation loss.
    """

        if data is None:
            print('Using cached validation data')
            raw_data = DataCache.get('valid')
        else:
            raw_data = self._batch_data(data)

        inputs = raw_data['inputs']
        outputs = raw_data['outputs']
        active_entries = self._get_active_locations(raw_data['active_entries'])

        metric_values = self.model.evaluate(
            x=inputs,
            y=np.concatenate([outputs, outputs, outputs], axis=-1),
            sample_weight=active_entries,
            workers=16,
            use_multiprocessing=True)

        metrics = pd.Series(metric_values, self.model.metrics_names)

        return metrics[eval_metric]

    def predict(self, df, return_targets=False, multi_processing=True):
        """Computes predictions for a given input dataset.

    Args:
      df: Input dataframe
      return_targets: Whether to also return outputs aligned with predictions to
        faciliate evaluation

    Returns:
      Input dataframe or tuple of (input dataframe, algined output dataframe).
    """

        data = self._batch_data(df)

        inputs = data['inputs']
        time = data['time']
        identifier = data['identifier']
        outputs = data['outputs']

        combined = self.model.predict(
            inputs,
            workers=16 if multi_processing else 1,
            use_multiprocessing=multi_processing,
            batch_size=self.minibatch_size)

        # Format output_csv
        if self.output_size != 1:
            raise NotImplementedError('Current version only supports 1D targets!')

        output_map = self.create_output_map(combined, data, return_targets)

        return output_map

    def create_output_map(self, quantile_prediction, data, return_targets=True):
        outputs = data['outputs']
        # Extract predictions for each quantile into different entries
        process_map = {
            'p{}'.format(int(q * 100)):
                quantile_prediction[Ellipsis, i * self.output_size:(i + 1) * self.output_size]
            for i, q in enumerate(self.quantiles)
        }

        if return_targets:
            # Add targets if relevant
            process_map['targets'] = outputs

        return {k: self.format_outputs(process_map[k], data) for k in process_map}

    def predict_all(self, df, batch_size):

        print('Batching data...')
        data = self._batch_data(df)
        inputs = data['inputs']

        batches = []
        for i in range((inputs.shape[0] // batch_size) + 1):
            batches.append(inputs[i * batch_size:min(i * batch_size + batch_size, inputs.shape[0]), Ellipsis])

        print('Prediction All Outputs...')
        outputs = []
        for batch in batches:
            outputs.append([out.numpy() for out in self.model(batch)])

        concat_outputs = []
        for i in range(len(outputs[0])):
            concat_outputs.append(np.concatenate([out[i] for out in outputs], axis=0 if i != 1 else 1))

        result = {}
        for label, output in zip(self.output_labels, concat_outputs):
            result[label] = output

        output_map = self.create_output_map(result['quantiles'], data, return_targets=True)
        return result, output_map, data

    def format_outputs(self, prediction, data):
        """Returns formatted dataframes for prediction."""
        time = data['time']
        identifier = data['identifier']

        flat_prediction = pd.DataFrame(
            prediction[:, :, 0],
            columns=[
                't+{}'.format(i + 1)
                for i in range(self.time_steps - self.num_encoder_steps)
            ])
        cols = list(flat_prediction.columns)
        flat_prediction['forecast_time'] = time[:, self.num_encoder_steps - 1, 0]
        flat_prediction['identifier'] = identifier[:, 0, 0]

        # Arrange in order
        return flat_prediction[['forecast_time', 'identifier'] + cols]

    # Serialisation.
    def reset_temp_folder(self):
        """Deletes and recreates folder with temporary Keras training outputs."""
        print('Resetting temp folder...')
        utils.create_folder_if_not_exist(self._temp_folder)
        shutil.rmtree(self._temp_folder)
        os.makedirs(self._temp_folder)

    def get_keras_saved_path(self, model_folder):
        """Returns path to keras checkpoint."""
        return os.path.join(model_folder, '{}.ckpt'.format(self.name))

    def save(self, model_folder):
        """Saves optimal TFT weights.

    Args:
      model_folder: Location to serialze model.
    """
        # Allows for direct serialisation of tensorflow variables to avoid spurious
        # issue with Keras that leads to different performance evaluation results
        # when model is reloaded (https://github.com/keras-team/keras/issues/4875).
        save_path = self.get_keras_saved_path(model_folder)
        self.model.save_weights(save_path)


    def load(self, model_folder, use_keras_loadings=False):
        """Loads TFT weights.

    Args:
      model_folder: Folder containing serialized models.
      use_keras_loadings: Whether to load from Keras checkpoint.

    Returns:

    """
        print('serialisation_path: {}'.format(self.get_keras_saved_path(model_folder)))
        if use_keras_loadings:
            # Loads temporary Keras model saved during training.
            serialisation_path = self.get_keras_saved_path(model_folder)
            print('Loading model from {}'.format(serialisation_path))
            self.model.load_weights(serialisation_path).expect_partial()
        else:
            # Loads tensorflow graph for optimal models.
            utils.load(
                tf.compat.v1.keras.backend.get_session(),
                model_folder,
                cp_name=self.name,
                scope=self.name)

    @classmethod
    def get_hyperparm_choices(cls):
        """Returns hyperparameter ranges for random search."""
        return {
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
            'hidden_layer_size': [10, 20, 40, 80, 160, 240, 320],
            'minibatch_size': [64, 128, 256],
            'learning_rate': [1e-4, 1e-3, 1e-2],
            'max_gradient_norm': [0.01, 1.0, 100.0],
            'num_heads': [1, 4],
            'stack_size': [1],
        }
