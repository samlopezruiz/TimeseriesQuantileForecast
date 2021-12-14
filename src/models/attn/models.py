import gc
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tqdm import tqdm

from src.models.attn.base import TSModel
from src.models.attn.nn_funcs import DataCache, gated_residual_network, get_lstm, apply_gating_layer, add_and_norm, \
    InterpretableMultiHeadAttention, get_decoder_mask, temporal_selection, static_combine_and_mask
from src.models.wavenet.func import dcnn_1st_layer, dcnn_layer

concat = tf.keras.backend.concatenate
K = tf.keras.backend
from tensorflow import keras


# TFT model definitions.
class TFTModel(TSModel):
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
    """

        self.name = self.__class__.__name__

        params = dict(raw_params)  # copy locally

        self.dropout_rate = float(params['dropout_rate'])
        self.max_gradient_norm = float(params['max_gradient_norm'])
        self._prediction_parts = None
        self.num_stacks = int(params['stack_size'])
        self.num_heads = int(params['num_heads'])
        self._attention_components = None

        super().__init__(raw_params, verbose=verbose, tb_callback=tb_callback)

    def _build_base_graph(self, all_inputs):
        """Returns graph defining layers of the TFT."""

        # # Size definitions.
        time_steps = self.time_steps
        encoder_steps = self.num_encoder_steps

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.get_embeddings(all_inputs)

        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            historical_inputs = concat([unknown_inputs[:, :encoder_steps, :],
                                        known_combined_layer[:, :encoder_steps, :],
                                        obs_inputs[:, :encoder_steps, :]
                                        ], axis=-1)
        else:
            historical_inputs = concat([known_combined_layer[:, :encoder_steps, :],
                                        obs_inputs[:, :encoder_steps, :]
                                        ], axis=-1)

        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        static_encoder, static_weights = static_combine_and_mask(static_inputs,
                                                                 self.hidden_layer_size,
                                                                 self.dropout_rate)

        static_context_variable_selection = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_enrichment = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_state_h = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_state_c = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)

        historical_features, historical_flags, _ = temporal_selection(historical_inputs,
                                                                      static_context_variable_selection,
                                                                      self.hidden_layer_size,
                                                                      self.dropout_rate)
        # lstm_combine_and_mask(historical_inputs)
        future_features, future_flags, _ = temporal_selection(future_inputs,
                                                              static_context_variable_selection,
                                                              self.hidden_layer_size,
                                                              self.dropout_rate)

        history_lstm, state_h, state_c \
            = get_lstm(return_state=True,
                       hidden_layer_size=self.hidden_layer_size)(historical_features,
                                                                 initial_state=[
                                                                     static_context_state_h,
                                                                     static_context_state_c])

        future_lstm = get_lstm(return_state=False,
                               hidden_layer_size=self.hidden_layer_size)(
            future_features, initial_state=[state_h, state_c])

        lstm_layer = concat([history_lstm, future_lstm], axis=1)

        # Apply gated skip connection
        input_embeddings = concat([historical_features, future_features], axis=1)

        lstm_layer, _ = apply_gating_layer(
            lstm_layer, self.hidden_layer_size, self.dropout_rate, activation=None)
        temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings])

        # Static enrichment layers
        expanded_static_context = K.expand_dims(static_context_enrichment, axis=1)
        enriched, _ = gated_residual_network(
            temporal_feature_layer,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=expanded_static_context,
            return_gate=True)

        # Decoder self attention
        self_attn_layer = InterpretableMultiHeadAttention(
            self.num_heads, self.hidden_layer_size, dropout=self.dropout_rate)

        mask = get_decoder_mask(enriched)
        x, self_att = self_attn_layer(enriched, enriched, enriched, mask=mask)

        x, _ = apply_gating_layer(
            x,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            activation=None)
        x = add_and_norm([x, enriched])

        # Nonlinear processing on outputs
        decoder = gated_residual_network(
            x,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True)

        # Final skip connection
        decoder, _ = apply_gating_layer(
            decoder, self.hidden_layer_size, activation=None)
        transformer_layer = add_and_norm([decoder, temporal_feature_layer])

        # Attention components for explainability
        attention_components = {
            # Temporal attention weights
            'decoder_self_attn': self_att,
            # Static variable selection weights
            'static_flags': static_weights[Ellipsis, 0],
            # Variable selection weights of past inputs
            'historical_flags': historical_flags[Ellipsis, 0, :],
            # Variable selection weights of future inputs
            'future_flags': future_flags[Ellipsis, 0, :]
        }

        transformer_output = transformer_layer[Ellipsis, self.num_encoder_steps:, :]

        return transformer_output, attention_components

    def get_attention(self, df=None):
        """Computes TFT attention weights for a given dataset.

    Args:
      df: Input dataframe

    Returns:
        Dictionary of numpy arrays for temporal attention weights and variable
          selection weights, along with their identifiers and time indices
    """
        if df is None:
            print('Using cached data')
            data = DataCache.get('valid')
        else:
            data = self._batch_data(df)

        inputs = data['inputs']
        identifiers = data['identifier']
        time = data['time']

        def get_batch_attention_weights(input_batch):
            """Returns weights for a given minibatch of data."""
            # input_placeholder = self._input_placeholder
            # attention_weights = {}
            y_pred, self_att, static_weights, historical_flags, future_flags, trans_output = self.model(
                input_batch.astype(np.float32), training=False)
            attention_weights = {
                'decoder_self_attn': self_att,
                'static_flags': static_weights,  # static_weights[Ellipsis, 0],
                'historical_flags': historical_flags,  # historical_flags[Ellipsis, 0, :],
                'future_flags': future_flags  # future_flags[Ellipsis, 0, :]
            }
            return attention_weights

        # Compute number of batches
        batch_size = self.minibatch_size
        n = inputs.shape[0]
        num_batches = n // batch_size
        if n - (num_batches * batch_size) > 0:
            num_batches += 1

        # Split up inputs into batches
        batched_inputs = [
            inputs[i * batch_size:(i + 1) * batch_size, Ellipsis]
            for i in range(num_batches)
        ]

        # Get attention weights, while avoiding large memory increases
        print('Computing Attention Weights')
        attention_by_batch = [
            get_batch_attention_weights(batch) for batch in tqdm(batched_inputs)
        ]
        attention_weights = {}
        for k in self._attention_components:
            attention_weights[k] = []
            for batch_weights in attention_by_batch:
                attention_weights[k].append(batch_weights[k])

            if len(attention_weights[k][0].shape) == 4:
                tmp = np.concatenate(attention_weights[k], axis=1)
            else:
                tmp = np.concatenate(attention_weights[k], axis=0)

            del attention_weights[k]
            gc.collect()
            attention_weights[k] = tmp

        attention_weights['identifiers'] = identifiers[:, 0, 0]
        attention_weights['time'] = time[:, :, 0]

        return attention_weights

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


class LSTMModel(TSModel):

    def __init__(self, raw_params, verbose=2, tb_callback=True):
        """Builds TFT from parameters.

    Args:
      raw_params: Parameters to define MODEL
    """

        self.name = self.__class__.__name__

        params = dict(raw_params)  # copy locally

        self.dropout_rate = float(params['dropout_rate'])
        self.max_gradient_norm = float(params['max_gradient_norm'])
        self._prediction_parts = None
        self._attention_components = None

        super().__init__(raw_params, verbose=verbose, tb_callback=tb_callback)

    def _build_base_graph(self, all_inputs):
        """Returns graph defining layers of the TFT."""

        # # Size definitions.
        time_steps = self.time_steps
        encoder_steps = self.num_encoder_steps

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.get_embeddings(all_inputs)

        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            historical_inputs = concat([unknown_inputs[:, :encoder_steps, :],
                                        known_combined_layer[:, :encoder_steps, :],
                                        obs_inputs[:, :encoder_steps, :]
                                        ], axis=-1)
        else:
            historical_inputs = concat([known_combined_layer[:, :encoder_steps, :],
                                        obs_inputs[:, :encoder_steps, :]
                                        ], axis=-1)

        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        static_encoder, static_weights = static_combine_and_mask(static_inputs,
                                                                 self.hidden_layer_size,
                                                                 self.dropout_rate)

        static_context_variable_selection = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_enrichment = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_state_h = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_state_c = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)

        historical_features, historical_flags, _ = temporal_selection(historical_inputs,
                                                                      static_context_variable_selection,
                                                                      self.hidden_layer_size,
                                                                      self.dropout_rate)
        # lstm_combine_and_mask(historical_inputs)
        future_features, future_flags, _ = temporal_selection(future_inputs,
                                                              static_context_variable_selection,
                                                              self.hidden_layer_size,
                                                              self.dropout_rate)

        history_lstm, state_h, state_c \
            = get_lstm(return_state=True,
                       hidden_layer_size=self.hidden_layer_size)(historical_features,
                                                                 initial_state=[
                                                                     static_context_state_h,
                                                                     static_context_state_c])

        future_lstm = get_lstm(return_state=False,
                               hidden_layer_size=self.hidden_layer_size)(
            future_features, initial_state=[state_h, state_c])

        lstm_layer = concat([history_lstm, future_lstm], axis=1)

        # Apply gated skip connection
        input_embeddings = concat([historical_features, future_features], axis=1)

        lstm_layer, _ = apply_gating_layer(
            lstm_layer, self.hidden_layer_size, self.dropout_rate, activation=None)
        temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings])

        # Static enrichment layers
        expanded_static_context = K.expand_dims(static_context_enrichment, axis=1)
        enriched, _ = gated_residual_network(
            temporal_feature_layer,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=expanded_static_context,
            return_gate=True)

        # Attention components for explainability
        attention_components = {
            # Static variable selection weights
            'static_flags': static_weights[Ellipsis, 0],
            # Variable selection weights of past inputs
            'historical_flags': historical_flags[Ellipsis, 0, :],
            # Variable selection weights of future inputs
            'future_flags': future_flags[Ellipsis, 0, :]
        }

        output = enriched[Ellipsis, self.num_encoder_steps:, :]

        return output, attention_components


class DCNNModel(TSModel):

    def __init__(self, raw_params, verbose=2, tb_callback=True):
        """Builds TFT from parameters.

    Args:
      raw_params: Parameters to define TFT
    """

        self.name = self.__class__.__name__

        params = dict(raw_params)  # copy locally

        self.dropout_rate = float(params['dropout_rate'])
        self.n_layers = int(params['n_layers'])
        self.reg = params['reg']
        self.n_kernel = int(params['n_kernel'])
        self.n_filters = int(params['hidden_layer_size'])

        self._prediction_parts = None
        self._attention_components = None

        assert params['num_encoder_steps'] > 2 ** (self.n_layers - 1) * self.n_kernel
        self.stddev = math.sqrt(2 / (self.n_kernel * self.n_filters))

        super().__init__(raw_params, verbose=verbose, tb_callback=tb_callback)

    def _build_base_graph(self, all_inputs):
        """Returns graph defining layers of the TFT."""

        # # Size definitions.
        encoder_steps = self.num_encoder_steps

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.get_embeddings(all_inputs)

        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            historical_inputs = concat([unknown_inputs[:, :encoder_steps, :],
                                        known_combined_layer[:, :encoder_steps, :],
                                        obs_inputs[:, :encoder_steps, :]
                                        ], axis=-1)
        else:
            historical_inputs = concat([known_combined_layer[:, :encoder_steps, :],
                                        obs_inputs[:, :encoder_steps, :]
                                        ], axis=-1)

        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        static_encoder, static_weights = static_combine_and_mask(static_inputs,
                                                                 self.hidden_layer_size,
                                                                 self.dropout_rate)

        static_context_variable_selection = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_enrichment = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)

        historical_features, historical_flags, _ = temporal_selection(historical_inputs,
                                                                      static_context_variable_selection,
                                                                      self.hidden_layer_size,
                                                                      self.dropout_rate)
        # lstm_combine_and_mask(historical_inputs)
        future_features, future_flags, _ = temporal_selection(future_inputs,
                                                              static_context_variable_selection,
                                                              self.hidden_layer_size,
                                                              self.dropout_rate)

        input_embeddings = concat([historical_features, future_features], axis=1)

        # ARCHITECTURE
        n_filters = self.n_filters
        n_kernel = self.n_kernel
        n_layers = self.n_layers
        reg = self.reg
        stddev = self.stddev


        x = dcnn_1st_layer(n_filters, n_kernel, 1, None, reg=reg)(input_embeddings)
        for layer in range(1, n_layers):
            x = dcnn_layer(n_filters, n_kernel, 2 ** layer, None, reg=reg)(x)
        out_conv = keras.layers.Conv1D(n_filters, 1,
                                       padding='same', use_bias=True,
                                       activation='relu')(x)

        # Apply gated skip connection
        dcnn_out, _ = apply_gating_layer(out_conv,
                                         self.hidden_layer_size,
                                         self.dropout_rate,
                                         activation=None)

        temporal_feature_layer = add_and_norm([dcnn_out, input_embeddings])

        # Static enrichment layers
        expanded_static_context = K.expand_dims(static_context_enrichment, axis=1)
        enriched, _ = gated_residual_network(
            temporal_feature_layer,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=expanded_static_context,
            return_gate=True)

        # Attention components for explainability
        attention_components = {
            # Static variable selection weights
            'static_flags': static_weights[Ellipsis, 0],
            # Variable selection weights of past inputs
            'historical_flags': historical_flags[Ellipsis, 0, :],
            # Variable selection weights of future inputs
            'future_flags': future_flags[Ellipsis, 0, :]
        }

        output = enriched[Ellipsis, self.num_encoder_steps:, :]

        return output, attention_components
