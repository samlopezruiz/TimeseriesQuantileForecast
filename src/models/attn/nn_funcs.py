import tensorflow as tf

from src.models.attn.utils import tensorflow_quantile_loss

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


# Layer definitions.
# LSTM layer
def get_lstm(return_state, hidden_layer_size):
    """Returns LSTM cell initialized with default parameters."""
    lstm = tf.keras.layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        return_state=return_state,
        stateful=False,
        # Additional params to ensure LSTM matches CuDNN, See TF 2.0 :
        # (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
        activation='tanh',
        recurrent_activation='sigmoid',
        recurrent_dropout=0,
        unroll=False,
        use_bias=True)
    return lstm

# Layer utility functions.
def linear_layer(size,
                 activation=None,
                 use_time_distributed=False,
                 use_bias=True):
    """Returns simple Keras linear layer.

  Args:
    size: Output size
    activation: Activation function to apply if required
    use_time_distributed: Whether to apply layer across time
    use_bias: Whether bias should be included in layer
  """
    linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear


def apply_mlp(inputs,
              hidden_size,
              output_size,
              output_activation=None,
              hidden_activation='tanh',
              use_time_distributed=False):
    """Applies simple feed-forward network to an input.

  Args:
    inputs: MLP inputs
    hidden_size: Hidden state size
    output_size: Output size of MLP
    output_activation: Activation function to apply on output
    hidden_activation: Activation function to apply on input
    use_time_distributed: Whether to apply across time

  Returns:
    Tensor for MLP outputs.
  """
    if use_time_distributed:
        hidden = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_size, activation=hidden_activation))(
            inputs)
        return tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(output_size, activation=output_activation))(
            hidden)
    else:
        hidden = tf.keras.layers.Dense(
            hidden_size, activation=hidden_activation)(
            inputs)
        return tf.keras.layers.Dense(
            output_size, activation=output_activation)(
            hidden)


def apply_gating_layer(x,
                       hidden_layer_size,
                       dropout_rate=None,
                       use_time_distributed=True,
                       activation=None):
    """Applies a Gated Linear Unit (GLU) to an input.

  Args:
    x: Input to gating layer
    hidden_layer_size: Dimension of GLU
    dropout_rate: Dropout rate to apply if any
    use_time_distributed: Whether to apply across time
    activation: Activation function to apply to the linear feature transform if
      necessary

  Returns:
    Tuple of tensors for: (GLU output, gate)
  """

    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation=activation))(x)
        gated_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(x)
    else:
        activation_layer = tf.keras.layers.Dense(
            hidden_layer_size, activation=activation)(x)
        gated_layer = tf.keras.layers.Dense(
            hidden_layer_size, activation='sigmoid')(x)

    return tf.keras.layers.Multiply()([activation_layer, gated_layer]), gated_layer


def add_and_norm(x_list):
    """Applies skip connection followed by layer normalisation.

  Args:
    x_list: List of inputs to sum for skip connection

  Returns:
    Tensor output from layer.
  """
    tmp = Add()(x_list)
    tmp = LayerNorm()(tmp)
    return tmp


def gated_residual_network(x,
                           hidden_layer_size,
                           output_size=None,
                           dropout_rate=None,
                           use_time_distributed=True,
                           additional_context=None,
                           return_gate=False):
    """Applies the gated residual network (GRN) as defined in paper.

  Args:
    x: Network inputs
    hidden_layer_size: Internal state size
    output_size: Size of output layer
    dropout_rate: Dropout rate if dropout is applied
    use_time_distributed: Whether to apply network across time dimension
    additional_context: Additional context vector to use if relevant
    return_gate: Whether to return GLU gate for diagnostic purposes

  Returns:
    Tuple of tensors for: (GRN output, GLU gate)
  """

    # Setup skip connection
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = Dense(output_size)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        skip = linear(x)

    # Apply feedforward network
    hidden = linear_layer(hidden_layer_size,
                          activation=None,
                          use_time_distributed=use_time_distributed)(x)
    if additional_context is not None:
        hidden = hidden + linear_layer(hidden_layer_size,
                                       activation=None,
                                       use_time_distributed=use_time_distributed,
                                       use_bias=False)(additional_context)
    hidden = tf.keras.layers.Activation('elu')(hidden)
    hidden = linear_layer(hidden_layer_size,
                          activation=None,
                          use_time_distributed=use_time_distributed)(hidden)

    gating_layer, gate = apply_gating_layer(hidden,
                                            output_size,
                                            dropout_rate=dropout_rate,
                                            use_time_distributed=use_time_distributed,
                                            activation=None)

    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])


# Attention Components.
def get_decoder_mask(self_attn_inputs):
    """Returns causal mask to apply for self-attention layer.

  Args:
    self_attn_inputs: Inputs to self attention layer to determine mask shape
  """
    len_s = tf.shape(input=self_attn_inputs)[1]
    bs = tf.shape(input=self_attn_inputs)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask

class ScaledDotProductAttention():
    """Defines scaled dot product attention layer.

  Attributes:
    dropout: Dropout rate to use
    activation: Normalisation function for scaled dot product attention (e.g.
      softmax by default)
  """

    def __init__(self, attn_dropout=0.0):
        self.dropout = Dropout(attn_dropout)
        self.activation = Activation('softmax')

    def __call__(self, q, k, v, mask):
        """Applies scaled dot product attention.

    Args:
      q: Queries
      k: Keys
      v: Values
      mask: Masking if required -- sets softmax to very large value

    Returns:
      Tuple of (layer outputs, attention weights)
    """
        temper = tf.sqrt(tf.cast(tf.shape(input=k)[-1], dtype='float32'))
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([q, k])  # shape=(batch, q, k)
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(mask)  # setting to infinity
            attn = Add()([attn, mmask])
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class InterpretableMultiHeadAttention():
    """Defines interpretable multi-head attention layer.

  Attributes:
    n_head: Number of heads
    d_k: Key/query dimensionality per head
    d_v: Value dimensionality
    dropout: Dropout rate to apply
    qs_layers: List of queries across heads
    ks_layers: List of keys across heads
    vs_layers: List of values across heads
    attention: Scaled dot product attention layer
    w_o: Output weight matrix to project internal state to the original TFT
      state size
  """

    def __init__(self, n_head, d_model, dropout):
        """Initialises layer.

    Args:
      n_head: Number of heads
      d_model: TFT state dimensionality
      dropout: Dropout discard rate
    """

        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        # Use same value layer to facilitate interp
        vs_layer = Dense(d_v, use_bias=False)

        for _ in range(n_head):
            self.qs_layers.append(Dense(d_k, use_bias=False))
            self.ks_layers.append(Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)  # use same vs_layer

        self.attention = Attention(causal=True)  # ScaledDotProductAttention()
        self.w_o = Dense(d_model, use_bias=False)

    def __call__(self, q, k, v, mask=None):
        """Applies interpretable multihead attention.

    Using T to denote the number of time steps fed into the transformer.

    Args:
      q: Query tensor of shape=(?, T, d_model)
      k: Key of shape=(?, T, d_model)
      v: Values of shape=(?, T, d_model)
      mask: Masking if required with shape=(?, T, T)

    Returns:
      Tuple of (layer outputs, attention weights)
    """
        n_head = self.n_head

        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            # head, attn = self.attention(qs, ks, vs, mask)
            head, attn = self.attention([qs, ks, vs], return_attention_scores=True)

            head_dropout = Dropout(self.dropout)(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = K.stack(heads) if n_head > 1 else heads[0]
        attn = K.stack(attns)

        outputs = K.mean(head, axis=0) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = Dropout(self.dropout)(outputs)  # output dropout

        return outputs, attn


def static_combine_and_mask(embedding,
                            hidden_layer_size,
                            dropout_rate):
    """Applies variable selection network to static inputs.

  Args:
    embedding: Transformed static inputs

  Returns:
    Tensor output for variable selection network
  """

    # Add temporal features
    _, num_static, _ = embedding.get_shape().as_list()

    flatten = tf.keras.layers.Flatten()(embedding)

    # Nonlinear transformation with gated residual network.
    mlp_outputs = gated_residual_network(
        flatten,
        hidden_layer_size,
        output_size=num_static,
        dropout_rate=dropout_rate,
        use_time_distributed=False,
        additional_context=None)

    sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)
    sparse_weights = K.expand_dims(sparse_weights, axis=-1)

    trans_emb_list = []
    for i in range(num_static):
        e = gated_residual_network(
            embedding[:, i:i + 1, :],
            hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=False)
        trans_emb_list.append(e)

    transformed_embedding = concat(trans_emb_list, axis=1)

    combined = tf.keras.layers.Multiply()([sparse_weights, transformed_embedding])

    static_vec = K.sum(combined, axis=1)

    return static_vec, sparse_weights


def temporal_selection(embedding,
                       static_context_variable_selection,
                       hidden_layer_size,
                       dropout_rate):
    """Apply temporal variable selection networks.

  Args:
    embedding: Transformed inputs.

  Returns:
    Processed tensor outputs.
  """

    # Add temporal features
    _, time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()

    flatten = K.reshape(embedding, [-1, time_steps, embedding_dim * num_inputs])

    expanded_static_context = K.expand_dims(static_context_variable_selection, axis=1)

    # Variable selection weights
    mlp_outputs, static_gate = gated_residual_network(
        flatten,
        hidden_layer_size,
        output_size=num_inputs,
        dropout_rate=dropout_rate,
        use_time_distributed=True,
        additional_context=expanded_static_context,
        return_gate=True)

    sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)
    sparse_weights = tf.expand_dims(sparse_weights, axis=2)

    # Non-linear Processing & weight application
    trans_emb_list = []
    for i in range(num_inputs):
        grn_output = gated_residual_network(
            embedding[Ellipsis, i],
            hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=True)
        trans_emb_list.append(grn_output)

    transformed_embedding = stack(trans_emb_list, axis=-1)

    combined = tf.keras.layers.Multiply()([sparse_weights, transformed_embedding])
    temporal_ctx = K.sum(combined, axis=-1)

    return temporal_ctx, sparse_weights, static_gate


class DataCache(object):
    """Caches data for the TFT."""

    _data_cache = {}

    @classmethod
    def update(cls, data, key):
        """Updates cached data.

    Args:
      data: Source to update
      key: Key to dictionary location
    """
        cls._data_cache[key] = data

    @classmethod
    def get(cls, key):
        """Returns data stored at key location."""
        return cls._data_cache[key].copy()

    @classmethod
    def contains(cls, key):
        """Retuns boolean indicating whether key is present in cache."""

        return key in cls._data_cache



class QuantileLossCalculator(object):
    """Computes the combined quantile loss for prespecified quantiles.

Attributes:
quantiles: Quantiles to compute losses
"""

    def __init__(self, quantiles, output_size):
        """Initializes computer with quantiles for loss calculations.

Args:
quantiles: Quantiles to use for computations.
"""
        self.quantiles = quantiles
        self.output_size = output_size

    def quantile_loss(self, a, b):
        """Returns quantile loss for specified quantiles.

Args:
a: Targets
b: Predictions
"""
        quantiles_used = set(self.quantiles)

        loss = 0.
        for i, quantile in enumerate(self.quantiles):
            if quantile in quantiles_used:
                loss += tensorflow_quantile_loss(
                    a[Ellipsis, self.output_size * i:self.output_size * (i + 1)],
                    b[Ellipsis, self.output_size * i:self.output_size * (i + 1)], quantile)
        return loss

