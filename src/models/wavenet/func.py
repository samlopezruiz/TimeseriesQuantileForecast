import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras import initializers
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def dcnn_layer(channels, kernel_size, dilation_rate, name, reg='l2'):
    stddev = math.sqrt(2 / (kernel_size * channels))
    filter_name = 'filter_' + name if name is not None else None
    resid_name = 'resid_' + name if name is not None else None
    def f(input_):
        filter_out = keras.layers.Conv1D(channels, kernel_size,
                                         strides=1, dilation_rate=dilation_rate,
                                         kernel_initializer=initializers.RandomNormal(stddev=stddev),
                                         padding='valid', use_bias=True, kernel_regularizer=reg,
                                         activation='relu', name=filter_name)(input_)
        filter_padded = keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0))(filter_out)
        output = keras.layers.Add(name=resid_name)([filter_padded, input_])
        return output
    return f


def dcnn_1st_layer(channels, kernel_size, dilation_rate, name, reg='l2'):
    stddev = math.sqrt(2 / (kernel_size * channels))
    conv1_name = 'filter_' + name if name is not None else None
    skip_name = 'param_skip_' + name if name is not None else None
    add_name = 'add_' + name if name is not None else None
    def f(input_):
        filter_out = keras.layers.Conv1D(channels, kernel_size,
                                         strides=1, dilation_rate=dilation_rate,
                                         padding='valid', use_bias=True, kernel_regularizer=reg,
                                         kernel_initializer=initializers.RandomNormal(stddev=stddev),
                                         activation='relu', name=conv1_name)(input_)
        filter_padded = keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0))(filter_out)
        param_skip = keras.layers.Conv1D(channels, 1,
                                         padding='same', use_bias=False,
                                         activation='linear', name=skip_name)(input_)
        output = keras.layers.Add(name=add_name)([filter_padded, param_skip])
        return output
    return f


def dcnn_build(cfg, n_features):
    n_steps_in, n_steps_out, n_filters = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_epochs, n_batch = cfg['n_kernel'], cfg['n_epochs'], cfg['n_batch']
    n_layers, reg = cfg['n_layers'], cfg['reg']
    input_shape = (n_steps_in, n_features)

    assert n_steps_in > 2 ** (n_layers - 1) * n_kernel

    stddev = math.sqrt(2 / (n_kernel * n_filters))

    # ARCHITECTURE
    sequence = keras.layers.Input(shape=input_shape, name='sequence')
    x = dcnn_1st_layer(n_filters, n_kernel, 1, '0', reg=reg)(sequence)
    for layer in range(1, n_layers):
        x = dcnn_layer(n_filters, n_kernel, 2**layer, str(layer), reg=reg)(x)
    out_conv = keras.layers.Conv1D(n_filters, 1,
                                    padding='same', use_bias=True,
                                    activation='relu', name='conv1x1')(x)
    preoutput = keras.layers.Flatten()(out_conv)
    preoutput = keras.layers.Dense(n_steps_out*2, kernel_regularizer=reg, name='preoutput',
                                   kernel_initializer=initializers.RandomNormal(stddev=stddev))(preoutput)
    output = keras.layers.Dense(n_steps_out, kernel_regularizer=reg, name='output',
                                   kernel_initializer=initializers.RandomNormal(stddev=stddev))(preoutput)

    return keras.models.Model(inputs=[sequence], outputs=output)

def dcnn_build2(cfg, n_features):
    n_steps_in, n_steps_out, n_filters = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_epochs, n_batch = cfg['n_kernel'], cfg['n_epochs'], cfg['n_batch']
    n_layers, reg = cfg['n_layers'], cfg['reg']
    input_shape = (n_steps_in, n_features)

    assert n_steps_in > 2 ** (n_layers - 1) * n_kernel

    stddev = math.sqrt(2 / (n_kernel * n_filters))

    # ARCHITECTURE
    sequence = keras.layers.Input(shape=input_shape, name='sequence')
    x = dcnn_1st_layer(n_filters, n_kernel, 1, '0', reg=reg)(sequence)
    for layer in range(1, n_layers):
        x = dcnn_layer(n_filters, n_kernel, 2**layer, str(layer), reg=reg)(x)
    out_conv = keras.layers.Conv1D(n_filters, 1,
                                    padding='same', use_bias=True,
                                    activation='relu', name='conv1x1')(x)
    # out_conv = keras.layers.Conv1D(n_filters // 2, 1,
    #                                padding='same', use_bias=True,
    #                                activation='relu', name='conv1x1')(out_conv)
    preoutput = keras.layers.Flatten()(out_conv)
    preoutput = keras.layers.Dense(n_steps_out, kernel_regularizer=reg, name='preoutput',
                                   kernel_initializer=initializers.RandomNormal(stddev=stddev))(preoutput)
    output = keras.layers.Dense(n_steps_out, kernel_regularizer=reg, name='output',
                                   kernel_initializer=initializers.RandomNormal(stddev=stddev))(preoutput)

    return keras.models.Model(inputs=[sequence], outputs=output)


def dcnn_build_cond(cfg, n_features):
    n_steps_in, n_steps_out, n_filters = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_epochs, n_batch = cfg['n_kernel'], cfg['n_epochs'], cfg['n_batch']
    n_layers, reg = cfg['n_layers'], cfg['reg']
    # input_shape = (n_steps_in, n_features)
    input_shape = (n_steps_in, 1)
    assert n_steps_in > 2 ** (n_layers - 1) * n_kernel

    stddev = math.sqrt(2 / (n_kernel * n_filters))

    # ARCHITECTURE
    sequence = keras.layers.Input(shape=input_shape, name='sequence')
    in_sequence = dcnn_1st_layer(n_filters, n_kernel, 1, '0', reg=reg)(sequence)
    conditions = [keras.layers.Input(shape=input_shape, name='condition'+str(i)) for i in range(n_features-1)]
    in_conditions = [dcnn_1st_layer(n_filters, n_kernel, 1, 'cond'+str(i), reg=reg)(cond) for i, cond in enumerate(conditions)]

    if n_features > 1:
        x = keras.layers.Add(name='input_layer')([in_sequence] + in_conditions)
    else:
        x = in_sequence

    for layer in range(1, n_layers):
        x = dcnn_layer(n_filters, n_kernel, 2**layer, str(layer), reg=reg)(x)
    out_conv = keras.layers.Conv1D(n_filters, 1,
                                    padding='same', use_bias=True,
                                    activation='relu', name='conv1x1')(x)
    preoutput = keras.layers.Flatten()(out_conv)
    preoutput = keras.layers.Dense(n_steps_out*2, kernel_regularizer=reg, name='preoutput',
                                   kernel_initializer=initializers.RandomNormal(stddev=stddev))(preoutput)
    output = keras.layers.Dense(n_steps_out, kernel_regularizer=reg, name='output',
                                   kernel_initializer=initializers.RandomNormal(stddev=stddev))(preoutput)

    return keras.models.Model(inputs=[sequence] + conditions, outputs=output)


def wavenet_layer(channels, hidden_channels, kernel_size, dilation_rate, name):
    def f(input_):
        filter_out = keras.layers.Conv1D(hidden_channels, kernel_size,
                                         strides=1, dilation_rate=dilation_rate,
                                         padding='valid', use_bias=True,
                                         activation='tanh', name='filter_' + name)(input_)
        gate_out = keras.layers.Conv1D(hidden_channels, kernel_size,
                                       strides=1, dilation_rate=dilation_rate,
                                       padding='valid', use_bias=True,
                                       activation='sigmoid', name='gate_' + name)(input_)
        mult = keras.layers.Multiply(name='mult_' + name)([filter_out, gate_out])

        mult_padded = keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0))(mult)

        transformed = keras.layers.Conv1D(channels, 1,
                                          padding='same', use_bias=True,
                                          activation='linear', name='trans_' + name)(mult_padded)
        skip_out = keras.layers.Conv1D(channels, 1,
                                       padding='same', use_bias=True,
                                       activation='relu', name='skip_' + name)(mult_padded)
        output = keras.layers.Add(name='resid_' + name)([transformed, input_])
        return output, skip_out
    return f


def wavenet_build(cfg, n_features):
    n_steps_in, n_steps_out, n_filters = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    hidden_channels, n_layers = cfg['hidden_channels'], cfg['n_layers']
    n_kernel, n_epochs, n_batch = cfg['n_kernel'], cfg['n_epochs'], cfg['n_batch']
    input_shape = (n_steps_in, n_features)

    assert n_steps_in > 2**(n_layers - 1) * n_kernel

    sequence = keras.layers.Input(shape=input_shape, name='sequence')

    x = s0 = keras.layers.Conv1D(n_filters, 1,
                                 padding='same', use_bias=True,
                                 activation='linear', name='input_expanded')(sequence)

    skip_layers = []
    for layer in range(n_layers):
        x, s = wavenet_layer(n_filters, hidden_channels, n_kernel, 2**layer, str(layer))(x)
        skip_layers.append(s)
    skip_overall = keras.layers.Add(name='skip_overall')([s0]+skip_layers)

    skip_act = keras.activations.relu(skip_overall)
    skip_conv = keras.layers.Conv1D(n_filters, 1,
                                    padding='same', use_bias=True,
                                    activation='relu', name='conv1x1')(skip_act)
    preoutput = keras.layers.Flatten()(skip_conv)
    preoutput = keras.layers.Dense(n_steps_out, name='preoutput')(preoutput)
    output = keras.layers.Dense(n_steps_out, name='output')(preoutput)

    return keras.models.Model(inputs=[sequence], outputs=output)


def wavenet_build2(cfg, n_features):
    n_steps_in, n_steps_out, n_filters = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    hidden_channels, n_layers = cfg['hidden_channels'], cfg['n_layers']
    n_kernel, n_epochs, n_batch = cfg['n_kernel'], cfg['n_epochs'], cfg['n_batch']
    input_shape = (n_steps_in, n_features)

    assert n_steps_in > 2**(n_layers - 1) * n_kernel

    sequence = keras.layers.Input(shape=input_shape, name='sequence')

    x = s0 = keras.layers.Conv1D(n_filters, 1,
                                 padding='same', use_bias=True,
                                 activation='linear', name='input_expanded')(sequence)

    skip_layers = []
    for layer in range(n_layers):
        x, s = wavenet_layer(n_filters, hidden_channels, n_kernel, 2**layer, str(layer))(x)
        skip_layers.append(s)
    skip_overall = keras.layers.Add(name='skip_overall')([s0]+skip_layers)

    skip_act = keras.activations.relu(skip_overall)
    skip_conv = keras.layers.Conv1D(n_filters, 1,
                                    padding='same', use_bias=True,
                                    activation='relu', name='conv1x1')(skip_act)
    preoutput = keras.layers.Flatten()(skip_conv)
    preoutput = keras.layers.Dense(n_steps_out, name='preoutput')(preoutput)
    output = keras.layers.Dense(n_steps_out, name='output')(preoutput)

    return keras.models.Model(inputs=[sequence], outputs=output)


def get_wnet_steps_cfgs(l_range, k_range):
    cfgs_steps_in = []
    for l in range(*l_range):
        for k in range(*k_range):
            cfgs_steps_in.append((2**(l-1)*k+1, l, k))
    x = np.array(cfgs_steps_in)
    return {'n_steps_in': x[:, 0].tolist(), 'n_layers': x[:, 1].tolist(), "n_kernel": x[:, 2].tolist()}
