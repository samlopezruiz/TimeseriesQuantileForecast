import copy
import datetime as dte
import multiprocessing
import os
from contextlib import contextmanager

import tensorflow as tf

from src.models.attn import utils
from src.models.attn.hyperparam_opt import HyperparamOptManager
from src.models.attn.models import TFTModel, LSTMModel, DCNNModel
from src.models.attn.utils import extract_numerical_data
from src.timeseries.data_formatter.configs import ExperimentConfig
from src.timeseries.utils.config import read_config
from src.timeseries.utils.moo import get_last_layer_weights


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.pool.ThreadPool(*args, **kwargs)
    yield pool
    pool.terminate()


'''
EXAMPLE CODE TO USE MULTIPROCESSING
partial_model_pred = partial(model_pred, model, model_cfg, model_func, model_n_steps_out, ss,
                             test_x_pp, training_cfg, unscaled_test_y, use_regimes, verbose)
with poolcontext(processes=cpu_count()) as pool:
    result = pool.map(partial_model_pred, range(len(test_x_pp)))
'''


def get_model_data_config(project, experiment_cfg, model_params, fixed_params):
    config = ExperimentConfig(project, experiment_cfg)
    formatter = config.make_data_formatter()
    formatter.update_model_params(model_params)
    formatter.update_fixed_params(fixed_params)
    model_folder = os.path.join(config.model_folder, experiment_cfg['experiment_name'])

    return config, formatter, model_folder


def train_test_model(use_gpu,
                     architecture,
                     prefetch_data,
                     model_folder,
                     data_config,
                     data_formatter,
                     use_testing_mode=False,
                     predict_eval=True,
                     tb_callback=True,
                     use_best_params=False,
                     indicators_use_time_subset=False,
                     split_data=None,
                     n_train_samples=None
                     ):
    """Trains tft based on defined model params.
  Args:
      :param split_data:
      :param indicators_use_time_subset:
      :param architecture:
      :param use_best_params:
      :param tb_callback:
      :param predict_eval:
      :param use_gpu: Whether to run tensorflow with GPU operations
      :param use_testing_mode: Uses a smaller models and data sizes for testing purposes
      :param data_formatter: Dataset-specific data fromatter
      :param data_config: Data input file configurations
      :param model_folder: Folder path where models are serialized
      :param prefetch_data: Prefetch data for training
  """

    Model = get_model(architecture)

    print("Loading & splitting data...")
    if split_data is None:
        train, valid, test = data_formatter.split_data(data_config,
                                                       indicators_use_time_subset=indicators_use_time_subset)
    else:
        train, valid, test = split_data

    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    print('shape: {}'.format(train.shape))

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Parameter overrides for testing only! Small sizes used to speed up script.
    if use_testing_mode:
        fixed_params["num_epochs"] = 1
        params["hidden_layer_size"] = 5
        train_samples, valid_samples = 100, 10

    if n_train_samples is not None:
        train_samples, valid_samples = n_train_samples, n_train_samples // 10

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)

    # Training -- one iteration only
    print("*** Running calibration ***")
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    # best_loss = np.Inf

    with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):

        params = opt_manager.get_next_parameters()
        # model = TemporalFusionTransformer(params, use_cudnn=use_gpu, tb_callback=tb_callback)
        model = Model(params, tb_callback=tb_callback)

        if not model.training_data_cached():
            model.cache_batched_data(train, "train", num_samples=train_samples)
            model.cache_batched_data(valid, "valid", num_samples=valid_samples)

        model.fit(prefetch_data)
        fit_history = copy.deepcopy(model.fit_history)

        val_loss = model.evaluate()

        # if val_loss < best_loss:
        opt_manager.update_score(params, val_loss, model)
            # best_loss = val_loss

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Validation loss = {}".format(val_loss))

    if predict_eval:
        if use_best_params:
            print("*** Running tests ***")
            params = opt_manager.get_best_params()
            model = Model(params, tb_callback=tb_callback)
            # model = TemporalFusionTransformer(params, use_cudnn=use_gpu)

            model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

            print("Computing best validation loss")
            val_loss = model.evaluate(valid)
            print("Best Validation loss = {}".format(val_loss))

        print("Training completed @ {}".format(dte.datetime.now()))
        print("Best validation loss = {}".format(val_loss))

        return predict_from_model(params, data_formatter, model, test, val_loss, fit_history)
    else:
        return val_loss


#
def predict_from_model(best_params, data_formatter, model, test, val_loss, fit_history):
    print("Computing test loss")
    output_map = model.predict(test, return_targets=True)
    unscaled_output_map = {}
    for k, df in output_map.items():
        unscaled_output_map[k] = data_formatter.format_predictions(df)

    losses = {}
    weighted_errors, eq_weighted_errors = {}, {}
    targets = unscaled_output_map['targets']
    for q in model.quantiles:
        key = 'p{}'.format(int(q * 100))
        losses[key + '_loss'] = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(unscaled_output_map[key]), q)
        weighted_errors[key] = utils.numpy_normalised_weighted_errors(
            extract_numerical_data(targets), extract_numerical_data(unscaled_output_map[key]), q)
        eq_weighted_errors[key] = utils.numpy_normalised_weighted_errors(
            extract_numerical_data(targets), extract_numerical_data(unscaled_output_map[key]), 0.5)

        weighted_errors[key]['forecast_time'] = unscaled_output_map[key]['forecast_time']
        weighted_errors[key]['identifier'] = unscaled_output_map[key]['identifier']
        eq_weighted_errors[key]['forecast_time'] = unscaled_output_map[key]['forecast_time']
        eq_weighted_errors[key]['identifier'] = unscaled_output_map[key]['identifier']

    print("Params:")
    for k in best_params:
        print(k, " = ", best_params[k])

    test_loss = [p_loss.mean() for k, p_loss in losses.items()]
    print("\nNormalised Quantile Losses for Test Data: {}".format(test_loss))

    results = {'quantiles': model.quantiles,
               'forecasts': unscaled_output_map,
               'weighted_errors': weighted_errors,
               'eq_weighted_errors': eq_weighted_errors,
               'val_loss': val_loss,
               'test_loss': test_loss,
               'losses': losses,
               'learning_rate': model.learning_rate,
               'fit_history': fit_history,
               'target': data_formatter.test_true_y.columns[0] if data_formatter.test_true_y is not None else None,
               'fixed_params': data_formatter.get_fixed_params(),
               'model_params': data_formatter.get_default_model_params()}

    return results


def load_predict_model(use_gpu,
                       architecture,
                       model_folder,
                       data_config,
                       data_formatter,
                       use_all_data=False,
                       last_layer_weights=None):
    """Trains tft based on defined model params.
  Args:
      :param use_all_data:
      :param use_gpu: Whether to run tensorflow with GPU operations
      :param use_testing_mode: Uses a smaller models and data sizes for testing purposes
      :param data_formatter: Dataset-specific data fromatter
      :param data_config: Data input file configurations
      :param model_folder: Folder path where models are serialized

  """
    Model = get_model(architecture)

    print("Loading & splitting data...")
    if use_all_data:
        test = data_formatter.process_data(data_config)
    else:
        train, valid, test = data_formatter.split_data(data_config)

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)

    print("*** Running tests ***")
    params = opt_manager.get_next_parameters()

    with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):

        model = Model(params)
        model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

        # manually set last layer weights (used in multi-objective optimization)
        if last_layer_weights is not None:
            weights, last_layer = get_last_layer_weights(model)
            last_layer.set_weights(last_layer_weights)

        return predict_from_model(params, data_formatter, model, test, None, None), test


def get_model(architecture):
    architecture_options = ['TFTModel', 'LSTMModel', 'DCNNModel']
    if architecture not in architecture_options:
        raise Exception('{} not a valid option. \nOptions: {}'.format(architecture, architecture_options))

    if architecture == 'TFTModel':
        model = TFTModel
    elif architecture == 'LSTMModel':
        model = LSTMModel
    elif architecture == 'DCNNModel':
        model = DCNNModel

    return model


def get_attention_model(use_gpu,
                        architecture,
                        model_folder,
                        data_config,
                        data_formatter,
                        get_attentions=False,
                        samples=None):
    """Trains tft based on defined model params.

  Args:
    expt_name: Name of experiment
    use_gpu: Whether to run tensorflow with GPU operations
    model_folder: Folder path where models are serialized
    data_csv_path: Path to csv file containing data
    data_formatter: Dataset-specific data fromatter (see
      expt_settings.dataformatter.GenericDataFormatter)
    use_testing_mode: Uses a smaller models and data sizes for testing purposes
      only -- switch to False to use original default settings
  """
    Model = get_model(architecture)

    print("Loading & splitting data...")
    train, valid, test = data_formatter.split_data(data_config)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    if samples is not None:
        valid_samples = samples

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)

    params = opt_manager.get_next_parameters()
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):
        model = Model(params)
        model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

        if not model.training_data_cached():
            model.cache_batched_data(valid, "valid", num_samples=valid_samples)

        attentions = model.get_attention() if get_attentions else None

    results = {'attentions': attentions,
               'params': params}
    return results
