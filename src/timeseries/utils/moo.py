import copy
import datetime

import numpy as np
import pandas as pd
from deap.tools._hypervolume import hv
from pymoo.factory import get_performance_indicator

from src.models.attn import utils
from src.models.attn.utils import extract_numerical_data
from src.timeseries.utils.util import array_from_lists

import numpy as np
# from pymcdm import methods as mcdm_methods
# from pymcdm import weights as mcdm_weights
# from pymcdm.helpers import rankdata


def get_moo_args(params):
    return (params['name'],
            params['algo_cfg'],
            params['prob_cfg'],
            params.get('problem', None),
            params.get('plot', False),
            params.get('file_path', ['img', 'res']),
            params.get('save_plots', False),
            params.get('verbose', 1),
            params.get('seed', None))

def get_hv_hist_vs_n_evals(algos_runs, algos_hv_hist_runs):
    algos_n_evals_runs = [[gen.evaluator.n_eval for gen in algo_runs[0]['result']['res'].history]
                          for algo_runs in algos_runs]
    mean_hv_hist = [np.mean(hv_hist_runs, axis=0) for hv_hist_runs in algos_hv_hist_runs]
    series = [pd.DataFrame(hv, index=n_eval) for hv, n_eval in zip(mean_hv_hist, algos_n_evals_runs)]
    df2 = pd.concat(series, ignore_index=False, join='outer', axis=1)
    return df2.values.T

def loss_wo_middle_row(loss):
    return np.array([loss[0, 0], loss[0, 1], loss[2, 0], loss[2, 1]])


def get_loss_to_obj_function(type_func):
    if type_func == 'mean_across_quantiles':
        return lambda x: np.mean(x, axis=0)
    elif type_func == 'ind_loss_woP50':
        return loss_wo_middle_row


def aggregate_qcd_qee(ql):
    ql_2k = np.empty((ql.shape[0], 2))
    ql_2k[:, 0] = ql[:, 0] + ql[:, 2]
    ql_2k[:, 1] = ql[:, 1] + ql[:, 3]
    return ql_2k


# def rank_solutions(matrix, weights=None, types=None):
#     if weights is None:
#         weights = mcdm_weights.equal_weights(matrix)
#
#     if types is None:
#         types = np.array([-1] * matrix.shape[1])
#
#     topsis = mcdm_methods.TOPSIS()
#     ranks = rankdata(topsis(matrix, np.array(weights), types), reverse=True)
#
#     return np.argsort(ranks)


def sort_1st_col(X, F, eq_F=None):
    ix = np.argsort(F, axis=0)
    X_sorted = X[ix[:, 0], :]
    F_sorted = F[ix[:, 0], :]
    if eq_F is not None:
        eq_F_sorted = eq_F[ix[:, 0], :]
        return X_sorted, F_sorted, eq_F_sorted
    else:
        return X_sorted, F_sorted


def get_selected_ix(quantiles_loss, risk, upper=True):
    valid_keys = ['qcru', 'qeru'] if upper else ['qcrl', 'qerl']
    valid_ix = {}

    for key, value in risk.items():
        if key in valid_keys:
            valid_ix[key] = [np.argmax(np.array(valid_keys) == key), value]

    if not valid_keys:
        raise ValueError('{} does have valid options, must contain {}'.format(risk, valid_keys))

    # consider only first element for bound
    for key, value in valid_ix.items():
        return np.argmin(np.abs(quantiles_loss[:, value[0]] - value[1]))

def valid_test_model(data_formatter, model, test, valid):
    print("Computing best validation loss")
    val_loss = model.evaluate(valid)

    print("Computing test loss")
    unscaled_output_map = predict_model(data_formatter, model, test)
    q_losses = compute_q_loss(model.quantiles, unscaled_output_map)

    print("Training completed @ {}".format(datetime.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("\nNormalised Quantile Losses for Test Data: {}".format(q_losses))

    return q_losses, unscaled_output_map


def q_loss_model(data_formatter, model, test, return_output_map=False):
    unscaled_output_map = predict_model(data_formatter, model, test)
    q_losses = compute_q_loss(model.quantiles, unscaled_output_map)

    if return_output_map:
        return q_losses, unscaled_output_map
    else:
        return q_losses


def moo_q_loss_model(data_formatter, model, test, return_output_map=False, multi_processing=False,
                     eq_weights=False):
    unscaled_output_map = predict_model(data_formatter, model, test, multi_processing)
    q_losses = compute_moo_q_loss(model.quantiles, unscaled_output_map,
                                  overwrite_q=np.ones_like(model.quantiles) * 0.5 if eq_weights else None)

    if return_output_map:
        return q_losses, unscaled_output_map
    else:
        return q_losses


def compute_q_loss(quantiles, unscaled_output_map):
    targets = unscaled_output_map['targets']
    losses = {}
    for q in quantiles:
        key = 'p{}'.format(int(q * 100))
        losses[key + '_loss'] = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(unscaled_output_map[key]), q)
    q_losses = [p_loss.mean() for k, p_loss in losses.items()]
    return q_losses


def compute_moo_q_loss(quantiles, unscaled_output_map, overwrite_q=None):
    if overwrite_q is None:
        overwrite_q = quantiles

    targets = unscaled_output_map['targets']
    losses, losses_eq_weight = {}, {}
    for q, new_q in zip(quantiles, overwrite_q):
        key = 'p{}'.format(int(q * 100))
        losses[key + '_loss'] = utils.numpy_normalised_quantile_loss_moo(
            extract_numerical_data(targets), extract_numerical_data(unscaled_output_map[key]), new_q)

    #     if output_eq_loss:
    #         losses_eq_weight[key + '_loss'] = utils.numpy_normalised_quantile_loss_moo(
    #             extract_numerical_data(targets), extract_numerical_data(unscaled_output_map[key]), 0.5)
    #
    q_losses = [[obj.mean() for obj in p_loss] for k, p_loss in losses.items()]
    #
    # if output_eq_loss:
    #     q_eq_losses = [[obj.mean() for obj in p_loss] for k, p_loss in losses_eq_weight.items()]
    #
    #     return np.array(q_losses), np.array(q_eq_losses)
    #
    # else:
    return np.array(q_losses)


def predict_model(data_formatter, model, test, multi_processing=False):
    output_map = model.predict(test, return_targets=True, multi_processing=multi_processing)
    unscaled_output_map = {}
    for k, df in output_map.items():
        unscaled_output_map[k] = data_formatter.format_predictions(df)
    return unscaled_output_map


def dense_layer_output(weights, X):
    b = np.repeat(weights[1].reshape(1, -1), X.shape[1], axis=0)
    ans = []
    for x in X:
        ans.append(np.expand_dims(np.matmul(x, weights[0]) + b, 0))

    return np.concatenate(ans, axis=0)

def get_last_layer_weights(model, layer_name='quantiles'):
    relevant_layers = [l for l in model.model.layers if layer_name in l.name]
    if len(relevant_layers) > 1:
        raise Exception('More than one layer found')
    else:
        last_layer = relevant_layers[0]
        return last_layer.get_weights(), last_layer

def get_new_weights(original_weights, selected_weights):
    new_weights = copy.deepcopy(original_weights)
    new_weights[0][:, 0] = selected_weights['lq'][:-1]
    new_weights[0][:, 2] = selected_weights['uq'][:-1]
    new_weights[1][0] = selected_weights['lq'][-1]
    new_weights[1][2] = selected_weights['uq'][-1]
    return new_weights

def params_conversion_weights(weights):
    shapes = [w.shape for w in weights]
    flatten_dim = [np.multiply(*s) if len(s) > 1 else s[0] for s in shapes]

    ind = np.concatenate([w.flatten() for w in weights]).reshape(1, -1)
    params = {
        'shapes': shapes,
        'flatten_dim': flatten_dim
    }
    return ind, params


def get_ix_ind_from_weights(weights, ix_weight):
    w = weights[0][:, ix_weight].reshape(1, -1)
    b = weights[1][ix_weight].reshape(1, 1)
    ind = np.hstack([w, b])
    return ind


def reconstruct_weights(ind, params):
    shapes, flatten_dim = params['shapes'], params['flatten_dim']
    reconstruct = []
    ind = ind.reshape(-1, )
    for i in range(len(shapes)):
        if i == 0:
            reconstruct.append(ind[:flatten_dim[i]].reshape(shapes[i]))
        else:
            reconstruct.append(ind[flatten_dim[i - 1]:flatten_dim[i - 1] + flatten_dim[i]].reshape(shapes[i]))

    return reconstruct


def create_output_map(prediction,
                      quantiles,
                      output_size,
                      data_map,
                      time_steps,
                      num_encoder_steps):
    # Extract predictions for each quantile into different entries
    process_map = {
        'p{}'.format(int(q * 100)):
            prediction[Ellipsis, i * output_size:(i + 1) * output_size]
        for i, q in enumerate(quantiles)
    }

    process_map['targets'] = data_map['outputs']

    return {k: format_outputs(process_map[k],
                              data_map,
                              time_steps,
                              num_encoder_steps) for k in process_map}


def format_outputs(prediction,
                   data_map,
                   time_steps,
                   num_encoder_steps):
    """Returns formatted dataframes for prediction."""
    time = data_map['time']
    identifier = data_map['identifier']

    flat_prediction = pd.DataFrame(
        prediction[:, :, 0],
        columns=[
            't+{}'.format(i + 1)
            for i in range(time_steps - num_encoder_steps)
        ])
    cols = list(flat_prediction.columns)
    flat_prediction['forecast_time'] = time[:, num_encoder_steps - 1, 0]
    flat_prediction['identifier'] = identifier[:, 0, 0]

    # Arrange in order
    return flat_prediction[['forecast_time', 'identifier'] + cols]


def run_moo_nn(x,
               quantiles,
               output_size,
               data_map,
               time_steps,
               num_encoder_steps,
               transformer_output,
               w_params,
               loss_to_obj,
               p50_w,
               p50_b,
               output_eq_loss=False):
    new_weights = reconstruct_weights(x, w_params)

    if p50_w is not None and p50_b is not None:
        new_weights[0] = np.vstack([new_weights[0][:, 0],
                                    p50_w,
                                    new_weights[0][:, 1]]).T
        new_weights[1] = np.array([new_weights[1][0],
                                   p50_b,
                                   new_weights[1][1]])

    prediction = dense_layer_output(new_weights, transformer_output)
    unscaled_output_map = create_output_map(prediction,
                                            quantiles,
                                            output_size,
                                            data_map,
                                            time_steps,
                                            num_encoder_steps)

    losses = compute_moo_q_loss(quantiles, unscaled_output_map, output_eq_loss=output_eq_loss)

    if output_eq_loss:
        return loss_to_obj(losses[0]), loss_to_obj(losses[1])
    else:
        return loss_to_obj(losses)


def run_single_w_nn(x,
                    quantiles,
                    output_size,
                    data_map,
                    time_steps,
                    num_encoder_steps,
                    transformer_output,
                    ix_weight,
                    original_weights,
                    overwrite_q=None):

    new_weights = copy.deepcopy(original_weights)
    weights, b = x[:-1], x[-1]
    new_weights[0][:, ix_weight] = weights
    new_weights[1][ix_weight] = b

    prediction = dense_layer_output(new_weights, transformer_output)
    unscaled_output_map = create_output_map(prediction,
                                            quantiles,
                                            output_size,
                                            data_map,
                                            time_steps,
                                            num_encoder_steps)

    losses = compute_moo_q_loss(quantiles,
                                unscaled_output_map,
                                overwrite_q=overwrite_q)

    # if output_eq_loss:
    #     return losses[0][ix_weight, :], losses[1][ix_weight, :]
    # else:
    return losses[ix_weight, :]


def get_deap_pops_obj(logbook):
    pops = logbook.select('pop')
    pops_obj = [np.array([ind.fitness.values for ind in pop]) for pop in pops]
    ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) #+ 1
    return pops_obj, ref

def get_pymoo_pops_obj(res):
    pops = [pop.pop for pop in res.history]
    pops_obj = [np.array([ind.F for ind in pop]) for pop in pops]
    ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) #+ 1
    return pops_obj, ref

def get_deap_pop_hist(logbook):
    pop_hist = []
    for gen in logbook:
        pop = gen['pop']
        pop_hist.append(np.array([ind.fitness.values for ind in gen['pop']]))
    return pop_hist


def get_fitnesses(pop):
    fitnesses = np.array([ind.fitness.wvalues for ind in pop])
    fitnesses *= -1
    return fitnesses

def hypervolume(individuals, ref=None):
    # front = tools.sortLogNondominated(individuals, len(individuals), first_front_only=True)
    wobjs = np.array([ind.fitness.wvalues for ind in individuals]) * -1
    if ref is None:
        ref = np.max(wobjs, axis=0)  # + 1
    return hv.hypervolume(wobjs, ref)


def get_hypervolume(pop, ref=None):
    F = pop if isinstance(pop, np.ndarray) else get_fitnesses(pop)
    ref = np.max(F, axis=0) if ref is None else np.array(ref)
    hypervol = hv.hypervolume(F, ref)
    return hypervol


def get_hvs_from_log(hist, lib='deap'):
    pops_obj, ref = get_deap_pops_obj(hist) if lib == 'deap' else get_pymoo_pops_obj(hist)
    hv = get_performance_indicator("hv", ref_point=ref)
    hypervols = [hv.do(pop_obj) for pop_obj in pops_obj]
    return hypervols


def hv_hist_from_runs(runs, ref=None):
    pop_hist_runs = [run['result']['pop_hist'] for run in runs]
    hv_hist_runs = []
    for pop_hist in pop_hist_runs:
        hv_hist_runs.append([get_hypervolume(pop, ref=ref) for pop in pop_hist])
    return array_from_lists(hv_hist_runs)