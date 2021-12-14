import copy
import gc
import os
import time

import joblib
import numpy as np
from pymoo.core.problem import Problem
import tensorflow as tf
# from algorithms.tft2.harness.train_test import moo_q_loss_model, compute_moo_q_loss
# from algorithms.tft2.libs.hyperparam_opt import HyperparamOptManager
# from timeseries.experiments.market.moo.utils.model import get_last_layer_weights, run_single_w_nn, \
#     get_ix_ind_from_weights, create_output_map
# from timeseries.experiments.market.utils.filename import get_result_folder
# from timeseries.experiments.market.utils.harness import get_model, get_model_data_config
# from timeseries.utils.parallel import repeat_different_args
from src.models.attn.hyperparam_opt import HyperparamOptManager
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.harness import get_model, get_model_data_config
from src.timeseries.utils.moo import get_last_layer_weights, moo_q_loss_model, get_ix_ind_from_weights, \
    create_output_map, compute_moo_q_loss, run_single_w_nn
from src.timeseries.utils.parallel import repeat_different_args


class DualQuantileWeights:

    def __init__(self,
                 architecture,
                 model_folder,
                 data_formatter,
                 data_config,
                 use_gpu=True,
                 parallelize_pop=True,
                 constraints_limits=None,
                 optimize_eq_weights=False):
        Model = get_model(architecture)
        train, valid, test = data_formatter.split_data(data_config)

        # Sets up default params
        fixed_params = data_formatter.get_experiment_params()
        params = data_formatter.get_default_model_params()
        params["model_folder"] = model_folder

        # Sets up hyperparam manager
        print("\n*** Loading hyperparm manager ***")
        opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)
        model_params = opt_manager.get_next_parameters()

        with tf.device("/cpu:0"):
            # with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):
            model = Model(model_params)
            model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)
            weights, last_layer = get_last_layer_weights(model)

            print("\n*** Test Q-Loss with original weights ***")
            losses, unscaled_output_map = moo_q_loss_model(data_formatter, model, valid,
                                                           return_output_map=True,
                                                           eq_weights=optimize_eq_weights)

            # swap columns when calculating equal weights with lower quantile
            if optimize_eq_weights:
                losses[0, 0], losses[0, 1] = copy.copy(losses[0, 1]), copy.copy(losses[0, 0])

            outputs, output_map, data = model.predict_all(valid, batch_size=128)
            transformer_output = outputs['transformer_output']

        if use_gpu:
            transformer_output = tf.convert_to_tensor(transformer_output, dtype=tf.float32)

        self.use_gpu = use_gpu
        self.transformer_output = transformer_output
        self.data_map = data
        self.data_formatter = data_formatter
        self.valid = valid
        self.original_weights = weights
        self.last_layer = last_layer
        self.original_losses = losses
        self.parallelize_pop = parallelize_pop
        self.quantiles = copy.copy(model.quantiles)
        self.output_size = copy.copy(model.output_size)
        self.time_steps = copy.copy(model.time_steps)
        self.num_encoder_steps = copy.copy(model.num_encoder_steps)
        self.constraints_limits = constraints_limits
        self.optimize_eq_weights = optimize_eq_weights

        ind_lq = get_ix_ind_from_weights(self.original_weights, 0)
        ind_uq = get_ix_ind_from_weights(self.original_weights, 2)

        self.lower_quantile_problem = SingleQuantileWeights(ind_lq,
                                                            self.quantiles,
                                                            self.output_size,
                                                            self.data_map,
                                                            self.time_steps,
                                                            self.num_encoder_steps,
                                                            self.transformer_output,
                                                            0,  # index of lower quantile
                                                            self.original_weights,
                                                            self.parallelize_pop,
                                                            self.original_losses,
                                                            self.use_gpu,
                                                            self.constraints_limits,
                                                            self.optimize_eq_weights)

        self.upper_quantile_problem = SingleQuantileWeights(ind_uq,
                                                            self.quantiles,
                                                            self.output_size,
                                                            self.data_map,
                                                            self.time_steps,
                                                            self.num_encoder_steps,
                                                            self.transformer_output,
                                                            2,  # index of lower quantile
                                                            self.original_weights,
                                                            self.parallelize_pop,
                                                            self.original_losses,
                                                            self.use_gpu,
                                                            self.constraints_limits,
                                                            self.optimize_eq_weights)

        # self.n_var = ind.shape[1]
        # self.n_obj = len(self.original_losses)
        # super().__init__(self.n_var, self.n_obj, n_constr=0, xl=-1.0, xu=1.0, **kwargs)

        gc.collect()

    def get_problems(self):
        return self.lower_quantile_problem, self.upper_quantile_problem


class SingleQuantileWeights(Problem):

    def __init__(self,
                 original_ind,
                 quantiles,
                 output_size,
                 data_map,
                 time_steps,
                 num_encoder_steps,
                 transformer_output,
                 ix_weight,
                 original_weights,
                 parallelize_pop,
                 original_losses,
                 use_gpu,
                 constraints_limits,
                 optimize_eq_weights,
                 **kwargs):
        self.ini_ind = original_ind
        self.quantiles = quantiles
        self.output_size = output_size
        self.data_map = data_map
        self.time_steps = time_steps
        self.num_encoder_steps = num_encoder_steps
        self.transformer_output = transformer_output
        self.ix_weight = ix_weight
        self.original_weights = original_weights
        self.parallelize_pop = parallelize_pop
        self.original_losses = original_losses[ix_weight, :]
        self.use_gpu = use_gpu
        self.constraints_limits = constraints_limits
        self.optimize_eq_weights = optimize_eq_weights

        n_var = original_ind.shape[1]
        n_obj = 2
        n_constr = len(constraints_limits) if constraints_limits is not None else 0
        super().__init__(n_var, n_obj, n_constr=n_constr, xl=-1.0, xu=1.0, **kwargs)

    def get_pred_func(self, X):
        new_weights = copy.deepcopy(self.original_weights)

        batch_w = np.repeat(np.expand_dims(new_weights[0], axis=0), X.shape[0], axis=0)
        batch_b = np.repeat(np.expand_dims(np.repeat(new_weights[1].reshape(1, -1),
                                                     self.transformer_output.shape[1], axis=0), axis=0), X.shape[0],
                            axis=0)

        for i, x in enumerate(X):
            w, b = x[:-1], x[-1]
            batch_w[i, :, self.ix_weight] = w
            batch_b[i, :, self.ix_weight] = b

        batch_w = tf.convert_to_tensor(batch_w, dtype=tf.float32)
        batch_b = tf.convert_to_tensor(batch_b, dtype=tf.float32)

        @tf.function(experimental_relax_shapes=True)
        def my_map():
            return tf.map_fn(lambda x: tf.matmul(self.transformer_output, x[0]) + x[1],
                             elems=(batch_w, batch_b),
                             dtype=tf.float32,
                             parallel_iterations=16)

        return my_map

    def _evaluate(self, X, out, *args, **kwargs):
        if self.use_gpu:
            my_map = self.get_pred_func(X)

            prediction = my_map().numpy()

            losses = []

            overwrite_q = np.ones_like(self.quantiles) * 0.5 if self.optimize_eq_weights else None
            for pred in prediction:
                unscaled_output_map = create_output_map(pred,
                                                        self.quantiles,
                                                        self.output_size,
                                                        self.data_map,
                                                        self.time_steps,
                                                        self.num_encoder_steps)

                losses.append(compute_moo_q_loss(self.quantiles,
                                                 unscaled_output_map,
                                                 overwrite_q=overwrite_q)[self.ix_weight, :])

            losses = np.array(losses)

            # swap columns when calculating equal weights with lower quantile
            if self.optimize_eq_weights and self.ix_weight == 0:
                losses[:, 0], losses[:, 1] = copy.copy(losses[:, 1]), copy.copy(losses[:, 0])

            out["F"] = losses

        else:
            args = [[x,
                     self.quantiles,
                     self.output_size,
                     self.data_map,
                     self.time_steps,
                     self.num_encoder_steps,
                     self.transformer_output,
                     self.ix_weight,
                     self.original_weights
                     ]
                    for x in X]

            F = repeat_different_args(run_single_w_nn,
                                      args,
                                      parallel=self.parallelize_pop,
                                      n_jobs=None,
                                      use_tqdm=False)

            # gc.collect()
            # calculate the function values in a parallelized manner and wait until done
            out["F"] = np.array(F)

        if self.constraints_limits is not None:
            G = np.empty_like(out["F"])
            if len(self.constraints_limits) != out["F"].shape[1]:
                raise ValueError('{} constraints is not '
                                 'consistent with {} objectives'.format(len(self.constraints_limits),
                                                                        out["F"].shape[1]))
            for obj in range(out["F"].shape[1]):
                G[:, obj] = out["F"][:, obj] - self.constraints_limits[obj]

            out['G'] = G

    def compute_eq_F(self, X):

        if self.use_gpu:
            with tf.device('/device:GPU:0'):
                my_map = self.get_pred_func(X)
                prediction = my_map().numpy()

            losses = []
            for pred in prediction:
                unscaled_output_map = create_output_map(pred,
                                                        self.quantiles,
                                                        self.output_size,
                                                        self.data_map,
                                                        self.time_steps,
                                                        self.num_encoder_steps)
                loss = compute_moo_q_loss(self.quantiles,
                                          unscaled_output_map,
                                          overwrite_q=np.ones_like(self.quantiles) * 0.5)

                losses.append(loss[self.ix_weight, :])

            losses = np.array(losses)

        else:

            args = [[x,
                     self.quantiles,
                     self.output_size,
                     self.data_map,
                     self.time_steps,
                     self.num_encoder_steps,
                     self.transformer_output,
                     self.ix_weight,
                     self.original_weights,
                     np.ones_like(self.quantiles) * 0.5]  # overwrite_q
                    for x in X]

            F_eq_F = repeat_different_args(run_single_w_nn,
                                           args,
                                           parallel=self.parallelize_pop,
                                           n_jobs=None,
                                           use_tqdm=False)

            losses = np.array(F_eq_F)

        # swap columns when calculating equal weights with lower quantile
        if self.ix_weight == 0:
            losses[:, 0], losses[:, 1] = copy.copy(losses[:, 1]), copy.copy(losses[:, 0])

        return losses


if __name__ == "__main__":
    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q357',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'
                   }

    model_results = joblib.load(os.path.join(get_result_folder(results_cfg, project), results_cfg['results'] + '.z'))

    config, data_formatter, model_folder = get_model_data_config(project,
                                                                 model_results['experiment_cfg'],
                                                                 model_results['model_params'],
                                                                 model_results['fixed_params'])

    experiment_cfg = model_results['experiment_cfg']

    type_func = 'mean_across_quantiles'  # 'ind_loss_woP50' #'mean_across_quantiles'

    dual_q_problem = DualQuantileWeights(architecture=experiment_cfg['architecture'],
                                         model_folder=model_folder,
                                         data_formatter=data_formatter,
                                         data_config=config.data_config,
                                         use_gpu=False,
                                         parallelize_pop=False,
                                         constraints_limits=[1., 1.],
                                         optimize_eq_weights=True)

    lower_q_problem, upper_q_problem = dual_q_problem.get_problems()

    # %%
    X = np.random.rand(200, lower_q_problem.n_var)

    print('Evaluating')
    res = {}
    t0 = time.time()
    lower_q_problem._evaluate(X, res)
    print('Eval time: {} s'.format(round(time.time() - t0, 4)))
    F_l = res['F']

    eq_F_l = lower_q_problem.compute_eq_F(X)

    print('Evaluating')
    res = {}
    t0 = time.time()
    upper_q_problem._evaluate(X, res)
    print('Eval time: {} s'.format(round(time.time() - t0, 4)))
    F_u = res['F']

    eq_F_u = upper_q_problem.compute_eq_F(X)