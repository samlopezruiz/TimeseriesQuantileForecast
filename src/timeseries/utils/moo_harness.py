import gc
import time

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_termination, get_problem, get_sampling, get_crossover, get_mutation, \
    get_reference_directions
from pymoo.optimize import minimize


# from algorithms.moo.smsemoa import SMSEMOA
# from algorithms.moo.utils.indicators import get_hypervolume, hv_hist_from_runs
# from algorithms.moo.utils.plot import plot_runs, plot_boxplot
# from algorithms.moo.utils.utils import get_moo_args, get_hv_hist_vs_n_evals
# from timeseries.data.market.files.utils import save_df
# from timeseries.experiments.utils.files import save_vars
#
# from timeseries.utils.utils import get_type_str, array_from_lists, mean_std_from_array
from src.models.moo.smsemoa import SMSEMOA
from src.timeseries.plot.moo import plot_runs, plot_boxplot
from src.timeseries.utils.files import save_vars, save_df
from src.timeseries.utils.moo import sort_1st_col, get_hypervolume, get_moo_args, hv_hist_from_runs, \
    get_hv_hist_vs_n_evals
from src.timeseries.utils.parallel import repeat_different_args, repeat
from src.timeseries.utils.util import get_type_str, array_from_lists, mean_std_from_array


def run_moo(problem, algorithm, algo_cfg, verbose=1, seed=None, save_history=True):
    termination = get_termination(algo_cfg['termination'][0], algo_cfg['termination'][1])

    t0 = time.time()
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   save_history=save_history,
                   verbose=verbose >= 2)

    opt_time = time.time() - t0
    if verbose >= 1:
        print('{} with {} finished in {}s'.format(get_type_str(problem), get_type_str(algorithm), round(opt_time, 4)))

    pop_hist = [gen.pop.get('F') for gen in res.history] if save_history else None
    result = {'res': res, 'pop_hist': pop_hist, 'opt_time': opt_time}
    return result

def run_dual_moo_weights(moo_method,
                         algo_cfg,
                         general_cfg,
                         prob_cfg,
                         lower_q_problem,
                         upper_q_problem,
                         dual_q_problem,
                         model_results,
                         verbose=0):

    results, times = {}, []
    for bound, problem in zip(['lq', 'uq'], [lower_q_problem, upper_q_problem]):
        t0 = time.time()
        sampling = np.tile(problem.ini_ind, (algo_cfg['pop_size'], 1)) if algo_cfg['use_sampling'] else None
        algorithm = get_algorithm(moo_method,
                                  algo_cfg,
                                  n_obj=problem.n_obj,
                                  sampling=sampling)

        prob_cfg['n_var'], prob_cfg['n_obj'] = problem.n_var, problem.n_obj
        prob_cfg['hv_ref'] = [5] * problem.n_obj
        algo_cfg['name'] = get_type_str(algorithm)
        moo_result = run_moo(problem,
                             algorithm,
                             algo_cfg,
                             verbose=verbose,
                             save_history=general_cfg['save_history'])

        eq_F = problem.compute_eq_F(moo_result['res'].pop.get('X'))

        # swap columns because q < 0.5
        # if bound == 'lq':
        #     eq_F[:, 0], eq_F[:, 1] = copy.copy(eq_F[:, 1]), copy.copy(eq_F[:, 0])

        X_sorted, F_sorted, eq_F_sorted = sort_1st_col(moo_result['res'].pop.get('X'),
                                                       moo_result['res'].pop.get('F'),
                                                       eq_F)

        times.append(round((time.time() - t0) / 60, 0))

        results[bound] = {'X': X_sorted,
                          'F': F_sorted,
                          'eq_F': eq_F_sorted,
                          'original_losses': problem.original_losses,
                          'original_weights': dual_q_problem.original_weights,
                          'loss_to_obj_type': None,
                          'quantiles': problem.quantiles,
                          'experiment_cfg': model_results['experiment_cfg'],
                          'model_params': model_results['model_params'],
                          'fixed_params': model_results['fixed_params'],
                          'moo_method': moo_method,
                          'algo_cfg': algo_cfg,
                          'pop_hist': moo_result['pop_hist']}

        gc.collect()

    return {'results': results,
            'times': times}



def create_title(prob_cfg, algo_cfg, algorithm):
    return prob_cfg['name'] + ' Optimum Solutions' + '<br>' + \
           get_type_str(algorithm) + ' CFG: ' + str(algo_cfg)


def get_algorithm(name, algo_cfg, n_obj=3, sampling=None):
    if sampling is None:
        sampling = get_sampling("real_random")
    algo_options = ['SMSEMOA', 'MOEAD', 'NSGA2', 'NSGA3']
    if name not in algo_options:
        raise Exception('Algorithm {} not valid. Options are: {}'.format(name, str(algo_options)))

    cbx_pb = algo_cfg.get('cbx_pb', 0.9)
    cbx_eta = algo_cfg.get('cbx_eta', 15)
    mut_eta = algo_cfg.get('mut_eta', 20)
    ref_dirs = get_reference_directions("energy", n_obj, algo_cfg['pop_size'])
    if name == 'SMSEMOA':
        algorithm = SMSEMOA(
            ref_point=algo_cfg['hv_ref'],
            pop_size=algo_cfg['pop_size'],
            sampling=sampling,
            crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
            mutation=get_mutation("real_pm", eta=mut_eta),
            eliminate_duplicates=True,
        )
    elif name == 'MOEAD':
        algorithm = MOEAD(
            sampling=sampling,
            crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
            mutation=get_mutation("real_pm", eta=mut_eta),
            ref_dirs=ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
        )
    elif name == 'NSGA2':
        algorithm = NSGA2(
            pop_size=algo_cfg['pop_size'],
            sampling=sampling,
            crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
            mutation=get_mutation("real_pm", eta=mut_eta),
            eliminate_duplicates=True,
            ref_dirs=ref_dirs,
        )
    elif name == 'NSGA3':
        algorithm = NSGA3(
            pop_size=algo_cfg['pop_size'],
            sampling=sampling,
            crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
            mutation=get_mutation("real_pm", eta=mut_eta),
            eliminate_duplicates=True,
            ref_dirs=ref_dirs
        )

    return algorithm


def run_moo_problem(name,
                    algo_cfg,
                    prob_cfg,
                    problem=None,
                    plot=True,
                    file_path=['img', 'opt_res'],
                    save_plots=False,
                    verbose=1,
                    seed=None,
                    use_date=False):
    problem = get_problem(prob_cfg['name'],
                          n_var=prob_cfg['n_variables'],
                          n_obj=prob_cfg['n_obj']) if problem is None else problem

    algorithm = get_algorithm(name, algo_cfg, n_obj=prob_cfg['n_obj'])

    algo_cfg['name'] = get_type_str(algorithm)
    result = run_moo(problem, algorithm, algo_cfg, verbose=verbose, seed=seed)

    # if plot:
    #     plot_results_moo(result['res'],
    #                      file_path=file_path,
    #                      title=create_title(prob_cfg, algo_cfg, algorithm),
    #                      save_plots=save_plots,
    #                      use_date=use_date)

    hv_pop = get_hypervolume(result['pop_hist'][-1], prob_cfg['hv_ref'])
    hv_opt = None #get_hypervolume(result['res'].opt.get('F'), prob_cfg['hv_ref'])

    problem_result = {'result': result, 'hv_pop': hv_pop, 'hv_opt': hv_opt}
    return problem_result


def repeat_moo(params, n_repeat, is_reproductible=False, parallel=True):
    if is_reproductible:
        args = []
        for i in range(n_repeat):
            # Specify seed for each run for reproducibility
            params['seed'] = i
            args.append(get_moo_args(params))
        runs = repeat_different_args(run_moo_problem, args, parallel=parallel)
    else:
        runs = repeat(run_moo_problem, get_moo_args(params), n_repeat=n_repeat, parallel=parallel)
    return runs


def run_multiple_problems(probs, algos, general_cfg, params, algo_cfg, prob_cfg, folder_cfg, parallel=True):
    for problem, k in probs:
        print('\nRunning Optimization for problem: {} k={}'.format(problem, k))
        prob_cfg['name'], prob_cfg['n_obj'] = problem, k
        prob_cfg['hv_ref'] = [5] * prob_cfg['n_obj']
        algo_cfg['hv_ref'] = [10] * prob_cfg['n_obj']

        algos_runs, algos_hv_hist_runs = [], []
        for algo in algos:
            t0 = time.time()
            params['name'] = algo
            runs = repeat_moo(params, general_cfg['n_repeat'], general_cfg['is_reproductible'], parallel=parallel)
            algos_runs.append(runs)
            algos_hv_hist_runs.append(hv_hist_from_runs(runs, ref=prob_cfg['hv_ref']))
            print('\t{} with {} finished in {}s'.format(problem, algo, round(time.time() - t0, 2)))

        if general_cfg['save_stats']:
            pops = [algo_runs['result']['res'].pop for algo_runs in algos_runs]
            sv = {'algos': algos, 'problem': problem, 'k': k, 'prob_cfg': prob_cfg, 'pops': pops,
                  'algo_cfg': algo_cfg, 'algos_hv_hist_runs': algos_hv_hist_runs}
            save_vars(sv,
                      path=['output',
                            folder_cfg['experiment'],
                            folder_cfg['results'],
                                 '{}_k{}_res'.format(problem, k)],
                      use_date=general_cfg['use_date'])

        if general_cfg['plot_ind_algorithms']:
            for i, hv_hist_runs in enumerate(algos_hv_hist_runs):
                plot_runs(hv_hist_runs, np.mean(hv_hist_runs, axis=0),
                          x_label='Generations',
                          y_label='Hypervolume',
                          title='{} k={} convergence plot with {}. Ref={}'.format(problem, k,
                                                                                  algos[i],
                                                                                  str(prob_cfg['hv_ref'])),
                          size=(15, 9),
                          file_path=['output',
                                     folder_cfg['experiment'],
                                     folder_cfg['images'],
                                     '{}_{}_k{}'.format(algos[i], problem, k)],
                          save=general_cfg['save_ind_algorithms'],
                          use_date=general_cfg['use_date'])

        if algo_cfg['termination'][0] == 'n_eval':
            algos_mean_hv_hist = get_hv_hist_vs_n_evals(algos_runs, algos_hv_hist_runs)
        else:
            algos_mean_hv_hist = array_from_lists([np.mean(hv_hist_runs, axis=0)
                                                   for hv_hist_runs in algos_hv_hist_runs])

        plot_runs(algos_mean_hv_hist,
                  x_label='Function Evaluations' if algo_cfg['termination'][0] == 'n_eval' else 'Generations',
                  y_label='Hypervolume',
                  title='{} k={} convergence plot. Ref={}'.format(problem, k, str(prob_cfg['hv_ref'])),
                  size=(15, 9),
                  file_path=['output', folder_cfg['experiment'], folder_cfg['images'],
                             'Compare_{}_k{}'.format(problem, k)],
                  save=general_cfg['save_comparison_plot'],
                  legend_labels=algos,
                  use_date=general_cfg['use_date'])

        hvs_algos = array_from_lists([hv_hist_runs[:, -1] for hv_hist_runs in algos_hv_hist_runs])
        hvs_stats = mean_std_from_array(hvs_algos, labels=algos)
        exec_times = array_from_lists([[algo_run['result']['opt_time'] for algo_run in algo_runs]
                                       for algo_runs in algos_runs])
        exec_times_stats = mean_std_from_array(exec_times, labels=algos)
        if general_cfg['save_stats']:
            save_df(exec_times_stats, file_path=['output',
                                                 folder_cfg['experiment'],
                                                 folder_cfg['results'],
                                                 '{}_k{}_exec_t'.format(problem, k)],
                    use_date=general_cfg['use_date'])
            save_df(hvs_stats, file_path=['output',
                                          folder_cfg['experiment'],
                                          folder_cfg['results'],
                                          '{}_k{}_hv'.format(problem, k)],
                    use_date=general_cfg['use_date'])

        plot_boxplot(hvs_algos,
                     algos,
                     x_label='Algorithm',
                     y_label='Hypervolume',
                     title='{} k={} hypervolume history'.format(problem, k),
                     size=(15, 9),
                     file_path=['output',
                                  folder_cfg['experiment'],
                                  folder_cfg['images'],
                                  '{}_k{}_hv'.format(problem, k)],
                     save=general_cfg['save_stat_plots'],
                     show_grid=False,
                     use_date=general_cfg['use_date'])

        plot_boxplot(exec_times,
                     algos,
                     x_label='Algorithm',
                     y_label='Seconds',
                     title='{} k={} execution times'.format(problem, k),
                     size=(15, 9),
                     file_path=['output',
                                  folder_cfg['experiment'],
                                  folder_cfg['images'],
                                  '{}_k{}_exec_t'.format(problem, k)],
                     save=general_cfg['save_stat_plots'],
                     show_grid=False,
                     use_date=general_cfg['use_date'])

        gc.collect()
