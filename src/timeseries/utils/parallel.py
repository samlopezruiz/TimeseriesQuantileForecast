from multiprocessing import cpu_count

from joblib import Parallel, delayed
from numpy import mean, std
from tqdm import tqdm


def repeat(run_func, args, n_repeat, parallel=True, n_jobs=None):
    if parallel:
        executor = Parallel(n_jobs=cpu_count() if n_jobs is None else n_jobs)
        tasks = (delayed(run_func)(*args) for _ in tqdm(range(n_repeat)))
        result = executor(tasks)
    else:
        result = [run_func(*args) for _ in tqdm(range(n_repeat))]
    return result


def repeat_different_args(run_func, args, parallel=True, n_jobs=None, backend=None, use_tqdm=True):
    if not isinstance(args, list):
        raise Exception('args have to be a list')

    if parallel:
        executor = Parallel(n_jobs=cpu_count() if n_jobs is None else n_jobs, backend=backend)
        tasks = (delayed(run_func)(*arg) for arg in (tqdm(args) if use_tqdm else args))
        result = executor(tasks)
    else:
        result = [run_func(*arg) for arg in (tqdm(args) if use_tqdm else args)]
    return result


# def run_ga(ga, cfg):
#     best_ind, best_gen, best_eval, log = ga.run(cfg['n_gens'])
#     return best_gen, best_eval, log, best_ind


# def repeat_eval(ga, n_repeat, cfg, parallel=True):
#     if parallel:
#         executor = Parallel(n_jobs=-2, backend='loky') #cpu_count()
#         tasks = (delayed(run_ga)(ga, cfg) for _ in range(n_repeat))
#         logs = executor(tasks)
#     else:
#         logs = []
#         for _ in range(n_repeat):
#             best_ind, best_gen, best_eval, log = ga.run(cfg['n_gens'])
#             logs.append((best_gen, best_eval, log, best_ind))
#     return logs


# def print_eval(name, logs):
#     gens = [log[0] for log in logs]
#     f_evals = [log[1] for log in logs]
#     print('{} @ {} (+- {}) gens: f(x) = {} (+- {})'.format(name, round(mean(gens), 1), round(std(gens), 1),
#                                                            round(mean(f_evals), 4), round(std(f_evals), 4)))