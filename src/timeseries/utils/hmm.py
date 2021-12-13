import numpy as np
from hmmlearn.hmm import GaussianHMM, GMMHMM

from src.timeseries.utils.dataframe import append_to_df, relabel_col


def fitHMM(Q, n_iter, n_components=2):
    shape = np.array(Q).shape
    obs = np.argmax(shape)
    f = np.argmin(shape)
    Q = np.reshape(Q, [shape[obs], shape[f]])
    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=n_components, n_iter=n_iter).fit(Q)  # np.reshape(Q, [len(Q), 1])
    # GaussianHMM or GMMHMM
    # classify each observation as state 0 or 1
    hidden_states = model.predict(Q)  # np.reshape(Q, [len(Q), 1])
    hidden_proba = model.predict_proba(Q)
    # find parameters of Gaussian HMM
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)

    # find log-likelihood of Gaussian HMM
    logProb = model.score(Q)  # np.reshape(Q, [len(Q), 1])

    # generate nSamples from Gaussian HMM
    # samples = model.sample(100)

    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    # if mus[0] > mus[1]:
    #     mus = np.flipud(mus)
    #     sigmas = np.flipud(sigmas)
    #     P = np.fliplr(np.flipud(P))
    #     hidden_states = 1 - hidden_states

    return hidden_states, mus, sigmas, P, logProb, model, hidden_proba


def relabel_sort_var(label_cfg, vars, mus):
    ix_var = np.argmax(np.array(vars) == label_cfg[0])
    ix_sort = np.argsort(mus[:, ix_var])
    ix = ix_sort if label_cfg[1] == 'max' else np.flip(ix_sort)
    map = [np.argmax(ix == i) for i in range(len(ix))]
    return map

def count_regimes(regimes):
    reg_chgs = 0
    for reg in regimes:
        reg_chgs += len(reg)
    return reg_chgs


def get_regimes(states, print_=False):
    regimes = [[] for i in range(int(max(states) + 1))]
    t_1, k_1 = states.index[0], states.iloc[0]
    regimes[int(k_1)].append(t_1)

    for t, k in states.items():
        if k_1 != k:
            # end of regime
            regimes[int(k_1)].append(t)
            # change of regime
            reg = k
            regimes[int(k)].append(t)

        t_1, k_1 = t, k

    for i in range(len(regimes)):
        regimes[i] = regimes[i][:len(regimes[i]) - len(regimes[i]) % 2]

    if print_:
        reg_chgs = count_regimes(regimes)
        print('regimes changes: {}'.format(reg_chgs))

    return regimes


def append_hmm_states(df, hmm_vars, n_states, label_cfg=None, regime_col='state'):
    X = df.loc[:, hmm_vars]
    hidden_states, mus, sigmas, P, logProb, model, hidden_proba = fitHMM(X, 100, n_components=n_states)
    append_to_df(df, hidden_states, regime_col)

    for i in range(n_states):
        append_to_df(df, hidden_proba[:, i], 'p_'+str(i))
    n_regimes = get_regimes(df[regime_col], print_=True)
    if label_cfg is not None:
        map = relabel_sort_var(label_cfg, hmm_vars, mus)
        relabel_col(df, regime_col, map=map)
    df_proba = df.loc[:, ['p_'+str(i) for i in range(n_states)]].copy()
    return df, n_regimes, df_proba