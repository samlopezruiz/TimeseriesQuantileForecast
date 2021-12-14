import numpy as np
import pandas as pd
from scipy.stats import ranksums


def wilcoxon_rank(arr, labels, alternative='less'):
    '''
    :param alternative:  ‘less’: the distribution underlying x is stochastically less than the distribution underlying y.
    :param labels: list of test labels
    :param arr: rows are tests and columns are samples
    :return: matrix with wilcoxon p values
    '''
    n_tests = arr.shape[0]
    wilcoxon_rank = np.zeros((n_tests, n_tests))
    wilcoxon_rank.fill(np.nan)
    for i in range(n_tests):
        for j in range(n_tests):
            x, y = arr[i, :], arr[j, :]
            wilcoxon_rank[i, j] = ranksums(x, y, alternative=alternative).pvalue

    return pd.DataFrame(wilcoxon_rank, index=labels, columns=labels)


class Winners:
    def __init__(self, metrics, labels):
        '''
        metrics: rows are problem samples, cols are algorithms
        '''
        self.cd_mat_calculated = False
        self.labels = labels
        self.rankings = np.empty_like(metrics).astype(int)
        self.metrics = metrics
        self.n_elements = metrics.shape[1]
        self.ranking_count = np.zeros((self.n_elements, self.n_elements))
        self.condorcet_mat = np.zeros((self.n_elements, self.n_elements))
        self.wr_mat = None
        self.calc_rankings()
        self.count_rankings()

    def calc_rankings(self):
        for i, res in enumerate(self.metrics):
            self.rankings[i, :] = np.argsort(np.argsort(res))

    def count_rankings(self):
        for a in range(self.n_elements):
            for r in range(self.n_elements):
                self.ranking_count[r, a] = np.sum(self.rankings[:, a] == r)

    def get_ranking_count(self):
        pos = np.array(range(self.n_elements)) + 1
        return pd.DataFrame(self.ranking_count,
                            columns=self.labels,
                            index=['{} place'.format(p) for p in pos]).astype(int)

    def border_count(self):
        weights = np.array(range(self.n_elements))[::-1] + 1
        border_score = np.multiply(self.ranking_count, weights.reshape(-1, 1))
        ix = np.argsort(np.sum(border_score, axis=0))
        bd = pd.DataFrame(np.sum(border_score, axis=0), columns=['score'], index=self.labels)

        return [self.labels[i] for i in ix[::-1]], bd

    def condorcet_matrix(self, problem_samples=None, alternative='less'):
        '''
        :param problem_samples: shape [problems, algos, n_samples]
        :param alternative: what is better, 'less' or 'greater'
        :return: condorcet_matrix
        '''
        if problem_samples is None:
            for x in range(self.n_elements):
                for y in range(self.n_elements):
                    for res in self.rankings:
                        self.condorcet_mat[x, y] += int(res[x] < res[y])
        else:
            wr_mat = np.zeros((problem_samples.shape[0], self.n_elements, self.n_elements))
            for i, algos_hvs in enumerate(problem_samples):
                wr_mat[i, :, :] = wilcoxon_rank(algos_hvs, self.labels, alternative=alternative).values
            p_value = 0.05
            self.wr_mat = wr_mat
            wr_mat_test = (wr_mat <= p_value).astype(int)
            self.condorcet_mat = np.sum(wr_mat_test, axis=0)

        self.cd_mat_calculated = True

    def get_condorcet(self, problem_samples=None, alternative='less'):
        if not self.cd_mat_calculated:
            self.condorcet_matrix(problem_samples, alternative)
        return pd.DataFrame(self.condorcet_mat,
                            columns=self.labels,
                            index=self.labels).astype(int)

    def condercet_winner(self, problem_samples=None, alternative='less'):
        if not self.cd_mat_calculated:
            self.condorcet_matrix(problem_samples, alternative)

        ind_wins = np.zeros((self.n_elements, self.n_elements))
        for x in range(self.n_elements):
            for y in range(self.n_elements):
                if x != y:
                    ind_wins[x, y] += int(self.condorcet_mat[x, y] >
                                          self.condorcet_mat[y, x])
        ind_wins = np.sum(ind_wins, axis=1)
        cs = pd.DataFrame(ind_wins, columns=['individual wins'], index=self.labels).astype(int)
        condorcet_winner = [self.labels[i] for i, wins in enumerate(ind_wins) if wins == self.n_elements - 1]
        return condorcet_winner, cs

    def score(self, problem_samples=None, alternative='less'):
        sorted_algos, border_scores = self.border_count()
        condorcet_winner, pairwise_wins = self.condercet_winner(problem_samples, alternative)
        return {
            'ranking_counts': self.get_ranking_count(),
            'sorted_algos': sorted_algos,
            'border_scores': border_scores,
            'condorcet_scores': self.get_condorcet(),
            'wr_scores': self.get_wr(),
            'condorcet_winner': condorcet_winner,
            'pairwise_wins': pairwise_wins
        }

    def get_wr(self):
        wr_mats = []
        for i, w in enumerate(self.wr_mat):
            wr_mats.append(pd.DataFrame(w,
                                        columns=self.labels,
                                        index=['{} ({})'.format(label, i) for label in self.labels]))
        wr_mat = pd.concat(wr_mats, axis=0)
        wr_mat.sort_index(axis=0, inplace=True)
        wr_mat.sort_index(axis=1, inplace=True)
        return wr_mat.round(4)

