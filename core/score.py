#
#   Class for score (objective) functions
#

from numpy import ndarray, sum as npsum, sqrt as npsqrt, mean as npmean
from pandas import DataFrame
from math import lgamma
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

from core.common import ln
from core.timing import Timing
from fileio.data import Data
from fileio.oracle import Oracle

ENTROPY_SCORES = ['loglik', 'bic', 'aic']  # categorical entropy scores
BAYESIAN_SCORES = ['bde', 'k2', 'bdj', 'bds']  # categorical Bayesian
GAUSSIAN_SCORES = ['bic-g']  # Gaussian scores

SCORES = {'loglik': {'base'},  # Supported scores and their allowed parameters
          'aic': {'base', 'k'},
          'bic': {'base', 'k'},
          'bic-g': {'base', 'k'},
          'bde': {'iss', 'prior'},
          'bds': {'iss', 'prior'},
          'bdj': {},
          'k2': {}}

SCORE_PARAMS = {'base': 'e',  # score parameters and default values
                'k': 1,
                'iss': 1,
                'prior': 'uniform',
                'unistate_ok': True}


def bayesian_score(counts, q_i, type, params):
    """
        Return Bayesian based scores for marginal counts (for a node)

        :param ndarray counts: 2D instance counts [row, col]
        :param int q_i: maximum possible number of parental value combinations
                        (even if they don't all necessarily occur in the data)
        :param str type: bayesian score type required e.g. bde, k2
        :param dict params: parameters used in score computation: 'base' is
                            logarithm base

        :raises TypeError: if any argument has invalid type
        :raises ValueError: if any argument has invalid value

        :returns dict: of requested scores {score: value}
    """
    if (not isinstance(counts, ndarray)
            or not isinstance(q_i, int) or isinstance(q_i, bool)
            or not isinstance(type, str)
            or not isinstance(params, dict)):
        raise TypeError('Bad arg types for bayesian_score')

    if type not in BAYESIAN_SCORES or 'iss' not in params:
        raise ValueError('Bad arg values for bayesian_score')

    iss = params['iss']   # imaginary sample size
    score = 0.0  # will accrue the score value

    # Use variable names to match literature. NP_ijk is N-prime subscript ijk,
    # the prior count for kth state and jth parental value combination

    r_i = counts.shape[0]

    if type == 'bds':  # for BDS use actual number of parental combos in data
        q_i = counts.shape[1]

    if type in ['bde', 'bds']:
        NP_ijk = iss / (q_i * r_i)
        NP_ij = iss / q_i
    else:
        NP_ijk = 1.0 if type == 'k2' else 0.5  # 1 or 0.5 for K2 or BDJ
        NP_ij = r_i if q_i == 1 or type == 'k2' else 0.5 * r_i

    for j in range(counts.shape[1]):
        N_ij = counts[:, j].sum()
        score += lgamma(NP_ij) - lgamma(NP_ij + N_ij)
        for k in range(counts.shape[0]):
            N_ijk = counts[k, j]
            score += lgamma(NP_ijk + N_ijk) - lgamma(NP_ijk)

    return score


def entropy_scores(counts, types, params, N, free_params):
    """
        Return entropy based scores for marginal counts (for a node)

        :param ndarray counts: 2-D array [rows, cols] of instance counts
        :param str/list types: entropy based score(s) required
        :param dict params: parameters used in score computation: 'base' is
                            logarithm base
        :param int N: number of cases (instances)
        :param int free_params: number of free parameters

        :raises TypeError: if any argument has invalid type
        :raises ValueError: if any argument has invalid value

        :returns dict: of requested scores {score: value}
    """
    if isinstance(types, str):
        types = [types]

    if not isinstance(counts, ndarray) or not isinstance(params, dict) \
            or not isinstance(types, list)  \
            or not isinstance(N, int) or isinstance(N, bool) \
            or not isinstance(free_params, int) \
            or isinstance(free_params, bool) \
            or any([not isinstance(t, str) for t in types]):
        raise TypeError('Bad arg types for entropy_scores')

    if any([t not in ENTROPY_SCORES for t in types]) \
            or N < 1 or free_params < 0:
        raise ValueError('Bad arg values for entropy_scores')

    base = params['base'] if 'base' in params else 10  # default to base 10

    scores = {}

    # Compute log likelihood on which all entropy scores are based
    # Formula in lambda function is directly relateable to the literature:
    # - Nij is number of instances with jth parental value combo for node i
    # - Nijk is number of instances with jth parental value combo for node i
    #   where node i takes kth value.

    loglik = 0.0
    for j in range(counts.shape[1]):
        Nij = counts[:, j].sum()
        for k in range(counts.shape[0]):
            Nijk = counts[k, j]
            loglik += Nijk * ln(Nijk / Nij, base) if Nijk > 0 else 0.0

#    for combo in marginals.columns:
#        Nij = marginals[combo].sum()
#        loglik += marginals[combo].apply(lambda Nijk:
#                                         Nijk * ln(Nijk / Nij, base)
#                                         if Nijk > 0 else 0.0).sum()

    if 'loglik' in types:
        scores['loglik'] = loglik

    if 'bic' in types:
        scores['bic'] = loglik - (params['k'] * 0.5 * free_params
                                  * ln(N, base))

    if 'aic' in types:
        scores['aic'] = loglik - params['k'] * free_params

    return scores


def node_score(node, parents, types, params, data, counts_reqd=False):
    """
        Return specified types of decomposable scores for a node

        :param str node: node scores are required for
        :param dict parents: parents of non-orphan nodes {node: [parents]}
        :param list types: scores required e.g. ['loglik', 'aic', 'bde']
        :param dict params: parameters for scores e.g. logarithm base
        :param Data data: data to learn graph from
        :param bool counts_reqd: whether info about marginal counts returned

        :returns dict/tuple: scores {score_type: score} if counts not required
                             otherwise (scores, counts)
    """
    if data.dstype == 'categorical':
        return categorical_node_score(node, parents, types, params, data,
                                      counts_reqd)
    else:
        scores = gaussian_node_score(node, parents, types, params, data)
        return (scores if counts_reqd is False else
                (scores, {'mean': 0, 'max': 0, 'min': 0, 'lt5': 0, 'fpa': 0}))


def gaussian_node_score(node, parents, types, params, data):
    """
        Return specified types of decomposable scores for a categorical node.

        :param str node: node scores are required for
        :param dict parents: parents of non-orphan nodes {node: [parents]}
        :param list types: scores required e.g. ['loglik', 'aic', 'bde']
        :param dict params: parameters for scores e.g. logarithm base
        :param Data data: data to learn graph from
        :param bool counts_reqd: not relevant for Gaussian node, but retained
                                 so returned data has structure consistent with
                                 categorical nodes.

        :returns dict/tuple: scores {score_type: score} if counts not required
                             otherwise (scores, counts)
    """
    start = Timing.now()
    scale = len(parents[node]) + 1 if node in parents else 1

    if node not in parents:

        # Compute mean and corrected SD of values

        values = data.values((node,))[:, 0]
        mean = npmean(values)
        sd = npsqrt(npsum((values - mean) ** 2) / (data.N - 1))
        start = Timing.record('fit', scale, start)

        # sum log-likelihood of each value assuming Gaussian with
        # computed mean and SD.

        loglik = npsum(stats.norm.logpdf(values, loc=mean, scale=sd))
        start = Timing.record('loglik', scale, start)

        # print('\nOrphan node {}: mean {:.3f}, SD {:.3f}, loglik {:.6f}'
        #       .format(node, mean, sd, loglik))
        params = 2
    else:

        # Perform Linear Regression predicting child values from parents

        values = data.values(tuple([node] + parents[node]))
        model = LinearRegression().fit(values[:, 1:], values[:, 0])
        mean = model.intercept_.item()
        start = Timing.record('fit', scale, start)
        # coeffs = {p: model.coef_[i].item()
        #           for i, p in enumerate(parents[node])}
        # print('\nChild is {}, mean {:.6f} and coeffs {}'
        #       .format(node, mean, coeffs))

        # Compute residuals of actual - prediced value.
        # Square root of corrected mean of these is S.D. of noise

        residuals = values[:, 0] - model.predict(values[:, 1:])
        sd = npsqrt(npsum(residuals ** 2) / (data.N - 1))
        start = Timing.record('residuals', scale, start)
        # print('SD is {}'.format(sd))

        # Likelihood of predicted value is likelihood of residual using
        # Gaussian with zero mean and computed SD. NOTE can be > 1.0

        # loglik = sum([ln(stats.norm.pdf(r, 0.0, sd))
        #               for r in residuals])
        loglik = npsum(stats.norm.logpdf(residuals, loc=0.0, scale=sd))
        start = Timing.record('loglik', scale, start)
        # print('Node {} with parents {}: SD {:.3f}, loglik {:.6f}'
        #       .format(node, parents[node], sd, loglik))
        params = 2 + len(parents[node])

    # Subtract penalty value

    penalty = params * 0.5 * ln(data.N)
    bic_g = loglik - penalty
    # print('Penalty is {:.6f} making BIC {:.6f}'.format(penalty, bic_g))
    return {'bic-g': bic_g}


def categorical_node_score(node, parents, types, params, data,
                           counts_reqd=False):
    """
        Return specified types of decomposable scores for a categorical node.

        :param str node: node scores are required for
        :param dict parents: parents of non-orphan nodes {node: [parents]}
        :param list types: scores required e.g. ['loglik', 'aic', 'bde']
        :param dict params: parameters for scores e.g. logarithm base
        :param Data data: data to learn graph from
        :param bool counts_reqd: whether info about marginal counts returned

        :returns dict/tuple: scores {score_type: score} if counts not required
                             otherwise (scores, counts)
    """

    # get narginal counts from the data

    counts, q_i, _, _ = data.marginals(node, parents)

    # get statistics on contingency table counts

    if counts_reqd is True:
        counts_reqd = {'mean': counts.mean().item(),
                       'max': counts.max().item(),
                       'min': counts.min().item(),
                       'lt5': (counts < 5).sum() / counts.size,
                       'fpa': (counts.shape[0] - 1) * counts.shape[1]}

    scores = {}
    reqd_types = [t for t in types if t in ENTROPY_SCORES]
    if reqd_types:
        scores = entropy_scores(counts, reqd_types, params, data.N,
                                q_i * (counts.shape[0] - 1))

    for type in [t for t in types if t in BAYESIAN_SCORES]:
        scores[type] = bayesian_score(counts, q_i, type, params)

    return scores if counts_reqd is False else (scores, counts_reqd)


def dag_score(dag, data, types, params):
    """
        Returns per-node DAG+data network scores of specified types

        :param DAG dag: the DAG to score
        :param Data data: data to be scored
        :param list/str types: type(s) of score required e.g. 'bic', 'bde'
        :param dict params: parameters for scores e.g. logarithm base

        :raises TypeError: if args have bad types
        :raises ValueError: if args have bad values

        :returns DataFrame: scores (column=type) per node (rows)
    """
    if isinstance(types, str):
        types = [types]

    if type(dag).__name__ != 'DAG' or not isinstance(types, list) \
            or any([not isinstance(t, str) for t in types]) \
            or not isinstance(params, dict) or not isinstance(data, Data) \
            or isinstance(data, Oracle):
        raise TypeError('bad arg types for dag_score')

    if set(data.nodes) != set(dag.nodes):
        raise ValueError('data/dag variable mismatch for dag_score')
    allowed = (ENTROPY_SCORES + BAYESIAN_SCORES if data.dstype == 'categorical'
               else GAUSSIAN_SCORES)
    if not len(types) or any(t not in allowed for t in types):
        raise ValueError('bad score types for dag_score')

    params = params.copy()
    params = check_score_params(params, types)  # check params, set defaults

    # raise exception if any variables are single state if required

    if (('unistate_ok' not in params or params['unistate_ok'] is not True)
            and any([len(v.keys()) < 2 for n, v in data.node_values.items()])):
        raise ValueError('Some variables are single valued')

    # Loop over all nodes computing all required scores

    scores = {score_type: [] for score_type in types + ['node']}
    for node in dag.nodes:
        scores['node'].append(node)
        for score_type, value in \
                node_score(node, dag.parents, types, params,
                           data=data).items():
            scores[score_type].append(value)

    return DataFrame(scores).set_index('node')


def bn_score(bn, N, types, params):
    """
        Returns per-node scores of specified types for Bayesian Network
        assuming specified dataset size. Computes scores based on CPTs -
        hence is an "oracle" score

        :param bn BN: the BN to score
        :param int N: dataset size to use
        :param list/str types: type(s) of score required e.g. 'bic', 'bde'
        :param dict params: parameters for scores e.g. logarithm base

        :raises TypeError: if args have bad types
        :raises ValueError: if args have bad values

        :returns DataFrame: scores (column=type) per node (rows)
    """
    if isinstance(types, str):
        types = [types]

    if type(bn).__name__ != 'BN' or not isinstance(types, list) \
            or any([not isinstance(t, str) for t in types]) \
            or not isinstance(params, dict) or not isinstance(N, int) \
            or isinstance(N, bool):
        raise TypeError('bad arg types for bn_score')

    if not len(types) or any(t not in ENTROPY_SCORES + BAYESIAN_SCORES
                             for t in types):
        raise ValueError('bad score types for dag_score')

    if N < 1:
        raise ValueError('non-positive N for bn_score')

    params = params.copy()
    params = check_score_params(params, types)  # check params and set defaults

    # Loop over all nodes computing all required scores

    data = Oracle(bn=bn)
    data.set_N(N)
    scores = {score_type: [] for score_type in types + ['node']}
    for node in bn.dag.ordered_nodes():
        scores['node'].append(node)
        for score_type, value in node_score(node, bn.dag.parents, types,
                                            params, data).items():
            scores[score_type].append(value)

    return DataFrame(scores).set_index('node')


def check_score_params(params, scores=None):
    """
        Check score parameters specified by user and supply default values.
        Parameters supported are:
          * base: logarithm base used, 2, 10 or 'e'
          * iss: imaginary sample size, positive real
          * prior: structural prior, string 'uniform' or 'marginal'
          * k: complexity penalty multiplier

        :param dict params: parameters for scoring metrics
        :param list/None scores: score/list of scores which params used for

        :raises TypeError: if invalid types used
        :raises ValueError: if any invalid values, combinations specified

        :returns dict: of parameters {param: value}
    """
    if not isinstance(params, dict) or \
            (scores is not None and not isinstance(scores, list)):
        raise TypeError('Bad arg type for _check_score_params')

    if any([key not in SCORE_PARAMS for key in params]):
        raise ValueError('Unknown score parameter')

    # Determine required score parameters & check no irrelevant ones specified

    reqd = (list(SCORE_PARAMS) if scores is None
            else [p for s in scores for p in list(SCORES[s])])
    # if (set(params) - set(reqd)):
    #    raise ValueError('Irrelevant score parameters')

    # Use defaults if parameter not specified

    for p in reqd:
        if p not in params:
            params.update({p: SCORE_PARAMS[p]})

    # Check specified parameters have right type

    if 'base' in params and ((not isinstance(params['base'], int)
                              and not isinstance(params['base'], str))
                             or isinstance(params['base'], bool)):
        raise TypeError('Wrong type for base score parameter')
    if 'prior' in params and not isinstance(params['prior'], str):
        raise TypeError('Wrong type for prior base parameter')
    if 'iss' in params and not isinstance(params['iss'], int) \
            and not isinstance(params['iss'], float):
        raise TypeError('Wrong type for prior base parameter')
    if 'k' in params and not isinstance(params['k'], int) \
            and not isinstance(params['k'], float):
        raise TypeError('Wrong type for penalty multiplier')

    # Check specified parameters have valid values

    if 'base' in params and params['base'] not in [2, 10, 'e']:
        raise ValueError('Bad value for base score param')
    if 'prior' in params and params['prior'] not in ['uniform', 'marginal']:
        raise ValueError('Bad value for priot score param')
    if 'iss' in params and (params['iss'] < 1E-6 or params['iss'] > 1E6):
        raise ValueError('Bad value for prior score param')
    if 'k' in params and (params['k'] < 1E-6 or params['k'] > 1E6):
        raise ValueError('Bad value for penalty multiplier')

    return params


def free_params(graph, data, debug=False):
    """
        Compute number of free parameters for graph and data

        :param DAG graph: the dag to return value for
        :param DataFrame data: data for the graph
        :param bool debug: whether details of node parameters printed

        :returns: number of free params for graph with data
        :rtype: int
    """
    if debug:
        print('')

    value_counts = {}
    for col in data.columns:
        value_counts[col] = dict(data[col].value_counts())

    total_free_params = 0
    for node in graph.nodes:
        cardinality = len(value_counts[node])
        free_params = cardinality - 1
        if node in graph.parents:
            for parent in graph.parents[node]:
                free_params *= len(value_counts[parent])
        total_free_params += free_params
        if debug:
            num_parents = len(graph.parents[node]) if node in graph.parents \
                else 0
            if debug:
                print('{} has {} state(s), {} parent(s) and {} free param(s)'
                      .format(node, cardinality, num_parents, free_params))

    if debug:
        print('Graph has {} free params'.format(total_free_params))

    return total_free_params
