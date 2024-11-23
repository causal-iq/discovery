#
#   Class for metrics and comparisons of values, distributions, graphs
#   and networks
#

from pandas import Series, DataFrame
from math import floor, log10, isnan

from core.common import EdgeType, ln


def values_same(value1, value2, sf=3):
    """
    Tests whether two numeric values are the same to a specified number
    of significant figures.

    :param int/float/bool value1: first value in comparison
    :param int/float/bool value2: second value in comparison
    :param int sf: number of significant figures used in comparison

    :raises TypeError: if any arg not of required type. Since this function
                       must be very efficient we rely on the Python
                       standard functions to signal TypeErrors rather than
                       explicitly testing the argument types.

    :returns bool: whether two values are the same to specified number of s.f.
    """

    # Handle zero and NaNs explicitly  - all zeros are considered the same,
    # but unlike standard Python, two NaNs comapre as True

    if value1 == 0 or value2 == 0:
        return value1 == value2
    isnan_1 = isnan(value1)
    isnan_2 = isnan(value2)
    if isnan_1 or isnan_2:
        return isnan_1 == isnan_2

    # Quick pre-check: quickly determine if values differ by more than a factor
    # of bound_m which is an upper bound on the ratio of numbers at a specific
    # sf value.

    abs_value1 = abs(value1)
    abs_value2 = abs(value2)
    bound_m = 1.0 + 10 ** (1 - sf)
    if abs_value1 > bound_m * abs_value2 or abs_value2 > bound_m * abs_value1:
        return False

    # Compute the scaled values for comparison

    scale1 = round(value1, -int(floor(log10(abs_value1)) - (sf - 1)))
    scale2 = round(value2, -int(floor(log10(abs_value2)) - (sf - 1)))

    return scale1 == scale2


def dicts_same(dict1, dict2, sf=10, strict=True):
    """
        Returns whether two dicts of values have same values to a specified
        number of significant digits

        :param dict dict1: first dictionary of values
        :param dict dict2: second dictionary of values
        :param int sf: number of significant figures used in comparisons
        :param bool strict: whether two dicts must contain same keys

        :raises TypeError: if any arg not of required type

        :returns bool: whether two dicts are same
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict) \
            or not isinstance(sf, int) or isinstance(sf, bool) \
            or not isinstance(strict, bool):
        raise TypeError('Bad arg types for dicts_same')

    if strict and dict1.keys() != dict2.keys():
        raise TypeError('Two dicts have different keys and strict is True')

    same = True
    for key, value in dict1.items():
        if key not in dict2 or (dict1[key] is None and dict2[key] is None):
            continue
        if ((dict1[key] is None and dict2[key] is not None) or
            (dict1[key] is not None and dict2[key] is None) or
                not values_same(dict1[key], dict2[key], sf=sf)):
            same = False
            break

    return same


def dists_same(df1, df2, sf=10):
    """
        Test whether two distributions are the same, with their probabilities
        agreeing to a specified number of significant digits.

        :param DataFrame df1: first distribution
        :param DataFrame df2: second distrubution
        :param int sf: number of sig. figures used in probability comparisons

        :raises TypeError: if any arg not of required type

        :returns bool: whether the two distributions are same to the specified
                       number of significant figures.
    """
    if not isinstance(df1, DataFrame) or not isinstance(df2, DataFrame):
        raise TypeError('dists_same() bad arg types')

    if sorted(list(df1.index)) != sorted(list(df2.index)) \
            or df1.index.name != df2.index.name:
        print('\ndists_same: different primary variable/values')
        return False

    if list(df1.columns.names) != list(df2.columns.names):
        print('\ndists_same: different secondary variables')
        return False

    dict1 = df1.to_dict()
    dict2 = df2.to_dict()

    if dict1.keys() != dict2.keys():
        print('\ndists_same: different secondary values')
        return False

    for values, pmf in dict1.items():
        if not dicts_same(pmf, dict2[values], sf):
            print('\ndists_same: different probabilities')
            return False

    return True


def pdag_compare(graph, reference, bayesys=None, identify_edges=False):
    """
        Compare a pdag with a reference pdag

        :param PDAG graph: graph which is to be compared
        :param PDAG reference: reference graph for comparison
        :param str/None bayesys: version of Bayesys metrics to return, or None
                                 if not required
        :param bool identify_edges: whether edges in each low level category
                                    (e.g. arc_missing) are to be included in
                                    metrics returned.

        :raises TypeError: if bad argument types

        :returns dict: structural comparison metrics
    """
    def _metric(ref_type, type, reversed=False):  # identify count metric name
        if ref_type == EdgeType.DIRECTED and type == EdgeType.DIRECTED:
            return 'arc_matched' if not reversed else 'arc_reversed'
        elif ref_type == EdgeType.DIRECTED and type == EdgeType.UNDIRECTED:
            return 'edge_not_arc'
        elif ref_type == EdgeType.UNDIRECTED and type == EdgeType.DIRECTED:
            return 'arc_not_edge'
        elif ref_type == EdgeType.DIRECTED and type is None:
            return 'arc_missing'
        elif ref_type == EdgeType.UNDIRECTED and type is None:
            return 'edge_missing'
        else:
            return 'edge_matched'

    edges = graph.edges
    ref_edges = reference.edges

    metrics = {'arc_matched': 0, 'arc_reversed': 0, 'edge_not_arc': 0,
               'arc_not_edge': 0, 'edge_matched': 0, 'arc_extra': 0,
               'edge_extra': 0, 'arc_missing': 0, 'edge_missing': 0}
    metric_edges = {m: set() for m in metrics} if identify_edges else None

    # Loop over all edges in tested graph looking for match in reference graph
    # Include case of arcs that have same type but are oppositely orientated
    # Count edges/arcs in graph not in reference graph too

    for e, t in edges.items():
        if e in ref_edges:
            metric = _metric(ref_edges[e], t)
        elif (e[1], e[0]) in ref_edges:
            metric = _metric(ref_edges[(e[1], e[0])], t, reversed=True)
        else:
            metric = 'arc_extra' if t == EdgeType.DIRECTED else 'edge_extra'
        metrics[metric] += 1
        if identify_edges is True:
            metric_edges[metric].add(e)

    # loop over edges in reference not in graph

    for e, t in ref_edges.items():
        if e not in edges and (e[1], e[0]) not in edges:
            metric = _metric(t, None)
            metrics[metric] += 1
            if identify_edges is True:
                metric_edges[metric].add(e)

    max_edges = int(0.5 * len(reference.nodes) * (len(reference.nodes) - 1))
    metrics.update({'missing_matched': max_edges - sum(metrics.values())})

    # compute standard and edge SHD metrics and perform sanity check

    shd_e = metrics['arc_extra'] + metrics['edge_extra'] \
        + metrics['arc_missing'] + metrics['edge_missing']
    shd = shd_e + metrics['arc_reversed'] + metrics['arc_not_edge'] \
        + metrics['edge_not_arc']
    tp = metrics['arc_matched'] + metrics['edge_matched']

    # Alternative computation allowing weighting of reversed and edge/arc
    # to be varied more easily.

    mis = (1.0 * (metrics['arc_not_edge'] + metrics['edge_not_arc']) +
           1.0 * metrics['arc_reversed'])
    fp = metrics['arc_extra'] + metrics['edge_extra'] + mis
    fn = metrics['arc_missing'] + metrics['edge_missing'] + mis
    p = tp / (tp + fp) if tp + fp > 0 else None
    r = tp / (tp + fn) if tp + fn > 0 else None
    f1 = 0.0 if p is None or r is None or (p == 0 and r == 0) \
        else 2 * p * r / (p + r)
    if tp + metrics['missing_matched'] + shd != max_edges:
        raise RuntimeError('SHD sanity check: {}'.format(metrics))
    metrics.update({'shd': shd, 'p': p, 'r': r, 'f1': f1})

    # add in Bayesys metrics and edge details if required

    if bayesys is not None:
        metrics.update(bayesys_metrics(metrics, max_edges,
                                       len(reference.edges), bayesys))
    if identify_edges:
        metrics.update({'edges': metric_edges})

    return metrics


def kl(dist, ref_dist):
    """
        Returns the Kullback-Liebler Divergence of dist from ref_dist

        :param Series dist: distribution to compute KL from ...
        :param Series ref_dist: ... the reference/theoretical distribution

        :raises TypeError: if both arguments not Panda Series
        :raises ValueError: if dists have different indices or bad values

        :returns float: divergence value
    """
    if not isinstance(dist, Series) or not isinstance(ref_dist, Series):
        raise TypeError("kl() called with bad argument types")

    if set(dist.index) != set(ref_dist.index):
        raise ValueError("kl: dist and ref_dist indices different")
    if dist.hasnans or ref_dist.hasnans:
        raise ValueError("kl: distributions contain NaNs")
    if dist.max() > 1.000001 or dist.min() < -0.000001 or \
            ref_dist.max() > 1.000001 or ref_dist.min() <= -0.000001:
        raise ValueError("kl: distributions with bad values")

    result = 0.0
    for key, prob in dist.items():
        prob = prob if prob > 0 else 1E-16
        ref_prob = ref_dist[key] if ref_dist[key] > 0 else 1E-16
        result += prob * ln(prob / ref_prob)

    return result


def bayesys_metrics(metrics, max_edges, num_ref_edges, version):

    # Compute true/false postive/negatives
    # If reference has an edge but graph has arc this is considered a match
    # Bayesys comparison introduces concept of a "half-match" when graph has
    # an edge, or an oppositely orientated arc compared to reference.
    # Note edge_matched is not counted as TP giving incorrectly high shd-b for
    # CPDAG comparisons
    # This implementation has extra protection against divide by zero errors

    TP = float(metrics['arc_matched'] + metrics['arc_not_edge'])
    TP2 = float(metrics['arc_reversed'] + metrics['edge_not_arc'])
    FP = float(metrics['arc_extra'] + metrics['edge_extra'])
    TN = max_edges - num_ref_edges - FP
    FN = num_ref_edges - TP - 0.5 * TP2

    # Precision, recall and F1 but allowing for half-matches.

    precision = 1.0 if TP + TP2 + FP == 0 else \
        ((TP + 0.5 * TP2) / (TP + TP2 + FP) if version != 'v1.3'
         else (TP + 0.5 * TP2) / (TP + 0.5 * TP2 + FP))  # bug pre Bayesys v1.5
    recall = 1.0 if TP + TP2 + FN == 0 else \
        (TP + 0.5 * TP2) / (TP + 0.5 * TP2 + FN)
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall > 0 else 0.0  # Bayesys code sets this to 1.0!

    # SHD computed as usual but now includes half-matches
    # BSF and DDM as defined by Constantinou

    SHD = FP + FN
    DDM = (TP + 0.5 * TP2 - FN - FP) / num_ref_edges
    positive_worth = 1.0 / num_ref_edges
    negative_worth = 1.0 / (max_edges - num_ref_edges) if \
        max_edges != num_ref_edges else 1.0
    BSF = 0.5 * ((TP + 0.5 * TP2) * positive_worth + TN * negative_worth -
                 FP * negative_worth - FN * positive_worth)

    return {'tp-b': TP, 'tp2-b': TP2, 'fp-b': FP, 'tn-b': TN, 'fn-b': FN,
            'p-b': precision, 'r-b': recall, 'f1-b': f1,
            'shd-b': SHD, 'ddm': DDM, 'bsf': BSF}
