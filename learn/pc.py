
#   PC family of constraint-based structure learning algorithms

from pandas import DataFrame
from itertools import combinations

from fileio.pandas import Pandas
from core.bn import BN
from core.indep import indep


def _check_ci_args(data, bn, N):
    """
        Check arguments for CI learning functions are valid

        :param DataFrame/str data: data (file) to generate & score graphs from
        :param BN/str bn: BN (file) supplying CPT parameters for CI tests
        :param int N: Sample size to use when using BN parameters
        :param str type: Type of CI test to use: 'mi' or 'x2'
        :param float alpha: CI test significance level

        :raises TypeError: if bad argument types used
        :raises FileNotFoundError: if non-existent file specified
        :raises FileFormatError: if specified file has invalid format
        :raises ValueError: if data supplied unacceptable

        :return tuple: ([dependencies], [independencies])
    """
    if data is not None and bn is not None:
        raise TypeError('sgs_skeleton: both data and bn specified')

    if isinstance(data, str):
        data = Pandas.read(data, dstype='categorical').df
    if isinstance(bn, str):
        bn = BN.read(bn)

    if (bn is None and not isinstance(data, DataFrame)) \
            or (data is None and not isinstance(bn, BN)) \
            or (data is None and isinstance(N, bool)) \
            or (data is None and not isinstance(N, int)):
        raise TypeError('sgs_skeleton bad argument types')

    return (data, bn)


def sgs_skeleton(data, bn=None, N=1, type='mi', alpha=0.05):
    """
        Generate skeleton using CI tests and exhaustive approach of SGS. Can
        use either learn from data or BN parameters

        :param DataFrame/str data: data (file) to generate & score graphs from
        :param BN bn: BN supplying CPT parameters for CI tests
        :param int N: Sample size to use when using BN parameters
        :param str type: Type of CI test to use: 'mi' or 'x2'
        :param float alpha: CI test significance level

        :raises TypeError: if bad argument types used
        :raises FileNotFoundError: if non-existent file specified
        :raises FileFormatError: if specified file has invalid format
        :raises ValueError: if data supplied unacceptable

        :return tuple: ([dependencies], [independencies])
    """

    data, bn = _check_ci_args(data, bn, N)

    nodes = bn.dag.nodes if data is None else sorted(list(data.columns))

    sepsets = {}
    for edge in combinations(nodes, 2):
        rest = sorted(list(set(nodes) - set(edge)))
        max_p_value = None
        sepset = None
        for size in range(0, len(rest) + 1):
            if size == 0:
                cond_sets = [[]]
            elif size == len(rest):
                cond_sets = [rest]
            else:
                cond_sets = combinations(rest, size)
            for cond_set in cond_sets:
                p_value = (indep(edge[0], edge[1], list(cond_set), data)
                           )['mi'].to_dict()['p_value']
                if sepset is None or p_value > max_p_value:
                    max_p_value = p_value
                    sepset = cond_set
                print(edge, cond_set, p_value)
        sepsets[edge] = (sepset, max_p_value)

    return sepsets
