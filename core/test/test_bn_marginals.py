
import pytest
from itertools import combinations
from pandas import DataFrame

from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from core.graph import DAG
from core.bn import BN
from core.cpt import NodeValueCombinations
from core.metrics import dists_same


def global_distribution(bn):
    """
        Generate the global probability distribution for the specified BN.
        This method has been superseded by BN.marginals() which is more
        efficient and general. This version is just retained for testing.

        :param BN bn: The BN to generate the global distribution for.

        :returns DataFrame: global distribution in descending probability
                            (and then by ascending values)
    """

    # Generate possible values at every node {node: [poss values]}

    node_values = {n: c.node_values() for n, c in bn.cnds.items()}

    # Loop over all possible combinations of node values (i.e. a "case")
    # and collect the probability of each one

    values = {n: [] for n in bn.dag.nodes}
    probs = []
    for case in NodeValueCombinations(node_values):
        for node, value in case.items():
            values[node].append(value)
        lnprob = bn.lnprob_case(case)
        probs.append(0.0 if lnprob is None else 10 ** lnprob)

    # return DataFrame with correct dtypes and sorted by descending
    # probability, and then ascending value order

    return DataFrame(values, dtype='category') \
        .join(DataFrame({'': probs}, dtype='float64')) \
        .sort_values([''] + bn.dag.nodes, ignore_index=True,
                     ascending=[False] + [True] * len(bn.dag.nodes))


@pytest.fixture(scope="function")  # Heckerman BN used in many tests
def bn():
    # Creates DAG N1->N2 with 3 (1,1), 2 (1,2), 3 (2, 1) and 4 (2,2) cases
    dag = DAG(['N1', 'N2'], [('N1', '->', 'N2')])
    data = Pandas.read(TESTDATA_DIR + '/simple/heckerman.csv',
                       dstype='categorical')
    return BN.fit(dag, data)


def check_all_marginals(bn, bn_name):
    """
        Checks every marginal distribution for the specified bn. It does this
        by comparing the marginals created by module under test -
        bn.marginals() - with marginals created by doing groupby operations
        on the global distribution.

        :param BN bn: BN for which every possible marginal will be checked
        :param str desc: description of BN used in printed output

        :raises AssertionError: if any check fails
    """
    gd = global_distribution(bn)
    print('\n\nChecking all marginals for {}, global distribution:\n{}'
          .format(bn_name, gd))

    for n in range(1, len(bn.dag.nodes) + 1):
        for nodes in combinations(bn.dag.nodes, n):

            # Generate marginals using BN.marginals()

            nodes = list(nodes)
            marginals = bn.marginals(nodes)
            print('\nMarginals for {}:\n{}'.format(nodes, marginals))

            # Generate marginals using group by on global distribution

            # gd_marginals = gd.groupby(nodes).sum()
            gd_marginals = gd.groupby(nodes)[''].sum().to_frame()

            if len(nodes) > 1:

                # Multivariate distributions created by groupby have
                # MultiIndex row index. We need to change this so that row 
                # index is just the primary variable, and instead
                # MultIndex columns used for the other variables

                tmp = gd_marginals.to_dict()['']
                nvs = bn.cnds[nodes[0]].node_values()
                tmp = {vs[1:]: {nv: tmp[tuple([nv]) + vs[1:]]
                                for nv in nvs} for vs in tmp}
                gd_marginals = DataFrame(tmp)
                gd_marginals.index.name = nodes[0]
                gd_marginals.columns.names = nodes[1:]

            # Ensure two distributions are the same

            assert dists_same(marginals, gd_marginals)


def test_bn_marginals_type_error_1(bn):  # no args specified
    with pytest.raises(TypeError):
        bn.marginals()


def test_bn_marginals_type_error_2(bn):  # bad arg types
    with pytest.raises(TypeError):
        bn.marginals()
    with pytest.raises(TypeError):
        bn.marginals(77)
    with pytest.raises(TypeError):
        bn.marginals({})
    with pytest.raises(TypeError):
        bn.marginals(True)


def test_bn_marginals_value_error_1(bn):  # empty nodes argument
    with pytest.raises(ValueError):
        bn.marginals([])


def test_bn_marginals_value_error_2(bn):  # duplicates in nodes
    with pytest.raises(ValueError):
        bn.marginals(['N1', 'N1'])
    with pytest.raises(ValueError):
        bn.marginals(['N1', 'N2', 'N1'])


def test_bn_marginals_value_error_3(bn):  # nodes not in dag
    with pytest.raises(ValueError):
        bn.marginals(['unknown'])
    with pytest.raises(ValueError):
        bn.marginals(['N1', 'unknown'])


def test_bn_marginals_heckerman_ok(bn):  # Heckerman N1 -> N2
    check_all_marginals(bn, 'N1 -> N2')


def test_expt(bn):
    print()

    gd = bn.global_distribution().groupby(['N1', 'N2']).sum()
    print(gd)
    print(gd.to_dict()[''])

    print()
    md = bn.marginals(['N1', 'N2'])
    print(md)
    print(md.to_dict())

    print('\ntrying ...\n')
    gdm = gd.to_dict()['']
    print(gdm)
    res = {vs[1:]: {nv: gdm[tuple(nv) + vs[1:]] for nv in ["1", "2"]}
           for vs in gdm}
    res = DataFrame(res)
    print(res)


def test_bn_marginals_a_b_c_ok():  # A  B  C
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/a_b_c.dsc')
    check_all_marginals(bn, 'A  B  C')


def test_bn_marginals_ab_c_ok():  # A -> B  C
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_c.dsc')
    check_all_marginals(bn, 'A -> B  C')


def test_bn_marginals_abc_ok():  # A -> B -> C
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    check_all_marginals(bn, 'A -> B -> C')


def test_bn_marginals_ab_ac_ok():  # B <- A -> C
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_ac.dsc')
    check_all_marginals(bn, 'B <- A -> C')


def test_bn_marginals_ab_cb_ok():  # A -> B <- C
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_cb.dsc')
    check_all_marginals(bn, 'A -> B <- C')


def test_bn_marginals_abc_dual_ok():  # C <- A -> B -> C
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc_dual.dsc')
    check_all_marginals(bn, 'C <- A -> B -> C')


def test_bn_marginals_and4_10_ok():  # X1 -> X2 <- X3, X2 -> X4
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    check_all_marginals(bn, 'X1 -> X2 <- X3, X2 -> X4')


def test_bn_marginals_cancer_ok():  # Cancer
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    check_all_marginals(bn, 'Cancer')
