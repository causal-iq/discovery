
# Test the Pandas implementation of marginals

import pytest
from numpy import ndarray
from pandas import DataFrame, crosstab

from fileio.common import TESTDATA_DIR, EXPTS_DIR
from fileio.pandas import Pandas
from core.timing import Timing


@pytest.fixture(scope="module")  # AB, 2 rows
def ab2():
    df = DataFrame({'A': ['0', '1'], 'B': ['1', '1']}, dtype='category')
    return Pandas(df=df)


@pytest.fixture(scope="module")  # AB, 3 rows
def ab3():
    df = DataFrame({'A': ['1', '1', '0'], 'B': ['1', '0', '0']},
                   dtype='category')
    return Pandas(df=df)


@pytest.fixture(scope="module")  # AB, 8 rows
def ab8():
    df = DataFrame({'A': ['1', '1', '0', '1', '1', '1', '0', '0'],
                    'B': ['1', '0', '1', '1', '0', '0', '1', '1']},
                   dtype='category')
    return Pandas(df=df)


@pytest.fixture(scope="module")  # AB, 8 rows
def abc10():
    df = DataFrame({'A': ['1', '1', '0', '1', '1', '1', '0', '0', '0', '0'],
                    'B': ['1', '0', '1', '1', '0', '0', '1', '1', '1', '1'],
                    'C': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1']},
                   dtype='category')
    return Pandas(df=df)


@pytest.fixture(scope='module')  # Asia, 100 rows
def asia():
    return Pandas.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                       dstype='categorical', N=100)


def check(args, results, data):
    """
        Check results obtained match ones expected.

        :param tuple args: (node, parents, values_reqd) arguments
        :param tuple results: (counts, maxcol, rowval, colval) obtained
        :param Pandas data: underlying data used to check results
    """
    print(('\n\nNode: {}, parents: {}, values_reqd: {}\n\n' +
           'Counts:\n{}\nType: {}, memory: {} bytes\nMax parent values: {}'
           '\nChild values: {}\nParent values: {}')
          .format(args[0], args[1], args[2], results[0], results[0].dtype,
                  results[0].nbytes, results[1], results[2], results[3]))

    # assert results have correct types and match expected

    assert isinstance(results[0], ndarray)
    assert results[0].dtype == 'int32'

    # Check shape and maxcol returned by old marginals

    orig, orig_num_pvs = orig_marginals(data, args[0], args[1])
    assert results[0].shape == orig.shape
    assert results[1] == orig_num_pvs
    print('\n{}\n'.format(orig.head()))

    # if requesting values, then can compare old and new counts too

    if results[2] is not None:
        if results[3] is None:
            for i, xi in enumerate(results[2]):
                count = orig.loc[xi].values[0]
                assert count == results[0][i, 0]
        else:

            # get levels in original column multi-index which varies

            levels = {i: n for i, n in enumerate(orig.columns.names)}

            for j, pvs in enumerate(results[3]):

                # Get parental values in variable order in levels for
                # column key used in loc(). Then compare all row counts

                pvs = tuple(pvs[levels[i]] for i in levels)
                for i, xi in enumerate(results[2]):
                    assert orig.loc[xi, pvs] == results[0][i, j]


def orig_marginals(self, node, parents):
    """
        Original marginals code.

        :param str node: node for which marginals required.
        :param dict parents: {node: parents} parents of non-orphan nodes

        :returns tuple: marginals, DataFrame - marginal counts, row index
                        is node value, column multi-index are parent values
                        num_pvs int - number of parental value combinations
                        (which may not be all present in data though)
    """
    num_pvs = 1
    if node in parents:  # node has parents - get contingency table
        marginals = crosstab(self.sample[node], [self.sample[p] for p in
                                                 parents[node]]).copy()
        for p in parents[node]:
            num_pvs *= len(self.node_values[p].keys())

    else:  # node has no parents - simple value frequencies
        marginals = self.sample[node].value_counts().to_frame().copy()

    return (marginals, num_pvs)


def test_pandas_marginals_type_error_1(ab2):  # no arguments specified
    with pytest.raises(TypeError):
        ab2.marginals()


def test_pandas_marginals_type_error_2(ab2):  # bad or missing node
    with pytest.raises(TypeError):
        ab2.marginals(node=1, parents={})
    with pytest.raises(TypeError):
        ab2.marginals(node=12.9, parents={})
    with pytest.raises(TypeError):
        ab2.marginals(node=None, parents={})
    with pytest.raises(TypeError):
        ab2.marginals(node=['A'], parents={})
    with pytest.raises(TypeError):
        ab2.marginals(parents={})


def test_pandas_marginals_type_error_3(ab2):  # bad or missing parents
    with pytest.raises(TypeError):
        ab2.marginals(node='A', parents=2)
    with pytest.raises(TypeError):
        ab2.marginals(node='A', parents='B')
    with pytest.raises(TypeError):
        ab2.marginals(node='A', parents=None)
    with pytest.raises(TypeError):
        ab2.marginals(node='A')


def test_pandas_marginals_type_error_4(ab2):  # parent values bad type
    with pytest.raises(TypeError):
        ab2.marginals(node='A', parents={'A': 'B'})
    with pytest.raises(TypeError):
        ab2.marginals(node='A', parents={'A': {'B'}})
    with pytest.raises(TypeError):
        ab2.marginals(node='A', parents={'A': ('B', )})


def test_pandas_marginals_type_error_5(ab2):  # bad values_reqd type
    with pytest.raises(TypeError):
        ab2.marginals(node='A', parents={}, values_reqd=1)
    with pytest.raises(TypeError):
        ab2.marginals(node='A', parents={}, values_reqd=None)
    with pytest.raises(TypeError):
        ab2.marginals(node='A', parents={}, values_reqd={False})
    with pytest.raises(TypeError):
        ab2.marginals(node='A', parents={}, values_reqd=[False])
    with pytest.raises(TypeError):
        ab2.marginals(node='A', parents={}, values_reqd=(False, ))


def test_pandas_marginals_ab2_1_ok(ab2):
    args = ('A', {}, True)
    results = ab2.marginals(args[0], args[1], args[2])
    check(args, results, ab2)


def test_pandas_marginals_ab2_2_ok(ab2):
    args = ('A', {}, False)
    results = ab2.marginals(args[0], args[1], args[2])
    check(args, results, ab2)


def test_pandas_marginals_ab2_3_ok(ab2):
    args = ('B', {}, True)
    results = ab2.marginals(args[0], args[1], args[2])
    check(args, results, ab2)


def test_pandas_marginals_ab2_4_ok(ab2):
    args = ('A', {'A': ['B']}, True)
    results = ab2.marginals(args[0], args[1], args[2])
    check(args, results, ab2)


def test_pandas_marginals_ab2_5_ok(ab2):
    args = ('B', {'B': ['A']}, True)
    results = ab2.marginals(args[0], args[1], args[2])
    check(args, results, ab2)


def test_pandas_marginals_ab3_1_ok(ab3):
    args = ('A', {}, True)
    results = ab3.marginals(args[0], args[1], args[2])
    check(args, results, ab3)


def test_pandas_marginals_ab3_2_ok(ab3):
    args = ('B', {}, True)
    results = ab3.marginals(args[0], args[1], args[2])
    check(args, results, ab3)


def test_pandas_marginals_ab3_3_ok(ab3):
    args = ('A', {'A': ['B']}, True)
    results = ab3.marginals(args[0], args[1], args[2])
    check(args, results, ab3)


def test_pandas_marginals_ab3_4_ok(ab3):
    args = ('B', {'B': ['A']}, True)
    results = ab3.marginals(args[0], args[1], args[2])
    check(args, results, ab3)


def test_pandas_marginals_ab8_1_ok(ab8):
    args = ('A', {}, True)
    results = ab8.marginals(args[0], args[1], args[2])
    check(args, results, ab8)


def test_pandas_marginals_ab8_2_ok(ab8):
    args = ('B', {}, True)
    results = ab8.marginals(args[0], args[1], args[2])
    check(args, results, ab8)


def test_pandas_marginals_ab8_3_ok(ab8):
    args = ('A', {'A': ['B']}, True)
    results = ab8.marginals(args[0], args[1], args[2])
    check(args, results, ab8)


def test_pandas_marginals_ab8_4_ok(ab8):
    args = ('B', {'B': ['A']}, True)
    results = ab8.marginals(args[0], args[1], args[2])
    check(args, results, ab8)


def test_pandas_marginals_abc10_1_ok(abc10):
    args = ('A', {}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


def test_pandas_marginals_abc10_2_ok(abc10):
    args = ('A', {'A': ['B']}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


def test_pandas_marginals_abc10_3_ok(abc10):
    args = ('A', {'A': ['C']}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


def test_pandas_marginals_abc10_4_ok(abc10):
    args = ('A', {'A': ['B', 'C']}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)
    print('\n{}\n'.format(abc10.df))


def test_pandas_marginals_abc10_5_ok(abc10):
    args = ('B', {}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


def test_pandas_marginals_abc10_6_ok(abc10):
    args = ('B', {'B': ['A']}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


def test_pandas_marginals_abc10_7_ok(abc10):
    args = ('B', {'B': ['C']}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


def test_pandas_marginals_abc10_8_ok(abc10):
    args = ('B', {'B': ['A', 'C']}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)
    print('\n{}\n'.format(abc10.df))


def test_pandas_marginals_abc10_9_ok(abc10):
    args = ('C', {}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


def test_pandas_marginals_abc10_10_ok(abc10):
    args = ('C', {'C': ['A']}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


def test_pandas_marginals_abc10_11_ok(abc10):
    args = ('C', {'C': ['B']}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


def test_pandas_marginals_abc10_12_ok(abc10):
    args = ('C', {'C': ['A', 'B']}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


def test_pandas_marginals_asia_1_ok(asia):
    args = ('asia', {}, True)
    results = asia.marginals(args[0], args[1], args[2])
    check(args, results, asia)


def test_pandas_marginals_asia_2_ok(asia):
    args = ('asia', {'asia': ['bronc']}, True)
    results = asia.marginals(args[0], args[1], args[2])
    check(args, results, asia)


def test_pandas_marginals_asia_3_ok(asia):
    args = ('asia', {'asia': ['bronc', 'tub']}, True)
    results = asia.marginals(args[0], args[1], args[2])
    check(args, results, asia)


def test_pandas_marginals_asia_4_ok(asia):
    args = ('xray', {'xray': ['lung', 'smoke', 'either']}, True)
    results = asia.marginals(args[0], args[1], args[2])
    check(args, results, asia)


def test_pandas_marginals_asia_5_ok(asia):  # check operation of set_N
    asia = Pandas.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                       dstype='categorical', N=1000)
    asia.set_N(500)
    args = ('xray', {'xray': ['lung', 'smoke', 'either']}, True)
    results = asia.marginals(args[0], args[1], args[2])
    check(args, results, asia)


def test_pandas_marginals_covid_1_ok():
    covid = Pandas.read(EXPTS_DIR + '/datasets/covid.data.gz',
                        dstype='categorical', N=1000000)
    args = ('Lockdown', {'Lockdown': ['New_infections', 'Season']}, True)
    Timing.on(True)
    results = covid.marginals(args[0], args[1], args[2])
    check(args, results, covid)
    print(Timing)
