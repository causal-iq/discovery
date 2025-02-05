
# Testing the Pandas implementation of Data

import pytest
from pandas import DataFrame, read_csv
from numpy import ndarray, NaN

from fileio.common import TESTDATA_DIR
from fileio.data import Data
from fileio.pandas import Pandas
from core.bn import BN


@pytest.fixture(scope="module")
def data():
    return Pandas(df=read_csv(TESTDATA_DIR + '/simple/xyz_10.csv',
                              dtype='float32'))


# Constructor errors

def test_data_type_error_1():  # cannot call constructor directly
    with pytest.raises(TypeError):
        Data()


def test_constructor_type_error_1():  # no arguments specified
    with pytest.raises(TypeError):
        Pandas(df=None)


def test_constructor_type_error_2():  # bad df type
    with pytest.raises(TypeError):
        Pandas(df=None)
    with pytest.raises(TypeError):
        Pandas(df=2)
    with pytest.raises(TypeError):
        Pandas(df=False)
    with pytest.raises(TypeError):
        Pandas(df=12.7)
    with pytest.raises(TypeError):
        Pandas(df=[2])


def test_constructor_type_error_3():  # both df and bn specified
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    df = DataFrame({'A': ['0', '1'], 'B': ['0', '1']})
    with pytest.raises(TypeError):
        Pandas(df=df, bn=bn)


def test_constructor_value_error_1():  # missing data unsupported
    df = DataFrame({'A': ['0', NaN, '1'], 'B': ['0', '0', '1']},
                   dtype='category')
    with pytest.raises(ValueError):
        Pandas(df=df)


# Test constructor setting member variables correctly

def test_constructor_ab3_ok_1_():  # A,B dataset with 2 rows
    df = read_csv(TESTDATA_DIR + '/simple/ab_3.csv', dtype='category')

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert data.order == (0, 1)
    assert data.nodes == ('A', 'B')
    assert data.N == 3
    assert data.ext_to_orig == {'A': 'A', 'B': 'B'}
    assert data.node_values == {'A': {'0': 1, '1': 2},
                                'B': {'0': 2, '1': 1}}
    assert data.node_types == {'A': 'category',
                               'B': 'category'}
    assert data.dstype == 'categorical'

    assert data.get_order() == ('A', 'B')

    df = read_csv(TESTDATA_DIR + '/simple/ab_3.csv', dtype='category')
    assert (data.as_df() == df).all().all()


def test_constructor_abc5_1_ok():  # A,B,C dataset with 5 rows
    df = read_csv(TESTDATA_DIR + '/simple/abc_5.csv', dtype='category')

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2)
    assert data.nodes == ('A', 'B', 'C')
    assert data.N == 5
    # print('\n\n{}\n'.format(df))

    assert data.ext_to_orig == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.node_values == {'A': {'0': 2, '1': 3},
                                'B': {'0': 2, '1': 3},
                                'C': {'0': 4, '1': 1}}
    assert data.node_types == {'A': 'category',
                               'B': 'category',
                               'C': 'category'}
    assert data.dstype == 'categorical'

    assert data.get_order() == ('A', 'B', 'C')
    assert (data.as_df() == df).all().all()


def test_constructor_abc36_1_ok():  # A,B,C dataset with 36 rows
    df = read_csv(TESTDATA_DIR + '/simple/abc_36.csv', dtype='category')

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2)
    assert data.nodes == ('A', 'B', 'C')
    assert data.N == 36
    # print('\n\n{}\n'.format(df))

    assert data.ext_to_orig == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.node_values == {'A': {'0': 10, '1': 26},
                                'B': {'0': 14, '1': 22},
                                'C': {'0': 16, '1': 20}}
    assert data.node_types == {'A': 'category',
                               'B': 'category',
                               'C': 'category'}
    assert data.dstype == 'categorical'

    assert data.get_order() == ('A', 'B', 'C')
    assert (data.as_df() == df).all().all()


def test_constructor_xy3_1_ok():  # XY with 3 continuous rows
    df = read_csv(TESTDATA_DIR + '/simple/xy_3.csv', dtype='float32')

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1)
    assert data.nodes == ('F1', 'F2')
    assert data.N == 3

    assert data.ext_to_orig == {'F1': 'F1', 'F2': 'F2'}
    assert data.orig_to_ext == {'F1': 'F1', 'F2': 'F2'}
    assert data.node_values == {}
    assert data.node_types == {'F1': 'float32',
                               'F2': 'float32'}
    assert data.dstype == 'continuous'

    assert data.get_order() == ('F1', 'F2')
    assert (data.as_df() == df).all().all()


def test_constructor_cancer_1_ok():  # Cancer dataset with 10 rows
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                  dtype='category', nrows=10)

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2, 3, 4)
    assert data.nodes == ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray')
    assert data.N == 10
    assert data.ext_to_orig == \
        {'Cancer': 'Cancer', 'Dyspnoea': 'Dyspnoea', 'Pollution': 'Pollution',
         'Smoker': 'Smoker', 'Xray': 'Xray'}
    assert data.orig_to_ext == \
        {'Cancer': 'Cancer', 'Dyspnoea': 'Dyspnoea', 'Pollution': 'Pollution',
         'Smoker': 'Smoker', 'Xray': 'Xray'}
    assert data.node_values == \
        {'Cancer': {'False': 10},
         'Dyspnoea': {'False': 5, 'True': 5},
         'Pollution': {'low': 7, 'high': 3},
         'Smoker': {'False': 7, 'True': 3},
         'Xray': {'negative': 8, 'positive': 2}}
    assert data.node_types == {'Cancer': 'category',
                               'Dyspnoea': 'category',
                               'Pollution': 'category',
                               'Smoker': 'category',
                               'Xray': 'category'}
    assert data.dstype == 'categorical'

    assert data.get_order() == \
        ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray')
    assert (data.sample == df).all().all()


def test_constructor_asia_1_ok():  # Asia dataset with 100 rows
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.N == 100
    assert data.ext_to_orig == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.node_values == \
        {'asia': {'no': 97, 'yes': 3},
         'bronc': {'no': 56, 'yes': 44},
         'dysp': {'no': 56, 'yes': 44},
         'either': {'no': 92, 'yes': 8},
         'lung': {'no': 93, 'yes': 7},
         'smoke': {'no': 57, 'yes': 43},
         'tub': {'no': 99, 'yes': 1},
         'xray': {'no': 90, 'yes': 10}}
    assert data.node_types == \
        {'asia': 'category',
         'bronc': 'category',
         'dysp': 'category',
         'either': 'category',
         'lung': 'category',
         'smoke': 'category',
         'tub': 'category',
         'xray': 'category'}
    assert data.dstype == 'categorical'

    assert data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert (data.sample == df).all().all()


# Test set_N function

def test_set_N_type_error_1():  # Asia, N=100 - no args
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)

    with pytest.raises(TypeError):
        data.set_N()


def test_set_N_type_error_2():  # Asia, N=100 - non-integer arg
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)

    with pytest.raises(TypeError):
        data.set_N(2.1)
    with pytest.raises(TypeError):
        data.set_N(True)
    with pytest.raises(TypeError):
        data.set_N([2])


def test_set_N_type_error_3():  # Asia, N=100 - invalid seed type
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)

    with pytest.raises(TypeError):
        data.set_N(N=10, seed=True)
    with pytest.raises(TypeError):
        data.set_N(N=10, seed=[1])
    with pytest.raises(TypeError):
        data.set_N(N=10, seed=2.1)


def test_set_N_type_error_4():  # Asia, N=100 - invalid random_selection type
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)

    with pytest.raises(TypeError):
        data.set_N(N=10, random_selection=1)
    with pytest.raises(TypeError):
        data.set_N(N=10, random_selection=1)
    with pytest.raises(TypeError):
        data.set_N(N=10, random_selection={True})


def test_set_N_value_error_1():  # Asia, N=100 - set non-positive N
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)

    with pytest.raises(ValueError):
        data.set_N(0)
    with pytest.raises(ValueError):
        data.set_N(-3)


def test_set_N_value_error_2():  # Asia, N=100 - larger than amount of data
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)

    with pytest.raises(ValueError):
        data.set_N(101)


def test_set_N_value_error_3():  # Asia, N=100 - invalid seed values
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)

    with pytest.raises(ValueError):
        data.set_N(80, seed=-1)
    with pytest.raises(ValueError):
        data.set_N(80, seed=101)


def test_set_N_value_error_4():  # Asia, N=100 - random_selection unsupported
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)

    with pytest.raises(ValueError):
        data.set_N(80, random_selection=True)


def test_set_N_abc5_1_ok():  # ABC, 5 discrete rows, randomising order
    df = read_csv(TESTDATA_DIR + '/simple/abc_5.csv', dtype='category')
    data = Pandas(df=df)
    print('\n\nOriginal Dataset:\n{}\n'.format(data.as_df()))
    assert data.node_values == \
        {'A': {'1': 3, '0': 2},
         'B': {'1': 3, '0': 2},
         'C': {'0': 4, '1': 1}}
    assert data.as_df().to_dict(orient='list') == \
        {'A': ['0', '0', '1', '1', '1'],
         'B': ['0', '1', '0', '1', '1'],
         'C': ['0', '0', '0', '1', '0']}

    data.set_N(3)
    print('\n\nSetting N=3:\n{}\n'.format(data.as_df()))
    assert data.node_values == \
        {'A': {'0': 2, '1': 1},
         'B': {'0': 2, '1': 1},
         'C': {'0': 3}}
    assert data.as_df().to_dict(orient='list') == \
        {'A': ['0', '0', '1'],
         'B': ['0', '1', '0'],
         'C': ['0', '0', '0']}

    data.set_N(4, seed=1)
    print('\n\nSetting N=4, seed=1:\n{}\n'.format(data.as_df()))
    assert data.node_values == \
        {'A': {'0': 2, '1': 2},
         'B': {'0': 2, '1': 2},
         'C': {'0': 3, '1': 1}}
    assert data.as_df().to_dict(orient='list') == \
        {'A': ['1', '1', '0', '0'],
         'B': ['1', '0', '0', '1'],
         'C': ['1', '0', '0', '0']}

    data.set_N(4, seed=2)
    print('\n\nSetting N=4, seed=2:\n{}\n'.format(data.as_df()))
    assert data.node_values == \
        {'A': {'0': 2, '1': 2},
         'B': {'0': 2, '1': 2},
         'C': {'0': 3, '1': 1}}
    assert data.as_df().to_dict(orient='list') == \
        {'A': ['1', '1', '0', '0'],
         'B': ['0', '1', '1', '0'],
         'C': ['0', '1', '0', '0']}

    data.set_N(5)
    print('\n\nSetting N=5, no seed:\n{}\n'.format(data.as_df()))
    assert data.node_values == \
        {'A': {'1': 3, '0': 2},
         'B': {'1': 3, '0': 2},
         'C': {'0': 4, '1': 1}}
    assert data.as_df().to_dict(orient='list') == \
        {'A': ['0', '0', '1', '1', '1'],
         'B': ['0', '1', '0', '1', '1'],
         'C': ['0', '0', '0', '1', '0']}


def test_set_N_xyz10_1_ok():  # XYZ, 10 continuous rows, randomising order
    df = read_csv(TESTDATA_DIR + '/simple/xyz_10.csv', dtype='float32')
    data = Pandas(df=df)

    print('\n\nOriginal Dataset:\n{}\n'.format(data.as_df()))
    rdf = data.as_df().applymap(lambda x: round(x, 1))
    assert rdf.to_dict(orient='list') == \
        {'X': [1.1, 0.0, 0.2, 4.4, 0.6, 4.0, 2.2, 0.1, 7.1, 6.0],
         'Y': [0.3, 3.1, 5.4, 6.6, 2.8, 6.0, 3.1, 0.0, 3.9, 0.2],
         'Z': [0.3, 4.0, 1.7, 1.9, 9.9, 9.0, 0.8, 2.2, 1.4, 0.5]}

    data.set_N(6)
    print('\n\nSetting N=6, no seed:\n{}\n'.format(data.as_df()))
    rdf = data.as_df().applymap(lambda x: round(x, 1))
    assert rdf.to_dict(orient='list') == \
        {'X': [1.1, 0.0, 0.2, 4.4, 0.6, 4.0],
         'Y': [0.3, 3.1, 5.4, 6.6, 2.8, 6.0],
         'Z': [0.3, 4.0, 1.7, 1.9, 9.9, 9.0]}

    data.set_N(4, seed=3)
    print('\n\nSetting N=4, seed=3:\n{}\n'.format(data.as_df()))
    rdf = data.as_df().applymap(lambda x: round(x, 1))
    assert rdf.to_dict(orient='list') == \
        {'X': [4.4, 0.0, 1.1, 0.2],
         'Y': [6.6, 3.1, 0.3, 5.4],
         'Z': [1.9, 4.0, 0.3, 1.7]}

    data.set_N(10)
    print('\n\nSetting N=10, no seed:\n{}\n'.format(data.as_df()))
    rdf = data.as_df().applymap(lambda x: round(x, 1))
    # print(rdf.to_dict(orient='list'))
    assert rdf.to_dict(orient='list') == \
        {'X': [1.1, 0.0, 0.2, 4.4, 0.6, 4.0, 2.2, 0.1, 7.1, 6.0],
         'Y': [0.3, 3.1, 5.4, 6.6, 2.8, 6.0, 3.1, 0.0, 3.9, 0.2],
         'Z': [0.3, 4.0, 1.7, 1.9, 9.9, 9.0, 0.8, 2.2, 1.4, 0.5]}

    data.set_N(4, seed=3)
    print('\n\nSetting N=4, seed=3:\n{}\n'.format(data.as_df()))
    rdf = data.as_df().applymap(lambda x: round(x, 1))
    assert rdf.to_dict(orient='list') == \
        {'X': [4.4, 0.0, 1.1, 0.2],
         'Y': [6.6, 3.1, 0.3, 5.4],
         'Z': [1.9, 4.0, 0.3, 1.7]}


def test_set_N_asia_1_ok():  # Asia, N=100 - set N to 50
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)
    data.set_N(50)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.N == 50
    assert data.ext_to_orig == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.node_values == \
        {'asia': {'no': 47, 'yes': 3},
         'bronc': {'no': 29, 'yes': 21},
         'dysp': {'no': 28, 'yes': 22},
         'either': {'no': 45, 'yes': 5},
         'lung': {'no': 46, 'yes': 4},
         'smoke': {'yes': 26, 'no': 24},
         'tub': {'no': 49, 'yes': 1},
         'xray': {'no': 44, 'yes': 6}}
    assert data.node_types == \
        {'asia': 'category',
         'bronc': 'category',
         'dysp': 'category',
         'either': 'category',
         'lung': 'category',
         'smoke': 'category',
         'tub': 'category',
         'xray': 'category'}
    assert data.dstype == 'categorical'

    assert data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert len(data.sample) == 50
    assert (data.sample == df[:50]).all().all()

    # Note can increase sample size too

    data.set_N(80)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.N == 80
    assert data.ext_to_orig == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}

    assert data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert len(data.sample) == 80
    assert (data.sample == df[:80]).all().all()

    # Can increase size back up to original size

    data.set_N(100)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.N == 100
    assert data.ext_to_orig == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.node_values == \
        {'asia': {'no': 97, 'yes': 3},
         'bronc': {'no': 56, 'yes': 44},
         'dysp': {'no': 56, 'yes': 44},
         'either': {'no': 92, 'yes': 8},
         'lung': {'no': 93, 'yes': 7},
         'smoke': {'no': 57, 'yes': 43},
         'tub': {'no': 99, 'yes': 1},
         'xray': {'no': 90, 'yes': 10}}
    assert data.node_types == \
        {'asia': 'category',
         'bronc': 'category',
         'dysp': 'category',
         'either': 'category',
         'lung': 'category',
         'smoke': 'category',
         'tub': 'category',
         'xray': 'category'}
    assert data.dstype == 'categorical'

    assert data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert len(data.sample) == 100
    assert (data.sample == df).all().all()

    # Check first five rows

    print(data.sample.head())
    assert data.sample[:5].to_dict() == \
        {'asia': {0: 'no', 1: 'no', 2: 'no', 3: 'no', 4: 'no'},
         'bronc': {0: 'no', 1: 'yes', 2: 'yes', 3: 'no', 4: 'yes'},
         'dysp': {0: 'no', 1: 'yes', 2: 'yes', 3: 'no', 4: 'yes'},
         'either': {0: 'no', 1: 'no', 2: 'no', 3: 'yes', 4: 'no'},
         'lung': {0: 'no', 1: 'no', 2: 'no', 3: 'no', 4: 'no'},
         'smoke': {0: 'yes', 1: 'yes', 2: 'yes', 3: 'no', 4: 'no'},
         'tub': {0: 'no', 1: 'no', 2: 'no', 3: 'yes', 4: 'no'},
         'xray': {0: 'no', 1: 'no', 2: 'no', 3: 'yes', 4: 'no'}}


def test_set_N_asia_2_ok():  # Asia, N=100 - set N to 50, randomise rows
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)
    print('\nOriginal 5/100 randomised rows:\n{}'.format(data.as_df().head()))
    assert data.as_df()[:5].to_dict(orient='list') == \
        {'asia': ['no', 'no', 'no', 'no', 'no'],
         'bronc': ['no', 'yes', 'yes', 'no', 'yes'],
         'dysp': ['no', 'yes', 'yes', 'no', 'yes'],
         'either': ['no', 'no', 'no', 'yes', 'no'],
         'lung': ['no', 'no', 'no', 'no', 'no'],
         'smoke': ['yes', 'yes', 'yes', 'no', 'no'],
         'tub': ['no', 'no', 'no', 'yes', 'no'],
         'xray': ['no', 'no', 'no', 'yes', 'no']}

    data.set_N(50, seed=1)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.N == 50
    assert data.ext_to_orig == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.node_values == \
        {'asia': {'no': 47, 'yes': 3},
         'bronc': {'no': 29, 'yes': 21},
         'dysp': {'no': 28, 'yes': 22},
         'either': {'no': 45, 'yes': 5},
         'lung': {'no': 46, 'yes': 4},
         'smoke': {'yes': 26, 'no': 24},
         'tub': {'no': 49, 'yes': 1},
         'xray': {'no': 44, 'yes': 6}}

    assert data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')

    print('\n1st 5/50 randomised rows:\n{}'.format(data.as_df().head()))

    assert data.sample[:5].to_dict(orient='list') == \
        {'asia': ['no', 'no', 'yes', 'no', 'no'],
         'bronc': ['no', 'no', 'no', 'no', 'yes'],
         'dysp': ['no', 'no', 'no', 'no', 'yes'],
         'either': ['no', 'no', 'no', 'no', 'no'],
         'lung': ['no', 'no', 'no', 'no', 'no'],
         'smoke': ['yes', 'no', 'no', 'no', 'yes'],
         'tub': ['no', 'no', 'no', 'no', 'no'],
         'xray': ['no', 'no', 'no', 'no', 'no']}

    # Note can increase sample size too

    data.set_N(80, seed=2)

    assert isinstance(data, Pandas)
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.N == 80
    assert data.ext_to_orig == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}

    assert data.node_values == \
        {'asia': {'no': 77, 'yes': 3},
         'bronc': {'no': 45, 'yes': 35},
         'dysp': {'no': 43, 'yes': 37},
         'either': {'no': 74, 'yes': 6},
         'lung': {'no': 75, 'yes': 5},
         'smoke': {'no': 43, 'yes': 37},
         'tub': {'no': 79, 'yes': 1},
         'xray': {'no': 72, 'yes': 8}}

    assert data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')

    print('\n1st 5/80 randomised rows:\n{}'.format(data.sample.head()))
    assert data.as_df()[:5].to_dict(orient='list') == \
        {'asia': ['no', 'no', 'no', 'no', 'no'],
         'bronc': ['no', 'yes', 'no', 'yes', 'yes'],
         'dysp': ['no', 'yes', 'no', 'no', 'yes'],
         'either': ['no', 'no', 'no', 'no', 'no'],
         'lung': ['no', 'no', 'no', 'no', 'no'],
         'smoke': ['yes', 'yes', 'no', 'yes', 'yes'],
         'tub': ['no', 'no', 'no', 'no', 'no'],
         'xray': ['no', 'no', 'no', 'yes', 'no']}

    # Can increase size back up to original size, and without seed reverts
    # to original order.

    data.set_N(100)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.N == 100
    assert data.ext_to_orig == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.node_values == \
        {'asia': {'no': 97, 'yes': 3},
         'bronc': {'no': 56, 'yes': 44},
         'dysp': {'no': 56, 'yes': 44},
         'either': {'no': 92, 'yes': 8},
         'lung': {'no': 93, 'yes': 7},
         'smoke': {'no': 57, 'yes': 43},
         'tub': {'no': 99, 'yes': 1},
         'xray': {'no': 90, 'yes': 10}}

    assert data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert len(data.sample) == 100

    # Check first five rows

    print('\n1st 5/100 randomised rows:\n{}'.format(data.sample.head()))
    assert data.as_df()[:5].to_dict(orient='list') == \
        {'asia': ['no', 'no', 'no', 'no', 'no'],
         'bronc': ['no', 'yes', 'yes', 'no', 'yes'],
         'dysp': ['no', 'yes', 'yes', 'no', 'yes'],
         'either': ['no', 'no', 'no', 'yes', 'no'],
         'lung': ['no', 'no', 'no', 'no', 'no'],
         'smoke': ['yes', 'yes', 'yes', 'no', 'no'],
         'tub': ['no', 'no', 'no', 'yes', 'no'],
         'xray': ['no', 'no', 'no', 'yes', 'no']}


# Test set_order function

def test_set_order_type_error_1_ok():  # Asia, N=100 - no args
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)

    with pytest.raises(TypeError):
        data.set_order()


def test_set_order_type_error_2_ok():  # Asia, N=100 - bad arg type
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)

    with pytest.raises(TypeError):
        data.set_order(None)
    with pytest.raises(TypeError):
        data.set_order(True)
    with pytest.raises(TypeError):
        data.set_order(12)
    with pytest.raises(TypeError):
        data.set_order(list(data.nodes))
    with pytest.raises(TypeError):
        data.set_order(tuple([1, 2]))


def test_set_order_value_error_1_ok():  # Asia, N=100 - names mismatch
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)
    nodes = data.nodes

    with pytest.raises(ValueError):
        data.set_order(tuple())
    with pytest.raises(ValueError):
        data.set_order(tuple(list(nodes) + ['extra']))
    with pytest.raises(ValueError):
        data.set_order(tuple([n for n in nodes if n != 'asia']))


def test_set_order_asia_1_ok():  # Asia, N=100 - optimal/worst/original order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)

    data = Pandas(df=df)
    std_order = data.nodes

    # switch to optimal order

    order = tuple(bn.dag.ordered_nodes())
    assert order == \
        ('asia', 'smoke', 'bronc', 'lung', 'tub', 'either', 'dysp', 'xray')
    data.set_order(order)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()  # NB data.df unchanged
    assert data.order == (0, 5, 1, 4, 6, 3, 2, 7)
    assert data.nodes == std_order
    assert data.N == 100
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() and data.sample DO reflect new order

    assert data.get_order() == order
    assert tuple(data.sample.columns) == order
    assert data.sample.to_dict() == df.to_dict()

    # switch to worst order

    order = order[::-1]
    assert order == \
        ('xray', 'dysp', 'either', 'tub', 'lung', 'bronc', 'smoke', 'asia')
    data.set_order(order)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()  # NB data.df unchanged
    assert data.order == (7, 2, 3, 6, 4, 1, 5, 0)
    assert data.nodes == std_order
    assert data.N == 100
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() and data.sample DO reflect new order

    assert data.get_order() == order
    assert tuple(data.sample.columns) == order
    assert data.sample.to_dict() == df.to_dict()

    # revert to standard order

    data.set_order(std_order)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()  # NB data.df unchanged
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == std_order
    assert data.N == 100
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() and data.sample DO reflect new order

    assert data.get_order() == std_order
    assert tuple(data.sample.columns) == std_order
    assert data.sample.to_dict() == df.to_dict()


# Test randomise names

def test_rand_name_asia_1_ok():  # Asia, N=20 - randomise names
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    df = read_csv(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                  dtype='category', nrows=100)
    std_order = tuple(bn.dag.nodes)
    data = Pandas(df=df)

    data.randomise_names(seed=0)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()  # NB data.df unchanged
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)  # order unchanged
    assert data.nodes == std_order  # always original order
    assert data.N == 100  # size unchanged
    assert data.ext_to_orig == \
        {'X004asia': 'asia',
         'X007bronc': 'bronc',
         'X000dysp': 'dysp',
         'X006either': 'either',
         'X002lung': 'lung',
         'X003smoke': 'smoke',
         'X005tub': 'tub',
         'X001xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'X004asia',
         'bronc': 'X007bronc',
         'dysp': 'X000dysp',
         'either': 'X006either',
         'lung': 'X002lung',
         'smoke': 'X003smoke',
         'tub': 'X005tub',
         'xray': 'X001xray'}
    assert data.node_values == \
        {'X004asia': {'no': 97, 'yes': 3},
         'X007bronc': {'no': 56, 'yes': 44},
         'X000dysp': {'no': 56, 'yes': 44},
         'X006either': {'no': 92, 'yes': 8},
         'X002lung': {'no': 93, 'yes': 7},
         'X003smoke': {'no': 57, 'yes': 43},
         'X005tub': {'no': 99, 'yes': 1},
         'X001xray': {'no': 90, 'yes': 10}}
    assert data.node_types == \
        {'X004asia': 'category',
         'X007bronc': 'category',
         'X000dysp': 'category',
         'X006either': 'category',
         'X002lung': 'category',
         'X003smoke': 'category',
         'X005tub': 'category',
         'X001xray': 'category'}
    assert data.dstype == 'categorical'

    # Note get_order() and data.sample DO reflect new column names

    rev_map = {orig: ext for ext, orig in data.ext_to_orig.items()}
    new_cols = tuple([rev_map[data.nodes[i]] for i in data.order])
    assert data.get_order() == new_cols
    assert tuple(data.sample.columns) == new_cols
    assert df.to_dict() == \
        data.sample.rename(columns=data.ext_to_orig).to_dict()

    # Different seed produces different names

    data.randomise_names(seed=1)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()  # NB data.df unchanged
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)  # order unchanged
    assert data.nodes == std_order  # always original order
    assert data.N == 100  # size unchanged
    assert data.ext_to_orig == \
        {'X007asia': 'asia',
         'X002bronc': 'bronc',
         'X001dysp': 'dysp',
         'X003either': 'either',
         'X004lung': 'lung',
         'X000smoke': 'smoke',
         'X006tub': 'tub',
         'X005xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'X007asia',
         'bronc': 'X002bronc',
         'dysp': 'X001dysp',
         'either': 'X003either',
         'lung': 'X004lung',
         'smoke': 'X000smoke',
         'tub': 'X006tub',
         'xray': 'X005xray'}
    assert data.node_values == \
        {'X007asia': {'no': 97, 'yes': 3},
         'X002bronc': {'no': 56, 'yes': 44},
         'X001dysp': {'no': 56, 'yes': 44},
         'X003either': {'no': 92, 'yes': 8},
         'X004lung': {'no': 93, 'yes': 7},
         'X000smoke': {'no': 57, 'yes': 43},
         'X006tub': {'no': 99, 'yes': 1},
         'X005xray': {'no': 90, 'yes': 10}}
    assert data.node_types == \
        {'X007asia': 'category',
         'X002bronc': 'category',
         'X001dysp': 'category',
         'X003either': 'category',
         'X004lung': 'category',
         'X000smoke': 'category',
         'X006tub': 'category',
         'X005xray': 'category'}
    assert data.dstype == 'categorical'

    # Note get_order() and data.sample DO reflect new column names

    rev_map = {orig: ext for ext, orig in data.ext_to_orig.items()}
    new_cols = tuple([rev_map[data.nodes[i]] for i in data.order])
    assert data.get_order() == new_cols
    assert tuple(data.sample.columns) == new_cols
    assert df.to_dict() == \
        data.sample.rename(columns=data.ext_to_orig).to_dict()

    # Going back to seed 0 gives same results as before

    data.randomise_names(seed=0)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()  # NB data.df unchanged
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)  # order unchanged
    assert data.nodes == std_order  # always original order
    assert data.N == 100  # size unchanged
    assert data.ext_to_orig == \
        {'X004asia': 'asia',
         'X007bronc': 'bronc',
         'X000dysp': 'dysp',
         'X006either': 'either',
         'X002lung': 'lung',
         'X003smoke': 'smoke',
         'X005tub': 'tub',
         'X001xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'X004asia',
         'bronc': 'X007bronc',
         'dysp': 'X000dysp',
         'either': 'X006either',
         'lung': 'X002lung',
         'smoke': 'X003smoke',
         'tub': 'X005tub',
         'xray': 'X001xray'}
    assert data.node_values == \
        {'X004asia': {'no': 97, 'yes': 3},
         'X007bronc': {'no': 56, 'yes': 44},
         'X000dysp': {'no': 56, 'yes': 44},
         'X006either': {'no': 92, 'yes': 8},
         'X002lung': {'no': 93, 'yes': 7},
         'X003smoke': {'no': 57, 'yes': 43},
         'X005tub': {'no': 99, 'yes': 1},
         'X001xray': {'no': 90, 'yes': 10}}
    assert data.node_types == \
        {'X004asia': 'category',
         'X007bronc': 'category',
         'X000dysp': 'category',
         'X006either': 'category',
         'X002lung': 'category',
         'X003smoke': 'category',
         'X005tub': 'category',
         'X001xray': 'category'}
    assert data.dstype == 'categorical'

    # Note get_order() and data.sample DO reflect new column names

    rev_map = {orig: ext for ext, orig in data.ext_to_orig.items()}
    new_cols = tuple([rev_map[data.nodes[i]] for i in data.order])
    assert data.get_order() == new_cols
    assert tuple(data.sample.columns) == new_cols
    assert df.to_dict() == \
        data.sample.rename(columns=data.ext_to_orig).to_dict()

    # Using no seed value reverts back to original names

    data.randomise_names()

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()  # NB data.df unchanged
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)  # order unchanged
    assert data.nodes == std_order  # always original order
    assert data.N == 100  # size unchanged
    assert data.ext_to_orig == \
        {'asia': 'asia',
         'bronc': 'bronc',
         'dysp': 'dysp',
         'either': 'either',
         'lung': 'lung',
         'smoke': 'smoke',
         'tub': 'tub',
         'xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'asia',
         'bronc': 'bronc',
         'dysp': 'dysp',
         'either': 'either',
         'lung': 'lung',
         'smoke': 'smoke',
         'tub': 'tub',
         'xray': 'xray'}
    assert data.node_values == \
        {'asia': {'no': 97, 'yes': 3},
         'bronc': {'no': 56, 'yes': 44},
         'dysp': {'no': 56, 'yes': 44},
         'either': {'no': 92, 'yes': 8},
         'lung': {'no': 93, 'yes': 7},
         'smoke': {'no': 57, 'yes': 43},
         'tub': {'no': 99, 'yes': 1},
         'xray': {'no': 90, 'yes': 10}}
    assert data.node_types == \
        {'asia': 'category',
         'bronc': 'category',
         'dysp': 'category',
         'either': 'category',
         'lung': 'category',
         'smoke': 'category',
         'tub': 'category',
         'xray': 'category'}
    assert data.dstype == 'categorical'

    # Note get_order() and data.sample DO reflect new column names

    rev_map = {orig: ext for ext, orig in data.ext_to_orig.items()}
    new_cols = tuple([rev_map[data.nodes[i]] for i in data.order])
    assert data.get_order() == new_cols
    assert tuple(data.sample.columns) == new_cols
    assert df.to_dict() == \
        data.sample.rename(columns=data.ext_to_orig).to_dict()


# Test count number of unique combinations (approaches demonstrates that
# most networks have a hige number of combinations)

def test_unique_asia_1_ok():  # bn and df both defined
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                       dstype='categorical', N=1000)
    assert isinstance(data, Pandas)
    vcs = data.df.value_counts()
    assert len(vcs) == 36
    print('\n\n{} unique combinations in 1K rows of Asia are:\n{}'
          .format(len(vcs), vcs))


# Test values() functions

def test_values_type_error_1(data):  # no argument specified
    with pytest.raises(TypeError):
        data.values()


def test_values_type_error_2(data):  # bad nodes argument type
    with pytest.raises(TypeError):
        data.values(False)
    with pytest.raises(TypeError):
        data.values('X')
    with pytest.raises(TypeError):
        data.values(['X'])
    with pytest.raises(TypeError):
        data.values(12.7)


def test_values_value_error_1(data):  # duplicate node names
    with pytest.raises(ValueError):
        data.values(('X', 'X'))
    with pytest.raises(ValueError):
        data.values(('Y', 'X', 'Y'))


def test_values_value_error_2(data):  # nodes not in dataset
    with pytest.raises(ValueError):
        data.values(('X', 'Y', 'invalid'))
    with pytest.raises(ValueError):
        data.values(('badun',))


def test_values_value_error_3(data):  # Can't get values for categorical data
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Pandas(df=bn.generate_cases(10))
    with pytest.raises(ValueError):
        data.values(('tub', 'lung'))


def test_values_1_ok(data):  # Extract X
    nodes = ('X',)
    values = data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 1)
    assert (values[:, 0] == data.sample['X'].values).all()
    print('\n\nData for {} is:\n{}'.format(nodes, values))


def test_values_2_ok(data):  # Extract Y
    nodes = ('Y',)
    values = data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 1)
    assert (values[:, 0] == data.sample['Y'].values).all()
    print('\n\nData for {} is:\n{}'.format(nodes, values))


def test_values_3_ok(data):  # Extract Z
    nodes = ('Z',)
    values = data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 1)
    assert (values[:, 0] == data.sample['Z'].values).all()
    print('\n\nData for {} is:\n{}'.format(nodes, values))


def test_values_4_ok(data):  # Extract X, Y
    nodes = ('X', 'Y')
    values = data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 2)
    assert (values[:, 0] == data.sample['X'].values).all()
    assert (values[:, 1] == data.sample['Y'].values).all()
    print('\n\nData for {} is:\n{}'.format(nodes, values))


def test_values_5_ok(data):  # Extract Y, X
    nodes = ('Y', 'X')
    values = data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 2)
    assert (values[:, 0] == data.sample['Y'].values).all()
    assert (values[:, 1] == data.sample['X'].values).all()
    print('\n\nData for {} is:\n{}'.format(nodes, values))


def test_values_6_ok(data):  # Extract Y, Z, X
    nodes = ('Y', 'Z', 'X')
    values = data.values(nodes)
    assert isinstance(values, ndarray)
    assert values.shape == (10, 3)
    assert (values[:, 0] == data.sample['Y'].values).all()
    assert (values[:, 1] == data.sample['Z'].values).all()
    assert (values[:, 2] == data.sample['X'].values).all()
    print('\n\nData for {} is:\n{}'.format(nodes, values))
