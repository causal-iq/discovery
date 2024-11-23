
# Test the NumPy concrete implementation of Data

import pytest
from numpy import array, ndarray

from fileio.common import TESTDATA_DIR
from fileio.numpy import NumPy
from fileio.pandas import Pandas
from core.bn import BN


@pytest.fixture(scope="module")  # categorical AB, 3 rows
def ab3():
    data = array([[1, 1], [1, 0], [0, 0]], dtype='uint8')
    dstype = 'categorical'
    col_values = {'A': ('1', '0'), 'B': ('1', '0')}
    return {'d': data, 't': dstype, 'v': col_values}


@pytest.fixture(scope="module")  # categorical AB, 3 rows
def ab3_df():
    return Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv',
                       dstype='categorical').as_df()


@pytest.fixture(scope="module")  # categorical ABC, 5 rows
def abc5():
    data = array([[0, 0, 0], [0, 1, 0], [1, 0, 0],
                  [1, 1, 1], [1, 1, 0]], dtype='uint8')
    dstype = 'categorical'
    col_values = {'A': ('0', '1'), 'B': ('0', '1'), 'C': ('0', '1')}
    return {'d': data, 't': dstype, 'v': col_values}


@pytest.fixture(scope="module")  # categorical ABC, 36 rows
def abc36():
    data = array([[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0],
                  [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1],
                  [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                  [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
                  [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
                  [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1],
                  [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                  [1, 1, 1]], dtype='uint8')
    dstype = 'categorical'
    col_values = {'A': ('0', '1'), 'B': ('0', '1'), 'C': ('0', '1')}
    return {'d': data, 't': dstype, 'v': col_values}


@pytest.fixture(scope="module")  # continuous XY, 3 rows
def xy3():
    data = array([[1, 1], [1, 0], [0, 0]], dtype='float32')
    dstype = 'continuous'
    col_values = {'X': None, 'Y': None}
    return {'d': data, 't': dstype, 'v': col_values}


@pytest.fixture(scope="module")  # categorical XY, 3 rows
def xy3_df():
    return Pandas.read(TESTDATA_DIR + '/simple/xy_3.csv',
                       dstype='continuous').as_df()


@pytest.fixture(scope="module")  # continuous XYZ, 10 rows
def xyz10():
    pandas = Pandas.read(TESTDATA_DIR + '/simple/xyz_10.csv',
                         dstype='continuous')
    return NumPy.from_df(df=pandas.as_df(), dstype='continuous', keep_df=True)


# Constructor errors

def test_constructor_type_error_1_():  # no arguments provided
    with pytest.raises(TypeError):
        NumPy()


def test_constructor_type_error_2_(ab3):  # insufficent args
    with pytest.raises(TypeError):
        NumPy(ab3['d'], ab3['t'])
    with pytest.raises(TypeError):
        NumPy(ab3['d'])


def test_constructor_type_error_3_(ab3):  # data not an ndarray
    with pytest.raises(TypeError):
        NumPy(None, ab3['t'], ab3['v'])
    with pytest.raises(TypeError):
        NumPy(True, ab3['t'], ab3['v'])
    with pytest.raises(TypeError):
        NumPy(1, ab3['t'], ab3['v'])
    with pytest.raises(TypeError):
        NumPy([[2, 3], [2, 3]], ab3['t'], ab3['v'])


def test_constructor_type_error_4_(ab3):  # dstype not string or DatasetType
    with pytest.raises(TypeError):
        NumPy(ab3['d'], None, ab3['v'])
    with pytest.raises(TypeError):
        NumPy(ab3['d'], False, ab3['v'])
    with pytest.raises(TypeError):
        NumPy(ab3['d'], 'invalid', ab3['v'])
    with pytest.raises(TypeError):
        NumPy(ab3['d'], 2, ab3['v'])


def test_constructor_type_error_5_(ab3):  # col_types not dict
    with pytest.raises(TypeError):
        NumPy(ab3['d'], ab3['t'], True)
    with pytest.raises(TypeError):
        NumPy(ab3['d'], ab3['t'], None)
    with pytest.raises(TypeError):
        NumPy(ab3['d'], ab3['t'], 31.2)
    with pytest.raises(TypeError):
        NumPy(ab3['d'], ab3['t'], ['A', 'B'])


def test_constructor_type_error_6_(ab3):  # col_types keys not all strings
    with pytest.raises(TypeError):
        NumPy(ab3['d'], ab3['t'], {'A': ('Y', 'N'), 2: ('0', '1')})


def test_constructor_type_error_7_(ab3):  # col_types values not all tuples
    with pytest.raises(TypeError):
        NumPy(ab3['d'], ab3['t'], {'A': ['Y', 'N'], 'B': ['0', '1']})


def test_constructor_type_error_8_(ab3):  # tuple values not all strings
    with pytest.raises(TypeError):
        NumPy(ab3['d'], ab3['t'], {'A': ('Y', 'N'), 'B': (0, 1)})


def test_constructor_type_error_9_(ab3):  # col_types values not None
    with pytest.raises(TypeError):
        NumPy(ab3['d'], 'continuous', {'A': ('Y', 'N'), 'B': ('0', '1')})


def test_constructor_value_error_1_():  # less than two columns
    with pytest.raises(ValueError):
        NumPy(array([[1], [0]]), 'categorical', {'A': ('Y', 'N')})


def test_constructor_value_error_2_():  # less than two rows
    with pytest.raises(ValueError):
        NumPy(array([[0, 0]]), 'categorical', {'A': ('Y',), 'B': ('N',)})


def test_constructor_value_error_3_():  # data/col_values column count mismatch
    with pytest.raises(ValueError):
        NumPy(array([[0, 0], [1, 1]]), 'categorical',
              {'A': ('Y',), 'B': ('N',), 'C': ('1',)})


def test_constructor_value_error_4_(ab3):  # categorical, dtype not uint8
    with pytest.raises(ValueError):
        NumPy(ab3['d'], 'continuous', {'A': None, 'B': None})


def test_constructor_value_error_5_(ab3, xy3):  # continuous, dtype not float32
    with pytest.raises(ValueError):
        NumPy(xy3['d'], 'categorical', ab3['v'])


# Test constructor setting member variables correctly

def test_constructor_ab3_1_ok(ab3):
    data = NumPy(ab3['d'], ab3['t'], ab3['v'])

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'uint8'
    assert (data.data == array([[1, 1], [1, 0], [0, 0]])).all().all()
    assert data.nodes == ('A', 'B')
    assert data.order == (0, 1)
    assert data.ext_to_orig == {'A': 'A', 'B': 'B'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B'}
    assert data.N == 3
    assert data.dstype == 'categorical'
    assert (data.categories == (('1', '0'), ('1', '0'))).all().all()
    assert data.node_values == {'A': {'0': 2, '1': 1},
                                'B': {'0': 1, '1': 2}}
    assert data.node_types == {'A': 'category', 'B': 'category'}


def test_constructor_abc5_1_ok(abc5):  # A,B,C dataset with 5 rows
    data = NumPy(abc5['d'], abc5['t'], abc5['v'])

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'uint8'
    assert (data.data == array([[0, 0, 0], [0, 1, 0], [1, 0, 0],
                                [1, 1, 1], [1, 1, 0]])).all().all()
    assert data.nodes == ('A', 'B', 'C')
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.N == 5
    assert data.dstype == 'categorical'
    assert (data.categories == (('0', '1'), ('0', '1'),
                                ('0', '1'))).all().all()
    assert data.node_values == {'A': {'0': 2, '1': 3},
                                'B': {'0': 2, '1': 3},
                                'C': {'0': 4, '1': 1}}
    assert data.node_types == {'A': 'category', 'B': 'category',
                               'C': 'category'}


def test_constructor_abc36_1_ok(abc36):  # A,B,C dataset with 36 rows
    data = NumPy(abc36['d'], abc36['t'], abc36['v'])

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'uint8'
    assert (array([[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0],
                   [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1],
                   [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                   [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
                   [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
                   [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1],
                   [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                   [1, 1, 1]], dtype='uint8') == data.data).all().all()
    assert data.nodes == ('A', 'B', 'C')
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.N == 36
    assert data.dstype == 'categorical'
    assert (data.categories == (('0', '1'), ('0', '1'),
                                ('0', '1'))).all().all()
    assert data.node_values == {'A': {'0': 10, '1': 26},
                                'B': {'0': 14, '1': 22},
                                'C': {'0': 16, '1': 20}}
    assert data.node_types == {'A': 'category', 'B': 'category',
                               'C': 'category'}


def test_constructor_xy3_1_ok(xy3):  # XY with 3 continuous rows
    data = NumPy(xy3['d'], xy3['t'], xy3['v'])

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'float32'
    assert (data.data == array([[1.0, 1.0], [1.0, 0.0],
                               [0.0, 0.0]])).all().all()
    assert data.nodes == ('X', 'Y')
    assert data.order == (0, 1)
    assert data.ext_to_orig == {'X': 'X', 'Y': 'Y'}
    assert data.orig_to_ext == {'X': 'X', 'Y': 'Y'}
    assert data.N == 3
    assert data.dstype == 'continuous'
    assert data.categories is None
    assert data.node_values == {}
    assert data.node_types == {'X': 'float32', 'Y': 'float32'}


# Test from_df() function to instantiate from a Pandas DataFrame

def test_from_df_type_error_1_():  # no arguments
    with pytest.raises(TypeError):
        NumPy.from_df()


def test_from_df_type_error_2_(ab3_df):  # insufficient arguments
    with pytest.raises(TypeError):
        NumPy.from_df(ab3_df)
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype='categorical')
    with pytest.raises(TypeError):
        NumPy.from_df(ab3_df, keep_df=True)


def test_from_df_type_error_3_():  # df bad type
    with pytest.raises(TypeError):
        NumPy.from_df(df=True, dstype='categorical', keep_df=False)
    with pytest.raises(TypeError):
        NumPy.from_df(df=None, dstype='categorical', keep_df=False)
    with pytest.raises(TypeError):
        NumPy.from_df(df=2, dstype='categorical', keep_df=False)


def test_from_df_type_error_4_(ab3_df):  # dstype bad type
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype='invalid', keep_df=False)
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype=['categorical'], keep_df=False)
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype=True, keep_df=False)


def test_from_df_type_error_5_(ab3_df):  # keep_df bad type
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype='categorical', keep_df=None)
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype='categorical', keep_df=1)
    with pytest.raises(TypeError):
        NumPy.from_df(df=ab3_df, dstype='categorical', keep_df={True})


def test_from_df_value_error_1_(ab3_df):  # too few rows in df
    with pytest.raises(ValueError):
        NumPy.from_df(df=ab3_df[:1], dstype='categorical', keep_df=True)


def test_from_df_value_error_2_(ab3_df):  # too few columns in df
    with pytest.raises(ValueError):
        NumPy.from_df(df=ab3_df[['A']], dstype='categorical', keep_df=True)


def test_from_df_value_error_3_(ab3_df):  # type mismatch
    with pytest.raises(ValueError):
        NumPy.from_df(df=ab3_df, dstype='continuous', keep_df=True)


def test_from_df_value_error_4_(xy3_df):  # type mismatch
    with pytest.raises(ValueError):
        NumPy.from_df(df=xy3_df, dstype='categorical', keep_df=True)


def test_from_df_ab3_1_ok():  # AB 3 rows categorical data. keep_df = True
    dstype = 'categorical'
    df = Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', dstype=dstype).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'uint8'
    assert (data.data == array([[0, 0], [0, 1], [1, 1]])).all().all()
    assert data.nodes == ('A', 'B')
    assert data.order == (0, 1)
    assert data.ext_to_orig == {'A': 'A', 'B': 'B'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B'}
    assert data.N == 3
    assert data.dstype == 'categorical'
    assert (data.categories == (('1', '0'),
                                ('1', '0'))).all().all()
    assert data.node_values == {'A': {'0': 1, '1': 2},
                                'B': {'0': 2, '1': 1}}
    assert data.node_types == {'A': 'category', 'B': 'category'}

    assert (df == data.as_df()).all().all()


def test_from_df_ab3_2_ok():  # AB 3 rows categorical data. keep_df = False
    dstype = 'categorical'
    df = Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', dstype=dstype).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=False)

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'uint8'
    assert (data.data == array([[0, 0], [0, 1], [1, 1]])).all().all()
    assert data.nodes == ('A', 'B')
    assert data.order == (0, 1)
    assert data.ext_to_orig == {'A': 'A', 'B': 'B'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B'}
    assert data.N == 3
    assert data.dstype == 'categorical'
    assert (data.categories == (('1', '0'),
                                ('1', '0'))).all().all()
    assert data.node_values == {'A': {'0': 1, '1': 2},
                                'B': {'0': 2, '1': 1}}
    assert data.node_types == {'A': 'category', 'B': 'category'}

    assert not (df == data.as_df()).all().all()


def test_from_df_abc36_1_ok():  # ABC 36 rows categorical data, keep_df = True
    dstype = 'categorical'
    df = Pandas.read(TESTDATA_DIR + '/simple/abc_36.csv',
                     dstype=dstype).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'uint8'
    assert (array([[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0],
                   [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1],
                   [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                   [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
                   [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
                   [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1],
                   [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                   [1, 1, 1]]) == data.data).all().all()
    assert data.nodes == ('A', 'B', 'C')
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.N == 36
    assert data.dstype == 'categorical'
    assert (data.categories == (('0', '1'), ('0', '1'),
                                ('0', '1'))).all().all()
    assert data.node_values == {'A': {'0': 10, '1': 26},
                                'B': {'0': 14, '1': 22},
                                'C': {'0': 16, '1': 20}}
    assert data.node_types == {'A': 'category', 'B': 'category',
                               'C': 'category'}

    assert (df == data.as_df()).all().all()


def test_from_df_abc36_2_ok():  # ABC 36 rows categorical data, keep_df=False
    dstype = 'categorical'
    df = Pandas.read(TESTDATA_DIR + '/simple/abc_36.csv',
                     dstype=dstype).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=False)

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'uint8'
    assert (array([[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0],
                   [0, 1, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1],
                   [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                   [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
                   [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0],
                   [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1],
                   [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                   [1, 1, 1]]) == data.data).all().all()
    assert data.nodes == ('A', 'B', 'C')
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.N == 36
    assert data.dstype == 'categorical'
    assert (data.categories == (('0', '1'), ('0', '1'),
                                ('0', '1'))).all().all()
    assert data.node_values == {'A': {'0': 10, '1': 26},
                                'B': {'0': 14, '1': 22},
                                'C': {'0': 16, '1': 20}}
    assert data.node_types == {'A': 'category', 'B': 'category',
                               'C': 'category'}

    assert not (df == data.as_df()).all().all()


def test_from_df_cancer_1_ok():  # Cancer dataset with 10 rows
    df = Pandas.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                     dstype='categorical', N=10).as_df()
    data = NumPy.from_df(df=df, dstype='categorical', keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'uint8'
    assert (array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 0, 0],
                   [0, 0, 1, 0, 0], [0, 1, 1, 0, 1], [0, 0, 0, 0, 0],
                   [0, 1, 1, 0, 1], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0],
                   [0, 0, 1, 0, 0]]) == data.data).all().all()
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

    assert (df == data.as_df()).all().all()


def test_from_df_asia_1_ok():  # Asia dataset with 100 rows
    df = Pandas.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                     dstype='categorical', N=100).as_df()
    data = NumPy.from_df(df=df, dstype='categorical', keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'uint8'
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

    assert (df == data.as_df()).all().all()


def test_from_df_xy3_1_ok():  # XY 3 rows of continuous data, keep_df = True
    dstype = 'continuous'
    df = Pandas.read(TESTDATA_DIR + '/simple/xy_3.csv', dstype=dstype).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'float32'
    assert (data.data == array([[1.01, 1.21], [-0.45, 0.67],
                               [1.22, -1.41]], dtype='float32')).all().all()
    assert data.nodes == ('F1', 'F2')
    assert data.order == (0, 1)
    assert data.ext_to_orig == {'F1': 'F1', 'F2': 'F2'}
    assert data.orig_to_ext == {'F1': 'F1', 'F2': 'F2'}
    assert data.N == 3
    assert data.dstype == 'continuous'
    assert data.categories is None
    assert data.node_values == {}
    assert data.node_types == {'F1': 'float32', 'F2': 'float32'}

    assert (df == data.as_df()).all().all()


def test_from_df_xy3_2_ok():  # XY 3 rows of continuous data, keep_df = False
    dstype = 'continuous'
    df = Pandas.read(TESTDATA_DIR + '/simple/xy_3.csv', dstype=dstype).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=False)

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'float32'
    assert (data.data == array([[1.01, 1.21], [-0.45, 0.67],
                               [1.22, -1.41]], dtype='float32')).all().all()
    assert data.nodes == ('F1', 'F2')
    assert data.order == (0, 1)
    assert data.ext_to_orig == {'F1': 'F1', 'F2': 'F2'}
    assert data.orig_to_ext == {'F1': 'F1', 'F2': 'F2'}
    assert data.N == 3
    assert data.dstype == 'continuous'
    assert data.categories is None
    assert data.node_values == {}
    assert data.node_types == {'F1': 'float32', 'F2': 'float32'}

    assert (df == data.as_df()).all().all()  # df not modified for cont anyway


def test_from_df_xyz10_1_ok(xyz10):  # XYZ 10 rows of continuous data
    dstype = 'continuous'
    df = Pandas.read(TESTDATA_DIR + '/simple/xyz_10.csv',
                     dstype=dstype).as_df()

    data = NumPy.from_df(df, dstype=dstype, keep_df=True)

    assert isinstance(data, NumPy)
    assert data.data.dtype == 'float32'
    assert (data.data == array([[1.1, 0.3, 0.3], [0.0, 3.1, 4.0],
                                [0.2, 5.4, 1.7], [4.4, 6.6, 1.9],
                                [0.6, 2.8, 9.9], [4.0, 6.0, 9.0],
                                [2.2, 3.1, 0.8], [0.1, 0.0, 2.2],
                                [7.1, 3.9, 1.4], [6.0, 0.2, 0.5]],
                               dtype='float32')).all().all()
    assert data.nodes == ('X', 'Y', 'Z')
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
    assert data.orig_to_ext == {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
    assert data.N == 10
    assert data.dstype == 'continuous'
    assert data.categories is None
    assert data.node_values == {}
    assert data.node_types == {'X': 'float32', 'Y': 'float32', 'Z': 'float32'}

    assert (df == data.as_df()).all().all()


# Test as_df() function to return a Pandas DataFrame

def test_as_df_ab3_1_ok(ab3):
    data = NumPy(ab3['d'], ab3['t'], ab3['v'])

    df = data.as_df()

    assert (df == data.as_df()).all().all()
    df = df.to_dict(orient='list')

    print('\n\nab3 NumPy as dataframe: {}\n'.format(df))

    assert df == {'A': ['0', '0', '1'], 'B': ['0', '1', '1']}


def test_as_df_xy3_1_ok(xy3):
    data = NumPy(xy3['d'], xy3['t'], xy3['v'])

    df = data.as_df().to_dict(orient='list')
    print('\n\nxy3 NumPy as dataframe: {}\n'.format(df))

    assert df == {'X': [1.0, 1.0, 0.0], 'Y': [1.0, 0.0, 0.0]}


# Test set_N function

def test_set_N_type_error_1(ab3):  # AB 3 rows, no args
    data = NumPy(ab3['d'], ab3['t'], ab3['v'])

    with pytest.raises(TypeError):
        data.set_N()


def test_set_N_type_error_2(ab3):  # Invalid type for N
    data = NumPy(ab3['d'], ab3['t'], ab3['v'])

    with pytest.raises(TypeError):
        data.set_N(2.1)
    with pytest.raises(TypeError):
        data.set_N(True)
    with pytest.raises(TypeError):
        data.set_N(None)
    with pytest.raises(TypeError):
        data.set_N([2])


def test_set_N_type_error_3(ab3):  # Invalid type for N
    data = NumPy(ab3['d'], ab3['t'], ab3['v'])

    with pytest.raises(TypeError):
        data.set_N(N=3, seed=True)
    with pytest.raises(TypeError):
        data.set_N(N=2, seed=[1])
    with pytest.raises(TypeError):
        data.set_N(N=2, seed=2.1)


def test_set_N_value_error_1(ab3):  # set non-positive N
    data = NumPy(ab3['d'], ab3['t'], ab3['v'])

    with pytest.raises(ValueError):
        data.set_N(0)
    with pytest.raises(ValueError):
        data.set_N(-3)


def test_set_N_value_error_2(ab3):  # N larger than amount of data
    data = NumPy(ab3['d'], ab3['t'], ab3['v'])

    with pytest.raises(ValueError):
        data.set_N(4)


def test_set_N_value_error_3(ab3):  # invalid seed values
    data = NumPy(ab3['d'], ab3['t'], ab3['v'])

    with pytest.raises(ValueError):
        data.set_N(80, seed=-1)
    with pytest.raises(ValueError):
        data.set_N(80, seed=101)


def test_set_N_abc5_1_ok():  # ABC, 5 discrete rows, randomising order
    pandas = Pandas.read(TESTDATA_DIR + '/simple/abc_5.csv',
                         dstype='categorical')
    data = NumPy.from_df(df=pandas.as_df(), dstype='categorical', keep_df=True)

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
        {'A': ['0', '0', '1', '1'],
         'B': ['0', '1', '0', '1'],
         'C': ['0', '0', '0', '1']}

    data.set_N(4, seed=2)
    print('\n\nSetting N=4, seed=2:\n{}\n'.format(data.as_df()))
    assert data.node_values == \
        {'A': {'0': 2, '1': 2},
         'B': {'0': 2, '1': 2},
         'C': {'0': 3, '1': 1}}
    assert data.as_df().to_dict(orient='list') == \
        {'A': ['1', '1', '0', '0'],
         'B': ['1', '0', '0', '1'],
         'C': ['1', '0', '0', '0']}

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


def test_set_N_xyz10_1_ok():  # XYZ, 10 continuous rows, randmising order
    pandas = Pandas.read(TESTDATA_DIR + '/simple/xyz_10.csv',
                         dstype='continuous')
    data = NumPy.from_df(df=pandas.as_df(), dstype='continuous', keep_df=True)

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
        {'X': [4.4, 0.2, 0.0, 1.1],
         'Y': [6.6, 5.4, 3.1, 0.3],
         'Z': [1.9, 1.7, 4.0, 0.3]}

    data.set_N(10)
    print('\n\nSetting N=10, no seed:\n{}\n'.format(data.as_df()))
    rdf = data.as_df().applymap(lambda x: round(x, 1))
    assert rdf.to_dict(orient='list') == \
        {'X': [1.1, 0.0, 0.2, 4.4, 0.6, 4.0, 2.2, 0.1, 7.1, 6.0],
         'Y': [0.3, 3.1, 5.4, 6.6, 2.8, 6.0, 3.1, 0.0, 3.9, 0.2],
         'Z': [0.3, 4.0, 1.7, 1.9, 9.9, 9.0, 0.8, 2.2, 1.4, 0.5]}

    data.set_N(4, seed=3)
    print('\n\nSetting N=4, seed=3:\n{}\n'.format(data.as_df()))
    rdf = data.as_df().applymap(lambda x: round(x, 1))
    assert rdf.to_dict(orient='list') == \
        {'X': [4.4, 0.2, 0.0, 1.1],
         'Y': [6.6, 5.4, 3.1, 0.3],
         'Z': [1.9, 1.7, 4.0, 0.3]}


def test_set_N_asia_1_ok():  # Asia, N=100 - set N to 50
    pandas = Pandas.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                         dstype='categorical', N=100)
    data = NumPy.from_df(df=pandas.as_df(), dstype='categorical', keep_df=True)

    data.set_N(50)

    assert isinstance(data, NumPy)
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

    assert (data.as_df() == pandas.as_df()[:50]).all().all()

    # Note can increase sample size too

    data.set_N(80)

    assert isinstance(data, NumPy)
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

    assert (data.as_df() == pandas.as_df()[:80]).all().all()

    # Can increase size back up to original size

    data.set_N(100)

    assert isinstance(data, NumPy)
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

    assert (data.as_df() == pandas.as_df()).all().all()

    # Check first five rows

    assert data.as_df()[:5].to_dict() == \
        {'asia': {0: 'no', 1: 'no', 2: 'no', 3: 'no', 4: 'no'},
         'bronc': {0: 'no', 1: 'yes', 2: 'yes', 3: 'no', 4: 'yes'},
         'dysp': {0: 'no', 1: 'yes', 2: 'yes', 3: 'no', 4: 'yes'},
         'either': {0: 'no', 1: 'no', 2: 'no', 3: 'yes', 4: 'no'},
         'lung': {0: 'no', 1: 'no', 2: 'no', 3: 'no', 4: 'no'},
         'smoke': {0: 'yes', 1: 'yes', 2: 'yes', 3: 'no', 4: 'no'},
         'tub': {0: 'no', 1: 'no', 2: 'no', 3: 'yes', 4: 'no'},
         'xray': {0: 'no', 1: 'no', 2: 'no', 3: 'yes', 4: 'no'}}


def test_set_N_asia_2_ok():  # Asia, N=100 - set N to 50, randomise rows
    pandas = Pandas.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                         dstype='categorical', N=100)
    data = NumPy.from_df(df=pandas.as_df(), dstype='categorical', keep_df=True)

    assert (data.as_df() == pandas.as_df()).all().all()

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

    assert isinstance(data, NumPy)
    assert (data.as_df() != pandas).all().all()
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

    assert not (data.as_df() == pandas.as_df()[:50]).all().all()

    print('\n1st 5/50 randomised rows:\n{}'.format(data.as_df().head()))

    assert data.as_df()[:5].to_dict(orient='list') == \
        {'asia': ['no', 'no', 'yes', 'no', 'no'],
         'bronc': ['no', 'no', 'yes', 'no', 'yes'],
         'dysp': ['yes', 'no', 'yes', 'no', 'yes'],
         'either': ['yes', 'no', 'no', 'no', 'no'],
         'lung': ['yes', 'no', 'no', 'no', 'no'],
         'smoke': ['yes', 'yes', 'yes', 'no', 'yes'],
         'tub': ['no', 'no', 'no', 'no', 'no'],
         'xray': ['yes', 'no', 'no', 'no', 'no']}

    # Note can increase sample size too

    data.set_N(80, seed=2)

    assert isinstance(data, NumPy)
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

    print('\n1st 5/80 randomised rows:\n{}'.format(data.as_df().head()))

    assert data.as_df()[:5].to_dict(orient='list') == \
        {'asia': ['no', 'no', 'no', 'no', 'no'],
         'bronc': ['yes', 'yes', 'yes', 'yes', 'yes'],
         'dysp': ['yes', 'no', 'yes', 'yes', 'yes'],
         'either': ['no', 'no', 'no', 'no', 'no'],
         'lung': ['no', 'no', 'no', 'no', 'no'],
         'smoke': ['no', 'yes', 'yes', 'no', 'no'],
         'tub': ['no', 'no', 'no', 'no', 'no'],
         'xray': ['no', 'no', 'no', 'no', 'no']}

    # Can increase size back up to original size, and without seed reverts
    # to original order.

    data.set_N(100)

    assert isinstance(data, NumPy)
    assert (data.as_df() == pandas.as_df()).all().all()
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

    print('\n1st 5/100 randomised rows:\n{}'.format(data.as_df().head()))
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

def test_set_order_type_error_1_(ab3):  # no args
    data = NumPy(ab3['d'], ab3['t'], ab3['v'])
    with pytest.raises(TypeError):
        data.set_order()


def test_set_order_type_error_2_(ab3):  # bad arg type
    data = NumPy(ab3['d'], ab3['t'], ab3['v'])
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


def test_set_order_value_error_1_(ab3):  # AB bad arg values
    data = NumPy(ab3['d'], ab3['t'], ab3['v'])
    with pytest.raises(ValueError):
        data.set_order(tuple())
    with pytest.raises(ValueError):
        data.set_order(('A',))
    with pytest.raises(ValueError):
        data.set_order(('B',))
    with pytest.raises(ValueError):
        data.set_order(('A', 'B', 'extra'))


def test_set_order_abc36_1_ok(abc36):  # ABC 36 rows
    data = NumPy(abc36['d'], abc36['t'], abc36['v'])
    print('\n\nOriginal ABC order:\n{}'.format(data.as_df().head()))

    order = ('B', 'C', 'A')
    data.set_order(order)
    print('\n{} order:\n{}'.format(order, data.as_df().head()))
    assert data.order == (1, 2, 0)
    assert data.get_order() == order
    assert data.node_values == \
        {'A': {'0': 10, '1': 26},
         'B': {'0': 14, '1': 22},
         'C': {'0': 16, '1': 20}}

    order = ('C', 'A', 'B')
    data.set_order(order)
    print('\n{} order:\n{}'.format(order, data.as_df().head()))
    assert data.order == (2, 0, 1)
    assert data.get_order() == order
    assert data.node_values == \
        {'A': {'0': 10, '1': 26},
         'B': {'0': 14, '1': 22},
         'C': {'0': 16, '1': 20}}

    order = ('A', 'B', 'C')
    data.set_order(order)
    print('\n{} order:\n{}'.format(order, data.as_df().head()))
    assert data.order == (0, 1, 2)
    assert data.get_order() == order
    assert data.node_values == \
        {'A': {'0': 10, '1': 26},
         'B': {'0': 14, '1': 22},
         'C': {'0': 16, '1': 20}}

    assert (data.data == abc36['d']).all().all()


def test_set_order_xy3_1_ok(xy3):  # XY 3 rows
    data = NumPy(xy3['d'], xy3['t'], xy3['v'])
    print('\n\nOriginal XY order:\n{}'.format(data.as_df().head()))

    order = ('Y', 'X')
    data.set_order(order)
    print('\n{} order:\n{}'.format(order, data.as_df().head()))
    assert data.order == (1, 0)
    assert data.get_order() == order
    assert data.node_values == {}

    order = ('X', 'Y')
    data.set_order(order)
    print('\n{} order:\n{}'.format(order, data.as_df().head()))
    assert data.order == (0, 1)
    assert data.get_order() == order
    assert data.node_values == {}

    assert (data.data == xy3['d']).all().all()


def test_set_order_asia_1_ok():  # Asia, N=100 - optimal/worst/original order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    pandas = Pandas.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                         dstype='categorical', N=100)
    data = NumPy.from_df(df=pandas.as_df(), dstype='categorical', keep_df=True)
    orig_data = data.data.copy()

    std_order = data.nodes

    # switch to optimal order

    order = tuple(bn.dag.ordered_nodes())
    assert order == \
        ('asia', 'smoke', 'bronc', 'lung', 'tub', 'either', 'dysp', 'xray')
    data.set_order(order)

    assert isinstance(data, NumPy)
    assert data.order == (0, 5, 1, 4, 6, 3, 2, 7)
    assert data.nodes == std_order
    assert data.N == 100
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() and as_df() DO reflect new order, but data unchanged

    assert data.get_order() == order
    assert tuple(data.as_df().columns) == order
    assert (data.data == orig_data).all().all()

    # switch to worst order

    order = order[::-1]
    assert order == \
        ('xray', 'dysp', 'either', 'tub', 'lung', 'bronc', 'smoke', 'asia')
    data.set_order(order)

    assert isinstance(data, NumPy)
    assert data.order == (7, 2, 3, 6, 4, 1, 5, 0)
    assert data.nodes == std_order
    assert data.N == 100
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() and as_df() DO reflect new order, but data unchanged

    assert data.get_order() == order
    assert tuple(data.as_df().columns) == order
    assert (data.data == orig_data).all().all()

    # revert to standard order

    data.set_order(std_order)

    assert isinstance(data, NumPy)
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == std_order
    assert data.N == 100
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() and as_df() DO reflect new order, but data unchanged

    assert data.get_order() == std_order
    assert tuple(data.as_df().columns) == std_order
    assert (data.data == orig_data).all().all()


# Test randomise names

def test_rand_name_abc36_1_ok(abc36):  # ABC, N=36 - randomise names
    data = NumPy(abc36['d'], abc36['t'], abc36['v'])
    print('\n\nOriginal ABC names:\n{}'.format(data.as_df().head()))
    assert data.get_order() == ('A', 'B', 'C')
    assert data.node_values == \
        {'A': {'0': 10, '1': 26},
         'B': {'0': 14, '1': 22},
         'C': {'0': 16, '1': 20}}
    assert data.node_types == \
        {'A': 'category',
         'B': 'category',
         'C': 'category'}

    data.randomise_names(seed=0)
    print('\n\nRandom names, seed 0:\n{}'.format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ('X001A', 'X002B', 'X000C')
    assert data.get_order() == ('X001A', 'X002B', 'X000C')
    assert data.node_values == \
        {'X001A': {'0': 10, '1': 26},
         'X002B': {'0': 14, '1': 22},
         'X000C': {'0': 16, '1': 20}}
    assert data.node_types == \
        {'X001A': 'category',
         'X002B': 'category',
         'X000C': 'category'}
    assert (data.data == abc36['d']).all().all()

    data.randomise_names(seed=1)
    print('\n\nRandom names, seed 1:\n{}'.format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ('X002A', 'X000B', 'X001C')
    assert data.get_order() == ('X002A', 'X000B', 'X001C')
    assert data.node_values == \
        {'X002A': {'0': 10, '1': 26},
         'X000B': {'0': 14, '1': 22},
         'X001C': {'0': 16, '1': 20}}
    assert data.node_types == \
        {'X002A': 'category',
         'X000B': 'category',
         'X001C': 'category'}
    assert (data.data == abc36['d']).all().all()

    data.randomise_names(seed=0)
    print('\n\nRandom names, seed 0:\n{}'.format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ('X001A', 'X002B', 'X000C')
    assert data.get_order() == ('X001A', 'X002B', 'X000C')
    assert data.node_values == \
        {'X001A': {'0': 10, '1': 26},
         'X002B': {'0': 14, '1': 22},
         'X000C': {'0': 16, '1': 20}}
    assert data.node_types == \
        {'X001A': 'category',
         'X002B': 'category',
         'X000C': 'category'}
    assert (data.data == abc36['d']).all().all()

    data.randomise_names()
    print('\n\nRandom names, seed None:\n{}'.format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ('A', 'B', 'C')
    assert data.get_order() == ('A', 'B', 'C')
    assert data.node_values == \
        {'A': {'0': 10, '1': 26},
         'B': {'0': 14, '1': 22},
         'C': {'0': 16, '1': 20}}
    assert data.node_types == \
        {'A': 'category',
         'B': 'category',
         'C': 'category'}
    assert (data.data == abc36['d']).all().all()


def test_rand_name_xyz10_1_ok():  # XYZ, N10 - randomise names
    pandas = Pandas.read(TESTDATA_DIR + '/simple/xyz_10.csv',
                         dstype='continuous')
    data = NumPy.from_df(df=pandas.as_df(), dstype='continuous', keep_df=True)
    orig_data = data.data.copy()
    assert tuple(data.as_df().columns) == ('X', 'Y', 'Z')
    assert data.get_order() == ('X', 'Y', 'Z')
    assert data.node_values == {}
    assert data.node_types == {'X': 'float32', 'Y': 'float32', 'Z': 'float32'}
    print('\n\nOriginal XYZ names:\n{}'.format(data.as_df().head()))

    data.randomise_names(seed=0)
    print('\n\nRandom names, seed 0:\n{}'.format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ('X001X', 'X002Y', 'X000Z')
    assert data.get_order() == ('X001X', 'X002Y', 'X000Z')
    assert data.node_values == {}
    assert data.node_types == {'X001X': 'float32', 'X002Y': 'float32',
                               'X000Z': 'float32'}
    assert (data.data == orig_data).all().all()

    data.randomise_names(seed=1)
    print('\n\nRandom names, seed 1:\n{}'.format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ('X002X', 'X000Y', 'X001Z')
    assert data.get_order() == ('X002X', 'X000Y', 'X001Z')
    assert data.node_values == {}
    assert data.node_types == {'X002X': 'float32', 'X000Y': 'float32',
                               'X001Z': 'float32'}
    assert (data.data == orig_data).all().all()

    data.randomise_names(seed=0)
    print('\n\nRandom names, seed 0:\n{}'.format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ('X001X', 'X002Y', 'X000Z')
    assert data.get_order() == ('X001X', 'X002Y', 'X000Z')
    assert data.node_values == {}
    assert data.node_types == {'X001X': 'float32', 'X002Y': 'float32',
                               'X000Z': 'float32'}
    assert (data.data == orig_data).all().all()

    data.randomise_names()
    print('\n\nRandom names, seed None:\n{}'.format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ('X', 'Y', 'Z')
    assert tuple(data.as_df().columns) == ('X', 'Y', 'Z')
    assert data.get_order() == ('X', 'Y', 'Z')
    assert data.node_values == {}
    assert data.node_types == {'X': 'float32', 'Y': 'float32', 'Z': 'float32'}
    assert (data.data == orig_data).all().all()


def test_rand_name_asia_1_ok():  # Asia, N=20 - randomise names
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    pandas = Pandas.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                         dstype='categorical', N=100)
    data = NumPy.from_df(df=pandas.as_df(), dstype='categorical', keep_df=True)
    std_order = tuple(bn.dag.nodes)
    orig_data = data.data.copy()

    data.randomise_names(seed=0)

    assert isinstance(data, NumPy)
    assert (data.data == orig_data).all().all()  # NB data.df unchanged
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
    assert data.get_order() == tuple(data.ext_to_orig)
    assert tuple(data.as_df().columns) == tuple(data.ext_to_orig)

    # Different seed produces different names

    data.randomise_names(seed=1)

    assert isinstance(data, NumPy)
    assert (data.data == orig_data).all().all()  # NB data.df unchanged
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
    assert data.get_order() == tuple(data.ext_to_orig)
    assert tuple(data.as_df().columns) == tuple(data.ext_to_orig)

    # Check can go from one random set to another

    data.randomise_names(seed=2)

    assert isinstance(data, NumPy)
    assert (data.data == orig_data).all().all()  # NB data.df unchanged
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)  # order unchanged
    assert data.nodes == std_order  # always original order
    assert data.N == 100  # size unchanged
    assert data.ext_to_orig == \
        {'X001asia': 'asia',
         'X002bronc': 'bronc',
         'X005dysp': 'dysp',
         'X006either': 'either',
         'X000lung': 'lung',
         'X004smoke': 'smoke',
         'X003tub': 'tub',
         'X007xray': 'xray'}
    assert data.node_values == \
        {'X001asia': {'no': 97, 'yes': 3},
         'X002bronc': {'no': 56, 'yes': 44},
         'X005dysp': {'no': 56, 'yes': 44},
         'X006either': {'no': 92, 'yes': 8},
         'X000lung': {'no': 93, 'yes': 7},
         'X004smoke': {'no': 57, 'yes': 43},
         'X003tub': {'no': 99, 'yes': 1},
         'X007xray': {'no': 90, 'yes': 10}}
    assert data.node_types == \
        {'X001asia': 'category',
         'X002bronc': 'category',
         'X005dysp': 'category',
         'X006either': 'category',
         'X000lung': 'category',
         'X004smoke': 'category',
         'X003tub': 'category',
         'X007xray': 'category'}
    assert data.dstype == 'categorical'
    assert data.get_order() == tuple(data.ext_to_orig)
    assert tuple(data.as_df().columns) == tuple(data.ext_to_orig)

    # Going back to seed 0 gives same results as before

    data.randomise_names(seed=0)

    assert isinstance(data, NumPy)
    assert (data.data == orig_data).all().all()  # NB data.df unchanged
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
    assert data.get_order() == tuple(data.ext_to_orig)
    assert tuple(data.as_df().columns) == tuple(data.ext_to_orig)

    # Using no seed value reverts back to original names

    data.randomise_names()

    assert isinstance(data, NumPy)
    assert (data.data == orig_data).all().all()  # NB data.df unchanged
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
    assert data.get_order() == tuple(data.ext_to_orig)
    assert tuple(data.as_df().columns) == tuple(data.ext_to_orig)


# Test sequences changing data in different ways for categorical data

def test_sequence_abc5_1_ok(abc5):  # ABC5 - test sequences of changes
    data = NumPy(abc5['d'], abc5['t'], abc5['v'])

    assert data.get_order() == ('A', 'B', 'C')
    assert data.N == 5
    assert data.data.dtype == 'uint8'
    assert (data.data == array([[0, 0, 0], [0, 1, 0], [1, 0, 0],
                                [1, 1, 1], [1, 1, 0]])).all().all()
    assert data.sample.dtype == 'uint8'
    assert (data.sample == array([[0, 0, 0], [0, 1, 0], [1, 0, 0],
                                  [1, 1, 1], [1, 1, 0]])).all().all()
    assert data.nodes == ('A', 'B', 'C')
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.dstype == 'categorical'
    assert (data.categories == (('0', '1'), ('0', '1'),
                                ('0', '1'))).all().all()
    assert data.node_values == \
        {'A': {'0': 2, '1': 3}, 'B': {'0': 2, '1': 3}, 'C': {'0': 4, '1': 1}}
    assert data.as_df().to_dict(orient='list') == \
        {'A': ['0', '0', '1', '1', '1'],
         'B': ['0', '1', '0', '1', '1'],
         'C': ['0', '0', '0', '1', '0']}

    # Set data size to 3 - column names should stay the same, but data only
    # has firts three rows, and as_df() and node_values should reflect this

    data.set_N(3)

    assert data.get_order() == ('A', 'B', 'C')
    assert data.N == 3
    assert data.data.dtype == 'uint8'
    assert (data.data == array([[0, 0, 0], [0, 1, 0], [1, 0, 0],
                                [1, 1, 1], [1, 1, 0]])).all().all()
    assert data.sample.dtype == 'uint8'
    assert (data.sample == array([[0, 0, 0], [0, 1, 0],
                                  [1, 0, 0]])).all().all()
    assert data.nodes == ('A', 'B', 'C')
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert data.dstype == 'categorical'
    assert (data.categories == (('0', '1'), ('0', '1'),
                                ('0', '1'))).all().all()
    assert data.node_values == \
        {'A': {'0': 2, '1': 1}, 'B': {'0': 2, '1': 1}, 'C': {'0': 3}}
    assert data.as_df().to_dict(orient='list') == \
        {'A': ['0', '0', '1'],
         'B': ['0', '1', '0'],
         'C': ['0', '0', '0']}

    # Randomise column names - data should stay the same but external node
    # names should change

    data.randomise_names(1)

    assert data.get_order() == ('X002A', 'X000B', 'X001C')
    assert data.N == 3
    assert data.data.dtype == 'uint8'
    assert (data.data == array([[0, 0, 0], [0, 1, 0], [1, 0, 0],
                                [1, 1, 1], [1, 1, 0]])).all().all()
    assert data.sample.dtype == 'uint8'
    assert (data.sample == array([[0, 0, 0], [0, 1, 0],
                                  [1, 0, 0]])).all().all()
    assert data.nodes == ('A', 'B', 'C')
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {'X002A': 'A', 'X000B': 'B', 'X001C': 'C'}
    assert data.orig_to_ext == {'A': 'X002A', 'B': 'X000B', 'C': 'X001C'}
    assert data.dstype == 'categorical'
    assert (data.categories == (('0', '1'), ('0', '1'),
                                ('0', '1'))).all().all()
    assert data.node_values == \
        {'X002A': {'0': 2, '1': 1},
         'X000B': {'0': 2, '1': 1},
         'X001C': {'0': 3}}
    assert data.as_df().to_dict(orient='list') == \
        {'X002A': ['0', '0', '1'],
         'X000B': ['0', '1', '0'],
         'X001C': ['0', '0', '0']}

    # Set N to 4 - external column names should stay the same, but fourth
    # row should be added back into data, and node value counts adjusted

    data.set_N(4)

    assert data.get_order() == ('X002A', 'X000B', 'X001C')
    assert data.N == 4
    assert data.data.dtype == 'uint8'
    assert (data.data == array([[0, 0, 0], [0, 1, 0], [1, 0, 0],
                                [1, 1, 1], [1, 1, 0]])).all().all()
    assert data.sample.dtype == 'uint8'
    assert (data.sample == array([[0, 0, 0], [0, 1, 0],
                                  [1, 0, 0], [1, 1, 1]])).all().all()
    assert data.nodes == ('A', 'B', 'C')
    assert data.order == (0, 1, 2)
    assert data.ext_to_orig == {'X002A': 'A', 'X000B': 'B', 'X001C': 'C'}
    assert data.orig_to_ext == {'A': 'X002A', 'B': 'X000B', 'C': 'X001C'}
    assert data.dstype == 'categorical'
    assert (data.categories == (('0', '1'), ('0', '1'),
                                ('0', '1'))).all().all()
    assert data.node_values == \
        {'X002A': {'0': 2, '1': 2},
         'X000B': {'0': 2, '1': 2},
         'X001C': {'0': 3, '1': 1}}
    assert data.as_df().to_dict(orient='list') == \
        {'X002A': ['0', '0', '1', '1'],
         'X000B': ['0', '1', '0', '1'],
         'X001C': ['0', '0', '0', '1']}

    # change processing order - just data.order, get_order() changed

    data.set_order(('X000B', 'X001C', 'X002A'))

    assert data.get_order() == ('X000B', 'X001C', 'X002A')
    assert data.N == 4
    assert data.data.dtype == 'uint8'
    assert (data.data == array([[0, 0, 0], [0, 1, 0], [1, 0, 0],
                                [1, 1, 1], [1, 1, 0]])).all().all()
    assert data.sample.dtype == 'uint8'
    assert (data.sample == array([[0, 0, 0], [0, 1, 0],
                                  [1, 0, 0], [1, 1, 1]])).all().all()
    assert data.nodes == ('A', 'B', 'C')
    assert data.order == (1, 2, 0)
    assert data.ext_to_orig == {'X002A': 'A', 'X000B': 'B', 'X001C': 'C'}
    assert data.orig_to_ext == {'A': 'X002A', 'B': 'X000B', 'C': 'X001C'}
    assert data.dstype == 'categorical'
    assert (data.categories == (('0', '1'), ('0', '1'),
                                ('0', '1'))).all().all()
    assert data.node_values == \
        {'X002A': {'0': 2, '1': 2},
         'X000B': {'0': 2, '1': 2},
         'X001C': {'0': 3, '1': 1}}
    assert data.as_df().to_dict(orient='list') == \
        {'X002A': ['0', '0', '1', '1'],
         'X000B': ['0', '1', '0', '1'],
         'X001C': ['0', '0', '0', '1']}

    # randomise row order - sample and node value counts change

    data.set_N(4, 3)

    assert data.get_order() == ('X000B', 'X001C', 'X002A')
    assert data.N == 4
    assert data.data.dtype == 'uint8'
    assert (data.data == array([[0, 0, 0], [0, 1, 0], [1, 0, 0],
                                [1, 1, 1], [1, 1, 0]])).all().all()
    assert data.sample.dtype == 'uint8'
    assert (data.sample == array([[1, 1, 1], [1, 0, 0],
                                  [0, 1, 0], [0, 0, 0]])).all().all()
    assert data.nodes == ('A', 'B', 'C')
    assert data.order == (1, 2, 0)
    assert data.ext_to_orig == {'X002A': 'A', 'X000B': 'B', 'X001C': 'C'}
    assert data.orig_to_ext == {'A': 'X002A', 'B': 'X000B', 'C': 'X001C'}
    assert data.dstype == 'categorical'
    assert (data.categories == (('0', '1'), ('0', '1'),
                                ('0', '1'))).all().all()
    assert data.node_values == \
        {'X002A': {'0': 2, '1': 2},
         'X000B': {'0': 2, '1': 2},
         'X001C': {'0': 3, '1': 1}}
    assert data.as_df().to_dict(orient='list') == \
        {'X002A': ['1', '1', '0', '0'],
         'X000B': ['1', '0', '1', '0'],
         'X001C': ['1', '0', '0', '0']}


# Test values() function

def test_values_type_error_1(xyz10):  # no argument specified
    with pytest.raises(TypeError):
        xyz10.values()


def test_values_type_error_2(xyz10):  # bad nodes argument type
    with pytest.raises(TypeError):
        xyz10.values(False)
    with pytest.raises(TypeError):
        xyz10.values('X')
    with pytest.raises(TypeError):
        xyz10.values(['X'])
    with pytest.raises(TypeError):
        xyz10.values(12.7)


def test_values_value_error_1(xyz10):  # duplicate node names
    with pytest.raises(ValueError):
        xyz10.values(('X', 'X'))
    with pytest.raises(ValueError):
        xyz10.values(('Y', 'X', 'Y'))


def test_values_xyz10_1_(xyz10):  # Extract X
    nodes = ('X',)
    values = xyz10.values(nodes)

    assert isinstance(values, ndarray)
    assert values.shape == (10, 1)
    assert (values == xyz10.data[:, [0]]).all()
    print('\n\nData for {} is:\n{}'.format(nodes, values))


def test_values_xyz10_2_(xyz10):  # Extract Z
    nodes = ('Z',)
    values = xyz10.values(nodes)

    assert isinstance(values, ndarray)
    assert values.shape == (10, 1)
    assert (values == xyz10.data[:, [2]]).all()
    print('\n\nData for {} is:\n{}'.format(nodes, values))


def test_values_xyz10_3_(xyz10):  # Extract Z, Y
    nodes = ('Z', 'Y')
    values = xyz10.values(nodes)

    assert isinstance(values, ndarray)
    assert values.shape == (10, 2)
    assert (values == xyz10.data[:, [2, 1]]).all()
    print('\n\nData for {} is:\n{}'.format(nodes, values))


def test_values_xyz10_4_(xyz10):  # Extract Y, Z, X
    nodes = ('Y', 'Z', 'X')
    values = xyz10.values(nodes)

    assert isinstance(values, ndarray)
    assert values.shape == (10, 3)
    assert (values == xyz10.data[:, [1, 2, 0]]).all()
    print('\n\nData for {} is:\n{}'.format(nodes, values))


# Test sequences changing data in different ways for continuous data

def test_sequence_xyz10_1_ok(xyz10):  # XYZ10 - test sequences of changes

    print()
    assert xyz10.get_order() == ('X', 'Y', 'Z')
    assert xyz10.N == 10
    assert xyz10.data.dtype == 'float32'
    assert (xyz10.data ==
            array([[1.1, 0.3, 0.3],
                   [0.,  3.1, 4.],
                   [0.2, 5.4, 1.7],
                   [4.4, 6.6, 1.9],
                   [0.6, 2.8, 9.9],
                   [4.,  6.,  9.],
                   [2.2, 3.1, 0.8],
                   [0.1, 0., 2.2],
                   [7.1, 3.9, 1.4],
                   [6., 0.2, 0.5]],
                  dtype='float32')).all().all()
    assert xyz10.sample.dtype == 'float32'
    assert (xyz10.sample ==
            array([[1.1, 0.3, 0.3],
                   [0.,  3.1, 4.],
                   [0.2, 5.4, 1.7],
                   [4.4, 6.6, 1.9],
                   [0.6, 2.8, 9.9],
                   [4.,  6.,  9.],
                   [2.2, 3.1, 0.8],
                   [0.1, 0., 2.2],
                   [7.1, 3.9, 1.4],
                   [6., 0.2, 0.5]],
                  dtype='float32')).all().all()
    assert xyz10.nodes == ('X', 'Y', 'Z')
    assert xyz10.order == (0, 1, 2)
    assert xyz10.ext_to_orig == {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
    assert xyz10.orig_to_ext == {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
    assert xyz10.dstype == 'continuous'
    assert xyz10.categories is None
    assert xyz10.node_values == {}
    assert tuple(xyz10.as_df().columns) == ('X', 'Y', 'Z')
    assert len(xyz10.as_df()) == 10

    # Set xyz10 size to 3 - column names should stay the same, but xyz10 only
    # has first three rows, and as_df() and node_values should reflect this

    xyz10.set_N(3)

    assert xyz10.get_order() == ('X', 'Y', 'Z')
    assert xyz10.N == 3
    assert (xyz10.sample ==
            array([[1.1, 0.3, 0.3],
                   [0.,  3.1, 4.],
                   [0.2, 5.4, 1.7]],
                  dtype='float32')).all().all()
    assert xyz10.nodes == ('X', 'Y', 'Z')
    assert xyz10.order == (0, 1, 2)
    assert xyz10.ext_to_orig == {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
    assert xyz10.orig_to_ext == {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
    assert xyz10.dstype == 'continuous'
    assert xyz10.categories is None
    assert xyz10.node_values == {}
    assert tuple(xyz10.as_df().columns) == ('X', 'Y', 'Z')
    assert len(xyz10.as_df()) == 3

    # Randomise column names - xyz10 should stay the same but external node
    # names should change

    xyz10.randomise_names(1)

    assert xyz10.get_order() == ('X002X', 'X000Y', 'X001Z')
    assert xyz10.N == 3
    assert xyz10.data.dtype == 'float32'
    assert (xyz10.sample ==
            array([[1.1, 0.3, 0.3],
                   [0.,  3.1, 4.],
                   [0.2, 5.4, 1.7]],
                  dtype='float32')).all().all()
    assert xyz10.nodes == ('X', 'Y', 'Z')
    assert xyz10.order == (0, 1, 2)
    assert xyz10.ext_to_orig == {'X002X': 'X', 'X000Y': 'Y', 'X001Z': 'Z'}
    assert xyz10.orig_to_ext == {'X': 'X002X', 'Y': 'X000Y', 'Z': 'X001Z'}
    assert xyz10.dstype == 'continuous'
    assert xyz10.categories is None
    assert (xyz10.values(('X001Z', 'X002X')) ==
            array([[0.3, 1.1], [4., 0.], [1.7, 0.2]],
                  dtype='float32')).all().all()

    # Set N to 4 - external column names should stay the same, but fourth
    # row should be added back into xyz10, and node value counts adjusted

    xyz10.set_N(4)

    assert xyz10.get_order() == ('X002X', 'X000Y', 'X001Z')
    assert xyz10.N == 4
    assert xyz10.data.dtype == 'float32'
    assert (xyz10.sample[:4, :] ==
            array([[1.1, 0.3, 0.3],
                   [0.,  3.1, 4.],
                   [0.2, 5.4, 1.7],
                   [4.4, 6.6, 1.9]],
                  dtype='float32')).all().all()
    assert xyz10.nodes == ('X', 'Y', 'Z')
    assert xyz10.order == (0, 1, 2)
    assert xyz10.ext_to_orig == {'X002X': 'X', 'X000Y': 'Y', 'X001Z': 'Z'}
    assert xyz10.orig_to_ext == {'X': 'X002X', 'Y': 'X000Y', 'Z': 'X001Z'}
    assert xyz10.dstype == 'continuous'
    assert xyz10.categories is None
    assert (xyz10.values(('X001Z', 'X002X')) ==
            array([[0.3, 1.1], [4., 0.], [1.7, 0.2], [1.9, 4.4]], 
                  dtype='float32')).all().all()

    # change processing order - just xyz10.order, get_order() changed

    xyz10.set_order(('X000Y', 'X001Z', 'X002X'))

    assert xyz10.get_order() == ('X000Y', 'X001Z', 'X002X')
    assert xyz10.N == 4
    assert xyz10.data.dtype == 'float32'
    assert (xyz10.sample ==
            array([[1.1, 0.3, 0.3],
                   [0.,  3.1, 4.],
                   [0.2, 5.4, 1.7],
                   [4.4, 6.6, 1.9]],
                  dtype='float32')).all().all()
    assert xyz10.nodes == ('X', 'Y', 'Z')
    assert xyz10.order == (1, 2, 0)
    assert xyz10.ext_to_orig == {'X002X': 'X', 'X000Y': 'Y', 'X001Z': 'Z'}
    assert xyz10.orig_to_ext == {'X': 'X002X', 'Y': 'X000Y', 'Z': 'X001Z'}
    assert xyz10.dstype == 'continuous'
    assert xyz10.categories is None
    assert (xyz10.values(('X001Z', 'X002X')) ==
            array([[0.3, 1.1], [4., 0.], [1.7, 0.2], [1.9, 4.4]], 
                  dtype='float32')).all().all()

    # randomise row order - sample and values change

    xyz10.set_N(4, 3)

    assert xyz10.get_order() == ('X000Y', 'X001Z', 'X002X')
    assert xyz10.N == 4
    assert xyz10.data.dtype == 'float32'
    print(xyz10.sample)
    return
    assert (xyz10.sample ==
            array([[1.1, 0.3, 0.3],
                   [0.,  3.1, 4.],
                   [0.2, 5.4, 1.7],
                   [4.4, 6.6, 1.9]],
                  dtype='float32')).all().all()
    assert xyz10.nodes == ('X', 'Y', 'Z')
    assert xyz10.order == (1, 2, 0)
    assert xyz10.ext_to_orig == {'X002X': 'X', 'X000Y': 'Y', 'X001Z': 'Z'}
    assert xyz10.orig_to_ext == {'X': 'X002X', 'Y': 'X000Y', 'Z': 'X001Z'}
    assert xyz10.dstype == 'continuous'
    assert xyz10.categories is None
    assert (xyz10.values(('X001Z', 'X002X')) ==
            array([[0.3, 1.1], [4., 0.], [1.7, 0.2], [1.9, 4.4]], 
                  dtype='float32')).all().all()
    return
