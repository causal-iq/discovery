
# Test the Pandas read and write methods

import pytest
from random import random
from os import remove
from pandas import DataFrame

from fileio.pandas import Pandas
from fileio.common import FileFormatError, TESTDATA_DIR, EXPTS_DIR
from core.metrics import dicts_same


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.csv'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpgzfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.csv.gz'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


def test_pandas_read_type_error_1():  # fails with no arguments
    with pytest.raises(TypeError):
        Pandas.read()


def test_pandas_read_type_error_2():  # fails with bad filename type
    with pytest.raises(TypeError):
        Pandas.read(666)
    with pytest.raises(TypeError):
        Pandas.read(['should not be an array'])


def test_pandas_read_type_error_3():  # fails with bad PandassetType
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', dstype=True)
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', dstype=37)
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', dstype='invalid')


def test_pandas_read_type_error_4():  # fails with bad N type
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', N='invalid')
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', N=True)
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', N=[1])


def test_pandas_read_filenotfound_error():  # fails with nonexistent file
    with pytest.raises(FileNotFoundError):
        Pandas.read('nonexistent.txt')


def test_pandas_read_fileformat_error():  # fails with binary file
    with pytest.raises(FileFormatError):
        Pandas.read(TESTDATA_DIR + '/misc/null.sys')


def test_pandas_read_value_error_1():  # invalid coercion to float
    with pytest.raises(ValueError):
        Pandas.read(TESTDATA_DIR + '/simple/mix_2.csv', dstype='continuous')


def test_pandas_read_value_error_2():  # invalid N values
    with pytest.raises(ValueError):
        Pandas.read(TESTDATA_DIR + '/simple/mix_2.csv', N=1)
    with pytest.raises(ValueError):
        Pandas.read(TESTDATA_DIR + '/simple/mix_2.csv', N=0)
    with pytest.raises(ValueError):
        Pandas.read(TESTDATA_DIR + '/simple/mix_2.csv', N=-1)


def test_pandas_read_value_error_3():  # invalid N bigger than dataset
    with pytest.raises(ValueError):
        Pandas.read(TESTDATA_DIR + '/simple/mix_2.csv', N=3)


def test_pandas_read_ab_1_ok():  # reads AB csv file
    data = Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv')
    assert isinstance(data, Pandas)
    print('\nPandas is:\n{}\n'.format(data.sample))
    assert data.N == 3
    assert data.nodes == \
        ('A', 'B')
    assert data.dstype == 'categorical'
    assert data.node_types == {'A': 'category', 'B': 'category'}
    assert data.node_values == \
        {'A': {'1': 2, '0': 1},
         'B': {'0': 2, '1': 1}}


def test_pandas_read_ab_2_ok():  # reads AB csv file, force categorical
    data = Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', dstype='categorical')
    assert isinstance(data, Pandas)
    print('\nPandas is:\n{}\n'.format(data.sample))
    assert data.N == 3
    assert data.nodes == \
        ('A', 'B')
    assert data.dstype == 'categorical'
    assert data.node_types == {'A': 'category', 'B': 'category'}
    assert data.node_values == \
        {'A': {'1': 2, '0': 1},
         'B': {'0': 2, '1': 1}}


def test_pandas_read_ab_3_ok():  # reads AB csv file, force to floats
    data = Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', dstype='continuous')
    assert isinstance(data, Pandas)
    print('\nPandas is:\n{}\n'.format(data.sample))
    assert data.N == 3
    assert data.nodes == \
        ('A', 'B')
    assert data.dstype == 'continuous'
    assert data.node_types == {'A': 'float32', 'B': 'float32'}
    assert data.node_values == {}
    assert dicts_same(data.sample['A'].to_dict(),
                      {0: 1.0, 1: 1.0, 2: 0.0}, sf=3)
    assert dicts_same(data.sample['B'].to_dict(),
                      {0: 1.0, 1: 0.0, 2: 0.0}, sf=3)


def test_pandas_read_xy_1_ok():  # reads a XY csv file - detect var types
    data = Pandas.read(TESTDATA_DIR + '/simple/xy_3.csv')
    assert isinstance(data, Pandas)
    print('\nPandas is:\n{}\n'.format(data.sample))
    assert data.N == 3
    assert data.nodes == \
        ('F1', 'F2')
    assert data.node_values == {}
    assert data.dstype == 'continuous'
    assert data.node_types == {'F1': 'float32', 'F2': 'float32'}
    assert dicts_same(data.sample['F1'].to_dict(),
                      {0: 1.01, 1: -0.45, 2: 1.22}, sf=3)
    assert dicts_same(data.sample['F2'].to_dict(),
                      {0: 1.21, 1: 0.67, 2: -1.41}, sf=3)


def test_pandas_read_xy_2_ok():  # reads a XY csv file, specify as continuous
    data = Pandas.read((TESTDATA_DIR + '/simple/xy_3.csv'), 'continuous')
    assert isinstance(data, Pandas)
    print('\nPandas is:\n{}\n'.format(data.sample))
    assert data.N == 3
    assert data.nodes == \
        ('F1', 'F2')
    assert data.dstype == 'continuous'
    assert data.node_values == {}
    assert data.node_types == {'F1': 'float32', 'F2': 'float32'}
    assert dicts_same(data.sample['F1'].to_dict(),
                      {0: 1.01, 1: -0.45, 2: 1.22}, sf=3)
    assert dicts_same(data.sample['F2'].to_dict(),
                      {0: 1.21, 1: 0.67, 2: -1.41}, sf=3)


def test_pandas_read_xy_3_ok():  # reads a XY csv file, force to categorical
    data = Pandas.read((TESTDATA_DIR + '/simple/xy_3.csv'), 'categorical')
    assert isinstance(data, Pandas)
    print('\nPandas is:\n{}\n'.format(data.sample))
    assert data.N == 3
    assert data.nodes == \
        ('F1', 'F2')
    assert data.node_values == {'F1': {'-0.45': 1, '1.01': 1, '1.22': 1},
                                'F2': {'-1.41': 1, '0.67': 1, '1.21': 1}}
    assert data.dstype == 'categorical'
    assert data.node_types == {'F1': 'category', 'F2': 'category'}
    assert data.sample['F1'].to_dict() == {0: '1.01', 1: '-0.45', 2: '1.22'}
    assert data.sample['F2'].to_dict() == {0: '1.21', 1: '0.67', 2: '-1.41'}


def test_pandas_read_cancer_1_ok():  # reads a gzipped Cancer file OK
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 1000
    assert data.nodes == \
        ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray')
    assert data.node_values == \
        {'Cancer': {'False': 986, 'True': 14},
         'Dyspnoea': {'False': 700, 'True': 300},
         'Pollution': {'low': 897, 'high': 103},
         'Smoker': {'False': 693, 'True': 307},
         'Xray': {'negative': 805, 'positive': 195}}
    assert all([t == 'category' for t in data.node_types.values()])
    assert data.dstype == 'categorical'

    print('\nMemory usage is:')
    for n in data.nodes:
        print('  {} needs {} bytes'
              .format(n, data.sample[n].memory_usage(index=False, deep=True)))


def test_pandas_read_cancer_2_ok():  # reads a gzipped Cancer file, N=10
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                       N=10)
    assert isinstance(data, Pandas)
    assert data.N == 10
    assert data.nodes == \
        ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray')
    assert data.node_values == \
        {'Cancer': {'False': 10},
         'Dyspnoea': {'False': 5, 'True': 5},
         'Pollution': {'low': 7, 'high': 3},
         'Smoker': {'False': 7, 'True': 3},
         'Xray': {'negative': 8, 'positive': 2}}
    assert all([t == 'category' for t in data.node_types.values()])
    assert data.dstype == 'categorical'


def test_pandas_read_mix_1_ok():  # reads a mixed file - detect var types
    data = Pandas.read(TESTDATA_DIR + '/simple/mix_2.csv')
    assert isinstance(data, Pandas)
    print('\nPandas is:\n{}\n'.format(data.sample))
    assert data.N == 2
    assert data.nodes == \
        ('C', 'I', 'F')
    assert data.node_values == {'C': {'A': 1, 'B': 1},
                                'I': {'-1': 1, '3': 1}}
    assert data.dstype == 'mixed'
    assert data.node_types == \
        {'C': 'category', 'I': 'category', 'F': 'float32'}


def test_pandas_read_mix_2_ok():  # reads a mixed file - force to category
    data = Pandas.read(TESTDATA_DIR + '/simple/mix_2.csv',
                       dstype='categorical')
    assert isinstance(data, Pandas)
    print('\nPandas is:\n{}\n'.format(data.sample))
    assert data.N == 2
    assert data.nodes == \
        ('C', 'I', 'F')
    assert data.node_values == {'C': {'A': 1, 'B': 1},
                                'I': {'-1': 1, '3': 1},
                                'F': {'2.7': 1, '-0.3': 1}}
    assert data.dstype == 'categorical'
    assert data.node_types == \
        {'C': 'category', 'I': 'category', 'F': 'category'}


def test_pandas_read_gauss_1_ok():  # reads gzipped Gaussian test data file
    data = Pandas.read(TESTDATA_DIR + '/simple/gauss.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 5000
    assert data.nodes == \
        ('A', 'B', 'C', 'D', 'E', 'F', 'G')
    assert data.node_values == {}
    assert all([t == 'float32' for t in data.node_types.values()])
    assert data.dstype == 'continuous'

    print('\nMemory usage is:')
    for n in data.nodes:
        print('  {} needs {} bytes'
              .format(n, data.sample[n].memory_usage(index=False, deep=True)))


def test_pandas_read_asia_1_ok():  # reads a gzipped Asia data file
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 1000
    assert data.nodes == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.node_values == \
        {'asia': {'no': 990, 'yes': 10},
         'bronc': {'no': 576, 'yes': 424},
         'dysp': {'no': 586, 'yes': 414},
         'either': {'no': 930, 'yes': 70},
         'lung': {'no': 943, 'yes': 57},
         'smoke': {'no': 503, 'yes': 497},
         'tub': {'no': 986, 'yes': 14},
         'xray': {'no': 889, 'yes': 111}}
    assert all([t == 'category' for t in data.node_types.values()])
    assert data.dstype == 'categorical'


def test_pandas_read_alarm_1_ok():  # reads a gzipped Alarm data file
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/alarm.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 1000
    assert data.nodes == \
        ('ANAPHYLAXIS', 'ARTCO2', 'BP', 'CATECHOL', 'CO', 'CVP', 'DISCONNECT',
         'ERRCAUTER', 'ERRLOWOUTPUT', 'EXPCO2', 'FIO2', 'HISTORY', 'HR',
         'HRBP', 'HREKG', 'HRSAT', 'HYPOVOLEMIA', 'INSUFFANESTH', 'INTUBATION',
         'KINKEDTUBE', 'LVEDVOLUME', 'LVFAILURE', 'MINVOL', 'MINVOLSET', 'PAP',
         'PCWP', 'PRESS', 'PULMEMBOLUS', 'PVSAT', 'SAO2', 'SHUNT',
         'STROKEVOLUME', 'TPR', 'VENTALV', 'VENTLUNG', 'VENTMACH', 'VENTTUBE')
    assert data.node_values == \
        {'ANAPHYLAXIS': {'FALSE': 990, 'TRUE': 10},
         'ARTCO2': {'HIGH': 746, 'LOW': 195, 'NORMAL': 59},
         'BP': {'HIGH': 411, 'LOW': 382, 'NORMAL': 207},
         'CATECHOL': {'HIGH': 903, 'NORMAL': 97},
         'CO': {'HIGH': 637, 'NORMAL': 186, 'LOW': 177},
         'CVP': {'NORMAL': 731, 'HIGH': 160, 'LOW': 109},
         'DISCONNECT': {'FALSE': 898, 'TRUE': 102},
         'ERRCAUTER': {'FALSE': 903, 'TRUE': 97},
         'ERRLOWOUTPUT': {'FALSE': 948, 'TRUE': 52},
         'EXPCO2': {'LOW': 863, 'NORMAL': 55, 'HIGH': 42, 'ZERO': 40},
         'FIO2': {'NORMAL': 954, 'LOW': 46},
         'HISTORY': {'FALSE': 943, 'TRUE': 57},
         'HR': {'HIGH': 808, 'NORMAL': 173, 'LOW': 19},
         'HRBP': {'HIGH': 753, 'LOW': 180, 'NORMAL': 67},
         'HREKG': {'HIGH': 729, 'LOW': 174, 'NORMAL': 97},
         'HRSAT': {'HIGH': 732, 'LOW': 171, 'NORMAL': 97},
         'HYPOVOLEMIA': {'FALSE': 796, 'TRUE': 204},
         'INSUFFANESTH': {'FALSE': 927, 'TRUE': 73},
         'INTUBATION': {'NORMAL': 919, 'ONESIDED': 47, 'ESOPHAGEAL': 34},
         'KINKEDTUBE': {'FALSE': 963, 'TRUE': 37},
         'LVEDVOLUME': {'NORMAL': 707, 'HIGH': 212, 'LOW': 81},
         'LVFAILURE': {'FALSE': 952, 'TRUE': 48},
         'MINVOL': {'ZERO': 700, 'HIGH': 205, 'LOW': 66, 'NORMAL': 29},
         'MINVOLSET': {'NORMAL': 891, 'HIGH': 55, 'LOW': 54},
         'PAP': {'NORMAL': 884, 'HIGH': 63, 'LOW': 53},
         'PCWP': {'NORMAL': 677, 'HIGH': 217, 'LOW': 106},
         'PRESS': {'HIGH': 519, 'LOW': 256, 'NORMAL': 196, 'ZERO': 29},
         'PULMEMBOLUS': {'FALSE': 988, 'TRUE': 12},
         'PVSAT': {'LOW': 786, 'HIGH': 184, 'NORMAL': 30},
         'SAO2': {'LOW': 783, 'HIGH': 182, 'NORMAL': 35},
         'SHUNT': {'NORMAL': 897, 'HIGH': 103},
         'STROKEVOLUME': {'NORMAL': 771, 'LOW': 188, 'HIGH': 41},
         'TPR': {'NORMAL': 412, 'LOW': 301, 'HIGH': 287},
         'VENTALV': {'ZERO': 681, 'HIGH': 201, 'LOW': 84, 'NORMAL': 34},
         'VENTLUNG': {'ZERO': 725, 'LOW': 235, 'HIGH': 29, 'NORMAL': 11},
         'VENTMACH': {'NORMAL': 819, 'LOW': 62, 'ZERO': 61, 'HIGH': 58},
         'VENTTUBE': {'LOW': 723, 'ZERO': 205, 'HIGH': 58, 'NORMAL': 14}}
    assert all([t == 'category' for t in data.node_types.values()])
    assert data.dstype == 'categorical'


# Tests with full experimental datasets - just check data types as expected

@pytest.mark.slow
def test_pandas_read_asia_2_ok():  # check all Asia data treated as category
    data = Pandas.read(EXPTS_DIR + '/datasets/asia.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_asia_3_ok():  # check all Asia data specified as category
    data = Pandas.read(EXPTS_DIR + '/datasets/asia.data.gz',
                       dstype='categorical')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_sports_2_ok():  # check all Sports treated as category
    data = Pandas.read(EXPTS_DIR + '/datasets/sports.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_sachs_2_ok():  # check all Sachs data treated as category
    data = Pandas.read(EXPTS_DIR + '/datasets/sachs.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_covid_2_ok():  # check all Covid data treated as category
    data = Pandas.read(EXPTS_DIR + '/datasets/covid.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_child_2_ok():  # check all Child is category
    data = Pandas.read(EXPTS_DIR + '/datasets/child.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_insurance_2_ok():  # check all insurance is category
    data = Pandas.read(EXPTS_DIR + '/datasets/insurance.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_property_2_ok():  # check all property is category
    data = Pandas.read(EXPTS_DIR + '/datasets/property.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_diarrhoea_2_ok():  # check all Diarrhoea is category
    data = Pandas.read(EXPTS_DIR + '/datasets/diarrhoea.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_water_2_ok():  # check all Water is category
    data = Pandas.read(EXPTS_DIR + '/datasets/water.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_mildew_2_ok():  # check all Mildew is category
    data = Pandas.read(EXPTS_DIR + '/datasets/mildew.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_alarm_2_ok():  # check all Alarm is category
    data = Pandas.read(EXPTS_DIR + '/datasets/alarm.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_barley_2_ok():  # check all Barley is category
    data = Pandas.read(EXPTS_DIR + '/datasets/barley.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_hailfinder_2_ok():  # check all Hailfinder is category
    data = Pandas.read(EXPTS_DIR + '/datasets/hailfinder.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_hepar2_2_ok():  # check all Hepar2 is category
    data = Pandas.read(EXPTS_DIR + '/datasets/hepar2.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_win95pts_2_ok():  # check all Win95pts is category
    data = Pandas.read(EXPTS_DIR + '/datasets/win95pts.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_formed_2_ok():  # check all Formed is category
    data = Pandas.read(EXPTS_DIR + '/datasets/formed.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


@pytest.mark.slow
def test_pandas_read_pathfinder_2_ok():  # check all Pathfinder is category
    data = Pandas.read(EXPTS_DIR + '/datasets/pathfinder.data.gz')
    assert isinstance(data, Pandas)
    assert data.N == 10000000
    assert all([d.__str__() == 'category' for d in data.sample.dtypes])


# Pandas write tests

def test_pandas_write_type_error_1():  # write with no arguments
    with pytest.raises(TypeError):
        Pandas.write()


def test_pandas_write_type_error_2():  # write with bad filename type
    with pytest.raises(TypeError):
        Pandas.write(3)
    with pytest.raises(TypeError):
        Pandas.write(None, compress=True)
    with pytest.raises(TypeError):
        Pandas.write([42])
    with pytest.raises(TypeError):
        Pandas.write('invalid')


def test_pandas_write_type_error_3():  # write with bad compress type
    data = Pandas(df=DataFrame({'A': ['1', '1'], 'B': ['1', '0']},
                               dtype='category'))
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', compress=None)
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + '/tmp/wont_createt.csv', compress=1)


def test_pandas_write_type_error_4():  # write with bad sf type
    data = Pandas(df=DataFrame({'A': ['1', '1'], 'B': ['1', '0']},
                               dtype='category'))
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=None)
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf='one')
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=(1,))
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=0.2)
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=True)


def test_pandas_write_type_error_5():  # write with bad zero type
    data = Pandas(df=DataFrame({'A': ['1', '1'], 'B': ['1', '0']},
                               dtype='category'))
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=2, zero=False)
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=2, zero=1)


def test_pandas_write_type_error_6():  # write with bad preserve type
    data = Pandas(df=DataFrame({'A': ['1', '1'], 'B': ['1', '0']},
                               dtype='category'))
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', preserve=1)


def test_pandas_write_value_error_1():  # sf not between 2 and 10
    data = Pandas(df=DataFrame({'A': ['1', '1'], 'B': ['1', '0']},
                               dtype='category'))
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=1)
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=0)
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=-1)
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=11)


def test_pandas_write_value_error_2():  # zero not between 1E-20 and 1E-1
    data = Pandas(df=DataFrame({'A': ['1', '1'], 'B': ['1', '0']},
                               dtype='category'))
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=2, zero=0.2)
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=2, zero=-0.01)
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=2, zero=1E-21)


def test_pandas_write_filenotfound_error_1():  # bad directory
    data = Pandas(df=DataFrame({'A': ['1', '1'], 'B': ['1', '0']},
                               dtype='category'))
    with pytest.raises(FileNotFoundError):
        data.write(TESTDATA_DIR + '/nonexistent/bad.csv')


def test_pandas_write_1_ok(tmpfile):  # write a discrete non-compressed file OK
    data = Pandas(df=DataFrame({'A': ['1', '1'], 'B': ['1', '0']},
                               dtype='category'))
    data.write(tmpfile)
    check = Pandas.read(tmpfile, dstype='categorical')
    assert check.df.to_dict() == data.df.to_dict()


def test_pandas_write_2_ok(tmpgzfile):  # write a discrete compressed file OK
    data = Pandas(df=DataFrame({'A': ['1', '1'], 'B': ['1', '0']},
                               dtype='category'))
    data.write(tmpgzfile, compress=True)
    check = Pandas.read(tmpgzfile, dstype='categorical')
    assert check.df.to_dict() == data.df.to_dict()


def test_pandas_write_3_ok(tmpgzfile):  # write a cont compressed file OK
    data = Pandas(df=DataFrame({'A': [1.0, -1.0], 'B': [1.5, 0.0]},
                               dtype='float32'))
    data.write(tmpgzfile, compress=True, preserve=True)
    check = Pandas.read(tmpgzfile, dstype='continuous')
    assert check.df.to_dict() == data.df.to_dict()


def test_pandas_write_4_ok(tmpgzfile):  # check rounding to 2 s.f., zero 0.01
    data = Pandas(df=DataFrame({'A': [1.04, -.00348],
                                'B': [132, 0.0000453]},
                               dtype='float32'))
    data.write(tmpgzfile, compress=True, sf=2)
    check = Pandas.read(tmpgzfile, dstype='continuous')
    assert check.df.to_dict(orient='list') == {'A': [1.0, 0.0],
                                               'B': [130.0, 0.0]}


def test_pandas_write_5_ok(tmpfile):  # check rounding to 2 s.f., zero 1E-5
    data = [{'A': 1.04, 'B': 132},
            {'A': -.00348066, 'B': 0.00045}]
    df = DataFrame.from_records(data).astype('float32')
    Pandas(df).write(tmpfile, compress=False, sf=2, zero=10e-5)
    check = Pandas.read(tmpfile, dstype='continuous')
    print('\nRounded df read back in:\n{}\n'.format(check.df))
    check = check.df.to_dict(orient='records')
    for row in zip(data, check):
        assert dicts_same(row[0], row[1], sf=2)


def test_pandas_write_6_ok(tmpfile):  # check rounding to 3 s.f., zero 1E-6
    data = [{'A': 1.04, 'B': 132},
            {'A': -.00348066, 'B': 0.00045},
            {'A': 1E-5, 'B': 43.12345}]
    df = DataFrame.from_records(data).astype('float32')
    Pandas(df).write(tmpfile, compress=False, sf=3, zero=1E-6)
    check = Pandas.read(tmpfile, dstype='continuous')
    print('\nRounded df read back in:\n{}\n'.format(check.df))
    check = check.df.to_dict(orient='records')
    for row in zip(data, check):
        assert dicts_same(row[0], row[1], sf=3)


def test_pandas_write_7_ok(tmpfile):  # check rounding to 7 s.f.,
    data = [{'A': 1.04, 'B': 132},
            {'A': -.00348066, 'B': 0.00045},
            {'A': 1E-5, 'B': 43.12345}]
    df = DataFrame.from_records(data).astype('float32')
    Pandas(df).write(tmpfile, compress=False, sf=7)
    check = Pandas.read(tmpfile, dstype='continuous')
    print('\nRounded df read back in:\n{}\n'.format(check.df))
    check = check.df.to_dict(orient='records')
    for row in zip(data, check):
        assert dicts_same(row[0], row[1], sf=7)
