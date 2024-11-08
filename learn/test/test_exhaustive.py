
import pytest
from pandas import DataFrame

from learn.exhaustive import exhaustive
from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from core.metrics import dicts_same
from core.bn import BN


def test_exhaustive_type_error_1():
    with pytest.raises(TypeError):
        exhaustive()
    with pytest.raises(TypeError):
        exhaustive(67)
    with pytest.raises(TypeError):
        exhaustive(['asf'])
    with pytest.raises(TypeError):  # boolean for data
        exhaustive(True)
    data = Pandas.read(TESTDATA_DIR + '/simple/heckerman.csv',
                       dstype='categorical')
    with pytest.raises(TypeError):
        exhaustive(data, 32)  # numeric score type
    with pytest.raises(TypeError):
        exhaustive(data, [1.0, 'a'])  # numeric array for score types
    with pytest.raises(TypeError):
        exhaustive(data, 'bic', True)  # boolean for score params
    with pytest.raises(TypeError):
        exhaustive(data, params=[True])  # list for score params


def test_exhaustive_value_error_1():  # too many variables in data
    data = Pandas.read(TESTDATA_DIR + '/alarm/ALARM_N_1k.csv.zip',
                       dstype='categorical')
    with pytest.raises(ValueError):
        exhaustive(data, 'bic')


def test_exhaustive_value_error_2():  # invalid score types
    data = Pandas.read(TESTDATA_DIR + '/simple/heckerman.csv',
                       dstype='categorical')
    with pytest.raises(ValueError):  # unknown score
        exhaustive(data, types='unknown')
    with pytest.raises(ValueError):  # unknown score in list
        exhaustive(data, types=['aic', 'unknown'])


def test_exhaustive_value_error_3():  # unknown scoring parameters
    data = Pandas.read(TESTDATA_DIR + '/simple/heckerman.csv',
                       dstype='categorical')
    with pytest.raises(ValueError):  # unknown score
        exhaustive(data, params={'unknown': 0})
    with pytest.raises(ValueError):  # unknown score
        exhaustive(data, params={'base': 2, 'unknown': 0})


def test_exhaustive_value_error_4():  # unknown scoring parameter values
    data = Pandas.read(TESTDATA_DIR + '/simple/heckerman.csv',
                       dstype='categorical')
    with pytest.raises(ValueError):  # unknown score
        exhaustive(data, params={'base': 0})
    with pytest.raises(ValueError):  # unknown score
        exhaustive(data, params={'iss': -1})


def test_exhaustive_ab_1_ok():
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['1', '0']},
                            dtype='category'))
    dags = exhaustive(data)
    print('\nAll possible DAGs with two nodes, two identical rows\n')
    print(dags)


def test_exhaustive_abc_1_ok():
    data = Pandas(DataFrame({'A': ['0', '1'], 'B': ['1', '0'],
                             'C': ['0', '1']}, dtype='category'))
    dags = exhaustive(data)
    print('\nDAGs with three nodes, data (0,1,0), (1,0,1), best BIC first\n')
    print(dags)


def test_exhaustive_ab_random_1_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/ab_random.csv',
                       dstype='categorical')
    dags = exhaustive(data, params={'base': 2})
    print('\nn=2, N=100, perfectly random data')
    print(dags)
    assert dicts_same(dags.loc['[A][B]'].to_dict(),
                      {'bic': -44.32192809, 'loglik': -40.0,
                       'bde': -31.19819182})


def test_exhaustive_ab_deterministic_1_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/ab_deterministic.csv',
                       dstype='categorical')
    dags = exhaustive(data, params={'base': 2})
    print('\nn=2, N=20, completely deterministic data')
    print(dags)


def test_exhaustive_ab_strong_1_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/ab_strong.csv',
                       dstype='categorical')
    dags = exhaustive(data, params={'base': 2})
    print('\nn=2, N=20, close to deterministic data')
    print(dags)


def test_exhaustive_ab_moderate_1_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/ab_moderate.csv',
                       dstype='categorical')
    dags = exhaustive(data, params={'base': 2})
    print('\nn=2, N=20, moderate correlation')
    print(dags)


def test_exhaustive_file_abc_36_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/abc_36.csv',
                       dstype='categorical')
    dags = exhaustive(data, types='loglik')
    print('\nAll possible DAGs with three nodes, BIC with abc_36 data')
    print(dags)


def test_exhaustive_ab_100_ok():

    #   Generate 100 rows of data for a BN with structure A-->B and
    #   exhaustively score and rank all possible DAGs

    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    global_dist = bn.global_distribution()
    print('\n\n{}'.format(global_dist))
    data = Pandas(bn.generate_cases(100))
    dags = exhaustive(data, types=['bic', 'loglik', 'bde'],
                      params={'base': 'e'}, normalise=False)
    print("CPDAG scores (base e) using 100 random rows from A->B:\n{}"
          .format(dags))

    # Check score of graphs with highest score and normalisation - agrees
    # with values produced by bnlearn HC diagnostics

    assert dicts_same(dags.iloc[0].to_dict(),
                      {'bic': -129.443485, 'loglik': -122.535730,
                       'bde': -130.239091}, sf=8)
    assert dicts_same(dags.iloc[-1].to_dict(),
                      {'bic': -129.596717, 'loglik': -124.991546,
                       'bde': -130.053854}, sf=8)


def test_exhaustive_abc_100_ok():

    #   Generate 100 rows of data for a BN with structure A-->B-->C and
    #   exhaustively score and rank all possible DAGs

    bn = BN.read(TESTDATA_DIR + '/dsc/abc.dsc')
    assert isinstance(bn, BN)
    assert bn.free_params == 5
    global_dist = bn.global_distribution()
    print('\n\n{}'.format(global_dist))
    data = Pandas(bn.generate_cases(100))
    dags = exhaustive(data, types=['bic', 'loglik', 'bde'],
                      params={'base': 10}, normalise=True)
    print("CPDAG scores using 100 random rows from A->B->C:\n{}".format(dags))

    # Check score of graphs with highest score and normalisation

    assert dicts_same(dags.iloc[0].to_dict(),
                      {'bic': 9.353281926, 'loglik': 11.35328193,
                       'bde': 20.80273205})
    assert dicts_same(dags.iloc[-1].to_dict(),
                      {'bic': -90.70458804, 'loglik': -87.70458804,
                       'bde': -209.5401274})


def test_exhaustive_abc_1K_ok():

    #   Generate 1000 rows of data for a BN with structure A-->B-->C and
    #   exhaustively score and rank all possible DAGs

    bn = BN.read(TESTDATA_DIR + '/dsc/abc.dsc')
    assert isinstance(bn, BN)
    assert bn.free_params == 5
    global_dist = bn.global_distribution()
    print('\n\n{}'.format(global_dist))
    data = Pandas(bn.generate_cases(1000))
    dags = exhaustive(data, types=['bic', 'loglik', 'bde'],
                      params={'base': 10}, normalise=True)
    print("CPDAG scores using 1000 random rows from A->B->C:\n{}".format(dags))

    # Check score of graphs with highest score and normalisation

    assert dicts_same(dags.iloc[0].to_dict(),
                      {'bic': 131.0081588, 'loglik': 134.0081588,
                       'bde': 301.1263241})
    assert dicts_same(dags.iloc[-1].to_dict(),
                      {'bic': -845.7026966, 'loglik': -841.2026966,
                       'bde': -1947.980603})


def test_exhaustive_ab_cb_100_ok():

    #   Generate 100 rows of data for a BN with structure A-->B<--C and
    #   exhaustively score and rank all possible DAGs

    bn = BN.read(TESTDATA_DIR + '/dsc/ab_cb.dsc')
    assert isinstance(bn, BN)
    assert bn.free_params == 6
    global_dist = bn.global_distribution()
    print('\n\n{}'.format(global_dist))
    data = Pandas(bn.generate_cases(100))
    dags = exhaustive(data, types=['bic', 'loglik', 'bde'],
                      params={'base': 10}, normalise=True)
    print("CPDAG scores using 100 random rows from A->B<-C:\n{}".format(dags))

    # Check score of graphs with highest score and normalisation

    assert dicts_same(dags.iloc[0].to_dict(),
                      {'bic': 6.952241981, 'loglik': 8.952241981,
                       'bde': 15.44100853})
    assert dicts_same(dags.iloc[-1].to_dict(),
                      {'bic': -82.77577341, 'loglik': -79.77577341,
                       'bde': -191.2841223})


def test_exhaustive_ab_cb_1K_ok():

    #   Generate 1000 rows of data for a BN with structure A-->B<--C and
    #   exhaustively score and rank all possible DAGs

    bn = BN.read(TESTDATA_DIR + '/dsc/ab_cb.dsc')
    assert isinstance(bn, BN)
    assert bn.free_params == 6
    global_dist = bn.global_distribution()
    print('\n\n{}'.format(global_dist))
    data = Pandas(bn.generate_cases(1000))
    dags = exhaustive(data, types=['bic', 'loglik', 'bde'],
                      params={'base': 10}, normalise=True)
    print("CPDAG scores using 1000 random rows from A->B<-C:\n{}".format(dags))

    # Check score of graphs with highest score and normalisation

    assert dicts_same(dags.iloc[0].to_dict(),
                      {'bic': 71.61461171, 'loglik': 76.11461171,
                       'bde': 164.0778562})
    assert dicts_same(dags.iloc[-1].to_dict(),
                      {'bic': -747.1540477, 'loglik': -742.6540477,
                       'bde': -1721.064061})
