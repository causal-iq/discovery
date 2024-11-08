
# Test DAG scoring in R code

import pytest
from pandas import DataFrame
from random import random
from os import remove

from call.bnlearn import bnlearn_score
import testdata.example_dags as dag
from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from core.metrics import dicts_same, values_same

TYPES = ['loglik', 'bic', 'aic', 'bde', 'k2', 'bdj', 'bds']  # scores to test


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.csv'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


def test_bnlearn_score_type_error():  # bad arg types
    with pytest.raises(TypeError):
        bnlearn_score()
    with pytest.raises(TypeError):
        bnlearn_score(6, 'a')
    with pytest.raises(TypeError):
        bnlearn_score('[A][B]', DataFrame({'A': ['1'], 'B': ['1']}))
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), 32)
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), {'A': ['1'], 'B': ['1']})
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      17)
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      ['bic', False], {'iss': 1.0})
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      ['bic', 'loglik'], 32)


def test_bnlearn_score_type_error_2():  # bad ISS type
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='bde', params={'iss': 'wrong type'})


def test_bnlearn_score_type_error_3():  # Illegal dstype
    data = DataFrame({'A': ['1', '0'], 'B': ['1', '0']})
    with pytest.raises(TypeError):
        bnlearn_score(dag.a_b(), data, TYPES, params={'iss': 1.0},
                      dstype='invalid')
    with pytest.raises(TypeError):
        bnlearn_score(dag.a_b(), data, TYPES, params={'iss': 1.0},
                      dstype='continuous')


def test_bnlearn_score_value_error_1():  # variable set mismatch
    with pytest.raises(ValueError):
        bnlearn_score(dag.abc(), DataFrame({'A': ['1', '0'], 'B': ['0', '1']}),
                      types='bic', params={'iss': 1.0})


def test_bnlearn_score_value_error_2():  # must be at least 2 variables
    with pytest.raises(ValueError):
        bnlearn_score(dag.a(), DataFrame({'A': ['1']}), types='loglik',
                      params={'iss': 1.0})


def test_bnlearn_score_value_error_3():  # some variables single-valued
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '1']}),
                      types='aic', params={'iss': 1.0})


def test_bnlearn_score_value_error_4():  # invalid score type
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['0', '1']}),
                      types='unsupported', params={'iss': 1.0})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['0', '1']}),
                      types=['bic', 'unsupported'], params={'iss': 1.0})


def test_bnlearn_score_value_error_5():  # bad k value
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='aic', params={'k': 0.0})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='aic', params={'k': 1E+10})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='aic', params={'k': 1E-8})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='aic', params={'k': -3})

# Test for irrelevant score parameters disabled for now (error_6, 7 & 8)


def xtest_bnlearn_score_value_error_6():  # bad ISS value
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='aic', params={'iss': 0.0})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='aic', params={'iss': 1E+10})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='aic', params={'iss': 0.9})


def xtest_bnlearn_score_value_error_7():  # specifying ISS for non bds/bde
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='aic', params={'iss': 1.0})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='bic', params={'iss': 1.0})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='loglik', params={'iss': 1.0})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='bdj', params={'iss': 1.0})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='k2', params={'iss': 1.0})


def xtest_bnlearn_score_value_error_8():  # specifying k for non aic/bic
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='bde', params={'k': 1.0})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='bds', params={'k': 1.0})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='loglik', params={'k': 1.0})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='bdj', params={'k': 1.0})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
                      types='k2', params={'k': 1.0})


def test_bnlearn_score_a_b_1_ok():  # A, B unconnected
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['1', '0']},
                            dtype='category'))
    bnscores = bnlearn_score(dag.a_b(), data.sample, TYPES,
                             params={'iss': 1.0})
    scores = dag.a_b().score(data, TYPES, {'base': 'e'})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_2_ok():  # A, B unconnected, ISS = 2
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['1', '0']},
                            dtype='category'))
    bnscores = bnlearn_score(dag.a_b(), data.sample, TYPES,
                             params={'iss': 2.0})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'iss': 2.0})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_3_ok():  # A, B unconnected, ISS = 10
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['1', '0']},
                            dtype='category'))
    bnscores = bnlearn_score(dag.a_b(), data.sample, TYPES,
                             params={'iss': 10.0})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'iss': 10.0})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_4_ok():  # A, B unconnected, ISS = 100
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['1', '0']},
                            dtype='category'))
    bnscores = bnlearn_score(dag.a_b(), data.sample, TYPES,
                             params={'iss': 100.0})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'iss': 100.0})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_5_ok():  # A, B unconnected, k=2
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['1', '0']},
                            dtype='category'))
    bnscores = bnlearn_score(dag.a_b(), data.sample, TYPES,
                             params={'k': 2})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'k': 2})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_6_ok():  # A, B unconnected, k=0.5
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['1', '0']},
                            dtype='category'))
    bnscores = bnlearn_score(dag.a_b(), data.sample, TYPES,
                             params={'k': 0.5})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'k': 0.5})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_7_ok():  # A, B unconnected, k=10.0
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['1', '0']},
                  dtype='category'))
    bnscores = bnlearn_score(dag.a_b(), data.sample, TYPES,
                             params={'k': 10.0})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'k': 10.0})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_8_ok():  # A, B unconnected, k=0.1
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['1', '0']},
                            dtype='category'))
    bnscores = bnlearn_score(dag.a_b(), data.sample, TYPES, params={'k': 0.1})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'k': 0.1})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_x_y_1_ok():  # scoring X, Y data
    data = Pandas(DataFrame({'X': [1.1, 2.2, -0.3], 'Y': [0.0, 1.7, 0.0]}))
    bnscores = bnlearn_score(dag.x_y(), data.sample, ['bic-g'],
                             params={'k': 1.0})
    scores = dag.x_y().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))

    # BIC scores very close because no linear regression involved

    assert dicts_same(bnscores, dict(scores.sum()), sf=6)


def test_bnlearn_score_x_y_2_ok():  # scoring X, Y data
    data = Pandas(DataFrame({'X': [-0.4, 5.1, -3.2, 2.0, -0.7],
                             'Y': [0.6, 1.7, 0.0, -0.8, 0.6]}))
    bnscores = bnlearn_score(dag.x_y(), data.sample, ['bic-g'],
                             params={'k': 1.0})
    scores = dag.x_y().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))

    # BIC scores very close because no linear regression involved

    assert dicts_same(bnscores, dict(scores.sum()), sf=6)


def test_bnlearn_score_xy_1_ok():  # scoring X, Y data
    data = Pandas(DataFrame({'X': [-0.4, 5.1, -3.2, 2.0, -0.7],
                             'Y': [0.6, 1.7, 0.0, -0.8, 0.6]}))
    bnscores = bnlearn_score(dag.xy(), data.sample, ['bic-g'],
                             params={'k': 1.0})
    scores = dag.xy().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))

    # BIC scores not very similar because of linear regression differences

    assert dicts_same(bnscores, dict(scores.sum()), sf=2)


def test_bnlearn_score_xyz_1_ok():  # scoring X --> Y --> Z DAG
    data = Pandas(DataFrame({'X': [0.2, 0.4, -0.1, 0.6, 0.9, 1.1],
                             'Y': [0.5, 0.7, 0.0, 1.4, 1.9, 2.4],
                             'Z': [2.4, 2.6, 1.9, 3.3, 3.6, 4.2]}))
    bnscores = bnlearn_score(dag.xyz(), data.sample, ['bic-g'],
                             params={'k': 1.0})
    scores = dag.xyz().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))

    # BIC scores not very similar because of linear regression differences

    assert values_same(bnscores['bic-g'], 3.334, sf=4)
    assert values_same(dict(scores.sum())['bic-g'], 3.673, sf=4)


def test_bnlearn_score_xy_zy_1_ok():  # scoring X --> Y <-- Z DAG
    data = Pandas(DataFrame({'X': [0.2, 0.4, -0.1, 0.6, 0.9, 1.1],
                             'Y': [0.5, 0.7, 0.0, 1.4, 1.9, 2.4],
                             'Z': [2.4, 2.6, 1.9, 3.3, 3.6, 4.2]}))
    bnscores = bnlearn_score(dag.xy_zy(), data.sample, ['bic-g'],
                             params={'k': 1.0})
    scores = dag.xy_zy().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))

    # BIC scores not very similar because of linear regression differences

    assert values_same(bnscores['bic-g'], -8.1976, sf=5)
    assert values_same(dict(scores.sum())['bic-g'], -7.6652, sf=5)


def test_bnlearn_score_gauss_1_ok():  # scoring BNLearn example Gaussian
    data = Pandas.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                       dstype='continuous')

    # the following line succeeds even though dstype is not set to 'continuous'
    # because all variables are float and so not changed to categories in the
    # R code. Correct way to do this would be to support dstype='continuous'

    bnscores = bnlearn_score(dag.gauss(), data.sample, ['bic-g'],
                             params={'k': 1.0})

    assert values_same(bnscores['bic-g'], -53221.35, sf=7)  # website value
    print()
    scores = dag.gauss().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))
    print('BIC: bnlearn {:.3f}, bnbench {:.3f}'
          .format(bnscores['bic-g'], dict(scores.sum())['bic-g']))

    # bnlearn and bnbech similar scores with so many rows

    assert dicts_same(bnscores, dict(scores.sum()), sf=7)
