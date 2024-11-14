
# Test DAG scoring in R code

import pytest
from numpy import array
from random import random
from os import remove

from call.bnlearn import bnlearn_score
import testdata.example_dags as dag
from fileio.common import TESTDATA_DIR
from fileio.numpy import NumPy
from core.metrics import dicts_same, values_same
from core.bn import BN

TYPES = ['loglik', 'bic', 'aic', 'bde', 'k2', 'bdj', 'bds']  # scores to test


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.csv'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


@pytest.fixture(scope="module")  # AB, 10 categorical rows
def ab10():
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    return NumPy.from_df(df=bn.generate_cases(10), dstype='categorical',
                         keep_df=False)


def test_bnlearn_score_type_error_1_():  # no arguments
    with pytest.raises(TypeError):
        bnlearn_score()


def test_bnlearn_score_type_error_2_(ab10):  # insufficient args
    with pytest.raises(TypeError):
        bnlearn_score(dag=dag.ab(), data=ab10, types=['bic'])
    with pytest.raises(TypeError):
        bnlearn_score(dag=dag.ab(), types=['bic'], params={'base': 'e'})
    with pytest.raises(TypeError):
        bnlearn_score(dag=dag.ab(), data=ab10, params={'base': 'e'})
    with pytest.raises(TypeError):
        bnlearn_score(data=ab10, types=['bic'], params={'base': 'e'})


def test_bnlearn_score_type_error_3_(ab10):  # bad dag type
    with pytest.raises(TypeError):
        bnlearn_score(False, ab10, ['bic'], params={'base': 'e'})
    with pytest.raises(TypeError):
        bnlearn_score(32, ab10, ['bic'], params={'base': 'e'})
    with pytest.raises(TypeError):
        bnlearn_score({dag.ab()}, ab10, ['bic'], params={'base': 'e'})


def test_bnlearn_score_type_error_4_(ab10):  # bad data type
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), ab10.as_df(), ['bic'], params={'base': 'e'})
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), True, ['bic'], params={'base': 'e'})
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), 32.1, ['bic'], params={'base': 'e'})
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), [ab10], ['bic'], params={'base': 'e'})


def test_bnlearn_score_type_error_5_(ab10):  # bad type for types
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), ab10, 6, params={'base': 'e'})
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), ab10, ('bic',), params={'base': 'e'})
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), ab10, {'bic'}, params={'base': 'e'})


def test_bnlearn_score_type_error_6_(ab10):  # bad type for params
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), ab10, ['bic'], params=['base', 'e'])
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), ab10, ['bic'], params=True)
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), ab10, ['bic'], params=30.1)


def test_bnlearn_score_type_error_7_(ab10):  # bad ISS type
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), ab10, types=['bde'],
                      params={'iss': 'wrong type'})


def test_bnlearn_score_type_error_8_(ab10):  # bad k type
    with pytest.raises(TypeError):
        bnlearn_score(dag.ab(), ab10, ['bic'], {'k': 'bad type'})


def test_bnlearn_score_value_error_1(ab10):  # variable set mismatch
    with pytest.raises(ValueError):
        bnlearn_score(dag.abc(), ab10, types=['bic'], params={'k': 1})


def test_bnlearn_score_value_error_2():  # some variables single-valued
    data = NumPy(array([[0, 0], [1, 0]], dtype='uint8'), dstype='categorical',
                 col_values={'A': ('N', 'Y'), 'B': ('0',)})
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), data, ['bic'], {'k': 1})


def test_bnlearn_score_value_error_3(ab10):  # invalid score type
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), ab10, ['invalid'], {'k': 1})


def test_bnlearn_score_value_error_4(ab10):  # bad k value
    with pytest.raises(ValueError):
        bnlearn_score(dag.ab(), ab10, ['bic'], {'k': -1})


def test_bnlearn_score_a_b_1_ok():  # A, B unconnected
    data = NumPy(array([[0, 0], [1, 1]], dtype='uint8'), dstype='categorical',
                 col_values={'A': ('1', '0'), 'B': ('1', '0')})
    bnscores = bnlearn_score(dag.a_b(), data, TYPES, params={'iss': 1.0})
    scores = dag.a_b().score(data, TYPES, {'base': 'e'})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_2_ok():  # A, B unconnected, ISS = 2
    data = NumPy(array([[0, 0], [1, 1]], dtype='uint8'), dstype='categorical',
                 col_values={'A': ('1', '0'), 'B': ('1', '0')})
    bnscores = bnlearn_score(dag.a_b(), data, TYPES, {'iss': 2.0})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'iss': 2.0})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_3_ok():  # A, B unconnected, ISS = 10
    data = NumPy(array([[0, 0], [1, 1]], dtype='uint8'), dstype='categorical',
                 col_values={'A': ('1', '0'), 'B': ('1', '0')})
    bnscores = bnlearn_score(dag.a_b(), data, TYPES, {'iss': 10.0})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'iss': 10.0})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_4_ok():  # A, B unconnected, ISS = 100
    data = NumPy(array([[0, 0], [1, 1]], dtype='uint8'), dstype='categorical',
                 col_values={'A': ('1', '0'), 'B': ('1', '0')})
    bnscores = bnlearn_score(dag.a_b(), data, TYPES, {'iss': 100.0})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'iss': 100.0})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_5_ok():  # A, B unconnected, k=2
    data = NumPy(array([[0, 0], [1, 1]], dtype='uint8'), dstype='categorical',
                 col_values={'A': ('1', '0'), 'B': ('1', '0')})
    print('\n\n{}'.format(data.as_df()))
    bnscores = bnlearn_score(dag.a_b(), data, TYPES, {'k': 2})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'k': 2})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_6_ok():  # A, B unconnected, k=0.5
    data = NumPy(array([[0, 0], [1, 1]], dtype='uint8'), dstype='categorical',
                 col_values={'A': ('1', '0'), 'B': ('1', '0')})
    print('\n\n{}'.format(data.as_df()))
    bnscores = bnlearn_score(dag.a_b(), data, TYPES, {'k': 0.5})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'k': 0.5})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_7_ok():  # A, B unconnected, k=10.0
    data = NumPy(array([[0, 0], [1, 1]], dtype='uint8'), dstype='categorical',
                 col_values={'A': ('1', '0'), 'B': ('1', '0')})
    print('\n\n{}'.format(data.as_df()))
    bnscores = bnlearn_score(dag.a_b(), data, TYPES, {'k': 10.0})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'k': 10.0})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_a_b_8_ok():  # A, B unconnected, k=0.1
    data = NumPy(array([[0, 0], [1, 1]], dtype='uint8'), dstype='categorical',
                 col_values={'A': ('1', '0'), 'B': ('1', '0')})
    print('\n\n{}'.format(data.as_df()))
    bnscores = bnlearn_score(dag.a_b(), data, TYPES, {'k': 0.1})
    scores = dag.a_b().score(data, TYPES, {'base': 'e', 'k': 0.1})
    print(scores)
    assert dicts_same(bnscores, dict(scores.sum()))


def test_bnlearn_score_x_y_1_ok():  # scoring X, Y data
    data = NumPy(array([[1.1, 0.0], [2.2, 1.7], [-0.3, 0.0]], dtype='float32'),
                 dstype='continuous', col_values={'X': None, 'Y': None})
    print('\n\n{}'.format(data.as_df()))
    bnscores = bnlearn_score(dag.x_y(), data, ['bic-g'], {'k': 1.0})
    scores = dag.x_y().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))

    # BIC scores very close because no linear regression involved

    assert dicts_same(bnscores, dict(scores.sum()), sf=6)


def test_bnlearn_score_x_y_2_ok():  # scoring X, Y data
    data = NumPy(array([[0.6, -0.4], [5.1, 1.7], [-3.2, 0.0],
                        [-2.0, -0.8], [-0.7, 0.6]], dtype='float32'),
                 dstype='continuous', col_values={'X': None, 'Y': None})
    print('\n\n{}'.format(data.as_df()))
    bnscores = bnlearn_score(dag.x_y(), data, ['bic-g'], {'k': 1.0})
    scores = dag.x_y().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))

    # BIC scores very close because no linear regression involved

    assert dicts_same(bnscores, dict(scores.sum()), sf=6)


def test_bnlearn_score_xy_1_ok():  # scoring X, Y data
    data = NumPy(array([[0.6, -0.4], [5.1, 1.7], [-3.2, 0.0],
                        [-2.0, -0.8], [-0.7, 0.6]], dtype='float32'),
                 dstype='continuous', col_values={'X': None, 'Y': None})
    print('\n\n{}'.format(data.as_df()))
    bnscores = bnlearn_score(dag.xy(), data, ['bic-g'], {'k': 1.0})
    scores = dag.xy().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))

    # BIC scores not very similar because of linear regression differences

    assert dicts_same(bnscores, dict(scores.sum()), sf=2)


def test_bnlearn_score_xyz_1_ok():  # scoring X --> Y --> Z DAG
    data = NumPy(array([[0.2, 0.5, 2.4],
                        [0.4, 0.7, 2.6],
                        [-0.1, 0.0, 1.9],
                        [0.6, 1.4, 3.3],
                        [0.9, 1.9, 3.6],
                        [1.1, 2.4, 4.2]], dtype='float32'),
                 dstype='continuous',
                 col_values={'X': None, 'Y': None, 'Z': None})
    bnscores = bnlearn_score(dag.xyz(), data, ['bic-g'], {'k': 1.0})
    scores = dag.xyz().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))

    # BIC scores not very similar because of linear regression differences

    assert values_same(bnscores['bic-g'], 3.334, sf=4)
    assert values_same(dict(scores.sum())['bic-g'], 3.673, sf=4)


def test_bnlearn_score_xy_zy_1_ok():  # scoring X --> Y <-- Z DAG
    data = NumPy(array([[0.2, 0.5, 2.4],
                        [0.4, 0.7, 2.6],
                        [-0.1, 0.0, 1.9],
                        [0.6, 1.4, 3.3],
                        [0.9, 1.9, 3.6],
                        [1.1, 2.4, 4.2]], dtype='float32'),
                 dstype='continuous',
                 col_values={'X': None, 'Y': None, 'Z': None})
    bnscores = bnlearn_score(dag.xy_zy(), data, ['bic-g'], {'k': 1.0})
    scores = dag.xy_zy().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))

    # BIC scores not very similar because of linear regression differences

    assert values_same(bnscores['bic-g'], -8.1976, sf=5)
    assert values_same(dict(scores.sum())['bic-g'], -7.6652, sf=5)


def test_bnlearn_score_gauss_1_ok():  # scoring BNLearn example Gaussian
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous', N=5000)
    bnscores = bnlearn_score(dag.gauss(), data, ['bic-g'], {'k': 1.0})

    assert values_same(bnscores['bic-g'], -53221.35, sf=7)  # website value
    print()
    scores = dag.gauss().score(data, 'bic-g')
    print('\n\nBnbench node scores are:\n{}'.format(scores['bic-g']))
    print('BIC: bnlearn {:.3f}, bnbench {:.3f}'
          .format(bnscores['bic-g'], dict(scores.sum())['bic-g']))

    # bnlearn and bnbench similar scores with so many rows

    assert dicts_same(bnscores, dict(scores.sum()), sf=7)
