
import pytest
from numpy import array

from core.metrics import values_same
from core.score import bayesian_score


def check(counts, q_i, type, expected, iss=1):
    score = bayesian_score(counts=array(counts), q_i=q_i, type=type,
                           params={'iss': iss})
    print('\n{} score (q_i={}, iss={}) for {} is {:.6f}'
          .format(type, q_i, iss, counts, score))
    assert values_same(score, expected, sf=10)


def test_bayesian_score_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        bayesian_score()


def test_bayesian_score_type_error_2():  # counts not a DataFrame
    with pytest.raises(TypeError):
        bayesian_score({'1': [1]}, 1, 'bde', {'iss': 1})
    with pytest.raises(TypeError):
        bayesian_score(None, 1, 'bde', {'iss': 1})
    with pytest.raises(TypeError):
        bayesian_score(2, 1, 'bde', {'iss': 1})
    with pytest.raises(TypeError):
        bayesian_score('invalid', 1, 'bde', {'iss': 1})


def test_bayesian_score_type_error_3():  # q_i not an int
    with pytest.raises(TypeError):
        bayesian_score(array([[1]]), False, 'bde', {'iss': 1})
    with pytest.raises(TypeError):
        bayesian_score(array([[1]]), None, 'bde', {'iss': 1})
    with pytest.raises(TypeError):
        bayesian_score(array([[1]]), [1], 'bde', {'iss': 1})


def test_bayesian_score_type_error_4():  # type not a string
    with pytest.raises(TypeError):
        bayesian_score(array([[1]]), 1, 1, {'iss': 1})
    with pytest.raises(TypeError):
        bayesian_score(array([[1]]), 1, None, {'iss': 1})
    with pytest.raises(TypeError):
        bayesian_score(array([[1]]), 1, ['bde'], {'iss': 1})


def test_bayesian_score_type_error_5():  # params not a dict
    with pytest.raises(TypeError):
        bayesian_score(array([[1]]), 1, 'bde', 1)
    with pytest.raises(TypeError):
        bayesian_score(array([[1]]), 1, 'bde', None)


def test_bayesian_score_value_error_1():  # invalid type value
    with pytest.raises(ValueError):
        bayesian_score(array([[1]]), 1, 'invalid', {'iss': 1})


def test_bayesian_score_value_error_2():  # 'iss' param not included
    with pytest.raises(ValueError):
        bayesian_score(array([[1]]), 1, 'bde', {'k': 1})


def test_bayesian_score_bde_1_ok():  # no parents, single-valued
    check([[1]], 1, 'bde', 0.0)
    check([[100]], 1, 'bde', 0.0)
    check([[1]], 1, 'bde', 0.0, iss=5)
    check([[100]], 1, 'bde', 0.0, iss=5)


def test_bayesian_score_k2_1_ok():  # no parents, single-valued
    check([[1]], 1, 'k2', 0.0)
    check([[100]], 1, 'k2', 0.0)


def test_bayesian_score_bds_1_ok():  # no parents, single-valued
    check([[1]], 1, 'bds', 0.0)
    check([[100]], 1, 'bds', 0.0)
    check([[1]], 1, 'bds', 0.0, iss=5)
    check([[100]], 1, 'bds', 0.0, iss=5)


def test_bayesian_score_bdj_1_ok():  # no parents, single-valued
    check([[1]], 1, 'bdj', -0.6931471806)
    check([[100]], 1, 'bdj', -2.876200031)


def test_bayesian_score_bde_2_ok():  # no parents, binary
    check([[1], [1]], 1, 'bde', -2.079441542)
    check([[1], [2]], 1, 'bde', -2.772588722)
    check([[1], [1]], 1, 'bde', -1.568615918, iss=5)
    check([[1], [2]], 1, 'bde', -2.261763098, iss=5)


def test_bayesian_score_k2_2_ok():  # no parents, binary
    check([[1], [1]], 1, 'k2', -1.791759469)
    check([[1], [2]], 1, 'k2', -2.484906650)


# Scores with one parent

def test_bayesian_score_1_parents_1_ok():  # single-parent, binary vars
    ct = array([[0, 1], [2, 3]])

    assert values_same(bayesian_score(ct, 2, 'bde', {'iss': 1}), -4.495355320,
                       10)
    assert values_same(bayesian_score(ct, 2, 'k2', {'iss': 1}), -4.094344562,
                       10)
    assert values_same(bayesian_score(ct, 2, 'bds', {'iss': 1}), -4.495355320,
                       10)
    assert values_same(bayesian_score(ct, 2, 'bdj', {'iss': 1}), -4.223421604,
                       10)


def test_bayesian_score_1_parents_2_ok():  # single-parent, triple value cars
    ct = array([[0, 1, 2], [3, 0, 4], [5, 6, 0]])

    assert values_same(bayesian_score(ct, 18, 'bde', {'iss': 1}), -24.94454640,
                       10)
    assert values_same(bayesian_score(ct, 18, 'k2', {'iss': 1}), -19.40169798,
                       10)
    assert values_same(bayesian_score(ct, 18, 'bds', {'iss': 1}), -20.62998939,
                       10)
    assert values_same(bayesian_score(ct, 18, 'bdj', {'iss': 1}), -19.13051445,
                       10)


# Scores with two parents

def test_bayesian_score_2_parents_1_ok():
    ct = array([[1, 0, 0, 0], [0, 0, 0, 1]])

    assert values_same(bayesian_score(ct, 4, 'bde', {'iss': 1}), -1.386294361,
                       10)
    assert values_same(bayesian_score(ct, 4, 'k2', {'iss': 1}), -1.386294361,
                       10)
    assert values_same(bayesian_score(ct, 4, 'bds', {'iss': 1}), -1.386294361,
                       10)
    assert values_same(bayesian_score(ct, 4, 'bdj', {'iss': 1}), -1.386294361,
                       10)


def test_bayesian_score_2_parents_2_ok():
    ct = array([[1, 0, 2, 1], [1, 0, 2, 1]])

    assert values_same(bayesian_score(ct, 4, 'bde', {'iss': 1}), -10.74121596,
                       10)
    assert values_same(bayesian_score(ct, 4, 'k2', {'iss': 1}), -6.984716320,
                       10)
    assert values_same(bayesian_score(ct, 4, 'bds', {'iss': 1}), -10.74121596,
                       10)
    assert values_same(bayesian_score(ct, 4, 'bdj', {'iss': 1}), -7.912301059,
                       10)


def test_bayesian_score_2_parents_3_ok():
    ct = array([[0, 1, 2, 3], [4, 4, 5, 5], [2, 0, 1, 0]])

    assert values_same(bayesian_score(ct, 8, 'bde', {'iss': 5}), -29.52041031,
                       10)
    assert values_same(bayesian_score(ct, 8, 'k2', {'iss': 5}), -27.45685571,
                       10)
    assert values_same(bayesian_score(ct, 8, 'bds', {'iss': 5}), -28.09537070,
                       10)
    assert values_same(bayesian_score(ct, 8, 'bdj', {'iss': 5}), -27.85813063,
                       10)
