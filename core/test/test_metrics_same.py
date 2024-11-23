
# test the low-level value, dict and distribution approximate compare functions

import pytest
from pandas import DataFrame, set_option

from core.metrics import values_same, dicts_same, dists_same


def test_metrics_values_same_bad_values():
    with pytest.raises(TypeError):
        values_same()
    with pytest.raises(TypeError):
        values_same(2)
    with pytest.raises(TypeError):
        values_same('1', '2')
    with pytest.raises(TypeError):
        values_same('1', 2.0)
    with pytest.raises(TypeError):
        values_same([1], 2)
    with pytest.raises(TypeError):
        values_same({3}, 1)
    with pytest.raises(TypeError):
        values_same(2, {'1': 0})


def test_metrics_values_same_bad_sf():
    with pytest.raises(TypeError):
        values_same(1, 1, None)
    with pytest.raises(TypeError):
        values_same(1, 1, '4')


def test_metrics_values_same_small_ints():
    assert values_same(0, 0) is True
    assert values_same(0, -0) is True
    assert values_same(-0, 0) is True
    assert values_same(-0, -0) is True
    assert values_same(0, 0, 1) is True
    assert values_same(0, 0, 10) is True
    assert values_same(1, 1, 1) is True
    assert values_same(1, 1, 10) is True
    assert values_same(0, 1) is False
    assert values_same(0, 1, 3) is False
    assert values_same(0, 1, 1) is False
    assert values_same(-1, 0) is False
    assert values_same(-1, 0, 1) is False
    assert values_same(-1, 0, 7) is False
    assert values_same(-1, 0, 10) is False
    assert values_same(-1, 0) is False
    assert values_same(-1, 0, 1) is False
    assert values_same(-1, 0, 7) is False
    assert values_same(-1, 0, 10) is False
    assert values_same(-1, 1) is False
    assert values_same(-1, 1, 1) is False
    assert values_same(-1, 1, 7) is False
    assert values_same(-1, 1, 10) is False


def test_metrics_values_same_med_ints():
    assert values_same(1234, 1233) is True
    assert values_same(1234, 1233, 1) is True
    assert values_same(1234, 1233, 2) is True
    assert values_same(1234, 1233, 3) is True
    assert values_same(10, 11, 1) is True
    assert values_same(123, 124, 2) is True
    assert values_same(13, 13, 6) is True
    assert values_same(-1234, -1233) is True
    assert values_same(-1234, -1233, 1) is True
    assert values_same(1234, 1236) is False
    assert values_same(1234, 1233, 4) is False


def test_metrics_values_same_large_ints():
    assert values_same(201089745563, 201089745563) is True
    assert values_same(201089745563, 201089745599) is True
    assert values_same(201089745563, 201089897234, 6) is True
    assert values_same(301089745563, 201089745563) is False
    assert values_same(201089745563, 201089745599, 10) is True
    assert values_same(201089745563, 201089897234, 6) is True


def test_metrics_values_same_small_floats():
    assert values_same(0.0, 0.0) is True
    assert values_same(0.0001001, 0.0001, 1) is True
    assert values_same(0.0001001, 0.000100, 3) is True
    assert values_same(0.667, 0.6666666666666) is True
    assert values_same(0.667, 0.6666666666666, 3) is True
    assert values_same(0.667, 0.6666666666666, 1) is True
    assert values_same(0.667, 0.6666666666666, 2) is True
    assert values_same(0.667, 0.6666666666666, 4) is False
    assert values_same(-0.001, 0.001, 2) is False
    assert values_same(0.43478260869565216, 0.435, 2) is True


def test_metrics_values_same_med_floats():
    assert values_same(671.0099, 671.1) is True
    assert values_same(671.14999999, 671.1) is True
    assert values_same(671.0099, 670.500001) is True
    assert values_same(671.0099, 671.1) is True
    assert values_same(-671.0099, -671.1) is True
    assert values_same(-671.14999999, -671.1) is True
    assert values_same(-671.0099, -670.500001) is True
    assert values_same(-671.0099, -671.1) is True


def test_metrics_values_same_large_floats():
    assert values_same(1.0004E+10, 1.0E+10) is True
    assert values_same(1.0004E+10, 1.0E+10, sf=4) is True
    assert values_same(1.0004E+10, 1.0E+10, sf=5) is False


def test_metrics_values_same_small_mixed():
    assert values_same(0.0, 0) is True
    assert values_same(-0.0, 0) is True
    assert values_same(0.0, -0) is True
    assert values_same(-0.0, -0) is True
    assert values_same(-1, -1.0, 1) is True
    assert values_same(-1, -1.1, 1) is True
    assert values_same(15.001, 15) is True
    assert values_same(15.001, 15, sf=4) is True
    assert values_same(15.001, 15, sf=5) is False


def test_metrics_values_same_nans():  # test nan comparisons
    assert values_same(float('nan'), float('nan')) is True
    assert values_same(1.3, float('nan')) is False
    assert values_same(float('nan'), -1.0) is False


def test_metrics_values_same_bools():  # bools treated as True=1, False=0
    assert values_same(2.0, True) is False
    assert values_same(0, True) is False
    assert values_same(1, True) is True
    assert values_same(0, False) is True
    assert values_same(1, False) is False
    assert values_same(0.0, True) is False
    assert values_same(1.0, True) is True
    assert values_same(0.0, False) is True
    assert values_same(1.0, False) is False
    assert values_same(False, True) is False
    assert values_same(True, True) is True
    assert values_same(False, False) is True
    assert values_same(True, False) is False


def test_metrics_values_zero_sf():  # sf <= 0 multiplies vals by 10**(1-sf)
    assert values_same(10, 70, 0) is False
    assert values_same(140, 70, 0) is False
    assert values_same(60, 70, 0) is True


def test_metrics_values_bool_sf():  # sf = True treated as sf = 1
    # assert values_same(1, 2, True) is False
    # assert values_same(1.1, 3, True) is False
    # assert values_same(2.6, 3.1, True) is True
    assert values_same(35, 37, False) is True

# Test dict comparisons

def test_metrics_dicts_same_bad_args():
    with pytest.raises(TypeError):
        dicts_same()
    with pytest.raises(TypeError):
        dicts_same(3)
    with pytest.raises(TypeError):
        dicts_same(3, {})


def test_metrics_dicts_same_diff_keys():
    with pytest.raises(TypeError):
        dicts_same({'A': 1}, {})
    with pytest.raises(TypeError):
        dicts_same({'A': 1}, {'B': 2})


def test_metrics_dicts_same_diff_keys_nonstrict():
    assert dicts_same({'A': 1}, {}, strict=False) is True
    assert dicts_same({'A': 1}, {'B': 2}, strict=False) is True


def test_metrics_dicts_same_true_1():
    assert dicts_same({'A': 1}, {'A': 1.0000000001}) is True
    assert dicts_same({'A': 1}, {'A': 1.0001}, sf=4) is True
    assert dicts_same({'A': 1.1}, {'A': 1.09}, sf=2) is True


def test_metrics_dicts_same_true_2():  # Nones compare as True
    assert dicts_same({'A': None}, {'A': None}) is True


def test_metrics_dicts_same_false_1():
    assert dicts_same({'A': 1}, {'A': 1.0001}, sf=5) is False
    assert dicts_same({'A': 1}, {'A': 1.0001}, sf=8) is False
    assert dicts_same({'A': -0.0001}, {'A': 0.0002}, sf=2) is False


def test_metrics_dicts_same_false_2():  # Numeric with None is False
    assert dicts_same({'A': 1}, {'A': None}, sf=5) is False
    assert dicts_same({'A': None}, {'A': 2.3}, sf=5) is False

#   Tests of distributions comparisons - exceptions


def test_metrics_dists_same_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        dists_same()


def test_metrics_dists_same_type_error_2():  # 1 argument
    with pytest.raises(TypeError):
        dists_same(DataFrame({'A': ['1']}))
    with pytest.raises(TypeError):
        dists_same(None, DataFrame({'A': ['1']}))


def test_metrics_dists_same_type_error_3():  # some bad types
    with pytest.raises(TypeError):
        dists_same(32, DataFrame({'A': ['1']}))
    with pytest.raises(TypeError):
        dists_same(-101.1, DataFrame({'A': ['1']}))
    with pytest.raises(TypeError):
        dists_same(True, DataFrame({'A': ['1']}))
    with pytest.raises(TypeError):
        dists_same({'A': ['1']}, DataFrame({'A': ['1']}))
    with pytest.raises(TypeError):
        dists_same(DataFrame({'A': ['1']}), 32)
    with pytest.raises(TypeError):
        dists_same(DataFrame({'A': ['1']}), 2.1)
    with pytest.raises(TypeError):
        dists_same(DataFrame({'A': ['1']}), False)
    with pytest.raises(TypeError):
        dists_same(DataFrame({'A': ['1']}), {'A': ['1']})


#   Tests of distributions comparisons - univariates which are the same

def test_metrics_dists_same_ok_us1_():  # compare univariate with itself
    dist1 = DataFrame({'': {'0': 0.5, '1': 0.5}})
    dist1.index.name = 'A'
    sf = 10
    same = dists_same(dist1, dist1, sf)
    print('\nUnivariate:\n{}\n{}SAME (to {} s.f.) as:\n{}'
          .format(dist1, '' if same else 'NOT ', sf, dist1))
    assert same


def test_metrics_dists_same_ok_us2_():  # identical univariate copies
    dist1 = DataFrame({'': {'0': 0.00001, '1': 0.99999}})
    dist1.index.name = 'A'
    dist2 = DataFrame.copy(dist1)
    sf = 10
    same = dists_same(dist1, dist2, sf)
    print('\nUnivariate:\n{}\n{}SAME (to {} s.f.) as:\n{}'
          .format(dist1, '' if same else 'NOT ', sf, dist2))
    assert same


def test_metrics_dists_same_ok_us3_():  # identical univariates
    dist1 = DataFrame({'': {'0': 0.3, '1': 0.7}})
    dist1.index.name = 'A'
    dist2 = DataFrame({'': {'0': 0.3, '1': 0.7}})
    dist2.index.name = 'A'
    sf = 10
    same = dists_same(dist1, dist2, sf)
    print('\nUnivariate:\n{}\n{}SAME (to {} s.f.) as:\n{}'
          .format(dist1, '' if same else 'NOT ', sf, dist2))
    assert same


def test_metrics_dists_same_ok_us4_():  # almost identical univariates
    dist1 = DataFrame({'': {'0': 0.3, '1': 0.7}})
    dist1.index.name = 'A'
    dist2 = DataFrame({'': {'0': 0.301, '1': 0.699}})
    dist2.index.name = 'A'
    sf = 2
    same = dists_same(dist1, dist2, sf)
    print('\nUnivariate:\n{}\n{}SAME (to {} s.f.) as:\n{}'
          .format(dist1, '' if same else 'NOT ', sf, dist2))
    assert same


def test_metrics_dists_same_ok_us5_():  # almost identical univariates
    dist1 = DataFrame({'': {'0': 0.3, '1': 0.7}})
    dist1.index.name = 'A'
    dist2 = DataFrame({'': {'0': 0.2999, '1': 0.6999}})
    dist2.index.name = 'A'
    sf = 3
    same = dists_same(dist1, dist2, sf)
    print('\nUnivariate:\n{}\n{}SAME (to {} s.f.) as:\n{}'
          .format(dist1, '' if same else 'NOT ', sf, dist2))
    assert same


def test_metrics_dists_same_ok_us6_():  # almost identical univariates
    dist1 = DataFrame({'': {'0': 0.3, '1': 0.7}})
    dist1.index.name = 'A'
    dist2 = DataFrame({'': {'0': 0.29999999996, '1': 0.7}})
    dist2.index.name = 'A'
    set_option('precision', 11)
    sf = 10
    same = dists_same(dist1, dist2, sf)
    print('\nUnivariate:\n{}\n{}SAME (to {} s.f.) as:\n{}'
          .format(dist1, '' if same else 'NOT ', sf, dist2))
    assert same


def test_metrics_dists_same_ok_ud1_():  # univariate, different primary var
    dist1 = DataFrame({'': {'0': 0.00001, '1': 0.99999}})
    dist1.index.name = 'A'
    dist2 = DataFrame.copy(dist1)
    dist2.index.name = 'B'
    sf = 10
    same = dists_same(dist1, dist2, sf)
    print('\nUnivariate:\n{}\n{}SAME (to {} s.f.) as:\n{}'
          .format(dist1, '' if same else 'NOT ', sf, dist2))
    assert not same


def test_metrics_dists_same_ok_ud2_():  # univariate, different secondary var
    dist1 = DataFrame({'': {'0': 0.00001, '1': 0.99999}})
    dist1.index.name = 'A'
    dist2 = DataFrame.copy(dist1)
    dist2.index.name = 'A'
    dist2.columns.names = ['A']
    sf = 10
    same = dists_same(dist1, dist2, sf)
    print('\nUnivariate:\n{}\n{}SAME (to {} s.f.) as:\n{}'
          .format(dist1, '' if same else 'NOT ', sf, dist2))
    assert not same


def test_metrics_dists_same_ok_ud3_():  # univariates, different primary values
    dist1 = DataFrame({'': {'0': 0.3, '1': 0.7}})
    dist1.index.name = 'A'
    dist2 = DataFrame({'': {'0': 0.3, '3': 0.7}})
    dist2.index.name = 'A'
    sf = 10
    same = dists_same(dist1, dist2, sf)
    print('\nUnivariate:\n{}\n{}SAME (to {} s.f.) as:\n{}'
          .format(dist1, '' if same else 'NOT ', sf, dist2))
    assert not same


def test_metrics_dists_same_ok_ud4_():  # univariates, different primary values
    dist1 = DataFrame({'': {'0': 0.3, '1': 0.7}})
    dist1.index.name = 'A'
    dist2 = DataFrame({'': {'0': 0.3, '1': 0.7, '2': 0.0}})
    dist2.index.name = 'A'
    sf = 10
    same = dists_same(dist1, dist2, sf)
    print('\nUnivariate:\n{}\n{}SAME (to {} s.f.) as:\n{}'
          .format(dist1, '' if same else 'NOT ', sf, dist2))
    assert not same


def test_metrics_dists_same_ok_11_():  # compare bivariate with itself
    dist1 = DataFrame({('0',): {'0': 0.2, '1': 0.3},
                       ('1',): {'0': 0.4, '1': 0.1}})
    dist1.index.name = 'B'
    dist1.columns.names = ['A']
    sf = 10
    same = dists_same(dist1, dist1, sf)
    print('\ndists_same:\n{}\n{}SAME (to {} s.f.) as:\n{}'
          .format(dist1, '' if same else 'NOT ', sf, dist1))
    assert same
