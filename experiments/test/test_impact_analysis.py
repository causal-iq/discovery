
# Module testing of impact analysis

import pytest

from experiments.impact_analysis import series_comparable


def test_series_comparable_type_error_1():  # no arguments given
    with pytest.raises(TypeError):
        series_comparable()


def test_series_comparable_type_error_2():  # one argument only
    with pytest.raises(TypeError):
        series_comparable({0: (1, 2)})


def test_series_comparable_type_error_3():  # two argument only
    with pytest.raises(TypeError):
        series_comparable({0: (1, 2)}, (3, ))


def test_series_comparable_type_error_4():  # required not dict type
    with pytest.raises(TypeError):
        series_comparable({0: (1, 2)}, 1, (2, ))


def test_series_comparable_type_error_5():  # keys of required not all int
    with pytest.raises(TypeError):
        series_comparable({'a': (1, 2)}, (4, 6), (4, 5))


def test_series_comparable_type_error_6():  # values of required not all tuples
    with pytest.raises(TypeError):
        series_comparable({1: [1, 2]}, (4, 6), (4, 5))


def test_series_comparable_type_error_7():  # props1 not tuple type
    with pytest.raises(TypeError):
        series_comparable({0: (1, 2)}, 1, (2, ))


def test_series_comparable_type_error_8():  # props2 not tuple type
    with pytest.raises(TypeError):
        series_comparable({0: (1, 2)}, (4, ), [1])


def test_series_comparable_value_error_1():  # required zero length
    with pytest.raises(ValueError):
        series_comparable({}, (1, 4), (1, 5))


def test_series_comparable_value_error_2():  # props1, props2 zero length
    with pytest.raises(ValueError):
        series_comparable({0: (1, 2)}, tuple(), tuple())


def test_series_comparable_value_error_3():  # props1, props2 diff lengths
    with pytest.raises(ValueError):
        series_comparable({0: (1, 2)}, (4, ), (3, 5))


def test_series_comparable_value_error_4():  # required tuple not length 2
    with pytest.raises(ValueError):
        series_comparable({1: (3,)}, (1, 4), (1, 5))
    with pytest.raises(ValueError):
        series_comparable({1: (3, 4, 5)}, (1, 4), (1, 5))


def test_series_comparable_value_error_5():  # required points outside of props
    with pytest.raises(ValueError):
        series_comparable({-1: (3, 4)}, (1, 4), (1, 5))
    with pytest.raises(ValueError):
        series_comparable({2: (3, 4)}, (1, 4), (1, 5))


def test_series_comparable_true_1():  # 1 same property, 1 diff
    assert series_comparable({0: (1, 2)}, (1, 3), (2, 3)) is True
    assert series_comparable({1: (1, 2)}, ('a', 1), ('a', 2)) is True
    assert series_comparable({2: ('a', 'b')}, ('c', 1, 'a'),
                             ('c', 1, 'b')) is True


def test_series_comparable_true_2():  # 2 same property, 1 diff
    assert series_comparable({1: (1, 4)}, ('c', 1, 'b'),
                             ('c', 4, 'b')) is True
    assert series_comparable({2: ('a', 'b')}, ('c', 1, 'a'),
                             ('c', 1, 'b')) is True


def test_series_comparable_true_3():  # 1 same, 2 different
    assert series_comparable({1: (1, 4), 2: (0.01, 0.05)},
                             ('c', 1, 0.01), ('c', 4, 0.05)) is True
    assert series_comparable({0: (1, 4), 1: (0.01, 0.05)},
                             (1, 0.01, 'c'), (4, 0.05, 'c')) is True


def test_series_comparable_false_1():  # non-required not same
    assert series_comparable({0: (1, 2)}, (3, 3), (2, 3)) is False
    assert series_comparable({1: (1, 4), 2: (0.01, 0.05)},
                             ('d', 1, 0.01), ('c', 4, 0.05)) is False
    assert series_comparable({0: (1, 4), 1: (0.01, 0.05)},
                             (1, 0.01, 6), (4, 0.05, 'c')) is False


def test_series_comparable_false_2():  # required not correct
    assert series_comparable({0: (1, 2)}, (1, 3), (6, 3)) is False
    assert series_comparable({1: (1, 2)}, ('a', 1), ('a', 'b')) is False
    assert series_comparable({2: ('a', 'b')}, ('c', 1, 'a'),
                             ('c', 1, 0.1)) is False
