
import pytest

from analysis.statistics import rank_values


def test_rank_values_type_error_1():  # no arguments supplied
    with pytest.raises(TypeError):
        rank_values()


def test_rank_values_type_error_2():  # type other than dict
    with pytest.raises(TypeError):
        rank_values(2)
    with pytest.raises(TypeError):
        rank_values([3, 2])
    with pytest.raises(TypeError):
        rank_values(True)


def test_rank_values_type_error_3():  # empty dict
    with pytest.raises(TypeError):
        rank_values({})


def test_rank_values_type_error_4():  # non-str keys
    with pytest.raises(TypeError):
        rank_values({1: 2, 'A': 3})


def test_rank_values_type_error_5():  # non-str/float/int values
    with pytest.raises(TypeError):
        rank_values({'B': 2, 'A': [3]})
    with pytest.raises(TypeError):
        rank_values({'B': 2, 'A': True})


def test_rank_values_1_ok():  # scores just contains one element
    ranks = rank_values({'one': 0.2})
    assert ranks == {'one': 1}
    ranks = rank_values({'one': 2})
    assert ranks == {'one': 1}
    ranks = rank_values({'one': 'failed'})
    assert ranks == {'one': 1}


def test_rank_values_2_ok():  # two different scores
    ranks = rank_values({'A': 0.2, 'B': 0.1})
    assert ranks == {'A': 1, 'B': 2}
    ranks = rank_values({'A': 2, 'B': 3})
    assert ranks == {'A': 2, 'B': 1}
    ranks = rank_values({'A': 2, 'B': 1.7})
    assert ranks == {'A': 1, 'B': 2}
    ranks = rank_values({'A': -1.2, 'B': 0})
    assert ranks == {'A': 2, 'B': 1}


def test_rank_values_3_ok():  # two equal scores
    ranks = rank_values({'A': 0.2, 'B': 0.2})
    assert ranks == {'A': 1, 'B': 1}
    ranks = rank_values({'A': 0, 'B': 0.0})
    assert ranks == {'A': 1, 'B': 1}
    ranks = rank_values({'A': 3, 'B': 3})
    assert ranks == {'A': 1, 'B': 1}


def test_rank_values_4_ok():  # three different scores
    ranks = rank_values({'A': 0.2, 'B': 0.3, 'C': 0.1})
    assert ranks == {'A': 2, 'B': 1, 'C': 3}
    ranks = rank_values({'A': 17, 'B': 14, 'C': -0.1})
    assert ranks == {'A': 1, 'B': 2, 'C': 3}


def test_rank_values_5_ok():  # three scores, two equal
    ranks = rank_values({'A': 0.2, 'B': 0.3, 'C': 0.2})
    assert ranks == {'A': 2, 'B': 1, 'C': 2}
    ranks = rank_values({'A': 16, 'B': 16, 'C': 12})
    assert ranks == {'A': 1, 'B': 1, 'C': 3}


def test_rank_values_6_ok():  # three scores, some failed
    ranks = rank_values({'A': 0.2, 'B': 0.2, 'C': 'failed'})
    assert ranks == {'A': 1, 'B': 1, 'C': 3}
    ranks = rank_values({'A': 'failed', 'B': 16, 'C': 'fail = 2'})
    assert ranks == {'A': 2, 'B': 1, 'C': 2}
    ranks = rank_values({'A': 'failed', 'B': 'bad', 'C': 'fail = 2'})
    assert ranks == {'A': 1, 'B': 1, 'C': 1}


def test_rank_values_7_ok():  # six scores, some ties and/or failures
    ranks = rank_values({'A': 0.2, 'B': 0.2, 'C': 'failed',
                         'D': 0.4,  'E': 0.1, 'F': 0})
    assert ranks == {'A': 2, 'B': 2, 'C': 6, 'D': 1, 'E': 4, 'F': 5}
    ranks = rank_values({'A': 15, 'B': 15, 'C': 'failed',
                         'D': 15,  'E': 0.1, 'F': 'failed'})
    assert ranks == {'A': 1, 'B': 1, 'C': 5, 'D': 1, 'E': 4, 'F': 5}
