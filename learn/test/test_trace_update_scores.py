
# Test the Trace update_scores function

import pytest

from core.metrics import values_same
from fileio.common import TESTDATA_DIR
from learn.trace import Trace


def test_trace_update_scores_type_error_1():  # no args
    with pytest.raises(TypeError):
        Trace.update_scores()


def test_trace_update_scores_type_error_2():  # missing args
    with pytest.raises(TypeError):
        Trace.update_scores(series='HC/STD', networks=['asia'])
    with pytest.raises(TypeError):
        Trace.update_scores(series='HC/ORD', score='bic')
    with pytest.raises(TypeError):
        Trace.update_scores(networks=['sports'], score='bic')


def test_trace_update_scores_type_error_3():  # bad arg type
    with pytest.raises(TypeError):
        Trace.update_scores(series='HC/STD', networks=['asia'])
    with pytest.raises(TypeError):
        Trace.update_scores(series='HC/ORD', score='bic')
    with pytest.raises(TypeError):
        Trace.update_scores(networks=['sports'], score='bic')


def test_trace_update_scores_value_error_1_():  # unknown score
    with pytest.raises(ValueError):
        Trace.update_scores(series='HC/ORDER/BASE', networks=['asia'],
                            score='invalid',
                            root_dir=TESTDATA_DIR + 'experiments')


def test_trace_update_scores_value_error_2_():  # wrong objective score
    with pytest.raises(ValueError):
        Trace.update_scores(series='HC/ORDER/BASE', networks=['asia'],
                            score='bde',
                            root_dir=TESTDATA_DIR + 'experiments')


def test_trace_update_scores_ok_1_():  # update asia, bic scores
    scores = Trace.update_scores(series='HC/ORDER/BASE', networks=['asia'],
                                 score='bic',
                                 root_dir=TESTDATA_DIR + 'experiments')
    assert len(scores) == 260
    assert values_same(scores[('asia', 'N10_0')][0], -45.44360605, sf=10)
    assert values_same(scores[('asia', 'N200_6')][1], -491.1460852, sf=10)
    assert values_same(scores[('asia', 'N1000000_9')][0], -2981777.553, sf=10)
    assert values_same(scores[('asia', 'N1000000_9')][1], -2238028.810, sf=10)


def test_trace_update_scores_ok_2_():  # update asia, loglik scores
    scores = Trace.update_scores(series='HC/ORDER/BASE', networks=['asia'],
                                 score='loglik',
                                 root_dir=TESTDATA_DIR + 'experiments')
    assert len(scores) == 260
    assert scores[('asia', 'N10_0')][0] is None
    assert values_same(scores[('asia', 'N200_6')][1], -451.4087050, sf=10)
    assert scores[('asia', 'N1000000_9')][0] is None
    assert values_same(scores[('asia', 'N1000000_9')][1], -2237842.300, sf=10)


def test_trace_update_scores_ok_3_():  # unknown series
    scores = Trace.update_scores(series='invalid', networks=['asia'],
                                 score='bic',
                                 root_dir=TESTDATA_DIR + 'experiments')
    assert scores == {}


def test_trace_update_scores_ok_4_():  # unknown network
    scores = Trace.update_scores(series='HC/ORDER/BASE', networks=['invalid'],
                                 score='bic',
                                 root_dir=TESTDATA_DIR + 'experiments')
    assert scores == {}


def test_trace_update_scores_ok_5_():  # update covid_c bic scores
    scores = Trace.update_scores(series='TABU/BASE3', networks=['covid_c'],
                                 score='bic',
                                 root_dir=TESTDATA_DIR + 'experiments')
    assert len(scores) == 100
    assert values_same(scores[('covid_c', 'N100_3')][0], -10192.77128, sf=10)
    assert values_same(scores[('covid_c', 'N1000_5')][1], -89947.08588, sf=10)
    assert values_same(scores[('covid_c', 'N100000_24')][0], -10101400.67,
                       sf=10)
    assert values_same(scores[('covid_c', 'N100000_24')][1], -8971737.443,
                       sf=10)


def test_trace_update_scores_ok_6_():  # update covid_c loglik scores
    scores = Trace.update_scores(series='TABU/BASE3', networks=['covid_c'],
                                 score='loglik',
                                 root_dir=TESTDATA_DIR + 'experiments')
    assert len(scores) == 100
    assert scores[('covid_c', 'N100_3')][0] is None
    assert values_same(scores[('covid_c', 'N1000_5')][1], -89722.58383, sf=10)
    assert scores[('covid_c', 'N100000_24')][0] is None
    assert values_same(scores[('covid_c', 'N100000_24')][1], -8971288.439,
                       sf=10)
