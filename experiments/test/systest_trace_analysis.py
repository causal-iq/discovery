
# system testing of trace analysis

import pytest
from random import random
from os import remove

from core.common import Randomise
from fileio.common import TESTDATA_DIR
from experiments.trace_analysis import trace_analysis


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.dsc'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


def systest_trace_analysis_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        trace_analysis()


def systest_trace_analysis_type_error_2():  # insufficient arguments
    series = ['HC/KW_L1', 'HC/BAD']
    networks = ['asia']
    with pytest.raises(TypeError):
        trace_analysis(series)
    with pytest.raises(TypeError):
        trace_analysis(series, networks)


def systest_trace_analysis_type_error_3():  # bad series type
    networks = ['asia']
    with pytest.raises(TypeError):
        trace_analysis('HC/KW_L1', networks, None)
    with pytest.raises(TypeError):
        trace_analysis(32, networks, None)
    with pytest.raises(TypeError):
        trace_analysis([1, 'HC/KW_L1'], networks, None)


def systest_trace_analysis_type_error_4():  # bad network type
    series = ['HC/KW_L1', 'HC/BAD']
    with pytest.raises(TypeError):
        trace_analysis(series, 'asia', None)
    with pytest.raises(TypeError):
        trace_analysis(series, 32, None)
    with pytest.raises(TypeError):
        trace_analysis(series, [1, 'asia'], None)


def systest_trace_analysis_type_error_5():  # bad N_range type
    series = ['HC/KW_L1', 'HC/BAD']
    networks = ['asia']
    with pytest.raises(TypeError):
        trace_analysis(series, networks, 6)
    with pytest.raises(TypeError):
        trace_analysis(series, networks, (10,))
    with pytest.raises(TypeError):
        trace_analysis(series, networks, 'invalid')


def systest_trace_analysis_type_error_6():  # bad param type
    series = ['HC/UNKNOWN']
    networks = ['asia']
    N_range = None
    with pytest.raises(TypeError):
        trace_analysis(series, networks, N_range, params=6)
    with pytest.raises(TypeError):
        trace_analysis(series, networks, N_range, params='bad type')
    with pytest.raises(TypeError):
        trace_analysis(series, networks, N_range, params=[{'a': 1}])


def systest_trace_analysis_value_error_1():  # bad params keys
    series = ['HC/UNKNOWN']
    networks = ['asia']
    Ns = [10]
    with pytest.raises(ValueError):
        trace_analysis(series, networks, Ns, params={'bad': 'value'})


def systest_trace_analysis_value_error_2():  # bad params chart values
    series = ['HC/UNKNOWN']
    networks = ['asia']
    Ns = [10, 20]
    with pytest.raises(ValueError):
        trace_analysis(series, networks, Ns, params={'chart': 'unknown'})


def systest_trace_analysis_value_error_3():  # bad params metrics values
    series = ['HC/UNKNOWN']
    networks = ['asia']
    Ns = [10, 20, 40]
    with pytest.raises(ValueError):
        trace_analysis(series, networks, Ns, params={'metrics': 'unknown'})
    with pytest.raises(ValueError):
        trace_analysis(series, networks, Ns, params={'metrics': 'unknown,omi'})
    with pytest.raises(ValueError):
        trace_analysis(series, networks, Ns,
                       params={'metrics': 'omi,unknown,mi'})


def systest_trace_analysis_know_asia_1():  # Unknown series
    series = ['HC/UNKNOWN']
    networks = ['asia']
    Ns = [100]
    root_dir = TESTDATA_DIR + '/experiments'
    summaries, trace, diffs = trace_analysis(series, networks, Ns, None, None,
                                             None, root_dir)
    assert summaries == []
    assert trace is None
    assert diffs is None


def systest_trace_analysis_know_asia_2():  # No traces in range
    series = ['HC/ORDER']
    networks = ['asia']
    Ns = [1000]
    root_dir = TESTDATA_DIR + '/experiments'
    summaries, trace, diffs = trace_analysis(series, networks, Ns, None, None,
                                             None, root_dir)
    assert summaries == []
    assert trace is None
    assert diffs is None


def systest_trace_analysis_know_asia_3():  # Single trace, no randomisation
    series = ['HC/STD']
    networks = ['asia']
    Ns = [10]
    root_dir = TESTDATA_DIR + '/experiments'
    summaries, trace, diffs = trace_analysis(series, networks, Ns, None, None,
                                             None, root_dir)
    assert trace.context == \
        {'in': 'experiments/bn/asia.dsc', 'id': 'HC/STD/asia/N10',
         'randomise': [], 'var_order': ['asia', 'bronc', 'dysp', 'either',
                                        'lung', 'smoke', 'tub', 'xray'],
         'score': -4.26044, 'algorithm': 'HC', 'N': 10,
         'params': {'score': 'bic', 'base': 'e', 'k': 1},
         'dataset': True, 'os': 'Windows v10.0.22621',
         'cpu': 'Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz',
         'python': '3.8.2.final.0 (64 bit)', 'ram': 16, 'software_version': 94}

    assert summaries == \
        [{'series': 'HC/STD', 'network': 'asia', 'sample': None, 'N': 10,
          'iter': 5, 'time': 2.9, 'score': -3.88071, 'type': 'DAG', 'n': 8,
          '|A|': 8, '|E|': 5, 'shd': 7, 'shd-s': 0.88, 'shd-e': 9,
          'shd-b': 0.75, 'a-ok': 2, 'a-rev': 2, 'a-eqv': 0, 'a-non': 2,
          'a-ext': 1, 'a-mis': 4, 'p': 0.4, 'r': 0.25, 'f1': 0.308,
          'f1-b': 0.462, 'bsf': 0.325, 'f1-e': 0.0, 'e-ori': 4,
          'loglik': None}]
    assert trace.trace == \
        {'time': [1.8073070049285889, 2.0106122493743896, 2.244967460632324,
                  2.4325642585754395, 2.674945116043091, 2.9089272022247314,
                  2.9245593547821045],
         'activity': ['init', 'add', 'add', 'add', 'add', 'add', 'stop'],
         'arc': [None, ('bronc', 'dysp'), ('either', 'xray'),
                 ('either', 'lung'), ('either', 'tub'), ('tub', 'smoke'),
                 None],
         'delta/score': [-4.544361, 0.307681, 0.194319, 0.071324, 0.071324,
                         0.018999, -3.880712],
         'activity_2': [None, 'add', 'add', 'add', 'add', 'add', 'add'],
         'arc_2': [None, ('dysp', 'bronc'), ('xray', 'either'),
                   ('either', 'tub'), ('tub', 'either'), ('xray', 'smoke'),
                   ('asia', 'bronc')],
         'delta_2': [None, 0.307681, 0.194319, 0.071324, 0.071324, 0.017699,
                     0.0],
         'min_N': [None, 2.0, 1.5, 0.5, 0.5, 1.5, None],
         'mean_N': [None, 3.8, 3.8, 3.8, 3.8, 3.8, None],
         'max_N': [None, 5.5, 7.0, 8.5, 8.5, 7.0, None],
         'lt5': [None, 0.6, 0.6, 0.6, 0.6, 0.6, None],
         'free_params': [None, 1.5, 1.5, 1.5, 1.5, 1.5, None],
         'knowledge': [None, None, None, None, None, None, None],
         'blocked': [None, None, None, None, None, None, None]}
    assert diffs is None


def systest_trace_analysis_know_asia_4():  # Two traces, randomisation
    series = ['HC/ORDER/BASE']
    networks = ['asia']
    Ns = [20]
    root_dir = TESTDATA_DIR + '/experiments'
    summaries, trace, diffs = trace_analysis(series, networks, Ns, None, None,
                                             None, root_dir)
    assert len(summaries) == 10
    assert summaries[0] == \
        {'series': 'HC/ORDER/BASE', 'network': 'asia', 'sample': 0, 'N': 20,
         'iter': 4, 'time': 0.9, 'score': -3.47691, 'type': 'DAG', 'n': 8,
         '|A|': 8, 'shd': 8, 'shd-s': 1.0, 'shd-e': 8, 'shd-b': 0.75,
         'a-ok': 0, 'a-rev': 4, 'a-eqv': 0, 'a-non': 4, 'a-ext': 0,
         'a-mis': 4, 'p': 0.0, 'r': 0.0, 'f1': 0.0, 'f1-e': 0.0,
         'f1-b': 0.333, 'bsf': 0.25, '|E|': 4, 'e-ori': 4, 'loglik': None}
    assert summaries[7] == \
        {'series': 'HC/ORDER/BASE', 'network': 'asia', 'sample': 7, 'N': 20,
         'iter': 4, 'time': 1.2, 'score': -3.47691, 'type': 'DAG', 'n': 8,
         '|A|': 8, 'shd': 6, 'shd-s': 0.75, 'shd-e': 8, 'shd-b': 0.62,
         'a-ok': 2, 'a-rev': 2, 'a-eqv': 0, 'a-non': 2, 'a-ext': 0,
         'a-mis': 4, 'p': 0.5, 'r': 0.25, 'f1': 0.333, 'f1-e': 0.0,
         'f1-b': 0.5, 'bsf': 0.375, '|E|': 4, 'e-ori': 4, 'loglik': None}
    assert trace.context == \
        {'in': 'experiments/bn/asia.dsc', 'id': 'HC/ORDER/BASE/asia/N20_9',
         'randomise': [Randomise.ORDER],
         'var_order': ['asia', 'dysp', 'xray', 'bronc', 'tub', 'lung', 'smoke',
                       'either'],
         'score': -3.66433, 'algorithm': 'HC', 'N': 20,
         'params': {'score': 'bic', 'base': 'e', 'k': 1}, 'dataset': True,
         'os': 'Windows v10.0.22000',
         'cpu': 'AMD Ryzen 9 5900X 12-Core Processor',
         'python': '3.8.2.final.0 (64 bit)', 'ram': 64, 'software_version': 61}
    assert trace.trace == \
        {'time': [0.03125143051147461, 0.7345306873321533, 0.8126587867736816,
                  0.9535980224609375, 1.1099379062652588, 1.1099379062652588],
         'activity': ['init', 'add', 'add', 'add', 'add', 'stop'],
         'arc': [None, ('xray', 'either'), ('either', 'lung'),
                 ('dysp', 'bronc'), ('either', 'tub'), None],
         'delta/score': [-4.025261, 0.235349, 0.154713, 0.130145, 0.028145,
                         -3.47691],
         'activity_2': [None, 'add', 'add', 'add', 'add', 'add'],
         'arc_2': [None, ('either', 'xray'), ('dysp', 'bronc'),
                   ('bronc', 'dysp'), ('xray', 'tub'), ('asia', 'dysp')],
         'delta_2': [None, 0.235349, 0.130145, 0.130145, 0.011155, 0.0],
         'min_N': [None, 1.5, 1.0, 4.5, 0.5, None],
         'mean_N': [None, 7.5, 7.5, 7.5, 7.5, None],
         'max_N': [None, 16.5, 17.5, 10.5, 18.0, None],
         'lt5': [None, 2.0, 2.0, 1.0, 2.0, None],
         'free_params': [None, 1.5, 1.5, 1.5, 1.5, None],
         'knowledge': [None, None, None, None, None, None],
         'blocked': [None, None, None, None, None, None]}
    assert diffs[0] == {}
    assert diffs[1] == [5]


def systest_trace_analysis_know_asia_5():  # First sub-sample only
    series = ['TABU/BASE']
    networks = ['asia']
    Ns = [10]
    Ss = (0, 0)
    root_dir = TESTDATA_DIR + '/experiments'
    summaries, trace, diffs = trace_analysis(series, networks, Ns, Ss, None,
                                             None, root_dir)
    assert len(summaries) == 1
    assert summaries[0] == \
        {'series': 'TABU/BASE', 'network': 'asia', 'sample': 0, 'N': 10,
         'iter': 15, 'time': 2.0, 'score': -3.88071, 'type': 'DAG', 'n': 8,
         '|A|': 8, '|E|': 5, 'shd': 9, 'shd-s': 1.12, 'shd-e': 9,
         'shd-b': 0.88, 'a-ok': 0, 'a-rev': 4, 'a-eqv': 0, 'a-non': 4,
         'a-ext': 1, 'a-mis': 4, 'p': 0.0, 'r': 0.0, 'f1': 0.0, 'f1-e': 0.0,
         'f1-b': 0.308, 'bsf': 0.2, 'e-ori': 4, 'loglik': None}
    assert trace.trace['delta/score'] == \
        [-4.544361, 0.307681, 0.194319, 0.071324, 0.071324, 0.018999, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.880712]
    assert diffs is None


def systest_trace_analysis_know_asia_6():  # First two sub-sample only
    series = ['TABU/BASE']
    networks = ['asia']
    Ns = [10]
    Ss = (0, 1)
    root_dir = TESTDATA_DIR + '/experiments'
    summaries, trace, diffs = trace_analysis(series, networks, Ns, Ss, None,
                                             None, root_dir)

    # Two summaries, identical grahs produced for these two subsamples

    assert len(summaries) == 2
    assert summaries[0] == \
        {'series': 'TABU/BASE', 'network': 'asia', 'sample': 0, 'N': 10,
         'iter': 15, 'time': 2.0, 'score': -3.88071, 'type': 'DAG', 'n': 8,
         '|A|': 8, '|E|': 5, 'shd': 9, 'shd-s': 1.12, 'shd-e': 9,
         'shd-b': 0.88, 'a-ok': 0, 'a-rev': 4, 'a-eqv': 0, 'a-non': 4,
         'a-ext': 1, 'a-mis': 4, 'p': 0.0, 'r': 0.0, 'f1': 0.0, 'f1-e': 0.0,
         'f1-b': 0.308, 'bsf': 0.2, 'e-ori': 4, 'loglik': None}
    assert summaries[1] == \
        {'series': 'TABU/BASE', 'network': 'asia', 'sample': 1, 'N': 10,
         'iter': 15, 'time': 0.7, 'score': -3.88071, 'type': 'DAG', 'n': 8,
         '|A|': 8, '|E|': 5, 'shd': 9, 'shd-s': 1.12, 'shd-e': 9,
         'shd-b': 0.88, 'a-ok': 0, 'a-rev': 4, 'a-eqv': 0, 'a-non': 4,
         'a-ext': 1, 'a-mis': 4, 'p': 0.0, 'r': 0.0, 'f1': 0.0, 'f1-e': 0.0,
         'f1-b': 0.308, 'bsf': 0.2, 'e-ori': 4, 'loglik': None}
    assert trace.trace['delta/score'] == \
        [-4.544361, 0.307681, 0.194319, 0.071324, 0.071324, 0.018999, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.880712]

    # Small differences in order of changes and later iterations

    assert diffs[0] == \
        {('add', 'extra'): {('xray', 'asia'): (6, 7)},
         ('add', 'order'): {('lung', 'asia'): (10, 6),
                            ('asia', 'tub'): (11, 8),
                            ('either', 'asia'): (9, 12),
                            ('asia', 'smoke'): (12, 10)},
         ('add', 'missing'): {('asia', 'lung'): (None, 14)},
         ('delete', 'order'): {('xray', 'asia'): (13, 15)},
         ('delete', 'missing'): {('lung', 'asia'): (None, 13)},
         ('delete', 'extra'): {('dysp', 'asia'): (14, None)},
         ('add', 'opposite'): {('dysp', 'asia'): [7, 9],
                               ('bronc', 'asia'): [8, 11]}}
