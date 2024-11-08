
# system testing of run_learn structure learning entry point
# - checking that traces are repeatable for TABU/BASE learning

import pytest

from run_learn import run_learn
from fileio.common import TESTDATA_DIR
from experiments.trace_analysis import trace_analysis


def systest_run_learn_tabu_base_asia_10_ok():  # TABU/BASE asia N=10
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/BASE',
                      'networks': 'asia', 'N': '10;;0-1', 'nodes': None},
                     root_dir)

    _, trace, _ = \
        trace_analysis(series=['TABU/BASE'], networks=['asia'], Ns=[10],
                       Ss=(0, 1), params=None, root_dir=root_dir)

    # Check keye elements of trace from subsample 1

    assert trace.trace['activity'] == \
        ['init', 'add', 'add', 'add', 'add', 'add', 'add', 'add', 'add', 'add',
         'add', 'add', 'add', 'delete', 'delete', 'add', 'stop']
    assert trace.trace['arc'] == \
        [None, ('dysp', 'bronc'), ('xray', 'either'), ('either', 'lung'),
         ('either', 'tub'), ('tub', 'smoke'), ('xray', 'asia'),
         ('dysp', 'asia'), ('bronc', 'asia'), ('either', 'asia'),
         ('lung', 'asia'), ('asia', 'tub'), ('asia', 'smoke'),
         ('xray', 'asia'), ('dysp', 'asia'), ('xray', 'asia'), None]
    assert trace.context['var_order'] == \
        ['xray', 'dysp', 'bronc', 'either', 'lung', 'asia', 'tub', 'smoke']


def systest_run_learn_tabu_base_asia_100_ok():  # TABU/BASE asia N=100
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/BASE',
                      'networks': 'asia', 'N': '100;;0-1', 'nodes': None},
                     root_dir)

    _, trace, _ = \
        trace_analysis(series=['TABU/BASE'], networks=['asia'], Ns=[100],
                       Ss=(0, 1), params=None, root_dir=root_dir)

    # Check keye elements of trace from subsample 1

    print(trace.trace['activity'])
    print(trace.trace['arc'])
    print(trace.context['var_order'])
    return
    assert trace.trace['activity'] == \
        ['init', 'add', 'add', 'add', 'add', 'add', 'add', 'add', 'add', 'add',
         'add', 'add', 'add', 'delete', 'delete', 'add', 'stop']
    assert trace.trace['arc'] == \
        [None, ('dysp', 'bronc'), ('xray', 'either'), ('either', 'lung'),
         ('either', 'tub'), ('tub', 'smoke'), ('xray', 'asia'),
         ('dysp', 'asia'), ('bronc', 'asia'), ('either', 'asia'),
         ('lung', 'asia'), ('asia', 'tub'), ('asia', 'smoke'),
         ('xray', 'asia'), ('dysp', 'asia'), ('xray', 'asia'), None]
    assert trace.context['var_order'] == \
        ['xray', 'dysp', 'bronc', 'either', 'lung', 'asia', 'tub', 'smoke']


def systest_run_learn_tabu_base_asia_1k_ok():  # TABU/BASE asia N=1k
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/BASE',
                      'networks': 'asia', 'N': '1K;;0-1', 'nodes': None},
                     root_dir)

    _, trace, _ = \
        trace_analysis(series=['TABU/BASE'], networks=['asia'], Ns=[1000],
                       Ss=(0, 1), params=None, root_dir=root_dir)

    # Check keye elements of trace from subsample 1

    assert trace.trace['activity'] == \
        ['init', 'add', 'add', 'add', 'add', 'add', 'add', 'add', 'add', 'add',
         'reverse', 'reverse', 'reverse', 'delete', 'reverse', 'reverse',
         'reverse', 'reverse', 'reverse', 'delete', 'reverse', 'reverse',
         'add', 'stop']
    assert trace.trace['arc'] == \
        [None, ('dysp', 'bronc'), ('either', 'lung'), ('xray', 'either'),
         ('bronc', 'smoke'), ('either', 'tub'), ('lung', 'tub'),
         ('either', 'dysp'), ('lung', 'smoke'), ('either', 'bronc'),
         ('xray', 'either'), ('dysp', 'bronc'), ('bronc', 'smoke'),
         ('either', 'bronc'), ('either', 'xray'), ('lung', 'tub'),
         ('xray', 'either'), ('either', 'tub'), ('either', 'lung'),
         ('tub', 'lung'), ('lung', 'smoke'), ('smoke', 'bronc'),
         ('either', 'asia'), None]
    assert trace.context['var_order'] == \
        ['xray', 'dysp', 'bronc', 'either', 'lung', 'asia', 'tub', 'smoke']


def systest_run_learn_tabu_base_sports_10_ok():  # TABU/BASE sports N=10
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/BASE',
                      'networks': 'sports', 'N': '10;;0-1', 'nodes': None},
                     root_dir)

    _, trace, _ = \
        trace_analysis(series=['TABU/BASE'], networks=['sports'], Ns=[10],
                       Ss=(0, 1), params=None, root_dir=root_dir)

    # Check keye elements of trace from subsample 1

    assert trace.trace['activity'] == \
        ['init', 'add', 'reverse', 'add', 'reverse', 'reverse', 'delete',
         'reverse', 'add', 'delete', 'add', 'reverse', 'delete', 'reverse',
         'delete', 'stop']
    assert trace.trace['arc'] == \
        [None, ('HDA', 'ATgoals'), ('HDA', 'ATgoals'),
         ('HDA', 'HTshotOnTarget'), ('ATgoals', 'HDA'),
         ('HDA', 'HTshotOnTarget'), ('HDA', 'ATgoals'),
         ('HTshotOnTarget', 'HDA'), ('HDA', 'RDlevel'),
         ('HDA', 'HTshotOnTarget'), ('HDA', 'ATgoals'),
         ('HDA', 'RDlevel'), ('RDlevel', 'HDA'),
         ('HDA', 'ATgoals'), ('ATgoals', 'HDA'), None]
    assert trace.context['var_order'] == \
        ['possession', 'ATshotsOnTarget', 'ATshots', 'HDA', 'HTgoals',
         'HTshotOnTarget', 'RDlevel', 'HTshots', 'ATgoals']


def systest_run_learn_tabu_base_sports_100_ok():  # TABU/BASE sports N=100
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/BASE',
                      'networks': 'sports', 'N': '100;;0-1', 'nodes': None},
                     root_dir)

    _, trace, _ = \
        trace_analysis(series=['TABU/BASE'], networks=['sports'], Ns=[100],
                       Ss=(0, 1), params=None, root_dir=root_dir)

    # Check keye elements of trace from subsample 1

    assert trace.trace['activity'] == \
        ['init', 'add', 'add', 'add', 'reverse', 'reverse', 'reverse', 'add',
         'reverse', 'reverse', 'reverse', 'reverse', 'reverse', 'add',
         'delete', 'delete', 'stop']
    assert trace.trace['arc'] == \
        [None, ('HDA', 'HTgoals'), ('HDA', 'ATgoals'),
         ('ATshotsOnTarget', 'ATshots'), ('ATshotsOnTarget', 'ATshots'),
         ('HDA', 'HTgoals'), ('ATshots', 'ATshotsOnTarget'),
         ('HDA', 'possession'), ('ATshotsOnTarget', 'ATshots'),
         ('HTgoals', 'HDA'), ('ATshots', 'ATshotsOnTarget'),
         ('HDA', 'possession'), ('ATshotsOnTarget', 'ATshots'),
         ('HDA', 'HTshotOnTarget'), ('possession', 'HDA'),
         ('HDA', 'HTshotOnTarget'), None]
    assert trace.context['var_order'] == \
        ['possession', 'ATshotsOnTarget', 'ATshots', 'HDA', 'HTgoals',
         'HTshotOnTarget', 'RDlevel', 'HTshots', 'ATgoals']


def systest_run_learn_tabu_base_sports_1k_ok():  # TABU/BASE sports N=1k
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/BASE',
                      'networks': 'sports', 'N': '1k;;0-1', 'nodes': None},
                     root_dir)

    _, trace, _ = \
        trace_analysis(series=['TABU/BASE'], networks=['sports'], Ns=[1000],
                       Ss=(0, 1), params=None, root_dir=root_dir)

    # Check keye elements of trace from subsample 1

    # print(trace.trace['activity'])
    # print(trace.trace['arc'])
    # print(trace.context['var_order'])
    # return
    assert trace.trace['activity'] == \
        ['init', 'add', 'add', 'add', 'add', 'add', 'add', 'add', 'add',
         'add', 'reverse', 'reverse', 'delete', 'reverse', 'reverse',
         'reverse', 'add', 'reverse', 'reverse', 'reverse', 'reverse',
         'delete', 'add', 'stop']
    assert trace.trace['arc'] == \
        [None, ('HDA', 'HTgoals'), ('ATgoals', 'HTgoals'), ('HDA', 'ATgoals'),
         ('ATshotsOnTarget', 'ATshots'), ('HTshotOnTarget', 'HTshots'),
         ('possession', 'RDlevel'), ('HTshots', 'possession'),
         ('HTgoals', 'HTshotOnTarget'), ('ATgoals', 'ATshotsOnTarget'),
         ('HDA', 'ATgoals'), ('HDA', 'HTgoals'), ('ATgoals', 'HTgoals'),
         ('HTgoals', 'HTshotOnTarget'), ('HTshotOnTarget', 'HTshots'),
         ('HTshots', 'possession'), ('ATshots', 'possession'),
         ('ATgoals', 'ATshotsOnTarget'), ('ATshotsOnTarget', 'ATshots'),
         ('ATshots', 'possession'), ('possession', 'RDlevel'),
         ('ATshotsOnTarget', 'ATgoals'), ('ATshots', 'ATgoals'), None]
    assert trace.context['var_order'] == \
        ['possession', 'ATshotsOnTarget', 'ATshots', 'HDA', 'HTgoals',
         'HTshotOnTarget', 'RDlevel', 'HTshots', 'ATgoals']
