
# system testing of run_learn structure learning entry point
# - checking that traces are repeatable for TABU AL learning

from run_learn import run_learn
from fileio.common import TESTDATA_DIR
from experiments.trace_analysis import trace_analysis


# Do some runs with relative sample sizes

def systest_run_learn_tabu_eqvp_asia_0_1_ok():  # TABU/BASE asia N=0.1
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/EQVP/L050',
                      'networks': 'asia', 'N': '0.1;;0-1', 'nodes': None},
                     root_dir)

    _, trace, _ = \
        trace_analysis(series=['TABU/EQVP/L050'], networks=['asia'], Ns=[2],
                       Ss=(0, 1), params=None, root_dir=root_dir)

    # Check key elements of trace from subsample 1

    assert trace.trace['activity'] == \
        ['init', 'add', 'add', 'add', 'add', 'add', 'add', 'add', 'add',
         'add', 'add', 'add', 'stop']
    assert trace.trace['arc'] == \
        [None, ('bronc', 'dysp'), ('xray', 'dysp'), ('xray', 'bronc'),
         ('xray', 'either'), ('xray', 'lung'), ('xray', 'asia'),
         ('xray', 'tub'), ('xray', 'smoke'), ('dysp', 'either'),
         ('dysp', 'lung'), ('dysp', 'asia'), None]
    assert trace.trace['knowledge'] == \
        [None, ('equiv_add', True, 'swap_best', ('dysp', 'bronc')), None, None,
         None, None, None, None, None, None, None, None, None]
    assert trace.context['var_order'] == \
        ['xray', 'dysp', 'bronc', 'either', 'lung', 'asia', 'tub', 'smoke']


def systest_run_learn_tabu_eqvp_asia_2_0_ok():  # TABU/EQVP/L050 asia N=2.0
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/EQVP/L050',
                      'networks': 'asia', 'N': '2.0;;0-1', 'nodes': None},
                     root_dir)

    _, trace, _ = \
        trace_analysis(series=['TABU/EQVP/L050'], networks=['asia'], Ns=[36],
                       Ss=(0, 1), params=None, root_dir=root_dir)

    # Check key elements of trace from subsample 1

    print(trace.trace['activity'])
    print(trace.trace['arc'])
    print(trace.trace['knowledge'])
    print(trace.context['var_order'])
    assert trace.trace['activity'] == \
        ['init', 'add', 'add', 'add', 'add', 'add', 'add', 'reverse',
         'reverse', 'reverse', 'delete', 'delete', 'add', 'add', 'reverse',
         'delete', 'reverse', 'add', 'reverse', 'delete', 'stop']
    assert trace.trace['arc'] == \
        [None, ('either', 'xray'), ('lung', 'either'), ('bronc', 'dysp'),
         ('tub', 'either'), ('smoke', 'bronc'), ('lung', 'smoke'),
         ('lung', 'smoke'), ('tub', 'either'), ('smoke', 'lung'),
         ('lung', 'smoke'), ('either', 'tub'), ('xray', 'tub'),
         ('lung', 'smoke'), ('lung', 'smoke'), ('xray', 'tub'),
         ('smoke', 'lung'), ('tub', 'either'), ('lung', 'smoke'),
         ('smoke', 'lung'), None]
    assert trace.trace['knowledge'][:9] == \
        [None,
         ('equiv_add', True, 'swap_best', ('xray', 'either')),
         ('equiv_add', True, 'swap_best', ('either', 'lung')),
         ('equiv_add', True, 'swap_best', ('dysp', 'bronc')),
         None,
         ('equiv_add', True, 'swap_best', ('bronc', 'smoke')),
         ('equiv_add', None, 'no_op', ('lung', 'smoke')),
         None,
         ('act_cache', True, 'stop_rev', ('smoke', 'bronc'))]
    assert trace.context['var_order'] == \
        ['xray', 'dysp', 'bronc', 'either', 'lung', 'asia', 'tub', 'smoke']
