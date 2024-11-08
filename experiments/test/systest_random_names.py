
# system testing of run_learn where node names are randomised

from run_learn import run_learn
from fileio.common import TESTDATA_DIR
from experiments.trace_analysis import trace_analysis


# Check BNBENCH/TABU insensitive to randomising node names - BASE2 series

def systest_run_learn_tabu_nam_asia_10_ok():  # TABU/NAM asia N=10
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/NAM',
                      'networks': 'asia', 'N': '10;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TABU/NAM'], networks=['asia'], Ns=[10],
                       Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004asia', 'X007bronc', 'X000dysp', 'X006either', 'X002lung',
         'X003smoke', 'X005tub', 'X001xray']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TABU/NAM'], networks=['asia'], Ns=[10],
                       Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X007asia', 'X002bronc', 'X001dysp', 'X003either', 'X004lung',
         'X000smoke', 'X006tub', 'X005xray']

    # Check trace from std run same as renamed trace

    s2, t2, diffs = \
        trace_analysis(series=['TABU/NAM', 'TABU/STD'], networks=['asia'],
                       Ns=[10], Ss=(0, 0), params=None, root_dir=root_dir)
    assert diffs is None


def systest_run_learn_tabu_nam_asia_100_ok():  # TABU/NAM asia N=100
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/NAM',
                      'networks': 'asia', 'N': '100;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TABU/NAM'], networks=['asia'], Ns=[100],
                       Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004asia', 'X007bronc', 'X000dysp', 'X006either', 'X002lung',
         'X003smoke', 'X005tub', 'X001xray']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TABU/NAM'], networks=['asia'], Ns=[100],
                       Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X007asia', 'X002bronc', 'X001dysp', 'X003either', 'X004lung',
         'X000smoke', 'X006tub', 'X005xray']

    # Check trace from std run same as renamed trace

    s2, t2, diffs = \
        trace_analysis(series=['TABU/NAM', 'TABU/STD'], networks=['asia'],
                       Ns=[100], Ss=(0, 0), params=None, root_dir=root_dir)
    assert diffs is None


def systest_run_learn_tabu_nam_asia_1k_ok():  # TABU/NAM asia N=1k
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/NAM',
                      'networks': 'asia', 'N': '1k;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TABU/NAM'], networks=['asia'], Ns=[1000],
                       Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004asia', 'X007bronc', 'X000dysp', 'X006either', 'X002lung',
         'X003smoke', 'X005tub', 'X001xray']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TABU/NAM'], networks=['asia'], Ns=[1000],
                       Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X007asia', 'X002bronc', 'X001dysp', 'X003either', 'X004lung',
         'X000smoke', 'X006tub', 'X005xray']

    # Check trace from std run same as renamed trace

    s2, t2, diffs = \
        trace_analysis(series=['TABU/NAM', 'TABU/STD'], networks=['asia'],
                       Ns=[1000], Ss=(0, 0), params=None, root_dir=root_dir)
    assert diffs is None


def systest_run_learn_tabu_nam_sports_10_ok():  # TABU/NAM sports N=10
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/NAM',
                      'networks': 'sports', 'N': '10;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TABU/NAM'], networks=['sports'], Ns=[10],
                       Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TABU/NAM'], networks=['sports'], Ns=[10],
                       Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace from std run same as renamed trace

    s2, t2, diffs = \
        trace_analysis(series=['TABU/NAM', 'TABU/STD'], networks=['sports'],
                       Ns=[10], Ss=(0, 0), params=None, root_dir=root_dir)
    assert diffs is None


def systest_run_learn_tabu_nam_sports_100_ok():  # TABU/NAM sports N=1090
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/NAM',
                      'networks': 'sports', 'N': '100;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TABU/NAM'], networks=['sports'], Ns=[100],
                       Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TABU/NAM'], networks=['sports'], Ns=[100],
                       Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace from std run same as renamed trace

    s2, t2, diffs = \
        trace_analysis(series=['TABU/NAM', 'TABU/STD'], networks=['sports'],
                       Ns=[100], Ss=(0, 0), params=None, root_dir=root_dir)
    assert diffs is None


def systest_run_learn_tabu_nam_sports_1k_ok():  # TABU/NAM sports N=1k
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TABU/NAM',
                      'networks': 'sports', 'N': '1k;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TABU/NAM'], networks=['sports'], Ns=[1000],
                       Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TABU/NAM'], networks=['sports'], Ns=[1000],
                       Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace from std run same as renamed trace

    s2, t2, diffs = \
        trace_analysis(series=['TABU/NAM', 'TABU/STD'], networks=['sports'],
                       Ns=[1000], Ss=(0, 0), params=None, root_dir=root_dir)
    assert diffs is None


# Check BNLEARN/TABU insensitive to randomising node names - TABU_NAM series

def systest_run_learn_tabu_bnl_asia_100_ok():  # asia N=100
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'BNLEARN/TABU_NAM',
                      'networks': 'asia', 'N': '100;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['BNLEARN/TABU_NAM'], networks=['asia'],
                       Ns=[100], Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004asia', 'X007bronc', 'X000dysp', 'X006either', 'X002lung',
         'X003smoke', 'X005tub', 'X001xray']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['BNLEARN/TABU_NAM'], networks=['asia'],
                       Ns=[100], Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X007asia', 'X002bronc', 'X001dysp', 'X003either', 'X004lung',
         'X000smoke', 'X006tub', 'X005xray']

    # Check trace from std run same as renamed trace

    s2, t2, diffs = \
        trace_analysis(series=['BNLEARN/TABU_NAM', 'BNLEARN/TABU_STD'],
                       networks=['asia'], Ns=[100], Ss=(0, 0), params=None,
                       root_dir=root_dir)
    assert diffs is None


def systest_run_learn_tabu_bnl_asia_1k_ok():  # asia N=1k
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'BNLEARN/TABU_NAM',
                      'networks': 'asia', 'N': '1K;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['BNLEARN/TABU_NAM'], networks=['asia'],
                       Ns=[1000], Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004asia', 'X007bronc', 'X000dysp', 'X006either', 'X002lung',
         'X003smoke', 'X005tub', 'X001xray']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['BNLEARN/TABU_NAM'], networks=['asia'],
                       Ns=[1000], Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X007asia', 'X002bronc', 'X001dysp', 'X003either', 'X004lung',
         'X000smoke', 'X006tub', 'X005xray']

    # Check trace from std run same as renamed trace

    s2, t2, diffs = \
        trace_analysis(series=['BNLEARN/TABU_NAM', 'BNLEARN/TABU_STD'],
                       networks=['asia'], Ns=[1000], Ss=(0, 0), params=None,
                       root_dir=root_dir)
    assert diffs is None


def systest_run_learn_tabu_bnl_sports_10_ok():  # sports N=10
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'BNLEARN/TABU_NAM',
                      'networks': 'sports', 'N': '10;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['BNLEARN/TABU_NAM'], networks=['sports'],
                       Ns=[10], Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['BNLEARN/TABU_NAM'], networks=['sports'],
                       Ns=[10], Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X008ATgoal', 'X002ATshot', 'X001ATshot', 'X003HDA', 'X004HTgoal',
         'X005HTshot', 'X007HTshot', 'X006RDleve', 'X000posses']

    # Check trace from std run same as renamed trace

    s2, t2, diffs = \
        trace_analysis(series=['BNLEARN/TABU_NAM', 'BNLEARN/TABU_STD'],
                       networks=['sports'], Ns=[10], Ss=(0, 0), params=None,
                       root_dir=root_dir)
    assert diffs is None


def systest_run_learn_tabu_bnl_sports_100_ok():  # sports N=100
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'BNLEARN/TABU_NAM',
                      'networks': 'sports', 'N': '100;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['BNLEARN/TABU_NAM'], networks=['sports'],
                       Ns=[100], Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['BNLEARN/TABU_NAM'], networks=['sports'],
                       Ns=[100], Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X008ATgoal', 'X002ATshot', 'X001ATshot', 'X003HDA', 'X004HTgoal',
         'X005HTshot', 'X007HTshot', 'X006RDleve', 'X000posses']

    # Check trace from std run same as renamed trace

    s2, t2, diffs = \
        trace_analysis(series=['BNLEARN/TABU_NAM', 'BNLEARN/TABU_STD'],
                       networks=['sports'], Ns=[100], Ss=(0, 0), params=None,
                       root_dir=root_dir)
    assert diffs is None


def systest_run_learn_tabu_bnl_sports_1k_ok():  # sports N=1K
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'BNLEARN/TABU_NAM',
                      'networks': 'sports', 'N': '1k;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['BNLEARN/TABU_NAM'], networks=['sports'],
                       Ns=[1000], Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['BNLEARN/TABU_NAM'], networks=['sports'],
                       Ns=[1000], Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X008ATgoal', 'X002ATshot', 'X001ATshot', 'X003HDA', 'X004HTgoal',
         'X005HTshot', 'X007HTshot', 'X006RDleve', 'X000posses']

    # Check trace from std run same as renamed trace

    s2, t2, diffs = \
        trace_analysis(series=['BNLEARN/TABU_NAM', 'BNLEARN/TABU_STD'],
                       networks=['sports'], Ns=[1000], Ss=(0, 0), params=None,
                       root_dir=root_dir)
    assert diffs is None


# Check TETRAD/FGES insenstive to node names

def systest_run_learn_fges_nam_asia_10_ok():  # asia N=10
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TETRAD/FGES_NAM',
                      'networks': 'asia', 'N': '10;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['asia'],
                       Ns=[10], Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004asia', 'X007bronc', 'X000dysp', 'X006either', 'X002lung',
         'X003smoke', 'X005tub', 'X001xray']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['asia'],
                       Ns=[10], Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X007asia', 'X002bronc', 'X001dysp', 'X003either', 'X004lung',
         'X000smoke', 'X006tub', 'X005xray']


def systest_run_learn_fges_nam_asia_100_ok():  # asia N=100
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TETRAD/FGES_NAM',
                      'networks': 'asia', 'N': '100;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['asia'],
                       Ns=[100], Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004asia', 'X007bronc', 'X000dysp', 'X006either', 'X002lung',
         'X003smoke', 'X005tub', 'X001xray']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['asia'],
                       Ns=[100], Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X007asia', 'X002bronc', 'X001dysp', 'X003either', 'X004lung',
         'X000smoke', 'X006tub', 'X005xray']


def systest_run_learn_fges_nam_asia_1k_ok():  # asia N=1k
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TETRAD/FGES_NAM',
                      'networks': 'asia', 'N': '1k;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['asia'],
                       Ns=[1000], Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004asia', 'X007bronc', 'X000dysp', 'X006either', 'X002lung',
         'X003smoke', 'X005tub', 'X001xray']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['asia'],
                       Ns=[1000], Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X007asia', 'X002bronc', 'X001dysp', 'X003either', 'X004lung',
         'X000smoke', 'X006tub', 'X005xray']


def systest_run_learn_fges_nam_sports_10_ok():  # sports N=10
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TETRAD/FGES_NAM',
                      'networks': 'sports', 'N': '10;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['sports'],
                       Ns=[10], Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['sports'],
                       Ns=[10], Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X008ATgoal', 'X002ATshot', 'X001ATshot', 'X003HDA', 'X004HTgoal',
         'X005HTshot', 'X007HTshot', 'X006RDleve', 'X000posses']


def systest_run_learn_fges_nam_sports_100_ok():  # sports N=100
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TETRAD/FGES_NAM',
                      'networks': 'sports', 'N': '100;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['sports'],
                       Ns=[100], Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['sports'],
                       Ns=[100], Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X008ATgoal', 'X002ATshot', 'X001ATshot', 'X003HDA', 'X004HTgoal',
         'X005HTshot', 'X007HTshot', 'X006RDleve', 'X000posses']


def systest_run_learn_fges_nam_sports_1k_ok():  # sports N=1K
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'TETRAD/FGES_NAM',
                      'networks': 'sports', 'N': '1k;;0-1', 'nodes': None},
                     root_dir=root_dir)

    # Check randomisation of node names (original names in alphabetic order)
    # in subsample 0

    _, t0, _ = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['sports'],
                       Ns=[1000], Ss=(0, 0), params=None, root_dir=root_dir)
    assert t0.context['var_order'] == \
        ['X004ATgoal', 'X008ATshot', 'X001ATshot', 'X007HDA', 'X002HTgoal',
         'X003HTshot', 'X006HTshot', 'X000RDleve', 'X005posses']

    # Check trace of subsample 0 and 1 are same

    s1, t1, diffs = \
        trace_analysis(series=['TETRAD/FGES_NAM'], networks=['sports'],
                       Ns=[1000], Ss=(0, 1), params=None, root_dir=root_dir)
    assert {k: v for k, v in s1[0].items() if k not in ['sample', 'time']} \
        == {k: v for k, v in s1[1].items() if k not in ['sample', 'time']}
    assert diffs is None
    assert t1.context['var_order'] == \
        ['X008ATgoal', 'X002ATshot', 'X001ATshot', 'X003HDA', 'X004HTgoal',
         'X005HTshot', 'X007HTshot', 'X006RDleve', 'X000posses']
