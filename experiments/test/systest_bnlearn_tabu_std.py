
# system testing of run_learn structure learning entry point
# - checking that traces are repeatable for BNLEARN/TABU_STD

from run_learn import run_learn
from fileio.common import TESTDATA_DIR
from experiments.trace_analysis import trace_analysis


# Series BNLEARN/TABU_STD

def systest_run_learn_bnlearn_tabu_std_asia_10_ok():  # asia N=10
    root_dir = TESTDATA_DIR + '/experiments'

    # single-valued columns so no graph produced

    assert run_learn({'action': 'compare', 'series': 'BNLEARN/TABU_STD',
                      'networks': 'asia', 'N': '10', 'nodes': None},
                     root_dir) is False


def systest_run_learn_bnlearn_tabu_std_asia_100_ok():  # asia N=100
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'BNLEARN/TABU_STD',
                      'networks': 'asia', 'N': '100', 'nodes': None},
                     root_dir)

    _, t0, _ = \
        trace_analysis(series=['BNLEARN/TABU_STD'], networks=['asia'],
                       Ns=[100], Ss=None, params=None, root_dir=root_dir)
    _, t1, _ = \
        trace_analysis(series=['TABU/STD'], networks=['asia'],
                       Ns=[100], Ss=None, params=None, root_dir=root_dir)

    # BNLEARN & BNBENCH learn same graph

    assert t0.result == t1.result


def systest_run_learn_bnlearn_tabu_std_asia_1k_ok():  # asia N=1k
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'BNLEARN/TABU_STD',
                      'networks': 'asia', 'N': '1k', 'nodes': None},
                     root_dir)

    _, t0, _ = \
        trace_analysis(series=['BNLEARN/TABU_STD'], networks=['asia'],
                       Ns=[1000], Ss=None, params=None, root_dir=root_dir)
    _, t1, _ = \
        trace_analysis(series=['TABU/STD'], networks=['asia'],
                       Ns=[1000], Ss=None, params=None, root_dir=root_dir)

    # BNLEARN & BNBENCH learn same graph

    assert t0.result == t1.result


def systest_run_learn_bnlearn_tabu_std_sports_10_ok():  # sports N=10
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'BNLEARN/TABU_STD',
                      'networks': 'sports', 'N': '10', 'nodes': None},
                     root_dir)

    _, t0, _ = \
        trace_analysis(series=['BNLEARN/TABU_STD'], networks=['sports'],
                       Ns=[10], Ss=None, params=None, root_dir=root_dir)
    _, t1, _ = \
        trace_analysis(series=['TABU/STD'], networks=['sports'],
                       Ns=[10], Ss=None, params=None, root_dir=root_dir)

    # BNLEARN & BNBENCH learn same graph

    assert t0.result == t1.result


def systest_run_learn_bnlearn_tabu_std_sports_100_ok():  # sports N=100
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'BNLEARN/TABU_STD',
                      'networks': 'sports', 'N': '100', 'nodes': None},
                     root_dir)

    _, t0, _ = \
        trace_analysis(series=['BNLEARN/TABU_STD'], networks=['sports'],
                       Ns=[100], Ss=None, params=None, root_dir=root_dir)
    _, t1, _ = \
        trace_analysis(series=['TABU/STD'], networks=['sports'],
                       Ns=[100], Ss=None, params=None, root_dir=root_dir)

    # BNLEARN & BNBENCH learn same graph

    assert t0.result == t1.result


def systest_run_learn_bnlearn_tabu_std_sports_1k_ok():  # sports N=1k
    root_dir = TESTDATA_DIR + '/experiments'
    assert run_learn({'action': 'compare', 'series': 'BNLEARN/TABU_STD',
                      'networks': 'sports', 'N': '1K', 'nodes': None},
                     root_dir)

    _, trace, _ = \
        trace_analysis(series=['BNLEARN/TABU_STD'], networks=['sports'],
                       Ns=[1000], Ss=None, params=None, root_dir=root_dir)

    # Check learnt graph (slightly different from TABU/STD)

    assert trace.result.to_string() == \
        ('[ATgoals|ATshotsOnTarget]' +
         '[ATshots|possession]' +
         '[ATshotsOnTarget|ATshots]' +
         '[HDA|ATgoals:HTgoals]' +
         '[HTgoals|HTshotOnTarget]' +
         '[HTshotOnTarget]' +
         '[HTshots|HTshotOnTarget]' +
         '[RDlevel|possession]' +
         '[possession|HTshots]')
