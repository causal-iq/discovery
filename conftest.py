# content of conftest.py

import pytest

EXPT_ARGS = {'action', 'series', 'metrics', 'nodes', 'networks', 'N',
             'maxtime', 'file', 'params'}


def pytest_addoption(parser):
    """
        Adds custom command line options for running/analysing experiments
    """
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run slow tests")
    parser.addoption("--action", action="store", default=None,
                     help="specific learn or analysis action required")
    parser.addoption("--series", action="store", default=None,
                     help="series to run experiments for")
    parser.addoption("--metrics", action="store", default=None,
                     help="metrics to analyse")
    parser.addoption("--nodes", action="store", default=None,
                     help="nodes to run analysis for")
    parser.addoption("--networks", action="store", default=None,
                     help="network(s) to run experiments for")
    parser.addoption("--N", action="store", default=None,
                     help="min, max sample size")
    parser.addoption("--maxtime", action="store", default=None,
                     help="maximum elapsed execution time")
    parser.addoption("--file", action="store", default=None,
                     help="Output file name")
    parser.addoption("--params", action="store", default=None,
                     help="action-specific parameters")


def pytest_collection_modifyitems(config, items):

    # --runslow command line option used, so don't skip any tests

    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
