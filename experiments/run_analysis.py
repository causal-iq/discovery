
#   Entry point to analysing experimental results

import pytest

from fileio.common import EXPTS_DIR
from experiments.common import process_args
from experiments.score_analysis import score_analysis, score2_analysis, \
    score3_analysis
from experiments.error_analysis import error_analysis
from experiments.trace_analysis import trace_analysis, check_traces
from experiments.metric_analysis import metric_analysis
from experiments.impact_analysis import impact_analysis, network_impact
from experiments.bn_analysis import bn_analysis
from experiments.summary_analysis import summary_analysis


@pytest.fixture()
def args(pytestconfig):
    args = {'action': pytestconfig.getoption("action"),
            'series': pytestconfig.getoption('series'),
            'metrics': pytestconfig.getoption('metrics'),
            'networks': pytestconfig.getoption('networks'),
            'nodes': pytestconfig.getoption('nodes'),
            'N': pytestconfig.getoption('N'),
            'maxtime': pytestconfig.getoption('maxtime'),
            'file': pytestconfig.getoption('file'),
            'params': pytestconfig.getoption('params')}
    return args


def run_analysis(args, root_dir=EXPTS_DIR):
    """
        Performs analysis specified by command line arguments.

        :param dict args: relevant command line arguments {name: value}:
                            - series: series to analyse, e.g. TABU_DEF
                            - action: to perform e.g. check (traces)
                            - networks: networks to use e.g. 'asia,cancer'
                            - nodes: nodes for score analysis
                            - N: range of sample sizes, e.g. 10-1:1,5;1-2
                            - maxtime: maximum execution time in minutes
                            - file: output file path
                            - N: range of sample sizes, e.g. 100,2000
                            - params: action-specific parameters
        :param str root_dir: root location of files

        :raises ValueError: if arguments have bad values
    """

    #   Check valid arguments supplied - if not, print help, and return

    action, series, metrics, networks, nodes, Ns, Ss, maxtime, file, params = \
        process_args(args, analyse=True)
    if action is None:
        return None

    res = None
    if action == 'trace':
        trace_analysis(series, networks, Ns, Ss, file, params)
    if action == 'summary':
        res = summary_analysis(series, networks, Ns, Ss, metrics, maxtime,
                               file, params, root_dir)
    elif action == 'check':
        check_traces(series, networks, Ns)
    elif action == 'metrics':
        metric_analysis(action == 'metrics', series, networks, metrics, Ns,
                        params, args)
    elif action == 'series':
        metric_analysis(action == 'metrics', series, networks, metrics, Ns,
                        params, args)
    elif action == 'score':
        score_analysis(series, networks, nodes, Ns)
    elif action == 'score2':
        score2_analysis(networks, Ns)
    elif action == 'score3':
        score3_analysis()
    elif action == 'impact':
        impact_analysis(series, networks, metrics, Ns, Ss, params, args)
    elif action == 'bn':
        bn_analysis()
    elif action == 'network':
        network_impact()
    elif action == 'error':
        error_analysis(series, networks, Ns, Ss, params)

    return res
