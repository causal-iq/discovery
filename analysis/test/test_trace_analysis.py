
#   Test the TraceAnalysis class

import pytest
from pandas import DataFrame

from analysis.trace import TraceAnalysis
from learn.trace import Trace
from fileio.common import TESTDATA_DIR, EXPTS_DIR
from core.metrics import dicts_same
from core.bn import BN
from core.cpt import CPT
from fileio.pandas import Pandas
import testdata.example_dags as ex_dag


#   Test TraceAnalysis constructor

def test_trace_analysis_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        TraceAnalysis()


def test_trace_analysis_type_error_2():  # invalid trace argument
    ab = ex_dag.ab()
    with pytest.raises(TypeError):
        TraceAnalysis(37, ab)
    with pytest.raises(TypeError):
        TraceAnalysis('bad type', ab)
    with pytest.raises(TypeError):
        TraceAnalysis(False, ab)
    with pytest.raises(TypeError):
        TraceAnalysis({'another': 'bad type'}, ab)


def test_trace_analysis_type_error_3():  # invalid ref argument
    trace = Trace.read('HC_N_1/asia', TESTDATA_DIR + '/experiments')['N10']
    with pytest.raises(TypeError):
        TraceAnalysis(trace, None)
    with pytest.raises(TypeError):
        TraceAnalysis(trace, 37)
    with pytest.raises(TypeError):
        TraceAnalysis(trace, 'bad type')
    with pytest.raises(TypeError):
        TraceAnalysis(trace, False)
    with pytest.raises(TypeError):
        TraceAnalysis(trace, {'another': 'bad type'})


def test_trace_analysis_type_error_4():  # invalid data argument
    trace = Trace.read('HC_N_1/asia', TESTDATA_DIR + '/experiments')['N10']
    ab = ex_dag.ab()
    with pytest.raises(TypeError):
        TraceAnalysis(trace, ab, 37)
    with pytest.raises(TypeError):
        TraceAnalysis(trace, ab, 'bad type')
    with pytest.raises(TypeError):
        TraceAnalysis(trace, ab, False)
    with pytest.raises(TypeError):
        TraceAnalysis(trace, ab, {'another': 'bad type'})


def test_trace_analysis_value_error_1():  # Trace has no result included
    trace = Trace.read('HC_N_1/asia', TESTDATA_DIR + '/experiments')['N10']
    ab = ex_dag.ab()
    with pytest.raises(ValueError):
        TraceAnalysis(trace, ab)


def test_trace_analysis_value_error_2():  # Trace and ref have different nodes
    trace = Trace.read('HC_N_1/asia', TESTDATA_DIR + '/experiments')['N10']
    ref = ex_dag.ab()
    with pytest.raises(ValueError):
        TraceAnalysis(trace, ref)


def test_trace_analysis_value_error_3():  # Ref and data have different nodes
    trace = Trace.read('HC_N_1/asia', TESTDATA_DIR + '/experiments')['N10']
    ref = ex_dag.asia()
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/sachs.data.gz',
                       dstype='categorical').sample
    with pytest.raises(ValueError):
        TraceAnalysis(trace, ref, data)


def test_trace_analysis_ab2_r1_0_ok1():  # A-->B 1:0 deterministic, no MI
    trace = Trace.read('TINY/AB2/R1_0',
                       TESTDATA_DIR + '/experiments')['N1000']
    ref = ex_dag.ab()
    analysis = TraceAnalysis(trace, ref)
    add_entry = DataFrame(analysis.trace).to_dict(orient='records')[1]
    assert {'time': 0.07861900329589844,
            'activity': 'add',
            'arc': ('A', 'B'),
            'delta/score': 0.000693,
            'activity_2': 'add',
            'arc_2': ('B', 'A'),
            'delta_2': 0.000693,
            'min_N': 242.0,
            'mean_N': 375.0,
            'max_N': 516.0,
            'lt5': 1.0,
            'free_params': 1.5,
            'knowledge': None,
            'blocked': None,
            'status': 'ok',
            'margin': 0.0} == add_entry


def test_trace_analysis_ab2_r1_0_ok2():  # A-->B 1:0 deterministic, Oracle MI
    trace = Trace.read('TINY/AB2/R1_0',
                       TESTDATA_DIR + '/experiments')['N1000']
    ab = ex_dag.ab()
    bn = BN(ab, {'A': (CPT, {'0': 0.5, '1': 0.5}),
                 'B': (CPT, [({'A': '0'}, {'0': 1.0, '1': 0.0}),
                             ({'A': '1'}, {'0': 0.0, '1': 1.0})])})

    # supply TraceAnalysis with bn to get Oracle MI
    # Not checking true MI as this will change with randomly generated data.

    analysis = TraceAnalysis(trace, bn)
    add_entry = DataFrame(analysis.trace).to_dict(orient='records')[1]
    assert {'time': 0.07861900329589844,
            'activity': 'add',
            'arc': ('A', 'B'),
            'delta/score': 0.000693,
            'activity_2': 'add',
            'arc_2': ('B', 'A'),
            'delta_2': 0.000693,
            'min_N': 242.0,
            'mean_N': 375.0,
            'max_N': 516.0,
            'lt5': 1.0,
            'free_params': 1.5,
            'knowledge': None,
            'blocked': None,
            'status': 'ok',
            'margin': 0.0,
            'Oracle MI': 0.693147} == add_entry


def test_trace_analysis_ab2_r1_1_ok1():  # A-->B 1:1 independent, no MI
    trace = Trace.read('TINY/AB2/R1_1',
                       TESTDATA_DIR + '/experiments')['N1000']
    ref = ex_dag.ab()
    analysis = TraceAnalysis(trace, ref)
    add_entry = DataFrame(analysis.trace).to_dict(orient='records')[1]
    assert {'time': 0.03161191940307617,
            'activity': 'add',
            'arc': ('A', 'B'),
            'delta/score': 2e-06,
            'activity_2': 'add',
            'arc_2': ('B', 'A'),
            'delta_2': 2e-06,
            'min_N': 360.5,
            'mean_N': 375.0,
            'max_N': 394.5,
            'lt5': 0.0,
            'free_params': 1.5,
            'knowledge': None,
            'blocked': None,
            'status': 'ok',
            'margin': 0.0} == add_entry


def test_trace_analysis_ab2_r1_1_ok2():  # A-->B 1:1 independent, Oracle MI
    trace = Trace.read('TINY/AB2/R1_1',
                       TESTDATA_DIR + '/experiments')['N1000']
    ab = ex_dag.ab()
    bn = BN(ab, {'A': (CPT, {'0': 0.5, '1': 0.5}),
                 'B': (CPT, [({'A': '0'}, {'0': 0.5, '1': 0.5}),
                             ({'A': '1'}, {'0': 0.5, '1': 0.5})])})

    # supply TraceAnalysis with bn to get Oracle MI
    # Not checking true MI as this will change with randomly generated data.

    analysis = TraceAnalysis(trace, bn)
    add_entry = DataFrame(analysis.trace).to_dict(orient='records')[1]
    assert {'time': 0.03161191940307617,
            'activity': 'add',
            'arc': ('A', 'B'),
            'delta/score': 2e-06,
            'activity_2': 'add',
            'arc_2': ('B', 'A'),
            'delta_2': 2e-06,
            'min_N': 360.5,
            'mean_N': 375.0,
            'max_N': 394.5,
            'lt5': 0.0,
            'free_params': 1.5,
            'knowledge': None,
            'blocked': None,
            'status': 'ok',
            'margin': 0.0,
            'Oracle MI': 0.0} == add_entry


def test_trace_analysis_cancer_ok1():  # Trace & ref from Cancer network, N=10
    trace = Trace.read('HC_N_1/cancer', TESTDATA_DIR + '/experiments')['N10']
    ref = ex_dag.cancer()
    analysis = TraceAnalysis(trace, ref)
    assert analysis.context == trace.context
    assert dicts_same(dict1=analysis.summary, strict=False,
                      dict2={'iter': 3, 'time': 0.6, 'a-rev': 1,
                             'bsf': -0.208, 'a-eqv': 0, 'a-non': 1})
    assert analysis.edges['result'] == \
        {'arc_matched': set(),
         'arc_extra': {('Dyspnoea', 'Pollution'), ('Dyspnoea', 'Xray')},
         'arc_missing': {('Cancer', 'Xray'), ('Pollution', 'Cancer'),
                         ('Smoker', 'Cancer')},
         'arc_reversed': {('Dyspnoea', 'Cancer')},
         'arc_nonequivalent': {('Dyspnoea', 'Cancer')},
         'arc_equivalent': set(),
         'arc_not_edge': set(),
         'edge_not_arc': set(),
         'edge_matched': set(),
         'edge_extra': set(),
         'edge_missing': set()}


def test_trace_analysis_cancer_ok2():  # Trace & ref from Cancer network, N=500
    trace = Trace.read('HC_N_1/cancer', TESTDATA_DIR + '/experiments')['N500']
    ref = ex_dag.cancer()
    analysis = TraceAnalysis(trace, ref)
    assert analysis.context == trace.context
    assert dicts_same(dict1=analysis.summary, strict=False,
                      dict2={'iter': 3, 'time': 0.5, 'a-rev': 1, 'a-eqv': 0,
                             'a-non': 1, 'a-ok': 2, 'a-mis': 1, 'a-ext': 0,
                             'f1': 0.571})
    assert analysis.edges['result'] == \
        {'arc_matched': {('Cancer', 'Dyspnoea'), ('Cancer', 'Xray')},
         'arc_extra': set(),
         'arc_missing': {('Pollution', 'Cancer')},
         'arc_reversed': {('Cancer', 'Smoker')},
         'arc_nonequivalent': {('Cancer', 'Smoker')},
         'arc_equivalent': set(),
         'arc_not_edge': set(),
         'edge_not_arc': set(),
         'edge_matched': set(),
         'edge_extra': set(),
         'edge_missing': set()}


def test_trace_analysis_cancer_ok3():  # ancer network, N=500, using BN
    trace = Trace.read('HC_N_1/cancer', TESTDATA_DIR + '/experiments')['N500']
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    analysis = TraceAnalysis(trace, bn)
    assert analysis.context == trace.context
    print(DataFrame(analysis.trace))
    assert dicts_same(dict1=analysis.summary, strict=False,
                      dict2={'iter': 3, 'time': 0.5, 'a-rev': 1, 'a-eqv': 0,
                             'a-non': 1, 'a-ok': 2, 'a-mis': 1, 'a-ext': 0,
                             'f1': 0.571})
    assert analysis.edges['result'] == \
        {'arc_matched': {('Cancer', 'Dyspnoea'), ('Cancer', 'Xray')},
         'arc_extra': set(),
         'arc_missing': {('Pollution', 'Cancer')},
         'arc_reversed': {('Cancer', 'Smoker')},
         'arc_nonequivalent': {('Cancer', 'Smoker')},
         'arc_equivalent': set(),
         'arc_not_edge': set(),
         'edge_not_arc': set(),
         'edge_matched': set(),
         'edge_extra': set(),
         'edge_missing': set()}


def test_trace_analysis_asia_1_ok():  # Asia network, N=10, No MI
    trace = Trace.read('HC_N_1/asia', TESTDATA_DIR + '/experiments')['N10']
    ref = ex_dag.asia()
    analysis = TraceAnalysis(trace, ref)
    assert analysis.context == trace.context
    assert dicts_same(dict1=analysis.summary, strict=False,
                      dict2={'iter': 6, 'time': 1.7, 'p': 0.333, 'a-ok': 2,
                             'a-mis': 4, 'a-rev': 2, 'a-non': 2, 'a-eqv': 0})
    assert analysis.edges['result'] == \
        {'arc_matched': {('bronc', 'dysp'), ('either', 'xray')},
         'arc_extra': {('bronc', 'asia'), ('tub', 'smoke')},
         'arc_missing': {('asia', 'tub'), ('either', 'dysp'),
                         ('smoke', 'lung'), ('smoke', 'bronc')},
         'arc_reversed': {('either', 'lung'), ('either', 'tub')},
         'arc_nonequivalent': {('either', 'lung'), ('either', 'tub')},
         'arc_equivalent': set(),
         'arc_not_edge': set(),
         'edge_not_arc': set(),
         'edge_matched': set(),
         'edge_extra': set(),
         'edge_missing': set()}


def test_trace_analysis_asia_2_ok():  # Asia network, N=40, No MI
    trace = Trace.read('HC_N_1/asia', TESTDATA_DIR + '/experiments')['N40']
    ref = ex_dag.asia()
    analysis = TraceAnalysis(trace, ref)
    assert analysis.context == trace.context
    assert dicts_same(dict1=analysis.summary, strict=False,
                      dict2={'iter': 5, 'time': 2.0, 'p': 0.4, 'a-ok': 2,
                             'a-ext': 0, 'a-mis': 3, 'a-rev': 3, 'a-non': 2,
                             'a-eqv': 1})
    assert analysis.edges['result'] == \
        {'arc_matched': {('bronc', 'dysp'), ('either', 'xray')},
         'arc_extra': set(),
         'arc_missing': {('asia', 'tub'), ('either', 'dysp'),
                         ('smoke', 'lung')},
         'arc_reversed': {('bronc', 'smoke'), ('either', 'lung'),
                          ('either', 'tub')},
         'arc_nonequivalent': {('either', 'lung'), ('either', 'tub')},
         'arc_equivalent': {('bronc', 'smoke')},
         'arc_not_edge': set(),
         'edge_not_arc': set(),
         'edge_matched': set(),
         'edge_extra': set(),
         'edge_missing': set()}


def test_trace_analysis_asia_3_ok():  # Asia, N=10K, bn used so get Oracke MI
    trace = Trace.read('HC/STD/asia', TESTDATA_DIR + '/experiments')['N10000']
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    analysis = TraceAnalysis(trace, ref)
    assert analysis.context == trace.context
    print('\n{}'.format(DataFrame(analysis.trace)))
    assert dicts_same(dict1=analysis.summary, strict=False,
                      dict2={'iter': 10, 'time': 2.8, 'p': 0.444, 'a-ok': 4,
                             'a-ext': 1, 'a-mis': 0, 'a-rev': 4, 'a-non': 2,
                             'a-eqv': 2})
    assert {'arc_matched': {('bronc', 'dysp'), ('either', 'xray'),
                            ('either', 'dysp'), ('smoke', 'bronc')},
            'arc_reversed': {('either', 'lung'), ('lung', 'smoke'),
                             ('either', 'tub'), ('tub', 'asia')},
            'edge_not_arc': set(),
            'arc_not_edge': set(),
            'edge_matched': set(),
            'arc_extra': {('lung', 'tub')},
            'edge_extra': set(),
            'arc_missing': set(),
            'edge_missing': set(),
            'arc_equivalent': {('lung', 'smoke'), ('tub', 'asia')},
            'arc_nonequivalent': {('either', 'lung'), ('either', 'tub')}} \
        == analysis.edges['result']

    # check the Oracle MI values

    assert [None, 0.250616, 0.185452, 0.155004, 0.046201, 0.029288, 1e-06,
            0.020493, 0.02244, 0.046201, 0.000405, None] \
        == analysis.trace['Oracle MI']


def test_trace_analysis_asia_4_ok():  # Asia, N=100, MI and OMI values
    trace = Trace.read('HC/STD/asia', TESTDATA_DIR + '/experiments')['N100']
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                       dstype='categorical', N=100).sample

    analysis = TraceAnalysis(trace, ref, data)
    assert analysis.context == trace.context
    print('\n{}'.format(DataFrame(analysis.trace)))

    # Check learnt graph metrics

    loglik = analysis.summary.pop('loglik')
    assert analysis.summary == \
        {'N': 100, 'sample': None, 'iter': 6, 'time': 3.1, 'score': -2.61419,
         'type': 'DAG', 'n': 8, '|A|': 8, '|E|': 6, 'shd': 5, 'shd-s': 0.62,
         'shd-e': 6, 'shd-b': 0.44, 'a-ok': 3, 'a-rev': 3, 'a-eqv': 1,
         'a-non': 2, 'a-ext': 0, 'a-mis': 2, 'p': 0.5, 'r': 0.375, 'f1': 0.429,
         'f1-e': 0.286, 'f1-b': 0.643, 'bsf': 0.562, 'e-ori': 4, 'bsf-e': 0.5,
         'shd-es': 0.75}

    # Check log likelihood (no defined for this trace)

    assert loglik is None

    # check learnt graph edges

    assert analysis.edges['result'] == \
        {'arc_matched': {('smoke', 'bronc'), ('either', 'xray'),
                         ('bronc', 'dysp')},
         'arc_reversed': {('either', 'lung'), ('lung', 'smoke'),
                          ('either', 'tub')},
         'edge_not_arc': set(),
         'arc_not_edge': set(),
         'edge_matched': set(),
         'arc_extra': set(),
         'edge_extra': set(),
         'arc_missing': {('either', 'dysp'), ('asia', 'tub')},
         'edge_missing': set(),
         'arc_equivalent': {('lung', 'smoke')},
         'arc_nonequivalent': {('either', 'lung'), ('either', 'tub')}}

    # Check MI and Oracle MI values - note large discrepancies between the two

    assert analysis.trace['Oracle MI'] == \
        [None, 0.151177, 0.198875, 0.248491, 0.035458, 0.046201, 0.028968,
         None]
    assert analysis.trace['MI'] == \
        [None, 0.228729, 0.223497, 0.216131, 0.062603, 0.054763, 0.02586, None]


def test_trace_analysis_asia_5_ok():  # Asia, N=1K, MI and OMI values
    trace = Trace.read('HC/STD/asia', TESTDATA_DIR + '/experiments')['N1000']
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                       dstype='categorical').sample

    analysis = TraceAnalysis(trace, ref, data)
    assert analysis.context == trace.context
    print('\n{}'.format(DataFrame(analysis.trace)))

    # Check learnt graph metrics

    loglik = analysis.summary.pop('loglik')
    assert analysis.summary == \
        {'N': 1000, 'sample': None, 'iter': 9, 'time': 3.5, 'score': -2.28197,
         'type': 'DAG', 'n': 8, '|A|': 8, '|E|': 9, 'shd': 8, 'shd-s': 1.0,
         'shd-e': 10, 'shd-b': 0.69, 'a-ok': 2, 'a-rev': 5, 'a-eqv': 2,
         'a-non': 3, 'a-ext': 2, 'a-mis': 1, 'p': 0.222, 'r': 0.25,
         'f1': 0.235, 'f1-e': 0.0, 'f1-b': 0.529, 'bsf': 0.462, 'e-ori': 7,
         'bsf-e': 0.462, 'shd-es': 1.25}

    # Check log likelihood (no defined for this trace)

    assert loglik is None

    # check learnt graph edges

    assert analysis.edges['result'] == \
        {'arc_matched': {('either', 'xray'), ('bronc', 'dysp')},
         'arc_reversed': {('either', 'lung'), ('lung', 'smoke'),
                          ('either', 'tub'), ('bronc', 'smoke'),
                          ('dysp', 'either')},
         'edge_not_arc': set(),
         'arc_not_edge': set(),
         'edge_matched': set(),
         'arc_extra': {('bronc', 'either'), ('lung', 'tub')},
         'edge_extra': set(),
         'arc_missing': {('asia', 'tub')},
         'edge_missing': set(),
         'arc_equivalent': {('lung', 'smoke'), ('bronc', 'smoke')},
         'arc_nonequivalent': {('either', 'tub'), ('either', 'lung'),
                               ('dysp', 'either')}}

    # Check MI and Oracle MI values - much closer than for N=100

    assert analysis.trace['Oracle MI'] == \
        [None, 0.251948, 0.185076, 0.157087, 0.046201, 0.028254, 0.000117,
         0.021107, 0.02244, 0.001505, None]
    assert analysis.trace['MI'] == \
        [None, 0.253031, 0.185035, 0.180532, 0.056591, 0.038635, 2.6e-05,
         0.021661, 0.021691, 0.003973, None]


def test_trace_analysis_sports_ok1():  # Sports, N=20, empty graph
    trace = Trace.read('HC/STD/sports', TESTDATA_DIR + '/experiments')['N20']
    ref = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc').dag
    analysis = TraceAnalysis(trace, ref)
    assert analysis.context == trace.context
    assert dicts_same(dict1=analysis.summary, strict=False,
                      dict2={'iter': 0, 'time': 2.5, 'p': None, 'r': 0.0,
                             'f1': 0, 'shd-s': 1, 'a-ok': 0, 'a-mis': 15,
                             'a-rev': 0, 'a-eqv': 0, 'a-non': 0})
    assert analysis.edges['result'] == \
        {'arc_matched': set(),
         'arc_extra': set(),
         'arc_missing': {('ATshotsOnTarget', 'ATgoals'),
                         ('RDlevel', 'ATgoals'), ('RDlevel', 'ATshots'),
                         ('possession', 'ATshots'),
                         ('ATshots', 'ATshotsOnTarget'),
                         ('RDlevel', 'ATshotsOnTarget'), ('ATgoals', 'HDA'),
                         ('HTgoals', 'HDA'), ('HTshotOnTarget', 'HTgoals'),
                         ('RDlevel', 'HTgoals'), ('HTshots', 'HTshotOnTarget'),
                         ('RDlevel', 'HTshotOnTarget'), ('RDlevel', 'HTshots'),
                         ('possession', 'HTshots'), ('RDlevel', 'possession')},
         'arc_reversed': set(),
         'arc_nonequivalent': set(),
         'arc_equivalent': set(),
         'arc_not_edge': set(),
         'edge_not_arc': set(),
         'edge_matched': set(),
         'edge_extra': set(),
         'edge_missing': set()}


def test_trace_analysis_sports_ok2():  # Sports, N=10, single arc
    trace = Trace.read('HC/STD/sports', TESTDATA_DIR + '/experiments')['N10']
    ref = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc').dag
    analysis = TraceAnalysis(trace, ref)
    assert analysis.context == trace.context
    assert dicts_same(dict1=analysis.summary, strict=False,
                      dict2={'iter': 1, 'time': 2.6, 'p': 1.0, 'r': 0.067,
                             'f1': 0.125, 'shd-s': 0.93, 'a-ok': 1,
                             'a-mis': 14, 'a-rev': 0, 'a-eqv': 0, 'a-non': 0,
                             'a-ext': 0, 'score': -14.70617})
    assert analysis.summary['type'] == 'DAG'
    assert analysis.edges['result'] == \
        {'arc_matched': {('ATgoals', 'HDA')},
         'arc_missing': {('ATshotsOnTarget', 'ATgoals'),
                         ('RDlevel', 'ATgoals'), ('RDlevel', 'ATshots'),
                         ('possession', 'ATshots'),
                         ('ATshots', 'ATshotsOnTarget'),
                         ('RDlevel', 'ATshotsOnTarget'),
                         ('HTgoals', 'HDA'), ('HTshotOnTarget', 'HTgoals'),
                         ('RDlevel', 'HTgoals'), ('HTshots', 'HTshotOnTarget'),
                         ('RDlevel', 'HTshotOnTarget'), ('RDlevel', 'HTshots'),
                         ('possession', 'HTshots'), ('RDlevel', 'possession')},
         'arc_reversed': set(),
         'arc_nonequivalent': set(),
         'arc_equivalent': set(),
         'arc_not_edge': set(),
         'arc_extra': set(),
         'edge_not_arc': set(),
         'edge_matched': set(),
         'edge_extra': set(),
         'edge_missing': set()}


def test_trace_analysis_sports_ok3():  # Sports, N=500, 8 arcs
    trace = Trace.read('HC/STD/sports', TESTDATA_DIR + '/experiments')['N500']
    ref = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc').dag
    analysis = TraceAnalysis(trace, ref)
    assert analysis.context == trace.context
    assert dicts_same(dict1=analysis.summary, strict=False,
                      dict2={'iter': 9, 'time': 7.6, 'p': 0.625, 'r': 0.333,
                             'f1': 0.435, 'shd-s': 0.73, 'a-ok': 5, 'a-mis': 8,
                             'a-rev': 2, 'a-eqv': 2, 'a-non': 0, 'a-ext': 1,
                             'score': -12.14179, 'sample': None, 'N': 500})
    assert analysis.summary['type'] == 'DAG'
    assert analysis.edges['result'] == \
        {'arc_matched': {('HTgoals', 'HDA'), ('possession', 'ATshots'),
                         ('RDlevel', 'possession'), ('ATgoals', 'HDA'),
                         ('ATshots', 'ATshotsOnTarget')},
         'arc_extra': {('HDA', 'RDlevel')},
         'arc_missing': {('ATshotsOnTarget', 'ATgoals'),
                         ('RDlevel', 'ATshots'),
                         ('RDlevel', 'ATshotsOnTarget'),
                         ('RDlevel', 'HTgoals'), ('RDlevel', 'ATgoals'),
                         ('RDlevel', 'HTshotOnTarget'), ('RDlevel', 'HTshots'),
                         ('possession', 'HTshots')},
         'arc_reversed': {('HTgoals', 'HTshotOnTarget'),
                          ('HTshotOnTarget', 'HTshots')},
         'arc_nonequivalent': set(),
         'arc_equivalent': {('HTgoals', 'HTshotOnTarget'),
                            ('HTshotOnTarget', 'HTshots')},
         'arc_not_edge': set(),
         'edge_not_arc': set(),
         'edge_matched': set(),
         'edge_extra': set(),
         'edge_missing': set()}


@pytest.mark.slow
def test_trace_analysis_mildew_10_ok():  # Mildew, 10 rows
    """
        Test Mildew with N=10 ... 'edge' case because DataFrame contingency
        tables have some zero-entries which had caused the log. likelihood
        comparisn to fail without DataFrame recreation to prevent zero
        marginals.
    """
    trace = Trace.read('HC/STD/mildew')['N10']
    ref = BN.read(EXPTS_DIR + '/bn/mildew.dsc').dag
    data = Pandas.read(EXPTS_DIR + '/datasets/mildew.data.gz', N=10,
                       dstype='categorical').sample

    analysis = TraceAnalysis(trace, ref, data)
    assert analysis.context == trace.context
    print('\n{}'.format(DataFrame(analysis.trace)))

    # Check learnt graph metrics

    loglik = analysis.summary.pop('loglik')
    assert analysis.summary == \
        {'N': 10, 'sample': None, 'iter': 13, 'time': 11.0, 'score': -60.87598,
         'type': 'DAG', 'n': 35, '|A|': 46, '|E|': 13, 'shd': 56,
         'shd-s': 1.22, 'shd-e': 57, 'shd-es': 1.24, 'shd-b': 1.21, 'a-ok': 1,
         'a-rev': 1, 'a-eqv': 0, 'a-non': 1, 'a-ext': 11, 'a-mis': 44,
         'p': 0.077, 'r': 0.022, 'f1': 0.034, 'f1-b': 0.051, 'bsf': 0.013,
         'bsf-e': 0.002, 'f1-e': 0.0, 'e-ori': 2}

    # Check log likelihood (no defined for this trace)

    assert loglik is None

    assert analysis.edges['result'] == \
        {'arc_matched': {('meldug_3', 'meldug_4')},
         'arc_reversed': {('mikro_1', 'nedboer_1')},
         'edge_not_arc': set(),
         'arc_not_edge': set(),
         'edge_matched': set(),
         'arc_extra': {('middel_2', 'mikro_1'),
                       ('temp_4', 'nedboer_3'), ('nedboer_2', 'straaling_2'),
                       ('straaling_4', 'temp_2'), ('middel_1', 'temp_3'),
                       ('straaling_3', 'temp_4'), ('straaling_3', 'middel_2'),
                       ('straaling_2', 'straaling_1'),
                       ('straaling_2', 'straaling_3'),
                       ('straaling_2', 'temp_1'), ('lai_4', 'nedboer_2')},
         'edge_extra': set(),
         'arc_missing': {('middel_3', 'meldug_4'), ('temp_2', 'foto_2'),
                         ('lai_0', 'lai_1'), ('dm_4', 'udbytte'),
                         ('lai_2', 'mikro_2'), ('middel_2', 'meldug_3'),
                         ('lai_1', 'foto_1'), ('meldug_1', 'lai_1'),
                         ('temp_4', 'foto_4'), ('lai_2', 'foto_2'),
                         ('temp_3', 'mikro_3'), ('meldug_2', 'meldug_3'),
                         ('foto_1', 'dm_1'), ('lai_4', 'foto_4'),
                         ('straaling_1', 'foto_1'), ('mikro_1', 'meldug_2'),
                         ('straaling_4', 'foto_4'), ('lai_1', 'mikro_1'),
                         ('temp_3', 'foto_3'), ('mikro_2', 'meldug_3'),
                         ('foto_2', 'dm_2'), ('foto_4', 'dm_4'),
                         ('lai_3', 'mikro_3'), ('lai_1', 'lai_2'),
                         ('dm_3', 'dm_4'), ('nedboer_3', 'mikro_3'),
                         ('meldug_3', 'lai_3'), ('straaling_3', 'foto_3'),
                         ('dm_2', 'dm_3'), ('temp_1', 'foto_1'),
                         ('lai_3', 'lai_4'), ('meldug_4', 'lai_4'),
                         ('lai_3', 'foto_3'), ('mikro_3', 'meldug_4'),
                         ('nedboer_2', 'mikro_2'), ('lai_2', 'lai_3'),
                         ('dm_1', 'dm_2'), ('foto_3', 'dm_3'),
                         ('straaling_2', 'foto_2'), ('meldug_1', 'meldug_2'),
                         ('middel_1', 'meldug_2'), ('temp_1', 'mikro_1'),
                         ('temp_2', 'mikro_2'), ('meldug_2', 'lai_2')},
         'edge_missing': set(),
         'arc_equivalent': set(),
         'arc_nonequivalent': {('mikro_1', 'nedboer_1')}}


#   Test TraceAnalysis.select() - selecting trace from series

def test_trace_analysis_select_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        TraceAnalysis.select()


def test_trace_analysis_select_type_error_2():  # 1 argument
    with pytest.raises(TypeError):
        TraceAnalysis.select('HC_N_1')
    with pytest.raises(TypeError):
        TraceAnalysis.select(None, 'asia')


def test_trace_analysis_select_type_error_3():  # bad arg types
    with pytest.raises(TypeError):
        TraceAnalysis.select('HC_N_1', 43)
    with pytest.raises(TypeError):
        TraceAnalysis.select(['HC_N_2'], 'asia')
    with pytest.raises(TypeError):
        TraceAnalysis.select(['HC_N_2'], True)
    with pytest.raises(TypeError):
        TraceAnalysis.select(-37.2, 'asia')


def test_trace_analysis_select_value_error_1():  # bad series/network
    with pytest.raises(ValueError):
        TraceAnalysis.select('HC_N_1', 'unknown')
    with pytest.raises(ValueError):
        TraceAnalysis.select('UNKNOWN', 'asia')


def test_trace_analysis_select_cancer_ok_1():  # should select 4th trace
    selected = TraceAnalysis.select('HC_N_1', 'cancer',
                                    TESTDATA_DIR + '/experiments')
    assert isinstance(selected, TraceAnalysis)
    assert dicts_same(dict1=selected.summary, strict=False,
                      dict2={'iter': 3, 'time': 0.5, 'p': 0.667, 'r': 0.50,
                             'shd-s': 0.50, 'f1': 0.571})
    print('\nSelected Cancer trace is:{}'.format(selected))


def test_trace_analysis_select_asia_ok_1():  # should select 2nd trace
    selected = TraceAnalysis.select('HC_N_1', 'asia',
                                    TESTDATA_DIR + '/experiments')
    assert isinstance(selected, TraceAnalysis)
    assert dicts_same(dict1=selected.summary, strict=False,
                      dict2={'iter': 5, 'time': 2.2, 'p': 0.4, 'r': 0.25,
                             'shd-s': 0.88, 'f1': 0.308})
    print('\nSelected Asia trace is:{}'.format(selected))
