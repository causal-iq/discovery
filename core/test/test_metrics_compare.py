
#   Test standard structural metrics

import pytest

from core.graph import PDAG, DAG
from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_compare
from fileio.common import TESTDATA_DIR
import fileio.bayesys as bayesys
import testdata.example_dags as ex_dag
import testdata.example_sdgs as ex_sdg

TRUE = TESTDATA_DIR + '/noisy/Graphs true/DAGs/DAGtrue_{}.csv'
LEARNT = TESTDATA_DIR + '/noisy/Graphs learned/{0:}/{1:}/' \
    + 'DAGlearned_{1:}_{0:}_N_{2:}k.csv'


# returns a baseline set of metrics
@pytest.fixture
def expected():
    return {'arc_matched': 0, 'arc_reversed': 0, 'edge_not_arc': 0,
            'arc_not_edge': 0, 'edge_matched': 0, 'arc_extra': 0,
            'edge_extra': 0, 'arc_missing': 0, 'edge_missing': 0,
            'missing_matched': 0, 'shd': 0, 'p': None, 'r': None, 'f1': 0.0}


# returns a baseline set of metrics including detailed edge metrics
@pytest.fixture
def expected2():
    return {'arc_matched': 0, 'arc_reversed': 0, 'edge_not_arc': 0,
            'arc_not_edge': 0, 'edge_matched': 0, 'arc_extra': 0,
            'edge_extra': 0, 'arc_missing': 0, 'edge_missing': 0,
            'missing_matched': 0, 'shd': 0, 'p': None, 'r': None, 'f1': 0.0,
            'edges': {'arc_matched': set(), 'arc_reversed': set(),
                      'edge_not_arc': set(), 'arc_not_edge': set(),
                      'edge_matched': set(), 'arc_extra': set(),
                      'edge_extra': set(), 'arc_missing': set(),
                      'edge_missing': set()}}


# helper to print out SHD results
@pytest.fixture
def print_shd():
    def _method(desc, metrics):
        print('{} SHD: standard: {:3d}, bayesys: {:5.1f}'
              .format(desc, metrics['shd'], metrics['shd-b']))
    return _method


# --- Failure cases

# bad argument type
def test_compared_to_type_error1():
    with pytest.raises(TypeError):
        ex_dag.empty().compared_to()
    with pytest.raises(TypeError):
        ex_dag.empty().compared_to(37)
    with pytest.raises(TypeError):
        ex_dag.empty().compared_to('bad arg type')
    with pytest.raises(TypeError):
        ex_dag.empty().compared_to(ex_sdg.ab())


# --- comparisons between simple internal test graphs

# empty to itself
def test_compared_to_empty_ok1(expected):
    metrics = ex_dag.empty().compared_to(ex_dag.empty())
    print('\nComparing empty with itself:\n{}\n'.format(metrics))
    assert metrics == expected


# empty to itself, with edge details
def test_compared_to_empty_ok2(expected2):
    metrics = ex_dag.empty().compared_to(ex_dag.empty(), identify_edges=True)
    print('\nComparing empty with itself:\n{}\n'.format(metrics))
    assert metrics == expected2


# single node to itself
@requires_r_and_bnlearn
def test_compared_to_a_ok1(expected):
    dag = ex_dag.a()
    metrics = dag.compared_to(dag)
    print('\nComparing A with itself:\n{}\n'.format(metrics))
    assert metrics == expected

    bnlearn = bnlearn_compare(dag, dag)
    assert metrics['shd'] == bnlearn['shd']
    if bnlearn['tp'] + bnlearn['fp'] == 0:
        assert metrics['p'] is None
    else:
        assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fp'])
    if bnlearn['tp'] + bnlearn['fn'] == 0:
        assert metrics['r'] is None
    else:
        assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fn'])


# single node to itself, with edge details
def test_compared_to_a_ok2(expected2):
    dag = ex_dag.a()
    metrics = dag.compared_to(dag, identify_edges=True)
    print('\nComparing A with itself:\n{}\n'.format(metrics))
    assert metrics == expected2


# A -> B with itself
@requires_r_and_bnlearn
def test_compared_to_ab_ok1(expected):
    dag1 = ex_dag.ab()
    metrics = dag1.compared_to(dag1)
    expected2 = dict(expected)
    expected.update({'arc_matched': 1, 'p': 1.0, 'r': 1.0, 'f1': 1.0})
    assert metrics == expected  # compare the DAGs

    cpdag1 = PDAG.toCPDAG(dag1)
    metrics2 = cpdag1.compared_to(cpdag1)
    expected2.update({'edge_matched': 1, 'p': 1.0, 'r': 1.0, 'f1': 1.0})
    print('\nComparing DAG A->B with itself:\n{}\n .. and CPDAGs:\n{}\n'
          .format(metrics, metrics2))
    assert metrics2 == expected2  # compare the CPDAGs

    bnlearn = bnlearn_compare(dag1, dag1)
    assert metrics2['shd'] == bnlearn['shd']
    if bnlearn['tp'] + bnlearn['fp'] == 0:
        assert metrics['p'] is None
    else:
        assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fp'])
    if bnlearn['tp'] + bnlearn['fn'] == 0:
        assert metrics['r'] is None
    else:
        assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fn'])


# A -> B with itself
def test_compared_to_ab_ok2(expected2):
    dag1 = ex_dag.ab()
    metrics = dag1.compared_to(dag1, identify_edges=True)
    expected2.update({'arc_matched': 1, 'p': 1.0, 'r': 1.0, 'f1': 1.0})
    expected2['edges'].update({'arc_matched': {('A', 'B')}})
    assert metrics == expected2  # compare the DAGs


# A -> B with A <- B
@requires_r_and_bnlearn
def test_compared_to_ab_ok3(expected):
    dag1 = ex_dag.ab()
    dag2 = ex_dag.ba()
    metrics = dag1.compared_to(dag2)
    expected2 = dict(expected)
    expected.update({'arc_reversed': 1, 'shd': 1, 'p': 0.0, 'r': 0.0})
    assert metrics == expected  # compare the DAGs

    cpdag1 = PDAG.toCPDAG(dag1)
    cpdag2 = PDAG.toCPDAG(dag2)
    metrics2 = cpdag1.compared_to(cpdag2)
    expected2.update({'edge_matched': 1, 'p': 1.0, 'r': 1.0, 'f1': 1.0})
    print('\nComparing DAG A->B with A<-B:\n{}\n .. and CPDAGs:\n{}\n'
          .format(metrics, metrics2))
    assert metrics2 == expected2  # compare the CPDAGs

    bnlearn = bnlearn_compare(dag1, dag2)
    assert metrics2['shd'] == bnlearn['shd']
    if bnlearn['tp'] + bnlearn['fp'] == 0:
        assert metrics['p'] is None
    else:
        assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fp'])
    if bnlearn['tp'] + bnlearn['fn'] == 0:
        assert metrics['r'] is None
    else:
        assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fn'])


# A -> B with A <- B
def test_compared_to_ab_ok4(expected2):
    dag1 = ex_dag.ab()
    dag2 = ex_dag.ba()
    metrics = dag1.compared_to(dag2, identify_edges=True)
    expected2.update({'arc_reversed': 1, 'shd': 1, 'p': 0.0, 'r': 0.0})
    expected2['edges'].update({'arc_reversed': {('A', 'B')}})
    assert metrics == expected2  # compare the DAGs


# A -> B with A <- B
def test_compared_to_ab_ok5(expected2):
    dag1 = ex_dag.ba()
    dag2 = ex_dag.ab()
    metrics = dag1.compared_to(dag2, identify_edges=True)
    expected2.update({'arc_reversed': 1, 'shd': 1, 'p': 0.0, 'r': 0.0})
    expected2['edges'].update({'arc_reversed': {('B', 'A')}})
    assert metrics == expected2  # compare the DAGs


# A -> B <- C with A  B -> C
@requires_r_and_bnlearn
def test_compared_to_abc_ok1(expected2):
    dag1 = DAG(['A', 'B', 'C'], [('A', '->', 'B'), ('C', '->', 'B')])
    dag2 = DAG(['A', 'B', 'C'], [('B', '->', 'C')])

    # dag1 has 1 extra, 1 reversed arc & shd=2 compared to dag2

    metrics = dag1.compared_to(dag2)
    assert metrics == {'arc_matched': 0, 'arc_reversed': 1, 'edge_not_arc': 0,
                       'arc_not_edge': 0, 'edge_matched': 0, 'arc_extra': 1,
                       'edge_extra': 0, 'arc_missing': 0, 'edge_missing': 0,
                       'missing_matched': 1, 'shd': 2, 'p': 0.0, 'r': 0.0,
                       'f1': 0.0}

    # dag1 has 1 extra, 1 arc not edge & shd=2 compared to dag2

    cpdag1 = PDAG.toCPDAG(dag1)
    cpdag2 = PDAG.toCPDAG(dag2)
    metrics2 = cpdag1.compared_to(cpdag2)
    assert metrics2 == {'arc_matched': 0, 'arc_reversed': 0, 'edge_not_arc': 0,
                        'arc_not_edge': 1, 'edge_matched': 0, 'arc_extra': 1,
                        'edge_extra': 0, 'arc_missing': 0, 'edge_missing': 0,
                        'missing_matched': 1, 'shd': 2, 'p': 0.0, 'r': 0.0,
                        'f1': 0.0}

    # when converted to CPDAGs dag 1 has two extra arcs (fp=2) and one missing
    # edge (fn=1), SHD is 2 because extra arc and arc_not_edge

    bnlearn = bnlearn_compare(dag1, dag2)
    assert bnlearn == {'tp': 0, 'fp': 2, 'fn': 1, 'shd': 2}


# 2>1<3<2<4 & 2<1<3>2<4
@requires_r_and_bnlearn
def test_compared_to_and4_12_13_ok1(expected):
    dag1 = ex_dag.and4_12()
    dag2 = ex_dag.and4_13()
    metrics = dag1.compared_to(dag2)
    expected2 = dict(expected)
    expected.update({'arc_reversed': 2, 'arc_matched': 2, 'missing_matched': 2,
                     'shd': 2, 'p': 0.5, 'r': 0.5, 'f1': 0.5})
    assert metrics == expected  # compare the DAGs

    cpdag1 = PDAG.toCPDAG(dag1)
    cpdag2 = PDAG.toCPDAG(dag2)
    metrics2 = cpdag1.compared_to(cpdag2)
    expected2.update({'edge_matched': 1, 'edge_not_arc': 3,
                      'missing_matched': 2, 'shd': 3, 'p': 0.25, 'r': 0.25,
                      'f1': 0.25})
    print('\nComparing DAG and4_12 with and4_13:\n{}\n .. and CPDAGs:\n{}\n'
          .format(metrics, metrics2))
    assert metrics2 == expected2  # compare the CPDAGs

    bnlearn = bnlearn_compare(dag1, dag2)
    assert metrics2['shd'] == bnlearn['shd']
    if bnlearn['tp'] + bnlearn['fp'] == 0:
        assert metrics['p'] is None
    else:
        assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fp'])
    if bnlearn['tp'] + bnlearn['fn'] == 0:
        assert metrics['r'] is None
    else:
        assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fn'])


# 2>1<3<2<4 & 2<1<3>2<4
def test_compared_to_and4_12_13_ok2(expected2):
    dag1 = ex_dag.and4_12()
    dag2 = ex_dag.and4_13()
    metrics = dag1.compared_to(dag2, identify_edges=True)
    expected2.update({'arc_reversed': 2, 'arc_matched': 2,
                      'missing_matched': 2, 'shd': 2, 'p': 0.5, 'r': 0.5,
                      'f1': 0.5})
    expected2['edges'].update({'arc_matched': {('X3', 'X1'), ('X4', 'X2')},
                               'arc_reversed': {('X2', 'X1'), ('X2', 'X3')}})
    assert metrics == expected2  # compare the DAGs


# 1>2<3 4 & 2<4>3>1>2, 4>1
@requires_r_and_bnlearn
def test_compared_to_and4_5_17_ok1(expected):
    dag1 = ex_dag.and4_5()
    dag2 = ex_dag.and4_17()
    metrics = dag1.compared_to(dag2)
    expected2 = dict(expected)
    expected.update({'arc_matched': 1, 'arc_missing': 4, 'arc_extra': 1,
                     'shd': 5, 'p': 0.5, 'r': 0.2, 'f1': 0.2 / 0.7})
    assert metrics == expected

    cpdag1 = PDAG.toCPDAG(dag1)
    cpdag2 = PDAG.toCPDAG(dag2)
    metrics2 = cpdag1.compared_to(cpdag2)
    expected2.update({'arc_not_edge': 1, 'arc_extra': 1, 'edge_missing': 4,
                      'shd': 6, 'p': 0.0, 'r': 0.0})
    print('\nComparing DAG and4_5 with and4_17:\n{}\n .. and CPDAGs:\n{}\n'
          .format(metrics, metrics2))
    assert metrics2 == expected2  # compare the CPDAGs

    bnlearn = bnlearn_compare(dag1, dag2)
    assert metrics2['shd'] == bnlearn['shd']
    if bnlearn['tp'] + bnlearn['fp'] == 0:
        assert metrics['p'] is None
    else:
        assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fp'])
    if bnlearn['tp'] + bnlearn['fn'] == 0:
        assert metrics['r'] is None
    else:
        assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fn'])


# 1>2<3 4 & 2<4>3>1>2, 4>1
def test_compared_to_and4_5_17_ok2(expected2):
    dag1 = ex_dag.and4_5()
    dag2 = ex_dag.and4_17()
    metrics = dag1.compared_to(dag2, identify_edges=True)
    expected2.update({'arc_matched': 1, 'arc_missing': 4, 'arc_extra': 1,
                      'shd': 5, 'p': 0.5, 'r': 0.2, 'f1': 0.2 / 0.7})
    expected2['edges'].update({'arc_matched': {('X1', 'X2')},
                               'arc_missing': {('X4', 'X2'), ('X4', 'X3'),
                                               ('X3', 'X1'), ('X4', 'X1')},
                               'arc_extra': {('X3', 'X2')}})
    assert metrics == expected2


# --- Larger graph shd comparisons with bnlearn

# d7a-fges & d7a-tabu
@requires_r_and_bnlearn
def test_compared_to_dhs1():
    dag1 = bayesys.read(TESTDATA_DIR + '/dhs/d7a/d7a-fges.csv')
    dag2 = bayesys.read(TESTDATA_DIR + '/dhs/d7a/d7a-tabu.csv')
    metrics = dag1.compared_to(dag2, bayesys='v1.5+')
    assert metrics['shd'] == 56
    assert metrics['p'] == 69 / (23 + 14 + 69)
    assert metrics['r'] == 69 / (23 + 19 + 69)
    assert metrics['f1'] == 2 * 69 / (23 + 14 + 69 + 23 + 19 + 69)

    cpdag1 = PDAG.toCPDAG(dag1)
    cpdag2 = PDAG.toCPDAG(dag2)
    metrics2 = cpdag1.compared_to(cpdag2, bayesys='v1.5+')
    print('\nComparing d7a-fges & d7a-tabu:\n{}\n .. and CPDAGs:\n{}\n'
          .format(metrics, metrics2))

    bnlearn = bnlearn_compare(dag1, dag2)
    assert metrics2['shd'] == bnlearn['shd']
    if bnlearn['tp'] + bnlearn['fp'] == 0:
        assert metrics['p'] is None
    else:
        assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fp'])
    if bnlearn['tp'] + bnlearn['fn'] == 0:
        assert metrics['r'] is None
    else:
        assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fn'])


# d8atr_fges cf d8atr_fges3
@requires_r_and_bnlearn
def test_compared_to_dhs2():
    dag1 = bayesys.read(TESTDATA_DIR + '/dhs/d8atr/d8atr-fges.csv')
    dag2 = bayesys.read(TESTDATA_DIR + '/dhs/d8atr/d8atr-fges3.csv')
    metrics = dag1.compared_to(dag2, bayesys='v1.5+')

    cpdag1 = PDAG.toCPDAG(dag1)
    cpdag2 = PDAG.toCPDAG(dag2)
    metrics2 = cpdag1.compared_to(cpdag2, bayesys='v1.5+')
    print('\nComparing d8atr-fges with d8atr-fges3:\n{}\n .. and CPDAGs:\n{}\n'
          .format(metrics, metrics2))

    bnlearn = bnlearn_compare(dag1, dag2)
    assert metrics2['shd'] == bnlearn['shd']
    if bnlearn['tp'] + bnlearn['fp'] == 0:
        assert metrics['p'] is None
    else:
        assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fp'])
    if bnlearn['tp'] + bnlearn['fn'] == 0:
        assert metrics['r'] is None
    else:
        assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] + bnlearn['fn'])


# ASIA: learnt against true
@pytest.mark.slow
@requires_r_and_bnlearn
def test_compared_to_asia(print_shd):
    print('\n\nSHD for ASIA learnt against true graphs')
    for algo in ['GS', 'HC', 'TABU']:
        for size in ['0.1', '1', '10', '100', '1000']:
            true = bayesys.read(TRUE.format('ASIA'))
            learnt = bayesys.read(LEARNT.format('ASIA', algo, size))
            learnt = DAG(true.nodes,
                         [(e[0], '->', e[1]) for e in learnt.edges.keys()])

            metrics = learnt.compared_to(true, bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k   DAG'.format(algo, size), metrics)

            true_cpdag = PDAG.toCPDAG(true)
            learnt_cpdag = PDAG.toCPDAG(learnt)
            metrics_cpdag = learnt_cpdag.compared_to(true_cpdag,
                                                     bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k CPDAG'.format(algo, size), metrics_cpdag)

            bnlearn = bnlearn_compare(learnt, true)
            assert metrics_cpdag['shd'] == bnlearn['shd']
            if bnlearn['tp'] + bnlearn['fp'] == 0:
                assert metrics['p'] is None
            else:
                assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fp'])
            if bnlearn['tp'] + bnlearn['fn'] == 0:
                assert metrics['r'] is None
            else:
                assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fn'])


# SPORTS: learnt against true
@pytest.mark.slow
@requires_r_and_bnlearn
def test_compared_to_sports(print_shd):
    print('\n\nSHD for SPORTS learnt against true graphs')
    for algo in ['GS', 'HC', 'TABU']:
        for size in ['0.1', '1', '10', '100', '1000']:
            true = bayesys.read(TRUE.format('SPORTS'))
            learnt = bayesys.read(LEARNT.format('SPORTS', algo, size))
            learnt = DAG(true.nodes,
                         [(e[0], '->', e[1]) for e in learnt.edges.keys()])

            metrics = learnt.compared_to(true, bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k   DAG'.format(algo, size), metrics)

            true_cpdag = PDAG.toCPDAG(true)
            learnt_cpdag = PDAG.toCPDAG(learnt)
            metrics_cpdag = learnt_cpdag.compared_to(true_cpdag,
                                                     bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k CPDAG'.format(algo, size), metrics_cpdag)

            bnlearn = bnlearn_compare(learnt, true)
            assert metrics_cpdag['shd'] == bnlearn['shd']
            if bnlearn['tp'] + bnlearn['fp'] == 0:
                assert metrics['p'] is None
            else:
                assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fp'])
            if bnlearn['tp'] + bnlearn['fn'] == 0:
                assert metrics['r'] is None
            else:
                assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fn'])


# ALARM: learnt against true
@pytest.mark.slow
@requires_r_and_bnlearn
def test_compared_to_alarm(print_shd):
    print('\n\nSHD for ALARM learnt against true graphs')
    for algo in ['GS', 'HC', 'TABU']:
        for size in ['0.1', '1', '10', '100', '1000']:
            true = bayesys.read(TRUE.format('ALARM'))
            learnt = bayesys.read(LEARNT.format('ALARM', algo, size))
            learnt = DAG(true.nodes,
                         [(e[0], '->', e[1]) for e in learnt.edges.keys()])

            metrics = learnt.compared_to(true, bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k   DAG'.format(algo, size), metrics)

            true_cpdag = PDAG.toCPDAG(true)
            learnt_cpdag = PDAG.toCPDAG(learnt)
            metrics_cpdag = learnt_cpdag.compared_to(true_cpdag,
                                                     bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k CPDAG'.format(algo, size), metrics_cpdag)

            bnlearn = bnlearn_compare(learnt, true)
            assert metrics_cpdag['shd'] == bnlearn['shd']
            if bnlearn['tp'] + bnlearn['fp'] == 0:
                assert metrics['p'] is None
            else:
                assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fp'])
            if bnlearn['tp'] + bnlearn['fn'] == 0:
                assert metrics['r'] is None
            else:
                assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fn'])


# PROPERTY: learnt cf. true
@pytest.mark.slow
@requires_r_and_bnlearn
def test_compared_to_property(print_shd):
    print('\n\nSHD for PROPERTY learnt against true graphs')
    for algo in ['GS', 'HC', 'TABU']:
        for size in ['0.1', '1', '10', '100', '1000']:
            true = bayesys.read(TRUE.format('PROPERTY'))
            learnt = bayesys.read(LEARNT.format('PROPERTY', algo, size))
            learnt = DAG(true.nodes,
                         [(e[0], '->', e[1]) for e in learnt.edges.keys()])

            metrics = learnt.compared_to(true, bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k   DAG'.format(algo, size), metrics)

            true_cpdag = PDAG.toCPDAG(true)
            learnt_cpdag = PDAG.toCPDAG(learnt)
            metrics_cpdag = learnt_cpdag.compared_to(true_cpdag,
                                                     bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k CPDAG'.format(algo, size), metrics_cpdag)

            bnlearn = bnlearn_compare(learnt, true)
            assert metrics_cpdag['shd'] == bnlearn['shd']
            if bnlearn['tp'] + bnlearn['fp'] == 0:
                assert metrics['p'] is None
            else:
                assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fp'])
            if bnlearn['tp'] + bnlearn['fn'] == 0:
                assert metrics['r'] is None
            else:
                assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fn'])


# FORMED: learnt against true
@pytest.mark.slow
@requires_r_and_bnlearn
def test_compared_to_formed(print_shd):
    print('\n\nSHD for FORMED learnt against true graphs')
    for algo in ['GS', 'HC', 'TABU']:
        for size in ['0.1', '1', '10', '100', '1000']:
            true = bayesys.read(TRUE.format('FORMED'))
            learnt = bayesys.read(LEARNT.format('FORMED', algo, size))
            learnt = DAG(true.nodes,
                         [(e[0], '->', e[1]) for e in learnt.edges.keys()])

            metrics = learnt.compared_to(true, bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k   DAG'.format(algo, size), metrics)

            true_cpdag = PDAG.toCPDAG(true)
            learnt_cpdag = PDAG.toCPDAG(learnt)
            metrics_cpdag = learnt_cpdag.compared_to(true_cpdag,
                                                     bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k CPDAG'.format(algo, size), metrics_cpdag)

            bnlearn = bnlearn_compare(learnt, true)
            assert metrics_cpdag['shd'] == bnlearn['shd']
            if bnlearn['tp'] + bnlearn['fp'] == 0:
                assert metrics['p'] is None
            else:
                assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fp'])
            if bnlearn['tp'] + bnlearn['fn'] == 0:
                assert metrics['r'] is None
            else:
                assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fn'])


# PATHFINDER
@pytest.mark.slow
@requires_r_and_bnlearn
def test_compared_to_pathfinder(print_shd):
    print('\n\nSHD for PATHFINDER learnt against true graphs')
    for algo in ['GS', 'HC', 'TABU']:
        for size in ['0.1', '1', '10', '100', '1000']:
            true = bayesys.read(TRUE.format('PATHFINDER'))
            learnt = bayesys.read(LEARNT.format('PATHFINDER', algo, size))
            learnt = DAG(true.nodes,
                         [(e[0], '->', e[1]) for e in learnt.edges.keys()])

            metrics = learnt.compared_to(true, bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k   DAG'.format(algo, size), metrics)

            true_cpdag = PDAG.toCPDAG(true)
            learnt_cpdag = PDAG.toCPDAG(learnt)
            metrics_cpdag = learnt_cpdag.compared_to(true_cpdag,
                                                     bayesys='v1.5+')
            print_shd('{:>4s} {:>4s}k CPDAG'.format(algo, size), metrics_cpdag)

            bnlearn = bnlearn_compare(learnt, true)
            assert metrics_cpdag['shd'] == bnlearn['shd']
            if bnlearn['tp'] + bnlearn['fp'] == 0:
                assert metrics['p'] is None
            else:
                assert metrics['p'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fp'])
            if bnlearn['tp'] + bnlearn['fn'] == 0:
                assert metrics['r'] is None
            else:
                assert metrics['r'] == bnlearn['tp'] / (bnlearn['tp'] +
                                                        bnlearn['fn'])
