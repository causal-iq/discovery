
#   Test calling the bnlearn with knowledge constraints

import pytest

from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_learn
from fileio.common import TESTDATA_DIR
from fileio.bayesys import read_constraints
from fileio.numpy import NumPy
from core.common import EdgeType
from core.graph import DAG
from core.bn import BN
from learn.knowledge import Knowledge
from learn.knowledge_rule import RuleSet


# Generate 10 categorical rows for A --> B
@pytest.fixture(scope="module")
def ab10():
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    data = NumPy.from_df(df=bn.generate_cases(10), dstype='categorical',
                         keep_df=False)
    return (data, bn)


# --- Failure cases

# bad Knowledge arg type
def test_reqd_type_error_1_(ab10):
    with pytest.raises(TypeError):
        bnlearn_learn('hc', ab10[0], knowledge=2)
    with pytest.raises(TypeError):
        bnlearn_learn('hc', ab10[0], knowledge=False)


# only reqd & tiers knowledge supported
def test_reqd_value_error_1_(ab10):
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': tuple([True])})
    with pytest.raises(ValueError):
        bnlearn_learn('hc', ab10[0], knowledge=knowledge)


# --- Successful learning cases

# A --> B data, No knowledge
@requires_r_and_bnlearn
def test_reqd_hc_ab_1_ok(ab10):
    dag, trace = bnlearn_learn('hc', ab10[0],
                               context={'in': 'in', 'id': 'hc_ab_1'})

    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')

    # DAG initialised to A --> B, and no changes made

    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -15.141340, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[1] == {'activity': 'add', 'arc': ('A', 'B'),
                         'delta/score': 0.798467, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[2] == {'activity': 'stop', 'arc': None,
                         'delta/score': -14.342880, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert trace.result == dag


# A --> B data, A --> B required
@requires_r_and_bnlearn
def test_reqd_hc_ab_2_ok(ab10):
    knowledge = Knowledge(rules=RuleSet.REQD_ARC,
                          params={'reqd': {('A', 'B'): True},
                                  'initial': ab10[1].dag})
    dag, trace = bnlearn_learn('hc', ab10[0], knowledge=knowledge,
                               context={'in': 'in', 'id': 'hc_ab_2'})

    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')

    # DAG initialised to A --> B, and no changes made

    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -14.342880, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[1] == {'activity': 'stop', 'arc': None,
                         'delta/score': -14.342880, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert trace.result == dag


# A --> B data, B --> A required
@requires_r_and_bnlearn
def test_reqd_hc_ab_3_ok(ab10):
    knowledge = Knowledge(rules=RuleSet.REQD_ARC,
                          params={'reqd': {('B', 'A'): True},
                                  'initial': ab10[1].dag})
    dag, trace = bnlearn_learn('hc', ab10[0], knowledge=knowledge,
                               context={'in': 'in', 'id': 'hc_ab_3'})

    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A|B][B]'  # HC learns correct answer
    print(trace)
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')

    # DAG initialised to A --> B, and no changes made

    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -14.342880, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[1] == {'activity': 'stop', 'arc': None,
                         'delta/score': -14.342880, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert trace.result == dag


# Cancer, no knowledge
@requires_r_and_bnlearn
def test_reqd_hc_cancer_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    dag, trace = bnlearn_learn('hc', data,
                               context={'in': 'in', 'id': 'hc_cancer_1'})

    print('\nDAG learnt from 1K rows of Cancer: {}'.format(dag))
    assert dag.to_string() == \
        '[Cancer][Dyspnoea|Cancer][Pollution][Smoker|Cancer][Xray|Cancer]'
    print(trace)
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')

    # DAG initialised to A --> B, and no changes made

    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -2143.494000, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[1] == {'activity': 'add', 'arc': ('Cancer', 'Xray'),
                         'delta/score': 14.757857, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[2] == {'activity': 'add', 'arc': ('Cancer', 'Smoker'),
                         'delta/score': 8.843660, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[3] == {'activity': 'add', 'arc': ('Cancer', 'Dyspnoea'),
                         'delta/score': 1.718316, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[4] == {'activity': 'stop', 'arc': None,
                         'delta/score': -2118.174000, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert trace.result == dag


# PC, Cancer data, No knowledge
@requires_r_and_bnlearn
def test_reqd_pc_cancer_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    pdag, trace = bnlearn_learn('pc.stable', data,
                                context={'in': 'in', 'id': 'pc_cancer_1'})

    print('\nPDAG learnt from 1K rows of Cancer: {}'.format(pdag))
    assert pdag.edges == \
        {('Smoker', 'Cancer'): EdgeType.DIRECTED,
         ('Xray', 'Cancer'): EdgeType.DIRECTED,
         ('Dyspnoea', 'Cancer'): EdgeType.DIRECTED}


# MMHC, Cancer data, No knowledge
@requires_r_and_bnlearn
def test_reqd_mmhc_cancer_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    pdag, trace = bnlearn_learn('mmhc', data,
                                context={'in': 'in', 'id': 'mmhc_cancer_1'})

    print('\nPDAG learnt from 1K rows of Cancer: {}'.format(pdag))
    assert pdag.edges == \
        {('Cancer', 'Smoker'): EdgeType.DIRECTED,
         ('Cancer', 'Xray'): EdgeType.DIRECTED,
         ('Cancer', 'Dyspnoea'): EdgeType.DIRECTED}


# H2PC, Cancer data, No knowledge
@requires_r_and_bnlearn
def test_reqd_h2pc_cancer_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    pdag, trace = bnlearn_learn('h2pc', data,
                                context={'in': 'in', 'id': 'h2pc_cancer_1'})

    print('\nPDAG learnt from 1K rows of Cancer: {}'.format(pdag))
    assert pdag.edges == \
        {('Cancer', 'Smoker'): EdgeType.DIRECTED,
         ('Cancer', 'Xray'): EdgeType.DIRECTED,
         ('Cancer', 'Dyspnoea'): EdgeType.DIRECTED}


# Cancer, S --> C, P --> C required
@requires_r_and_bnlearn
def test_reqd_hc_cancer_2_ok():
    bn = BN.read(TESTDATA_DIR + '/experiments/bn/cancer.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    initial = DAG(bn.dag.nodes, [('Smoker', '->', 'Cancer'),
                                 ('Pollution', '->', 'Cancer')])
    knowledge = Knowledge(rules=RuleSet.REQD_ARC,
                          params={'reqd': {('Smoker', 'Cancer'): True,
                                           ('Pollution', 'Cancer'): True},
                                  'initial': initial})
    dag, trace = bnlearn_learn('hc', data, context={'in': 'in',
                               'id': 'hc_cancer_2'}, knowledge=knowledge)

    print('\nDAG learnt from 1K rows of Cancer: {}'.format(dag))
    print(dag.to_string())
    assert dag.to_string() == \
        ('[Cancer|Pollution:Smoker][Dyspnoea|Cancer]' +
         '[Pollution][Smoker][Xray|Cancer]')
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')

    # DAG initialised to A --> B, and no changes made

    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -2137.683000, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[1] == {'activity': 'add', 'arc': ('Cancer', 'Xray'),
                         'delta/score': 14.757857, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[2] == {'activity': 'add', 'arc': ('Cancer', 'Dyspnoea'),
                         'delta/score': 1.718316, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[3] == {'activity': 'stop', 'arc': None,
                         'delta/score': -2121.207000, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert trace.result == dag


# PC, Cancer data, No knowledge
@requires_r_and_bnlearn
def test_reqd_pc_cancer_2_ok():
    bn = BN.read(TESTDATA_DIR + '/experiments/bn/cancer.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    initial = DAG(bn.dag.nodes, [('Smoker', '->', 'Cancer'),
                                 ('Pollution', '->', 'Cancer')])
    knowledge = Knowledge(rules=RuleSet.REQD_ARC,
                          params={'reqd': {('Smoker', 'Cancer'): True,
                                           ('Pollution', 'Cancer'): True},
                                  'initial': initial})
    pdag, _ = bnlearn_learn('pc.stable', data, knowledge=knowledge,
                            context={'in': 'in', 'id': 'pc_cancer_2'})
    print('\nPDAG learnt by PC from 1K rows of Cancer: {}'.format(pdag))
    assert pdag.edges == \
        {('Smoker', 'Cancer'): EdgeType.DIRECTED,
         ('Pollution', 'Cancer'): EdgeType.DIRECTED,
         ('Xray', 'Cancer'): EdgeType.DIRECTED}


# MMHC, Cancer data, 2 reqd
@requires_r_and_bnlearn
def test_reqd_mmhc_cancer_2_ok():
    bn = BN.read(TESTDATA_DIR + '/experiments/bn/cancer.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    initial = DAG(bn.dag.nodes, [('Smoker', '->', 'Cancer'),
                                 ('Pollution', '->', 'Cancer')])
    knowledge = Knowledge(rules=RuleSet.REQD_ARC,
                          params={'reqd': {('Smoker', 'Cancer'): True,
                                           ('Pollution', 'Cancer'): True},
                                  'initial': initial})
    pdag, _ = bnlearn_learn('mmhc', data, knowledge=knowledge,
                            context={'in': 'in', 'id': 'pc_cancer_2'})

    print('\nPDAG learnt by MMHC from 1K rows of Cancer: {}'.format(pdag))
    assert pdag.edges == \
        {('Cancer', 'Xray'): EdgeType.DIRECTED,
         ('Pollution', 'Cancer'): EdgeType.DIRECTED,
         ('Smoker', 'Cancer'): EdgeType.DIRECTED}


# H2PC, Cancer data, 2 reqd
@requires_r_and_bnlearn
def test_reqd_h2pc_cancer_2_ok():
    bn = BN.read(TESTDATA_DIR + '/experiments/bn/cancer.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    initial = DAG(bn.dag.nodes, [('Smoker', '->', 'Cancer'),
                                 ('Pollution', '->', 'Cancer')])
    knowledge = Knowledge(rules=RuleSet.REQD_ARC,
                          params={'reqd': {('Smoker', 'Cancer'): True,
                                           ('Pollution', 'Cancer'): True},
                                  'initial': initial})

    # This fails - suspect bug in bnlearn

    with pytest.raises(RuntimeError):
        bnlearn_learn('h2pc', data, knowledge=knowledge,
                      context={'in': 'in', 'id': 'h2pc_cancer_2'})


# H2PC, Cancer data, 1 reqd
@requires_r_and_bnlearn
def test_reqd_h2pc_cancer_3_ok():
    bn = BN.read(TESTDATA_DIR + '/experiments/bn/cancer.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    initial = DAG(bn.dag.nodes, [('Smoker', '->', 'Cancer'),
                                 ('Pollution', '->', 'Cancer')])
    knowledge = Knowledge(rules=RuleSet.REQD_ARC,
                          params={'reqd': {('Pollution', 'Cancer'): True},
                                  'initial': initial})
    pdag, _ = bnlearn_learn('h2pc', data, knowledge=knowledge,
                            context={'in': 'in', 'id': 'h2pc_cancer_2'})

    print('\nPDAG learnt by H2PC from 1K rows of Cancer: {}'.format(pdag))
    assert pdag.edges == \
        {('Cancer', 'Xray'): EdgeType.DIRECTED,
         ('Cancer', 'Dyspnoea'): EdgeType.DIRECTED,
         ('Pollution', 'Cancer'): EdgeType.DIRECTED,
         ('Smoker', 'Cancer'): EdgeType.DIRECTED}


# HC, Sports, no knowledge
@requires_r_and_bnlearn
def test_reqd_hc_sports_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    dag, trace = bnlearn_learn('hc', data, context={'in': 'in',
                               'id': 'hc_sports_1'})

    print('\nDAG learnt by HC from 1K rows of Sports: {}'.format(dag))
    print(trace)
    assert dag.edges == \
        {('RDlevel', 'possession'): EdgeType.DIRECTED,
         ('HTshotOnTarget', 'HTshots'): EdgeType.DIRECTED,
         ('HDA', 'HTgoals'): EdgeType.DIRECTED,
         ('possession', 'ATshots'): EdgeType.DIRECTED,
         ('HTgoals', 'HTshotOnTarget'): EdgeType.DIRECTED,
         ('ATgoals', 'HTgoals'): EdgeType.DIRECTED,
         ('HDA', 'RDlevel'): EdgeType.DIRECTED,
         ('ATgoals', 'HDA'): EdgeType.DIRECTED,
         ('ATshots', 'ATshotsOnTarget'): EdgeType.DIRECTED}


# Sports, 9 required arcs
@requires_r_and_bnlearn
def test_reqd_hc_sports_7_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    constraints = TESTDATA_DIR + '/bayesys/constraintsDirected_SPORTS_7.csv'
    knowledge = read_constraints(constraints, set(bn.dag.nodes))
    dag, _ = bnlearn_learn('hc', data, knowledge=knowledge,
                           context={'in': 'in', 'id': 'hc_sports_1'})

    print('\nDAG learnt from 1K rows of Sports: {}'.format(dag))
    assert dag.edges == \
        {('ATshots', 'ATshotsOnTarget'): EdgeType.DIRECTED,
         ('HTshots', 'HTshotOnTarget'): EdgeType.DIRECTED,
         ('HTgoals', 'HDA'): EdgeType.DIRECTED,
         ('possession', 'ATshots'): EdgeType.DIRECTED,
         ('HTshotOnTarget', 'HTgoals'): EdgeType.DIRECTED,
         ('ATgoals', 'HDA'): EdgeType.DIRECTED,
         ('RDlevel', 'possession'): EdgeType.DIRECTED,
         ('possession', 'HTshots'): EdgeType.DIRECTED,
         ('ATshotsOnTarget', 'ATgoals'): EdgeType.DIRECTED}


# PC, Sports, no knowledge
@requires_r_and_bnlearn
def test_reqd_pc_sports_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    pdag, trace = bnlearn_learn('pc.stable', data, context={'in': 'in',
                                'id': 'pc_sports_1'})

    print('\nPDAG learnt by PC from 1K rows of Sports: {}'.format(pdag))
    assert pdag.edges == \
        {('ATgoals', 'ATshotsOnTarget'): EdgeType.UNDIRECTED,
         ('HTgoals', 'HDA'): EdgeType.DIRECTED,
         ('HTgoals', 'HTshotOnTarget'): EdgeType.UNDIRECTED,
         ('HTshotOnTarget', 'HTshots'): EdgeType.UNDIRECTED,
         ('ATshots', 'ATshotsOnTarget'): EdgeType.UNDIRECTED,
         ('ATgoals', 'HDA'): EdgeType.DIRECTED}


# PC, Sports, 9 required arcs
@requires_r_and_bnlearn
def test_reqd_pc_sports_7_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    constraints = TESTDATA_DIR + '/bayesys/constraintsDirected_SPORTS_7.csv'
    knowledge = read_constraints(constraints, set(bn.dag.nodes))
    pdag, trace = bnlearn_learn('pc.stable', data, knowledge=knowledge,
                                context={'in': 'in', 'id': 'pc_sports_7'})

    print('\nPDAG learnt by PC from 1K rows of Sports: {}'.format(pdag))
    assert pdag.edges == \
        {('HTgoals', 'HDA'): EdgeType.DIRECTED,
         ('possession', 'HTshots'): EdgeType.DIRECTED,
         ('possession', 'ATshots'): EdgeType.DIRECTED,
         ('HTshots', 'HTshotOnTarget'): EdgeType.DIRECTED,
         ('ATshotsOnTarget', 'ATgoals'): EdgeType.DIRECTED,
         ('ATshots', 'ATshotsOnTarget'): EdgeType.DIRECTED,
         ('ATgoals', 'HDA'): EdgeType.DIRECTED,
         ('HTshotOnTarget', 'HTgoals'): EdgeType.DIRECTED}


# MMHC, Sports, no knowledge
@requires_r_and_bnlearn
def test_reqd_mmhc_sports_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    pdag, _ = bnlearn_learn('mmhc', data, context={'in': 'in',
                                                   'id': 'mmhc_sports_1'})

    print('\nPDAG learnt by MMHC from 1K rows of Sports: {}'.format(pdag))
    assert pdag.edges == \
        {('HTshotOnTarget', 'HTshots'): EdgeType.DIRECTED,
         ('HTgoals', 'HTshotOnTarget'): EdgeType.DIRECTED,
         ('ATshots', 'ATshotsOnTarget'): EdgeType.DIRECTED,
         ('ATgoals', 'HDA'): EdgeType.DIRECTED,
         ('ATshotsOnTarget', 'ATgoals'): EdgeType.DIRECTED,
         ('RDlevel', 'possession'): EdgeType.DIRECTED,
         ('HTgoals', 'HDA'): EdgeType.DIRECTED}


# MMHC, Sports, 9 required arcs
@requires_r_and_bnlearn
def test_reqd_mmhc_sports_7_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    constraints = TESTDATA_DIR + '/bayesys/constraintsDirected_SPORTS_7.csv'
    knowledge = read_constraints(constraints, set(bn.dag.nodes))
    pdag, _ = bnlearn_learn('mmhc', data, knowledge=knowledge,
                            context={'in': 'in', 'id': 'mmhc_sports_7'})

    print('\nPDAG learnt by MMHC from 1K rows of Sports: {}'.format(pdag))
    assert pdag.edges == \
        {('ATshots', 'ATshotsOnTarget'): EdgeType.DIRECTED,
         ('possession', 'ATshots'): EdgeType.DIRECTED,
         ('HTgoals', 'HDA'): EdgeType.DIRECTED,
         ('ATgoals', 'HDA'): EdgeType.DIRECTED,
         ('possession', 'HTshots'): EdgeType.DIRECTED,
         ('HTshotOnTarget', 'HTgoals'): EdgeType.DIRECTED,
         ('HTshots', 'HTshotOnTarget'): EdgeType.DIRECTED,
         ('ATshotsOnTarget', 'ATgoals'): EdgeType.DIRECTED}


# H2PC, Sports, no knowledge
@requires_r_and_bnlearn
def test_reqd_h2pc_sports_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    pdag, trace = bnlearn_learn('h2pc', data, context={'in': 'in',
                                'id': 'h2pc_sports_1'})

    print('\nPDAG learnt by H2PC from 1K rows of Sports: {}'.format(pdag))
    assert pdag.edges == \
        {('HTshotOnTarget', 'HTshots'): EdgeType.DIRECTED,
         ('ATshots', 'ATshotsOnTarget'): EdgeType.DIRECTED,
         ('ATgoals', 'HDA'): EdgeType.DIRECTED,
         ('RDlevel', 'possession'): EdgeType.DIRECTED,
         ('possession', 'ATshots'): EdgeType.DIRECTED,
         ('HTgoals', 'HDA'): EdgeType.DIRECTED}


# H2PC, Sports, 9 required arcs
@requires_r_and_bnlearn
def test_reqd_h2pc_sports_7_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    constraints = TESTDATA_DIR + '/bayesys/constraintsDirected_SPORTS_7.csv'
    knowledge = read_constraints(constraints, set(bn.dag.nodes))

    # Bug with bnlearn H2PC and specific constraints

    with pytest.raises(RuntimeError):
        bnlearn_learn('h2pc', data, knowledge=knowledge,
                      context={'in': 'in', 'id': 'h2pc_sports_7'})


# H2PC, Sports, 4 required arcs
@requires_r_and_bnlearn
def test_reqd_h2pc_sports_7a_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)

    # commented out required arcs cause R abend

    knowledge = \
        Knowledge(rules=RuleSet.REQD_ARC,
                  params={'reqd': {('ATshots', 'ATshotsOnTarget'): True,
                                   ('ATshotsOnTarget', 'ATgoals'): True,
                                   # ('HTshotsOnTarget', 'HTgoals'): True,
                                   ('HTgoals', 'HDA'): True,
                                   # ('HTshots', 'HTshotsOnTarget'): True,
                                   # ('possession', 'ATshots'): True,
                                   ('possession', 'HTshots'): True},
                          # ('ATgoals', 'HDA'): True},
                          'initial': bn.dag})
    print(knowledge.reqd)
    pdag, _ = bnlearn_learn('h2pc', data, knowledge=knowledge,
                            context={'in': 'in', 'id': 'h2pc_sports_7'})

    print('\nPDAG learnt by H2PC from 1K rows of Sports: {}'.format(pdag))
    assert pdag.edges == \
        {('ATshots', 'ATshotsOnTarget'): EdgeType.DIRECTED,
         ('possession', 'ATshots'): EdgeType.DIRECTED,
         ('HTgoals', 'HDA'): EdgeType.DIRECTED,
         ('ATgoals', 'HDA'): EdgeType.DIRECTED,
         ('possession', 'HTshots'): EdgeType.DIRECTED,
         ('RDlevel', 'possession'): EdgeType.DIRECTED,
         ('HTshots', 'HTshotOnTarget'): EdgeType.DIRECTED,
         ('ATshotsOnTarget', 'ATgoals'): EdgeType.DIRECTED}


# A --> B data, A --> B prohibited
@requires_r_and_bnlearn
def test_stop_hc_ab_2_ok(ab10):
    knowledge = Knowledge(rules=RuleSet.STOP_ARC,
                          params={'stop': {('A', 'B'): True}})
    dag, trace = bnlearn_learn('hc', ab10[0], knowledge=knowledge,
                               context={'in': 'in', 'id': 'stop_hc_ab_2'})

    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A|B][B]'  # HC learns correct answer
    print(trace)
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')

    # Arc B --> A added

    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -15.141340, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[1] == {'activity': 'add', 'arc': ('B', 'A'),
                         'delta/score': 0.798467, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[2] == {'activity': 'stop', 'arc': None,
                         'delta/score': -14.342880, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert trace.result == dag


# A --> B data, B --> A prohibited
@requires_r_and_bnlearn
def test_stop_hc_ab_3_ok(ab10):
    knowledge = Knowledge(rules=RuleSet.STOP_ARC,
                          params={'stop': {('B', 'A'): True}})
    dag, trace = bnlearn_learn('hc', ab10[0], knowledge=knowledge,
                               context={'in': 'in', 'id': 'stop_hc_ab_3'})

    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')

    # DAG initialised to A --> B, and no changes made

    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -15.141340, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[1] == {'activity': 'add', 'arc': ('A', 'B'),
                         'delta/score': 0.798467, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[2] == {'activity': 'stop', 'arc': None,
                         'delta/score': -14.342880, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert trace.result == dag


# Cancer, prohibit C --> S
@requires_r_and_bnlearn
def test_stop_hc_cancer_2_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    knowledge = Knowledge(rules=RuleSet.STOP_ARC,
                          params={'stop': {('Cancer', 'Smoker'): True}})
    dag, trace = bnlearn_learn('hc', data, context={'in': 'in',
                               'id': 'stop_hc_cancer_2'}, knowledge=knowledge)

    print('\nDAG learnt from 1K rows of Cancer: {}'.format(dag))
    print(trace)
    assert dag.to_string() == \
        ('[Cancer|Smoker][Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')

    # DAG initialised to A --> B, and no changes made

    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -2143.494000, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[1] == {'activity': 'add', 'arc': ('Cancer', 'Xray'),
                         'delta/score': 14.757857, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[2] == {'activity': 'add', 'arc': ('Smoker', 'Cancer'),
                         'delta/score': 8.843660, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[3] == {'activity': 'add', 'arc': ('Cancer', 'Dyspnoea'),
                         'delta/score': 1.718316, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[4] == {'activity': 'stop', 'arc': None,
                         'delta/score': -2118.174000, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert trace.result == dag


# PC, Cancer, prohibit D->C, X->C & C->S
@requires_r_and_bnlearn
def test_stop_pc_cancer_2_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    knowledge = Knowledge(rules=RuleSet.STOP_ARC,
                          params={'stop': {('Dyspnoea', 'Cancer'): True,
                                           ('Xray', 'Cancer'): True,
                                           ('Cancer', 'Smoker'): True}})
    pdag, _ = bnlearn_learn('pc.stable', data, knowledge=knowledge,
                            context={'in': 'in', 'id': 'stop_pc_cancer_2'})

    print('\nPDAG learnt by PC from 1K rows of Cancer: {}'.format(pdag))
    assert pdag.edges == \
        {('Smoker', 'Cancer'): EdgeType.DIRECTED,
         ('Cancer', 'Xray'): EdgeType.DIRECTED,
         ('Cancer', 'Dyspnoea'): EdgeType.DIRECTED}


# MMHC, Cancer, stop C->X, C->S
@requires_r_and_bnlearn
def test_stop_mmhc_cancer_2_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    knowledge = Knowledge(rules=RuleSet.STOP_ARC,
                          params={'stop': {('Cancer', 'Xray'): True,
                                           ('Cancer', 'Smoker'): True}})
    pdag, _ = bnlearn_learn('mmhc', data, knowledge=knowledge,
                            context={'in': 'in', 'id': 'stop_mmhc_cancer_2'})

    print('\nPDAG learnt by MMHC from 1K rows of Cancer: {}'.format(pdag))
    assert pdag.edges == \
        {('Cancer', 'Dyspnoea'): EdgeType.DIRECTED,
         ('Smoker', 'Cancer'): EdgeType.DIRECTED,
         ('Xray', 'Cancer'): EdgeType.DIRECTED}


# H2PC, Cancer data, 2 reqd
@requires_r_and_bnlearn
def test_stop_h2pc_cancer_2_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical', N=1000)
    knowledge = Knowledge(rules=RuleSet.STOP_ARC,
                          params={'stop': {('Cancer', 'Smoker'): True}})

    # This fails - suspect bug in bnlearn

    pdag, _ = bnlearn_learn('h2pc', data, knowledge=knowledge,
                            context={'in': 'in', 'id': 'stop_h2pc_cancer_2'})
    print('\nPDAG learnt by H2PC from 1K rows of Cancer: {}'.format(pdag))

    assert pdag.edges == \
        {('Cancer', 'Xray'): EdgeType.DIRECTED,
         ('Cancer', 'Dyspnoea'): EdgeType.DIRECTED,
         ('Smoker', 'Cancer'): EdgeType.DIRECTED}


# Sports, HC, 3 tiers
@requires_r_and_bnlearn
def test_tiers_hc_sports_7_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    constraints = TESTDATA_DIR + '/bayesys/constraintsTemporal_SPORTS_7.csv'
    knowledge = read_constraints(constraints, set(bn.dag.nodes))
    dag, _ = bnlearn_learn('hc', data, knowledge=knowledge,
                           context={'in': 'in', 'id': 'tiers_hc_sports_1'})

    print('\nDAG learnt from 1K rows of Sports: {}'.format(dag))

    assert dag.edges == \
        {('RDlevel', 'possession'): EdgeType.DIRECTED,
         ('ATgoals', 'HDA'): EdgeType.DIRECTED,
         ('HTshotOnTarget', 'HTgoals'): EdgeType.DIRECTED,
         ('ATshots', 'ATshotsOnTarget'): EdgeType.DIRECTED,
         ('ATshotsOnTarget', 'ATgoals'): EdgeType.DIRECTED,
         ('HTshots', 'HTshotOnTarget'): EdgeType.DIRECTED,
         ('possession', 'HTshots'): EdgeType.DIRECTED,
         ('possession', 'ATshots'): EdgeType.DIRECTED,
         ('HTgoals', 'HDA'): EdgeType.DIRECTED}


# PC, Sports, 3 tiers
@requires_r_and_bnlearn
def test_tiers_pc_sports_7_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    constraints = TESTDATA_DIR + '/bayesys/constraintsTemporal_SPORTS_7.csv'
    knowledge = read_constraints(constraints, set(bn.dag.nodes))
    pdag, _ = bnlearn_learn('pc.stable', data, knowledge=knowledge,
                            context={'in': 'in', 'id': 'tiers_pc_sports_7'})

    print('\nPDAG learnt by PC from 1K rows of Sports: {}'.format(pdag))
    assert pdag.edges == \
        {('HTgoals', 'HDA'): EdgeType.DIRECTED,
         ('ATshotsOnTarget', 'ATgoals'): EdgeType.DIRECTED,
         ('ATshots', 'ATshotsOnTarget'): EdgeType.DIRECTED,
         ('HTshotOnTarget', 'HTgoals'): EdgeType.DIRECTED,
         ('HTshots', 'HTshotOnTarget'): EdgeType.DIRECTED,
         ('ATgoals', 'HDA'): EdgeType.DIRECTED}


# MMHC, Sports, 3 tiers
@requires_r_and_bnlearn
def test_tiers_mmhc_sports_7_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    constraints = TESTDATA_DIR + '/bayesys/constraintsTemporal_SPORTS_7.csv'
    knowledge = read_constraints(constraints, set(bn.dag.nodes))
    pdag, _ = bnlearn_learn('mmhc', data, knowledge=knowledge,
                            context={'in': 'in', 'id': 'tiers_mmhc_sports_7'})

    print('\nPDAG learnt by MMHC from 1K rows of Sports: {}'.format(pdag))
    assert pdag.edges == \
        {('ATshots', 'ATshotsOnTarget'): EdgeType.DIRECTED,
         ('HTshotOnTarget', 'HTgoals'): EdgeType.DIRECTED,
         ('ATgoals', 'HDA'): EdgeType.DIRECTED,
         ('possession', 'HTshots'): EdgeType.DIRECTED,
         ('RDlevel', 'possession'): EdgeType.DIRECTED,
         ('ATshotsOnTarget', 'ATgoals'): EdgeType.DIRECTED,
         ('HTgoals', 'HDA'): EdgeType.DIRECTED,
         ('HTshots', 'HTshotOnTarget'): EdgeType.DIRECTED}
    return


# H2PC, Sports, 3 tiers
@requires_r_and_bnlearn
def test_tiers_h2pc_sports_7_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                      dstype='categorical', N=1000)
    constraints = TESTDATA_DIR + '/bayesys/constraintsTemporal_SPORTS_7.csv'
    knowledge = read_constraints(constraints, set(bn.dag.nodes))
    pdag, _ = bnlearn_learn('h2pc', data, knowledge=knowledge,
                            context={'in': 'in', 'id': 'tiers_h2pc_sports_7'})

    print('\nPDAG learnt by H2PC from 1K rows of Sports: {}'.format(pdag))
    assert pdag.edges == \
        {('ATgoals', 'HDA'): EdgeType.DIRECTED,
         ('ATshots', 'ATshotsOnTarget'): EdgeType.DIRECTED,
         ('HTshots', 'HTshotOnTarget'): EdgeType.DIRECTED,
         ('possession', 'HTshots'): EdgeType.DIRECTED,
         ('possession', 'ATshots'): EdgeType.DIRECTED,
         ('RDlevel', 'possession'): EdgeType.DIRECTED,
         ('HTgoals', 'HDA'): EdgeType.DIRECTED}
