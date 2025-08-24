#   Test calling the bnlearn H2PC structure learning algorithm

import pytest

from call.bnlearn import bnlearn_learn
from call.r import requires_r_and_bnlearn
from fileio.common import TESTDATA_DIR
from fileio.numpy import NumPy
from core.bn import BN


# Fixture creating 10 categorical rows for A --> B
@pytest.fixture(scope="module")
def ab10():
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    return NumPy.from_df(df=bn.generate_cases(10), dstype='categorical',
                         keep_df=False)


# --- Failure cases

# no arguments
def test_bnlearn_h2pc_type_error_1():
    with pytest.raises(TypeError):
        bnlearn_learn()


# only one argument
def test_bnlearn_h2pc_type_error_2():
    with pytest.raises(TypeError):
        bnlearn_learn(32.23)
    with pytest.raises(TypeError):
        bnlearn_learn('h2pc')


# bad algorithm type
def test_bnlearn_h2pc_type_error_3(ab10):
    with pytest.raises(TypeError):
        bnlearn_learn(True, ab10)
    with pytest.raises(TypeError):
        bnlearn_learn(6, ab10)
    with pytest.raises(TypeError):
        bnlearn_learn(ab10, ab10)


# bad data argument type
def test_bnlearn_h2pc_type_error_4(ab10):
    with pytest.raises(TypeError):
        bnlearn_learn('h2pc', 32.23)
    with pytest.raises(TypeError):
        bnlearn_learn('h2pc', [['A', 'B'], [1, 2]])
    with pytest.raises(TypeError):
        bnlearn_learn('h2pc', ab10.as_df())


# non-existent data file
def test_bnlearn_h2pc_filenotfound_error_1():
    with pytest.raises(FileNotFoundError):
        bnlearn_learn('h2pc', 'nonexistent.txt')


# -- successful learning with discrete data

# default BIC score
@requires_r_and_bnlearn
def test_bnlearn_h2pc_ab_10_ok_1(ab10):
    dag, trace = bnlearn_learn('h2pc', ab10, context={'in': 'in', 'id': 'id'})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'id'
    assert trace.context['algorithm'] == 'H2PC'
    assert trace.context['in'] == 'in'
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'bic', 'k': 1, 'base': 'e',
                                       'test': 'mi', 'alpha': 0.05}
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')
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


# default BIC score, no trace
@requires_r_and_bnlearn
def test_bnlearn_h2pc_ab_10_ok_2(ab10):
    dag, trace = bnlearn_learn('h2pc', ab10)
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    assert trace is None


# BDE score
@requires_r_and_bnlearn
def test_bnlearn_h2pc_ab_10_ok_3(ab10):
    dag, trace = bnlearn_learn('h2pc', ab10, context={'in': 'in', 'id': 'id'},
                               params={'score': 'bde'})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'id'
    assert trace.context['algorithm'] == 'H2PC'
    assert trace.context['in'] == 'in'
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'bde', 'iss': 1, 'alpha': 0.05,
                                       'prior': 'uniform', 'test': 'mi'}
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')
    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -15.646650, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[1] == {'activity': 'add', 'arc': ('A', 'B'),
                         'delta/score': 0.664229, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[2] == {'activity': 'stop', 'arc': None,
                         'delta/score': -14.982420, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None,
                         'free_params': None, 'lt5': None, 'knowledge': None,
                         'blocked': None}
    assert trace.result == dag


# Loglik score
@requires_r_and_bnlearn
def test_bnlearn_h2pc_ab_10_ok_4(ab10):
    dag, trace = bnlearn_learn('h2pc', ab10, context={'in': 'in', 'id': 'id'},
                               params={'score': 'loglik'})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'id'
    assert trace.context['algorithm'] == 'H2PC'
    assert trace.context['in'] == 'in'
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'loglik', 'base': 'e',
                                       'test': 'mi', 'alpha': 0.05}
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')
    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -12.83876, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[1] == {'activity': 'add', 'arc': ('A', 'B'),
                         'delta/score': 1.94976, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[2] == {'activity': 'stop', 'arc': None,
                         'delta/score': -10.88900, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert trace.result == dag


# AB, 100 rows
@requires_r_and_bnlearn
def test_bnlearn_h2pc_ab_100_ok():
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    data = NumPy.from_df(bn.generate_cases(100), dstype='categorical',
                         keep_df=False)
    dag, _ = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 100 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # H2PC learns correct answer


# A -> B <- C, 1k Rows
@requires_r_and_bnlearn
def test_bnlearn_h2pc_ab_cb_1k_ok():
    bn = BN.read(TESTDATA_DIR + '/dsc/ab_cb.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    dag, _ = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 1K rows of A->B<-C: {}'.format(dag))
    assert dag.to_string() == '[A][B|A:C][C]'  # H2PC correct


# 1->2->4, 3->2, 1K rows
@requires_r_and_bnlearn
def test_bnlearn_h2pc_and4_10_1k_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    print(bn.global_distribution())
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    dag, _ = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 1K rows of 1->2->4, 3->2: {}'.format(dag))
    assert dag.to_string() == '[X1][X2|X1][X3|X2][X4|X2]'  # only equivalent


# Cancer, 1K rows
@requires_r_and_bnlearn
def test_bnlearn_h2pc_cancer_1k_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    print(bn.global_distribution())
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    dag, _ = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 1K rows of Cancer: {}'.format(dag))
    assert '[Cancer][Dyspnoea|Cancer][Pollution][Smoker|Cancer][Xray|Cancer]' \
        == dag.to_string()  # incorrect NOT equivalent


# Cancer, 1K rows
@requires_r_and_bnlearn
def test_bnlearn_h2pc_asia_1k_ok_1():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    print(bn.global_distribution())
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    dag, _ = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 1K rows of Asia: {}'.format(dag))
    assert ('[asia][bronc][dysp|bronc][either][lung|either][smoke|bronc]' +
            '[tub|either][xray]') == dag.to_string()


# Cancer, 1K rows, BDE score
@requires_r_and_bnlearn
def test_bnlearn_h2pc_asia_1k_ok_2():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                      dstype='categorical')
    dag, trace = bnlearn_learn('h2pc', data, context={'in': 'in', 'id': 'id'},
                               params={'score': 'bde'})
    print('\nDAG learnt from 1K rows of Asia: {}'.format(dag))
    print(dag.to_string())
    assert ('[asia][bronc][dysp|bronc][either][lung|either][smoke|bronc]' +
            '[tub|either][xray]') == dag.to_string()
    print(trace)
    assert trace.context['N'] == 1000
    assert trace.context['id'] == 'id'
    assert trace.context['algorithm'] == 'H2PC'
    assert trace.context['in'] == 'in'
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == \
        {'score': 'bde', 'test': 'mi', 'iss': 1, 'prior': 'uniform',
         'alpha': 0.05}
    assert trace.result == dag


# --- successful learning with Gaussian datasets

# Gaussian example, 100 rows
@requires_r_and_bnlearn
def test_bnlearn_h2pc_gauss_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous', N=100)
    pdag, trace = bnlearn_learn('h2pc', data,
                                context={'in': 'in', 'id': 'gauss'},
                                params={'test': 'mi-g', 'score': 'bic-g'})
    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('B', '->', 'D'),
         ('F', '->', 'G')}


# Gaussian example, 100 rows, rev ord
@requires_r_and_bnlearn
def test_bnlearn_h2pc_gauss_2_ok():
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous', N=100)
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = bnlearn_learn('h2pc', data,
                                context={'in': 'in', 'id': 'gauss'},
                                params={'test': 'mi-g', 'score': 'bic-g'})
    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('D', '->', 'B'),
         ('G', '->', 'F')}


# Gaussian example, 5K rows
@requires_r_and_bnlearn
def test_bnlearn_h2pc_gauss_3_ok():
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous', N=5000)
    pdag, trace = bnlearn_learn('h2pc', data,
                                context={'in': 'in', 'id': 'gauss'},
                                params={'test': 'mi-g', 'score': 'bic-g'})
    print('\nPDAG learnt from 5K rows of gauss: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('B', '->', 'C'),
         ('B', '->', 'D'),
         ('A', '->', 'C'),
         ('D', '->', 'F'),
         ('G', '->', 'F'),
         ('E', '->', 'F'),
         ('A', '->', 'F')}


# Gaussian example, 5K rows, rev ord
@requires_r_and_bnlearn
def test_bnlearn_h2pc_gauss_4_ok():
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous', N=5000)
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = bnlearn_learn('h2pc', data,
                                context={'in': 'in', 'id': 'gauss'},
                                params={'test': 'mi-g', 'score': 'bic-g'})
    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('A', '->', 'C'),
         ('A', '->', 'F'),
         ('D', '->', 'B'),
         ('B', '->', 'C'),
         ('E', '->', 'F'),
         ('G', '->', 'F'),
         ('D', '->', 'F')}


# Sachs gauss example, 1K rows
@requires_r_and_bnlearn
def test_bnlearn_h2pc_sachs_c_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz',
                      dstype='continuous', N=1000)
    pdag, trace = bnlearn_learn('h2pc', data,
                                context={'in': 'in', 'id': 'gauss'},
                                params={'test': 'mi-g', 'score': 'bic-g'})
    print('\nPDAG rom 1K rows of sachs_c: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('Mek', '->', 'Raf'),
         ('PIP3', '->', 'PKA'),
         ('P38', '->', 'PKC'),
         ('P38', '->', 'PKA'),
         ('PIP2', '->', 'Plcg'),
         ('PIP3', '->', 'Plcg'),
         ('PIP2', '->', 'PIP3'),
         ('PKC', '->', 'Jnk'),
         ('PKA', '->', 'Erk'),
         ('PKA', '->', 'Mek'),
         ('Akt', '->', 'Erk')}


# Sachs gauss example, rev, 1K rows
@requires_r_and_bnlearn
def test_bnlearn_h2pc_sachs_c_2_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz',
                      dstype='continuous', N=1000)
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = bnlearn_learn('h2pc', data,
                                context={'in': 'in', 'id': 'gauss'},
                                params={'test': 'mi-g', 'score': 'bic-g'})
    print('\nPDAG rom 1K rows of sachs_c: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('Mek', '->', 'PKA'),
         ('P38', '->', 'PKA'),
         ('PIP3', '->', 'PIP2'),
         ('PIP3', '->', 'PKA'),
         ('PKC', '->', 'Jnk'),
         ('Plcg', '->', 'PIP3'),
         ('Raf', '->', 'Mek'),
         ('PKC', '->', 'P38'),
         ('PKA', '->', 'Erk'),
         ('Plcg', '->', 'PIP2'),
         ('Akt', '->', 'Erk')}
