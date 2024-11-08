
#   Test calling the bnlearn H2PC structure learning algorithm

import pytest

from call.bnlearn import bnlearn_learn
from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from core.bn import BN


def test_bnlearn_h2pc_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        bnlearn_learn()


def test_bnlearn_h2pc_type_error_2():  # single argument
    with pytest.raises(TypeError):
        bnlearn_learn('h2pc')
    with pytest.raises(TypeError):
        bnlearn_learn(6)
    with pytest.raises(TypeError):
        bnlearn_learn([['A', 'B'], [1, 2]])


def test_bnlearn_h2pc_type_error_3():  # bad algorithm type
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        bnlearn_learn(True, data)
    with pytest.raises(TypeError):
        bnlearn_learn(6, data)
    with pytest.raises(TypeError):
        bnlearn_learn(data, data)


def test_bnlearn_h2pc_type_error_4():  # bad data argument type
    with pytest.raises(TypeError):
        bnlearn_learn('h2pc', 32.23)
    with pytest.raises(TypeError):
        bnlearn_learn('h2pc', [['A', 'B'], [1, 2]])


def test_bnlearn_h2pc_type_error_5():  # bad context argument type
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        bnlearn_learn('h2pc', data, context=True)
    with pytest.raises(TypeError):
        bnlearn_learn('h2pc', data, context='test/ab/10')


def test_bnlearn_h2pc_type_error_6():  # bad params argument type
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        bnlearn_learn('h2pc', data, params=True)
    with pytest.raises(TypeError):
        bnlearn_learn('h2pc', data, params='bic')


def test_bnlearn_h2pc_value_error_1():  # bad context values
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        bnlearn_learn('h2pc', data, context={})
    with pytest.raises(ValueError):
        bnlearn_learn('h2pc', data, context={'invalid': 'bic'})


def test_bnlearn_h2pc_value_error_2():  # bad param name
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        bnlearn_learn('h2pc', data, {'invalid': 'bic'})


def test_bnlearn_h2pc_value_error_3():  # bad score specified
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        bnlearn_learn('h2pc', data, {'score': 'invalid'})


def test_bnlearn_h2pc_filenotfound_error_1():  # bad primary arg types
    with pytest.raises(FileNotFoundError):
        bnlearn_learn('h2pc', 'nonexistent.txt')


def test_bnlearn_h2pc_ab_10_ok_1():  # default BIC score
    _in = TESTDATA_DIR + '/dsc/ab.dsc'
    id = 'test/ab_10_ok_1'
    bn = BN.read(_in)
    data = bn.generate_cases(10)
    dag, trace = bnlearn_learn('h2pc', data, context={'in': _in, 'id': id})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == id
    assert trace.context['algorithm'] == 'H2PC'
    assert trace.context['in'] == _in
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


def test_bnlearn_h2pc_ab_10_ok_2():  # default BIC score, no trace
    _in = TESTDATA_DIR + '/dsc/ab.dsc'
    bn = BN.read(_in)
    data = bn.generate_cases(10)
    dag, trace = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    assert trace is None


def test_bnlearn_h2pc_ab_10_ok_3():  # BDE score
    _in = TESTDATA_DIR + '/dsc/ab.dsc'
    id = 'test/ab_10_ok_3'
    bn = BN.read(_in)
    data = bn.generate_cases(10)
    dag, trace = bnlearn_learn('h2pc', data, context={'in': _in, 'id': id},
                               params={'score': 'bde'})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == id
    assert trace.context['algorithm'] == 'H2PC'
    assert trace.context['in'] == _in
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


def test_bnlearn_h2pc_ab_10_ok_4():  # Loglik score
    _in = TESTDATA_DIR + '/dsc/ab.dsc'
    id = 'test/ab_10_ok_3'
    bn = BN.read(_in)
    data = bn.generate_cases(10)
    dag, trace = bnlearn_learn('h2pc', data, context={'in': _in, 'id': id},
                               params={'score': 'loglik'})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == id
    assert trace.context['algorithm'] == 'H2PC'
    assert trace.context['in'] == _in
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


def test_bnlearn_h2pc_ab_100_ok():
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    data = bn.generate_cases(100)
    dag, _ = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 100 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer


def test_bnlearn_h2pc_abc_100_ok():
    bn = BN.read(TESTDATA_DIR + '/dsc/abc.dsc')
    data = bn.generate_cases(100)
    dag, _ = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 100 rows of A->B->C: {}'.format(dag))
    assert dag.to_string() == '[A][B][C|B]'  # H2PC has missing arc


def test_bnlearn_h2pc_ab_cb_1k_ok():  # A -> B <- C, 1k Rows
    bn = BN.read(TESTDATA_DIR + '/dsc/ab_cb.dsc')
    data = bn.generate_cases(1000)
    dag, _ = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 1K rows of A->B<-C: {}'.format(dag))
    assert dag.to_string() == '[A][B|A:C][C]'  # H2PC correct


def test_bnlearn_h2pc_and4_10_1k_ok():  # 1->2->4, 3->2, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    print(bn.global_distribution())
    data = bn.generate_cases(1000)
    dag, _ = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 1K rows of 1->2->4, 3->2: {}'.format(dag))
    assert dag.to_string() == '[X1][X2|X1][X3|X2][X4|X2]'  # only equivalent


def test_bnlearn_h2pc_cancer_1k_ok():  # Cancer, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    print(bn.global_distribution())
    data = bn.generate_cases(1000)
    dag, _ = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 1K rows of Cancer: {}'.format(dag))
    assert '[Cancer][Dyspnoea|Cancer][Pollution][Smoker|Cancer][Xray|Cancer]' \
        == dag.to_string()  # incorrect NOT equivalent


def test_bnlearn_h2pc_asia_1k_ok_1():  # Cancer, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    print(bn.global_distribution())
    data = bn.generate_cases(1000)
    dag, _ = bnlearn_learn('h2pc', data)
    print('\nDAG learnt from 1K rows of Asia: {}'.format(dag))
    assert ('[asia][bronc][dysp|bronc][either][lung|either][smoke|bronc]' +
            '[tub|either][xray]') == dag.to_string()


def test_bnlearn_h2pc_asia_1k_ok_2():  # Cancer, 1K rows, BDE score
    _in = TESTDATA_DIR + '/discrete/small/asia.dsc'
    id = 'test/asia_1k'
    bn = BN.read(_in)
    data = bn.generate_cases(1000)
    dag, trace = bnlearn_learn('h2pc', data, context={'in': _in, 'id': id},
                               params={'score': 'bde'})
    print('\nDAG learnt from 1K rows of Asia: {}'.format(dag))
    print(dag.to_string())
    return
    assert ('[asia][bronc][dysp|bronc][either][lung|either][smoke|bronc]' +
            '[tub|either][xray]') == dag.to_string()
    print(trace)
    assert trace.context['N'] == 1000
    assert trace.context['id'] == id
    assert trace.context['algorithm'] == 'h2pc'
    assert trace.context['in'] == _in
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'bde'}
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')
    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -3032.945000, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None}
    assert _trace[4] == {'activity': 'add', 'arc': ('bronc', 'smoke'),
                         'delta/score': 52.652915, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None}
    assert _trace[9] == {'activity': 'reverse', 'arc': ('bronc', 'smoke'),
                         'delta/score': 6.999314, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None}
    assert _trace[10] == {'activity': 'stop', 'arc': None,
                          'delta/score': -2264.126, 'activity_2': None,
                          'arc_2': None, 'delta_2': None, 'min_N': None,
                          'mean_N': None, 'max_N': None, 'free_params': None,
                          'lt5': None, 'knowledge': None}
    assert trace.result == dag


# Gaussian datasets

def test_bnlearn_h2pc_gauss_1_ok():  # Gaussian example, 100 rows
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = Pandas.read(_in, dstype='continuous', N=100)
    pdag, trace = bnlearn_learn('h2pc', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
                                params={'test': 'mi-g', 'score': 'bic-g'})
    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('B', '->', 'D'),
         ('F', '->', 'G')}


def test_bnlearn_h2pc_gauss_2_ok():  # Gaussian example, 100 rows, rev ord
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = Pandas.read(_in, dstype='continuous', N=100)
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = bnlearn_learn('h2pc', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
                                params={'test': 'mi-g', 'score': 'bic-g'})
    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('D', '->', 'B'),
         ('G', '->', 'F')}


def test_bnlearn_h2pc_gauss_3_ok():  # Gaussian example, 5K rows
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = Pandas.read(_in, dstype='continuous')
    pdag, trace = bnlearn_learn('h2pc', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
                                params={'test': 'mi-g', 'score': 'bic-g'})
    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('B', '->', 'C'),
         ('B', '->', 'D'),
         ('A', '->', 'C'),
         ('D', '->', 'F'),
         ('G', '->', 'F'),
         ('E', '->', 'F'),
         ('A', '->', 'F')}


def test_bnlearn_h2pc_gauss_4_ok():  # Gaussian example, 5K rows, rev ord
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = Pandas.read(_in, dstype='continuous')
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = bnlearn_learn('h2pc', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
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


def test_bnlearn_h2pc_sachs_c_1_ok():  # Sachs gauss example, 1K rows
    _in = TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz'
    data = Pandas.read(_in, dstype='continuous')
    pdag, trace = bnlearn_learn('h2pc', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
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


def test_bnlearn_h2pc_sachs_c_2_ok():  # Sachs gauss example, rev, 1K rows
    _in = TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz'
    data = Pandas.read(_in, dstype='continuous')
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = bnlearn_learn('h2pc', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
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
