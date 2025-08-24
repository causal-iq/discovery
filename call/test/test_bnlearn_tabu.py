#   Test calling the bnlearn TABU structure learning algorithm

import pytest
from copy import deepcopy

from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_learn
from fileio.common import TESTDATA_DIR
from fileio.numpy import NumPy
from core.bn import BN


# AB, 10 categorical rows
@pytest.fixture(scope="module")
def ab10():
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    return NumPy.from_df(df=bn.generate_cases(10), dstype='categorical',
                         keep_df=False)


# empty trace entry
@pytest.fixture(scope='module')
def empty_entry():
    return {'activity': None, 'arc': None, 'delta/score': None,
            'activity_2': None, 'arc_2': None, 'delta_2': None,
            'min_N': None, 'mean_N': None, 'max_N': None, 'free_params': None,
            'lt5': None, 'knowledge': None, 'blocked': []}


# --- Failure cases

# no arguments
def test_bnlearn_tabu_type_error_1():
    with pytest.raises(TypeError):
        bnlearn_learn()


# single argument
def test_bnlearn_tabu_type_error_2():
    with pytest.raises(TypeError):
        bnlearn_learn('tabu')
    with pytest.raises(TypeError):
        bnlearn_learn(6)
    with pytest.raises(TypeError):
        bnlearn_learn([['A', 'B'], [1, 2]])


# bad algorithm type
def test_bnlearn_tabu_type_error_3(ab10):
    with pytest.raises(TypeError):
        bnlearn_learn(True, ab10)
    with pytest.raises(TypeError):
        bnlearn_learn(6, ab10)
    with pytest.raises(TypeError):
        bnlearn_learn(ab10, ab10)


# bad data argument type
def test_bnlearn_tabu_type_error_4(ab10):
    with pytest.raises(TypeError):
        bnlearn_learn('tabu', 32.23)
    with pytest.raises(TypeError):
        bnlearn_learn('tabu', [['A', 'B'], [1, 2]])
    with pytest.raises(TypeError):
        bnlearn_learn('tabu', ab10.as_df())


# bad context argument type
def test_bnlearn_tabu_type_error_5(ab10):
    with pytest.raises(TypeError):
        bnlearn_learn('tabu', ab10, context=True)
    with pytest.raises(TypeError):
        bnlearn_learn('tabu', ab10, context='test/ab/10')


# bad params argument type
def test_bnlearn_tabu_type_error_6(ab10):
    with pytest.raises(TypeError):
        bnlearn_learn('tabu', ab10, params=True)
    with pytest.raises(TypeError):
        bnlearn_learn('tabu', ab10, params='bic')
    with pytest.raises(TypeError):
        bnlearn_learn('tabu', ab10, params='bic')


# bad context values
def test_bnlearn_tabu_value_error_1(ab10):
    with pytest.raises(ValueError):
        bnlearn_learn('tabu', ab10, context={})
    with pytest.raises(ValueError):
        bnlearn_learn('tabu', ab10, context={'invalid': 'bic'})
    with pytest.raises(ValueError):
        bnlearn_learn('tabu', ab10, context={'in': 'in'})
    with pytest.raises(ValueError):
        bnlearn_learn('tabu', ab10, context={'id': 'id'})


# bad param name
def test_bnlearn_tabu_value_error_2(ab10):
    with pytest.raises(ValueError):
        bnlearn_learn('tabu', ab10, params={'invalid': 'bic'})


# bad score specified
def test_bnlearn_tabu_value_error_3(ab10):
    with pytest.raises(ValueError):
        bnlearn_learn('tabu', ab10, params={'score': 'invalid'})


# bad primary arg types
def test_bnlearn_tabu_filenotfound_error_1():
    with pytest.raises(FileNotFoundError):
        bnlearn_learn('tabu', 'nonexistent.txt')


# --- Successful cases with discrete data

# default BIC score
@requires_r_and_bnlearn
def test_bnlearn_tabu_ab_10_ok_1(ab10, empty_entry):
    dag, trace = bnlearn_learn('tabu', ab10, context={'in': 'in', 'id': 'id'})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # TABU learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'id'
    assert trace.context['algorithm'] == 'TABU'
    assert trace.context['in'] == 'in'
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'bic', 'k': 1, 'base': 'e'}
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')

    entry = deepcopy(empty_entry)
    entry.update({'activity': 'init', 'delta/score': -15.141340})
    assert _trace[0] == entry

    entry.update({'activity': 'add', 'delta/score': 0.798467,
                  'arc': ('A', 'B')})
    assert _trace[1] == entry

    entry.update({'activity': 'reverse', 'delta/score': 0.0,
                  'arc': ('A', 'B'),
                  'blocked': [('delete', ('A', 'B'), -0.798467,
                               {'elem': 1})]})
    assert _trace[2] == entry

    entry.update({'activity': 'stop', 'delta/score': -14.342880,
                  'arc': None,
                  'blocked': [('delete', ('B', 'A'), -0.798467,
                               {'elem': 1}),
                              ('reverse', ('B', 'A'), 0.0,
                               {'elem': 2})]})
    assert _trace[3] == entry

    assert trace.result == dag


# default BIC score, no trace
@requires_r_and_bnlearn
def test_bnlearn_tabu_ab_10_ok_2(ab10):
    dag, trace = bnlearn_learn('tabu', ab10)
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # TABU learns correct answer
    assert trace is None


# BDE score
@requires_r_and_bnlearn
def test_bnlearn_tabu_ab_10_ok_3(ab10, empty_entry):
    dag, trace = bnlearn_learn('tabu', ab10, context={'in': 'in', 'id': 'id'},
                               params={'score': 'bde'})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # TABU learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'id'
    assert trace.context['algorithm'] == 'TABU'
    assert trace.context['in'] == 'in'
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'bde', 'iss': 1,
                                       'prior': 'uniform'}
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')

    entry = deepcopy(empty_entry)
    entry.update({'activity': 'init', 'delta/score': -15.646650})
    assert _trace[0] == entry

    entry.update({'activity': 'add', 'arc': ('A', 'B'),
                  'delta/score': 0.664229})
    assert _trace[1] == entry

    entry.update({'activity': 'reverse', 'arc': ('A', 'B'),
                  'delta/score': 0.0,
                  'blocked': [('delete', ('A', 'B'), -0.664229,
                               {'elem': 1})]})
    assert _trace[2] == entry

    entry.update({'activity': 'stop', 'arc': None, 'delta/score': -14.982420,
                  'blocked': [('delete', ('B', 'A'), -0.664229,
                               {'elem': 1}),
                              ('reverse', ('B', 'A'), 0.0,
                               {'elem': 2})]})
    assert _trace[3] == entry

    assert trace.result == dag


# Loglik score
@requires_r_and_bnlearn
def test_bnlearn_tabu_ab_10_ok_4(ab10, empty_entry):
    dag, trace = bnlearn_learn('tabu', ab10, context={'in': 'in', 'id': 'id'},
                               params={'score': 'loglik'})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # TABU learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'id'
    assert trace.context['algorithm'] == 'TABU'
    assert trace.context['in'] == 'in'
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'loglik', 'base': 'e'}
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')

    entry = deepcopy(empty_entry)
    entry.update({'activity': 'init', 'delta/score': -12.83876})
    assert _trace[0] == entry

    entry.update({'activity': 'add', 'arc': ('A', 'B'),
                  'delta/score': 1.94976})
    assert _trace[1] == entry

    entry.update({'activity': 'reverse', 'arc': ('A', 'B'),
                  'delta/score': 0.0,
                  'blocked': [('delete', ('A', 'B'), -1.94976,
                               {'elem': 1})]})
    assert _trace[2] == entry

    entry.update({'activity': 'stop', 'arc': None, 'delta/score': -10.889,
                  'blocked': [('delete', ('B', 'A'), -1.94976,
                               {'elem': 1}),
                              ('reverse', ('B', 'A'), 0.0,
                               {'elem': 2})]})
    assert _trace[3] == entry

    assert trace.result == dag


# A --> B, 100 rows
@requires_r_and_bnlearn
def test_bnlearn_tabu_ab_100_ok():
    data = BN.read(TESTDATA_DIR + '/dsc/ab.dsc').generate_cases(100)
    data = NumPy.from_df(df=data, dstype='categorical', keep_df=False)
    dag, trace = bnlearn_learn('tabu', data, context={'in': 'in', 'id': 'id'})
    print('\nDAG learnt from 100 rows of A->B: {}\n{}'.format(dag, trace))
    assert dag.to_string() == '[A][B|A]'  # TABU learns correct answer


#   FOLLOWING TEST MAY LEARN DIFFERENT RESULTS ON LAPTOP & DESKTOP

# A --> B --> C, 100 rows
@requires_r_and_bnlearn
def test_bnlearn_tabu_abc_100_ok():
    data = BN.read(TESTDATA_DIR + '/dsc/abc.dsc').generate_cases(1000)
    data = NumPy.from_df(df=data, dstype='categorical', keep_df=True)
    dag, trace = bnlearn_learn('tabu', data, context={'in': 'in', 'id': 'id'})
    print('\nDAG learnt from 100 rows of A->B->C: {}\n{}'.format(dag, trace))
    assert dag.to_string() == '[A][B|A][C|B]'  # TABU learns correctly


# A -> B <- C, 1k Rows
@requires_r_and_bnlearn
def test_bnlearn_tabu_ab_cb_1k_ok():
    data = BN.read(TESTDATA_DIR + '/dsc/ab_cb.dsc').generate_cases(1000)
    data = NumPy.from_df(df=data, dstype='categorical', keep_df=True)
    dag, trace = bnlearn_learn('tabu', data, context={'in': 'in', 'id': 'id'})
    print('\nDAG learnt from 1K rows of A->B<-C: {}\n{}'.format(dag, trace))
    assert dag.to_string() == '[A][B|A:C][C]'  # TABU learns correctly


# 1->2->4, 3->2, 1K rows
@requires_r_and_bnlearn
def test_bnlearn_tabu_and4_10_1k_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    print(bn.global_distribution())
    data = bn.generate_cases(1000)
    data = NumPy.from_df(df=data, dstype='categorical', keep_df=True)
    dag, trace = bnlearn_learn('tabu', data, context={'in': 'in', 'id': 'id'})
    print('\nDAG learnt from 1K rows of 1->2->4, 3->2: {}\n{}'
          .format(dag, trace))
    assert dag.to_string() == '[X1][X2|X1][X3|X2][X4|X2]'  # only equivalent


# Cancer, 1K rows
@requires_r_and_bnlearn
def test_bnlearn_tabu_cancer_1k_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                      dstype='categorical')
    dag, trace = bnlearn_learn('tabu', data, context={'in': 'in', 'id': 'id'})
    print('\nDAG learnt from 1K rows of Cancer: {}\n{}'.format(dag, trace))
    assert '[Cancer][Dyspnoea|Cancer][Pollution][Smoker|Cancer][Xray|Cancer]' \
        == dag.to_string()  # incorrect NOT equivalent


# Asia, 1K rows
@requires_r_and_bnlearn
def test_bnlearn_tabu_asia_1k_ok_1():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                      dstype='categorical')
    dag, trace = bnlearn_learn('tabu', data, context={'in': 'in', 'id': 'id'})
    print('\nDAG learnt from 1K rows of Asia: {}\n{}'.format(dag, trace))
    assert ('[asia][bronc|smoke][dysp|bronc:either][either|lung:tub]' +
            '[lung][smoke|lung][tub][xray|either]') == dag.to_string()


#   FOLLOWING TEST MAY GIVE DIFFERENT RESULTS ON LAPTOP & DESKTOP

# Asia, 1K rows, BDE score
@requires_r_and_bnlearn
def test_bnlearn_tabu_asia_1k_ok_2():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                      dstype='categorical')
    dag, trace = bnlearn_learn('tabu', data, context={'in': 'in', 'id': 'id'},
                               params={'score': 'bde'})
    print('\nDAG learnt from 1K rows of Asia (BDeu): {}\n{}'
          .format(dag, trace))
    assert ("[asia][bronc|smoke][dysp|bronc:either][either|asia:lung:tub]" +
            "[lung][smoke|lung][tub][xray|either]") == dag.to_string()
    assert trace.context['N'] == 1000
    assert trace.context['id'] == 'id'
    assert trace.context['algorithm'] == 'TABU'
    assert trace.context['in'] == 'in'
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'bde', 'iss': 1,
                                       'prior': 'uniform'}
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')
    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -3032.945000, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': []}
    assert _trace[4] == {'activity': 'add', 'arc': ('bronc', 'smoke'),
                         'delta/score': 52.652915, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': []}
    assert _trace[19] == {'activity': 'reverse', 'arc': ('smoke', 'bronc'),
                          'delta/score': 0.0, 'activity_2': None,
                          'arc_2': None, 'delta_2': None, 'min_N': None,
                          'mean_N': None, 'max_N': None, 'lt5': None,
                          'free_params': None, 'knowledge': None,
                          'blocked': [('delete', ('either', 'asia'),
                                       0.731449, {'elem': 8}),
                                      ('reverse', ('either', 'asia'),
                                       0.741904, {'elem': 5})]}
    assert _trace[26] == {'activity': 'stop', 'arc': None,
                          'delta/score': -2262.281, 'activity_2': None,
                          'arc_2': None, 'delta_2': None, 'min_N': None,
                          'mean_N': None, 'max_N': None, 'free_params': None,
                          'lt5': None, 'knowledge': None, 'blocked': []}
    assert trace.result == dag


# --- Gaussian data experiments

# Gaussian example, 100 rows
@requires_r_and_bnlearn
def test_bnlearn_tabu_gauss_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous', N=100)
    dag, trace = bnlearn_learn('tabu', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'gauss'})
    print('\nDAG learnt from 100 rows of gauss: {}\n\n{}'.format(dag, trace))
    assert dag.to_string() == '[A][B][C|A:B][D|B:C][E|C][F|A:D:E:G][G]'


# Gaussian example, 100 rows, rev ord
@requires_r_and_bnlearn
def test_bnlearn_tabu_gauss_2_ok():
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous', N=100)
    data.set_order(tuple(list(data.get_order())[::-1]))
    dag, trace = bnlearn_learn('tabu', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'gauss'})
    print('\nDAG learnt from 100 rows of gauss: {}\n\n{}'.format(dag, trace))
    assert dag.to_string() == '[A][B|A:D][C|A:B][D][E|C][F|A:D:E:G][G]'


# Gaussian example, 5K rows
@requires_r_and_bnlearn
def test_bnlearn_tabu_gauss_3_ok():
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous')
    dag, trace = bnlearn_learn('tabu', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'gauss'})
    print('\nDAG learnt from 5K rows of gauss: {}\n\n{}'.format(dag, trace))
    assert dag.to_string() == '[A][B|A][C|A:B][D|B][E][F|A:C:D:E:G][G]'


# Gaussian example, 5K rows, rev ord
@requires_r_and_bnlearn
def test_bnlearn_tabu_gauss_4_ok():
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous')
    data.set_order(tuple(list(data.get_order())[::-1]))
    dag, trace = bnlearn_learn('tabu', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'gauss'})
    print('\nDAG learnt from 5K rows of gauss: {}\n\n{}'.format(dag, trace))
    assert dag.to_string() == '[A|B][B|D][C|A:B][D][E][F|A:C:D:E:G][G]'


# Sachs gauss example, 1K rows
@requires_r_and_bnlearn
def test_bnlearn_tabu_sachs_c_1_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz',
                      dstype='continuous')
    dag, trace = bnlearn_learn('tabu', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'sachs_c'})
    print('\nDAG rom 1K rows of sachs_c: {}\n\n{}'.format(dag, trace))
    assert dag.to_string() == \
        ('[Akt]' +
         '[Erk|Akt:PKA]' +
         '[Jnk|PKC]' +
         '[Mek|PKC]' +
         '[P38]' +
         '[PIP2]' +
         '[PIP3|PIP2]' +
         '[PKA|Jnk:Mek:P38:PIP3:PKC:Raf]' +
         '[PKC|Akt:P38:PIP3]' +
         '[Plcg|Akt:Mek:PIP2:PIP3]' +
         '[Raf|Akt:Jnk:Mek:PKC]')


# Sachs gauss example, rev, 1K rows
@requires_r_and_bnlearn
def test_bnlearn_tabu_sachs_c_2_ok():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz',
                      dstype='continuous')
    data.set_order(tuple(list(data.get_order())[::-1]))
    dag, trace = bnlearn_learn('tabu', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'sachs_c'})
    print('\nDAG rom 1K rows of sachs_c: {}\n\n{}'.format(dag, trace))
    assert dag.to_string() == \
        ('[Akt]' +
         '[Erk|Akt:PKA]' +
         '[Jnk|PKC]' +
         '[Mek|PKC]' +
         '[P38|Akt:PIP3:PKC]' +
         '[PIP2|Akt:Mek:PIP3:Plcg]' +
         '[PIP3|Plcg]' +
         '[PKA|Jnk:Mek:P38:PIP3:PKC:Raf]' +
         '[PKC]' +
         '[Plcg]' +
         '[Raf|Akt:Jnk:Mek:PKC]')
    return
