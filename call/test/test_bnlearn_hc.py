
#   Test calling the bnlearn HC structure learning algorithm

import pytest
from pandas import DataFrame

from call.bnlearn import bnlearn_learn
from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from fileio.numpy import NumPy
from core.bn import BN


@pytest.fixture(scope="module")  # AB, 10 categorical rows
def ab10():
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    return Pandas(df=bn.generate_cases(10))


def test_bnlearn_hc_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        bnlearn_learn()


def test_bnlearn_hc_type_error_2():  # single argument
    with pytest.raises(TypeError):
        bnlearn_learn('hc')
    with pytest.raises(TypeError):
        bnlearn_learn(6)
    with pytest.raises(TypeError):
        bnlearn_learn([['A', 'B'], [1, 2]])


def test_bnlearn_hc_type_error_3(ab10):  # bad algorithm type
    with pytest.raises(TypeError):
        bnlearn_learn(True, ab10)
    with pytest.raises(TypeError):
        bnlearn_learn(6, ab10)
    with pytest.raises(TypeError):
        bnlearn_learn(ab10, ab10)


def test_bnlearn_hc_type_error_4(ab10):  # bad data argument type
    with pytest.raises(TypeError):
        bnlearn_learn('hc', 32.23)
    with pytest.raises(TypeError):
        bnlearn_learn('hc', [['A', 'B'], [1, 2]])
    with pytest.raises(TypeError):
        bnlearn_learn('hc', ab10.as_df())


def test_bnlearn_hc_type_error_5(ab10):  # bad context argument type
    with pytest.raises(TypeError):
        bnlearn_learn('hc', ab10, context=True)
    with pytest.raises(TypeError):
        bnlearn_learn('hc', ab10, context='test/ab/10')


def test_bnlearn_hc_type_error_6(ab10):  # bad params argument type
    with pytest.raises(TypeError):
        bnlearn_learn('hc', ab10, params=True)
    with pytest.raises(TypeError):
        bnlearn_learn('hc', ab10, params='bic')


def test_bnlearn_hc_value_error_1(ab10):  # bad context values
    with pytest.raises(ValueError):
        bnlearn_learn('hc', ab10, context={})
    with pytest.raises(ValueError):
        bnlearn_learn('hc', ab10, context={'invalid': 'bic'})


def test_bnlearn_hc_value_error_2(ab10):  # bad param name
    with pytest.raises(ValueError):
        bnlearn_learn('hc', ab10, params={'invalid': 'bic'})


def test_bnlearn_hc_value_error_3(ab10):  # bad score specified
    with pytest.raises(ValueError):
        bnlearn_learn('hc', ab10, params={'score': 'invalid'})


def test_bnlearn_hc_value_error_4():  # mixed datasets unsupported
    data = Pandas(df=DataFrame({'A': ['0', '1'], 'X': [1.0, 2.2]}))
    with pytest.raises(ValueError):
        bnlearn_learn('hc', data)


def test_bnlearn_hc_value_error_5(ab10):  # bic-g incompatible categorical
    with pytest.raises(ValueError):
        bnlearn_learn('hc', ab10, params={'score': 'bic-g'})


def test_bnlearn_hc_value_error_6():  # must be bic-g for continuous
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = Pandas.read(_in, dstype='continuous', N=100)
    with pytest.raises(ValueError):
        bnlearn_learn('hc', data, params={'score': 'bic'})
    with pytest.raises(ValueError):
        bnlearn_learn('hc', data)


def test_bnlearn_hc_value_error_7():  # single-valued columns
    data = Pandas(df=DataFrame({'A': ['0', '1'], 'B': ['1', '1']},
                               dtype='category'))
    with pytest.raises(RuntimeError):
        bnlearn_learn('hc', data)


def test_bnlearn_hc_filenotfound_error_1():  # bad primary arg types
    with pytest.raises(FileNotFoundError):
        bnlearn_learn('hc', 'nonexistent.txt')


def test_bnlearn_hc_ab10_ok_1(ab10):  # default BIC score
    dag, trace = bnlearn_learn('hc', ab10, context={'in': 'in', 'id': 'id'})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'id'
    assert trace.context['algorithm'] == 'HC'
    assert trace.context['in'] == 'in'
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'bic', 'k': 1, 'base': 'e'}
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


def test_bnlearn_hc_ab_10_ok_2(ab10):  # default BIC score, no trace
    dag, trace = bnlearn_learn('hc', ab10)
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    assert trace is None


def test_bnlearn_hc_ab_10_ok_3(ab10):  # BDE score
    dag, trace = bnlearn_learn('hc', ab10, context={'in': 'in', 'id': 'id'},
                               params={'score': 'bde'})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'id'
    assert trace.context['algorithm'] == 'HC'
    assert trace.context['in'] == 'in'
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'bde', 'iss': 1,
                                       'prior': 'uniform'}
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
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert trace.result == dag


def test_bnlearn_hc_ab_10_ok_4(ab10):  # Loglik score
    dag, trace = bnlearn_learn('hc', ab10, context={'in': 'in', 'id': 'id'},
                               params={'score': 'loglik'})
    print('\nDAG learnt from 10 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer
    print(trace)
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'id'
    assert trace.context['algorithm'] == 'HC'
    assert trace.context['in'] == 'in'
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'loglik', 'base': 'e'}
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


def test_bnlearn_hc_ab_100_ok():
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    data = Pandas(df=bn.generate_cases(100))
    dag, _ = bnlearn_learn('hc', data)
    print('\nDAG learnt from 100 rows of A->B: {}'.format(dag))
    assert dag.to_string() == '[A][B|A]'  # HC learns correct answer


def test_bnlearn_hc_abc_100_ok():
    bn = BN.read(TESTDATA_DIR + '/dsc/abc.dsc')
    data = Pandas(df=bn.generate_cases(100))
    dag, _ = bnlearn_learn('hc', data)
    print('\nDAG learnt from 100 rows of A->B->C: {}'.format(dag))
    assert dag.to_string() == '[A][B|A][C|B]'  # HC learns correct answer


def test_bnlearn_hc_ab_cb_1k_ok():  # A -> B <- C, 1k Rows
    bn = BN.read(TESTDATA_DIR + '/dsc/ab_cb.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    dag, _ = bnlearn_learn('hc', data)
    print('\nDAG learnt from 1K rows of A->B<-C: {}'.format(dag))
    assert dag.to_string() == '[A][B|A][C|A:B]'  # incorrect and not-equivalent


def test_bnlearn_hc_and4_10_1k_ok():  # 1->2->4, 3->2, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    dag, _ = bnlearn_learn('hc', data)
    print('\nDAG learnt from 1K rows of 1->2->4, 3->2: {}'.format(dag))
    assert dag.to_string() == '[X1][X2|X1][X3|X2][X4|X2]'  # only equivalent


def test_bnlearn_hc_cancer_1k_ok():  # Cancer, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(bn.generate_cases(1000))
    dag, _ = bnlearn_learn('hc', data)
    print('\nDAG learnt from 1K rows of Cancer: {}'.format(dag))
    assert '[Cancer][Dyspnoea|Cancer][Pollution][Smoker|Cancer][Xray|Cancer]' \
        == dag.to_string()  # incorrect NOT equivalent


def test_bnlearn_hc_asia_1k_ok_1():  # Asia, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    print(bn.global_distribution())
    data = Pandas(bn.generate_cases(1000))
    dag, _ = bnlearn_learn('hc', data)
    print('\nDAG learnt from 1K rows of Asia: {}'.format(dag))
    assert ('[asia][bronc][dysp|bronc][either|bronc:dysp][lung|either][smoke' +
            '|bronc:lung][tub|either:lung][xray|either]') == dag.to_string()


def test_bnlearn_hc_asia_1k_ok_2():  # Cancer, 1K rows, BDE score
    _in = TESTDATA_DIR + '/discrete/small/asia.dsc'
    id = 'test/asia_1k'
    bn = BN.read(_in)
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    dag, trace = bnlearn_learn('hc', data, context={'in': _in, 'id': id},
                               params={'score': 'bde'})
    print('\nDAG learnt from 1K rows of Asia: {}'.format(dag))
    assert ('[asia][bronc|smoke][dysp|bronc:either][either][lung|either]' +
            '[smoke|lung][tub|either:lung][xray|either]') == dag.to_string()
    print(trace)
    assert trace.context['N'] == 1000
    assert trace.context['id'] == id
    assert trace.context['algorithm'] == 'HC'
    assert trace.context['in'] == _in
    assert trace.context['external'] == 'BNLEARN'
    assert trace.context['params'] == {'score': 'bde', 'prior': 'uniform',
                                       'iss': 1}
    _trace = trace.get().drop(labels='time', axis=1).to_dict('records')
    assert _trace[0] == {'activity': 'init', 'arc': None,
                         'delta/score': -3032.945000, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[4] == {'activity': 'add', 'arc': ('bronc', 'smoke'),
                         'delta/score': 52.652915, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[9] == {'activity': 'reverse', 'arc': ('bronc', 'smoke'),
                         'delta/score': 6.999314, 'activity_2': None,
                         'arc_2': None, 'delta_2': None, 'min_N': None,
                         'mean_N': None, 'max_N': None, 'free_params': None,
                         'lt5': None, 'knowledge': None, 'blocked': None}
    assert _trace[10] == {'activity': 'stop', 'arc': None,
                          'delta/score': -2264.126, 'activity_2': None,
                          'arc_2': None, 'delta_2': None, 'min_N': None,
                          'mean_N': None, 'max_N': None, 'free_params': None,
                          'lt5': None, 'knowledge': None, 'blocked': None}
    assert trace.result == dag


def test_bnlearn_hc_gauss_1_ok():  # Gaussian example, 100 rows
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = Pandas.read(_in, dstype='continuous', N=100)
    dag, trace = bnlearn_learn('hc', data,
                               context={'in': _in, 'id': 'gauss'},
                               params={'score': 'bic-g'})
    print('\nDAG learnt from 100 rows of gauss: {}\n\n{}'.format(dag, trace))
    assert dag.to_string() == '[A][B][C|A:B][D|B:C][E|C][F|A:D:E:G][G]'


def test_bnlearn_hc_gauss_2_ok():  # Gaussian example, 100 rows, rev ord
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = NumPy.read(_in, dstype='continuous', N=100)
    data.set_order(tuple(list(data.get_order())[::-1]))
    dag, trace = bnlearn_learn('hc', data,
                               context={'in': _in, 'id': 'gauss'},
                               params={'score': 'bic-g'})
    print('\nDAG learnt from 100 rows of gauss: {}\n\n{}'.format(dag, trace))
    assert dag.to_string() == '[A][B|A:D][C|A:B][D][E|C][F|A:D:E:G][G]'


def test_bnlearn_hc_gauss_3_ok():  # Gaussian example, 5K rows
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = Pandas.read(_in, dstype='continuous')
    dag, trace = bnlearn_learn('hc', data,
                               context={'in': _in, 'id': 'gauss'},
                               params={'score': 'bic-g'})
    print('\nDAG learnt from 5K rows of gauss: {}\n\n{}'.format(dag, trace))
    assert dag.to_string() == '[A][B|A][C|A:B][D|B][E][F|A:C:D:E:G][G]'


def test_bnlearn_hc_gauss_4_ok():  # Gaussian example, 5K rows, rev ord
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = NumPy.read(_in, dstype='continuous')
    data.set_order(tuple(list(data.get_order())[::-1]))
    dag, trace = bnlearn_learn('hc', data,
                               context={'in': _in, 'id': 'gauss'},
                               params={'score': 'bic-g'})
    print('\nDAG learnt from 5K rows of gauss: {}\n\n{}'.format(dag, trace))
    assert dag.to_string() == '[A|B][B|D][C|A:B][D][E][F|A:C:D:E:G][G]'


def test_bnlearn_hc_sachs_c_1_ok():  # Sachs gauss example, 1K rows
    _in = TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz'
    data = NumPy.read(_in, dstype='continuous')
    dag, trace = bnlearn_learn('hc', data,
                               context={'in': _in, 'id': 'gauss'},
                               params={'score': 'bic-g'})
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


def test_bnlearn_hc_sachs_c_2_ok():  # Sachs gauss example, rev, 1K rows
    _in = TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz'
    data = NumPy.read(_in, dstype='continuous')
    data.set_order(tuple(list(data.get_order())[::-1]))
    dag, trace = bnlearn_learn('hc', data,
                               context={'in': _in, 'id': 'gauss'},
                               params={'score': 'bic-g'})
    print('\nDAG rom 1K rows of sachs_c: {}\n\n{}'.format(dag, trace))
    assert dag.to_string() == \
        ('[Akt|Erk:PKA]' +
         '[Erk|PKA]' +
         '[Jnk|PKC]' +
         '[Mek|Raf]' +
         '[P38|PIP3:PKC]' +
         '[PIP2|Akt:Mek:PIP3:Plcg]' +
         '[PIP3|Plcg]' +
         '[PKA|Jnk:Mek:P38:PIP3:PKC:Raf]' +
         '[PKC|Mek:Raf]' +
         '[Plcg][Raf]')
