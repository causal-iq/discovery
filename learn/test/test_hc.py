
#   Test thebasics of hc hill-climbing structure learning

import pytest
from pandas import DataFrame, set_option

from fileio.common import TESTDATA_DIR
from fileio.oracle import Oracle
from fileio.pandas import Pandas
from core.bn import BN
from learn.hc import hc
from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_learn


# Set pandas options so that whole dataframe is printed
@pytest.fixture
def showall():
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)


# --- Failure cases

# bad arg types
def test_hc_type_error1():
    with pytest.raises(TypeError):
        hc()
    with pytest.raises(TypeError):
        hc(37)
    with pytest.raises(TypeError):
        hc('bad arg type')
    with pytest.raises(TypeError):
        data = DataFrame({'A': ['0', '1'], 'B': ['1', '2']})
        hc(data, params=77.7)
    with pytest.raises(TypeError):
        data = DataFrame({'A': ['0', '1'], 'B': ['1', '2']})
        hc(data, params=1)


# No longer possible to use dict
def test_hc_type_error2():
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    with pytest.raises(TypeError):
        hc({'N': N, 'bn': bn, 'order': bn.dag.nodes})


# params has bad type
def test_hc_type_error3():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(data, params='bic')
    with pytest.raises(TypeError):
        hc(data, params=True)


# Tabu param has bad type
def test_hc_type_error4():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': False})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': 12.7})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': 'bad type'})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': [12]})


# bnlearn has bad type
def test_hc_type_error5():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'bnlearn': 'invalid'})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'bnlearn': 0})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'bnlearn': [True]})


# knowledge has bad type
def test_hc_type_error6():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        dag, _ = hc(data, knowledge={'arcs': []})


# DataFrame only has one variable
def test_hc_value_error1():
    with pytest.raises(ValueError):
        hc(DataFrame({'A': ['0', '1']}))


# DataFrame only has one row
def test_hc_value_error2():
    with pytest.raises(ValueError):
        hc(DataFrame({'A': ['0'], 'B': ['1']}))


# only score parameter supported
def test_hc_value_error3():
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={'unknown': 3})


# invalid score specified
def test_hc_value_error4():
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={'score': 3})
    with pytest.raises(ValueError):
        hc(data, params={'score': 'invalid score'})


# invalid maxiter specified
def test_hc_value_error5():
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={'maxiter': 'invalid'})
    with pytest.raises(ValueError):
        hc(data, params={'maxiter': 0})


# invalid tabu specified
def test_hc_value_error6():
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 101, 'bnlearn': False})
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 0, 'bnlearn': False})


# invalid prefer parameter specified
def test_hc_value_error7():
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'prefer': 'invalid'})
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'prefer': None})
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'prefer': False})


# --- A->B learnt correctly for 10, 100 and 1K rows

# A->B 10 rows, no trace
@requires_r_and_bnlearn
def test_hc_ab_10_ok_1(showall):
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(10)
    dag, _ = hc(data)
    print('\nLearning DAG from 10 rows of A->B produces:\n{}'.format(dag))
    dag_bnlearn, _ = bnlearn_learn('hc', Pandas(data))
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn


# A->B 10 rows
@requires_r_and_bnlearn
def test_hc_ab_10_ok_2(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 10 rows, k is 2
@requires_r_and_bnlearn
def test_hc_ab_10_ok_2a(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    params = {'score': 'bic', 'k': 2}
    dag, trace = hc(data, context=context, params=params)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context, params=params)
    assert dag.to_string() == '[A][B]'  # higher complexity suppresses edge
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 10 rows, BDeu score
@requires_r_and_bnlearn
def test_hc_ab_10_ok_3(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    params = {'score': 'bde'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 10 rows, BDeu score, iss=5
@requires_r_and_bnlearn
def test_hc_ab_10_ok_3a(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    params = {'score': 'bde', 'iss': 5}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 10 rows, BDS score
@requires_r_and_bnlearn
def test_hc_ab_10_ok_4(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    params = {'score': 'bds'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 10 rows, BDS score, ISS=0.1
@requires_r_and_bnlearn
def test_hc_ab_10_ok_4a(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    params = {'score': 'bds', 'iss': 0.1}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 10 rows, Loglik score
@requires_r_and_bnlearn
def test_hc_ab_10_ok_5(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    params = {'score': 'loglik'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 100 rows
@requires_r_and_bnlearn
def test_hc_ab_100_ok_1(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_100', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 100 rows, BDeu score
@requires_r_and_bnlearn
def test_hc_ab_100_ok_2(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 100
    params = {'score': 'bde'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_100', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 100 rows, BDS score
@requires_r_and_bnlearn
def test_hc_ab_100_ok_3(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 100
    params = {'score': 'bds'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_100', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 100 rows, Log likelihood score
@requires_r_and_bnlearn
def test_hc_ab_100_ok_4(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 100
    params = {'score': 'loglik'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_100', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 1k rows
@requires_r_and_bnlearn
def test_hc_ab_1k_ok_1(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 1k rows, k = 0.5
@requires_r_and_bnlearn
def test_hc_ab_1k_ok_1a(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_1k', 'in': dsc}
    params = {'score': 'bic', 'k': 0.5}
    dag, trace = hc(data, context=context, params=params)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context, params=params)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 1k rows, BDeu score
@requires_r_and_bnlearn
def test_hc_ab_1k_ok_2(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    params = {'score': 'bde'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 1k rows, BDS score
@requires_r_and_bnlearn
def test_hc_ab_1k_ok_3(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    params = {'score': 'bds'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B 1k rows, Log-Likelihood score
@requires_r_and_bnlearn
def test_hc_ab_1k_ok_4(showall):
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    params = {'score': 'loglik'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- B->A always learnt as A->B because of equivalence and node order

# A<-B 10 rows
@requires_r_and_bnlearn
def test_hc_ba_10_ok(showall):
    dsc = '/discrete/tiny/ba.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ba_10', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A<-B 100 rows
@requires_r_and_bnlearn
def test_hc_ba_100_ok(showall):
    dsc = '/discrete/tiny/ba.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ba_100', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A<-B 1k rows
@requires_r_and_bnlearn
def test_hc_ba_1k_ok(showall):
    dsc = '/discrete/tiny/ba.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ba_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- A->B->C learnt correctly because of node order at 10, 100, 1K rows

# A->B->C 10 rows
@requires_r_and_bnlearn
def test_hc_abc_10_ok(showall):
    dsc = '/discrete/tiny/abc.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_10', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B->C 10 rows
@requires_r_and_bnlearn
def test_hc_abc_100_ok(showall):
    dsc = '/discrete/tiny/abc.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_100', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B->C 1k rows
@requires_r_and_bnlearn
def test_hc_abc_1k_ok_1(showall):
    dsc = '/discrete/tiny/abc.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B->C 1k rows, BDeu score
@requires_r_and_bnlearn
def test_hc_abc_1k_ok_2(showall):
    dsc = '/discrete/tiny/abc.dsc'
    N = 1000
    params = {'score': 'bde'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B->C 1k rows, BDS score
@requires_r_and_bnlearn
def test_hc_abc_1k_ok_3(showall):
    dsc = '/discrete/tiny/abc.dsc'
    N = 1000
    params = {'score': 'bds'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context, params=params)
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B->C 1k rows, Log-likelihood score
@requires_r_and_bnlearn
def test_hc_abc_1k_ok_4(showall):
    dsc = '/discrete/tiny/abc.dsc'
    N = 1000
    params = {'score': 'loglik'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A][C|A:B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B->C 1k rows
@requires_r_and_bnlearn
def test_hc_abc_3_1k_ok(showall):
    dsc = '/discrete/tiny/abc_3.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_3_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B<-C 10 row
@requires_r_and_bnlearn
def test_hc_ab_cb_10_ok(showall):
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_10', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A|C][B][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B<-C 100 rows
@requires_r_and_bnlearn
def test_hc_ab_cb_100_ok(showall):
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_100', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B][C|B]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B<-C 1k rows
@requires_r_and_bnlearn
def test_hc_ab_cb_1k_ok_1(showall):
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A][C|A:B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B<-C 1k rows, BDeu score
@requires_r_and_bnlearn
def test_hc_ab_cb_1k_ok_2(showall):
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    params = {'score': 'bde'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A][C|A:B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B<-C 1k rows, BDS score
@requires_r_and_bnlearn
def test_hc_ab_cb_1k_ok_3(showall):
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    params = {'score': 'bds'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A][C|A:B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B<-C 1k rows, Log-likelihood score
@requires_r_and_bnlearn
def test_hc_ab_cb_1k_ok_4(showall):
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    params = {'score': 'loglik'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A][C|A:B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B<-C 10k rows
@requires_r_and_bnlearn
def test_hc_ab_cb_10k_ok(showall):
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_10k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A][C|A:B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- Learning and4_10: X1->X2->X4, X3->X2 - variation with N

# X1->X2->X4, X3->X2 10 rows
@requires_r_and_bnlearn
def test_hc_and4_10_10_ok(showall):
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_10', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[X1][X2][X3][X4]'
    assert dag.number_components() == 4
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# X1->X2->X4, X3->X2 100 rows
@requires_r_and_bnlearn
def test_hc_and4_10_100_ok(showall):
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_100', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[X1][X2][X3|X2][X4|X2]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# X1->X2->X4, X3->X2 200 rows
@requires_r_and_bnlearn
def test_hc_and4_10_200_ok(showall):
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 200
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_200', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[X1][X2|X1][X3|X2][X4|X2]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# X1->X2->X4, X3->X2 1K rows
@requires_r_and_bnlearn
def test_hc_and4_10_1k_ok_1(showall):
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_1K', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[X1][X2|X1][X3|X2][X4|X2]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# X1->X2->X4, X3->X2 1K rows, BDeu score
@requires_r_and_bnlearn
def test_hc_and4_10_1k_ok_2(showall):
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 1000
    params = {'score': 'bde'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_1K', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[X1][X2|X1][X3|X2][X4|X2]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# X1->X2->X4, X3->X2 1K rows, BDS score
@requires_r_and_bnlearn
def test_hc_and4_10_1k_ok_3(showall):
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 1000
    params = {'score': 'bds'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_1K', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[X1][X2|X1][X3|X2][X4|X2]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# X1->X2->X4, X3->X2 1K rows, Loglik
@requires_r_and_bnlearn
def test_hc_and4_10_1k_ok_4(showall):
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 1000
    params = {'score': 'loglik'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_1K', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[X1][X2|X1][X3|X1:X2][X4|X1:X2:X3]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- Cancer

# Cancer 1K rows
@requires_r_and_bnlearn
def test_hc_cancer_1k_ok_1(showall):
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert '[Cancer][Dyspnoea|Cancer][Pollution][Smoker|Cancer][Xray|Cancer]' \
        == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Cancer 1K rows, BDeu score
@requires_r_and_bnlearn
def test_hc_cancer_1k_ok_2(showall):
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    params = {'score': 'bde'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert '[Cancer][Dyspnoea|Cancer][Pollution][Smoker|Cancer][Xray|Cancer]' \
        == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Cancer 1K rows, BDS score
@requires_r_and_bnlearn
def test_hc_cancer_1k_ok_3(showall):
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    params = {'score': 'bds'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert '[Cancer][Dyspnoea|Cancer][Pollution][Smoker|Cancer][Xray|Cancer]' \
        == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Cancer 1K rows, Log-likelihood score
@requires_r_and_bnlearn
def test_hc_cancer_1k_ok_4(showall):
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    params = {'score': 'loglik'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert ('[Cancer][Dyspnoea|Cancer:Pollution][Pollution|Cancer]' +
            '[Smoker|Cancer:Dyspnoea:Pollution:Xray]' +
            '[Xray|Cancer:Dyspnoea:Pollution]') == dag.to_string()
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- Asia 8-node model

# Asia 500 rows
@requires_r_and_bnlearn
def test_hc_asia_500_ok(showall):
    dsc = '/discrete/small/asia.dsc'
    N = 500
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_500', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert ('[asia][bronc][dysp|bronc][either|dysp][lung|either][smoke|bronc' +
            ':lung][tub|either:lung][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Asia 1K rows
@requires_r_and_bnlearn
def test_hc_asia_1k_ok_1(showall):
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert ('[asia][bronc][dysp|bronc][either|bronc:dysp][lung|either][smoke' +
            '|bronc:lung][tub|either:lung][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Asia 1K rows, BDeu score
@requires_r_and_bnlearn
def test_hc_asia_1k_ok_2(showall):
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    params = {'score': 'bde'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert ('[asia][bronc|smoke][dysp|bronc:either][either][lung|either]' +
            '[smoke|lung][tub|either:lung][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Asia 1K rows, BDS score
@requires_r_and_bnlearn
def test_hc_asia_1k_ok_3(showall):
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    params = {'score': 'bds'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert ('[asia][bronc|smoke][dysp|bronc:either][either][lung|either]' +
            '[smoke|lung][tub|either:lung][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Asia 1K rows, Loglik score
@requires_r_and_bnlearn
def test_hc_asia_1k_ok_4(showall):
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    params = {'score': 'loglik'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert ('[asia][bronc|asia:lung:tub]' +
            '[dysp|asia:bronc:either:lung:tub:xray][either|asia]' +
            '[lung|either][smoke|asia:bronc:dysp:lung:tub:xray]' +
            '[tub|either:lung][xray|asia:bronc:either]') == dag.to_string()
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- CHILD 20-node BN

# Child 1K rows
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_child_1k_ok(showall):
    ('display.max_rows', None)
    dsc = '/discrete/medium/child.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/child_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Child 10K rows
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_child_10k_ok_1(showall):
    dsc = '/discrete/medium/child.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/child_10k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Child 10K rows, BDeu score
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_child_10k_ok_2(showall):
    dsc = '/discrete/medium/child.dsc'
    N = 10000
    params = {'score': 'bic'}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/child_10k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               params=params, context=context)
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- INSURANCE 27-node BN

# Insurance 1K rows
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_insurance_1k_ok(showall):
    dsc = '/discrete/medium/insurance.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/insurance_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Insurance 10K rows
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_insurance_10k_ok(showall):
    dsc = '/discrete/medium/insurance.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/insurance_10k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- ALARM 37-node BN

# Alarm 1K rows
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_alarm_1k_ok(showall):
    dsc = '/discrete/medium/alarm.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/alarm_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Alarm 1K rows
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_alarm_10k_ok(showall):
    dsc = '/discrete/medium/alarm.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/alarm_10k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- HAILFINDER 56-node BN

# Hailfinder 10K rows
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_hailfinder_10k_ok(showall):
    dsc = '/discrete/large/hailfinder.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/hailfinder_10k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Hailfinder 25K rows
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_hailfinder_25k_ok(showall):
    dsc = '/discrete/large/hailfinder.dsc'
    N = 25000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/hailfinder_25k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- HEPAR 70-node BN

# HEPAR2 10K rows
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_hepar2_10k_ok(showall):
    dsc = '/discrete/large/hepar2.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/hepar2_10k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- PATHFINDER 109-node BN

# Pathfinder 1K rows
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_pathfinder_1k_ok(showall):
    dsc = '/discrete/verylarge/pathfinder.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    bn, data, removed = bn.remove_single_valued(data)
    print('\n\nNodes removed: {}'.format(removed))
    context = {'id': 'test/hc/pathfinder_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.number_components() == 6
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# Pathfinder 10K rows
@pytest.mark.slow
@requires_r_and_bnlearn
def test_hc_pathfinder_5k_ok(showall):
    dsc = '/discrete/verylarge/pathfinder.dsc'
    N = 5000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    bn, data, removed = bn.remove_single_valued(data)
    print('\n\nNodes removed: {}'.format(removed))
    context = {'id': 'test/hc/pathfinder_5k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('hc', Pandas(data),
                                               context=context)
    assert dag.number_components() == 4
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# --- Oracle-based tests

# Cancer 1K rows
def test_hc_oracle_cancer_1k_ok(showall):
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = Oracle(bn=bn)
    data.set_N(1000)
    data.set_order(tuple(['Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                          'Xray']))
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert '[Cancer][Dyspnoea][Pollution][Smoker|Cancer][Xray|Cancer]' \
        == dag.to_string()
    assert dag.number_components() == 3
