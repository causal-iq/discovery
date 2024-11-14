
#   Test thTabu hill-climbing structurlearning

import pytest
from pandas import set_option

from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from core.bn import BN
from learn.hc import hc
from call.bnlearn import bnlearn_learn
from core.graph import PDAG


@pytest.fixture
def showall():
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)


def test_tabu_type_error1():  # Tabu param has bad type
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


def test_tabu_type_error2():  # noinc param has bad type
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': 10, 'noinc': True})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': 10, 'noinc': 13.2})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': 10, 'noinc': 'bad type'})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': 10, 'noinc': [12]})


def test_tabu_type_error3():  # bnlearn param has bad type
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': 10, 'bnlearn': {True}})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': 10, 'bnlearn': 13.2})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': 10, 'bnlearn': 'bad type'})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'tabu': 10, 'bnlearn': [True]})


def test_tabu_value_error1():  # invalid tabu value specified
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 101})
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 0})


def test_tabu_value_error2():  # noinc specified without tabu
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={'noinc': 4})
    with pytest.raises(ValueError):
        hc(data, params={'noinc': 10})


def test_tabu_value_error3():  # invalid noinc value specified
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'noinc': 0})
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'noinc': -1})

# A->B learnt correctly for 10, 100 and 1K rows


def test_tabu_ab_10_1_ok(showall):  # A->B 10 rows, no trace, tabu=1
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(10)
    dag, _ = hc(data, params={'tabu': 1, 'bnlearn': False})
    print('\nLearning DAG from 10 rows of A->B produces:\n{}'.format(dag))
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1


def test_tabu_ab_10_2_ok(showall):  # A->B 10 rows, no trace, tabu=10
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(10)
    dag, _ = hc(data, params={'tabu': 10, 'bnlearn': False})
    print('\nLearning DAG from 10 rows of A->B produces:\n{}'.format(dag))
    dag_bnlearn, _ = bnlearn_learn('tabu', Pandas(data))
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn


def test_tabu_ab_10_3_ok(showall):  # A->B 10 rows
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_10_4_ok(showall):  # A->B 10 rows, k is 2, empty DAG best

    # Note, not comparing with bnlearn becausit erroneously doesn't return
    # thinitial DAG. Increasing k to 2 means empty DAG is highest scoring.

    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    params = {'score': 'bic', 'k': 2, 'tabu': 10, 'bnlearn': False}
    dag, trace = hc(data, context=context, params=params)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B]'
    assert dag.number_components() == 2


def test_tabu_ab_10_5_ok(showall):  # A->B 10 rows, BDeu score
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    params = {'score': 'bde', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_10_6_ok(showall):  # A->B 10 rows, BDeu score, iss=5
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    params = {'score': 'bde', 'iss': 5, 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_10_7_ok(showall):  # A->B 10 rows, BDS score

    # don't comparwith bnlearn as bnlearn chosB->A over equal scoring A->B
    # suspect just scoring rounding difference

    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    params = {'score': 'bds', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag_bnlearn.to_string() == '[A|B][B]'
    assert trace.diffs_from(trace_bnlearn, strict=False) is None
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_10_8_ok(showall):  # A->B 10 rows, BDS score, ISS=0.1
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    params = {'score': 'bds', 'iss': 0.1, 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_10_9_ok(showall):  # A->B 10 rows, Loglik score
    dsc = '/discrete/tiny/ab.dsc'
    N = 10
    params = {'score': 'loglik', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_10', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_100_1_ok(showall):  # A->B 100 rows
    dsc = '/discrete/tiny/ab.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_100', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_100_2_ok(showall):  # A->B 100 rows, BDeu score
    dsc = '/discrete/tiny/ab.dsc'
    N = 100
    params = {'score': 'bde', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_100', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_100_3_ok(showall):  # A->B 100 rows, BDS score
    dsc = '/discrete/tiny/ab.dsc'
    N = 100
    params = {'score': 'bds', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_100', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_100_4_ok(showall):  # A->B 100 rows, Log likelihood score
    dsc = '/discrete/tiny/ab.dsc'
    N = 100
    params = {'score': 'loglik', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_100', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_1k_1_ok(showall):  # A->B 1k rows
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_1k', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_1k_2_ok(showall):  # A->B 1k rows, k = 0.5
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_1k', 'in': dsc}
    params = {'score': 'bic', 'k': 0.5, 'tabu': 10, 'bnlearn': False}
    dag, trace = hc(data, context=context, params=params)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context, params=params)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_1k_3_ok(showall):  # A->B 1k rows, BDeu score
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    params = {'score': 'bde', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_1k_4_ok(showall):  # A->B 1k rows, BDS score
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    params = {'score': 'bds', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_1k_5_ok(showall):  # A->B 1k rows, Log-Likelihood score
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    params = {'score': 'loglik', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))

# B->A always learnt as A->B becausof equivalencand nodorder


def test_tabu_ba_10_ok(showall):  # A<-B 10 rows
    dsc = '/discrete/tiny/ba.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ba_10', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ba_100_ok(showall):  # A<-B 100 rows
    dsc = '/discrete/tiny/ba.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ba_100', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ba_1k_ok(showall):  # A<-B 1k rows
    dsc = '/discrete/tiny/ba.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ba_1k', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


# A->B->C learnt correctly becausof nodorder at 10, 100, 1K rows

def test_tabu_abc_10_1_ok(showall):  # A->B->C 10 rows
    dsc = '/discrete/tiny/abc.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_10', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_abc_10_2_ok(showall):  # A->B->C 10 rows, noinc=5
    dsc = '/discrete/tiny/abc.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_10', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False, 'noinc': 5})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag

    # noinc = 5 prevents later iterations

    major, _, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {('delete', 'missing'): {('B', 'C'): (None, 9),
                                    ('A', 'C'): (None, 13)},
            ('add', 'missing'): {('A', 'B'): (None, 10),
                                 ('B', 'C'): (None, 14)},
            ('reverse', 'missing'): {('C', 'A'): (None, 11),
                                     ('A', 'B'): (None, 12),
                                     ('B', 'A'): (None, 15)},
            ('stop', 'order'): {None: (9, 16)}}
    print(major)

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_abc_10_3_ok(showall):  # A->B->C 10 rows, noinc=2
    dsc = '/discrete/tiny/abc.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_10', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False, 'noinc': 2})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag

    # noinc = 5 prevents later iterations

    major, _, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {('delete', 'missing'): {('B', 'A'): (None, 5),
                                    ('B', 'C'): (None, 9),
                                    ('A', 'C'): (None, 13)},
            ('add', 'missing'): {('A', 'C'): (None, 6),
                                 ('A', 'B'): (None, 10),
                                 ('B', 'C'): (None, 14)},
            ('reverse', 'missing'): {('A', 'C'): (None, 7),
                                     ('C', 'B'): (None, 8),
                                     ('C', 'A'): (None, 11),
                                     ('A', 'B'): (None, 12),
                                     ('B', 'A'): (None, 15)},
            ('stop', 'order'): {None: (5, 16)}}

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_abc_100_ok(showall):  # A->B->C 100 rows

    # bnlearn and bnbench return different but equivalent DAGs

    dsc = '/discrete/tiny/abc.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_100', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert PDAG.fromDAG(dag) == PDAG.fromDAG(dag_bnlearn)
    assert trace.result == dag
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_abc_1k_1_ok(showall):  # A->B->C 1k rows

    # differences between scores and iteration 3 blocked ... manually checked
    # and bnbench HC implementation OK

    dsc = '/discrete/tiny/abc.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_1k', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_abc_1k_2_ok(showall):  # A->B->C 1k rows, BDeu score
    dsc = '/discrete/tiny/abc.dsc'
    N = 1000
    params = {'score': 'bde', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [3, 12, 14]  # differences in blicked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_abc_1k_3_ok(showall):  # A->B->C 1k rows, BDS score
    dsc = '/discrete/tiny/abc.dsc'
    N = 1000
    params = {'score': 'bds', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context, params=params)
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert PDAG.fromDAG(dag) == PDAG.fromDAG(dag_bnlearn)
    assert trace.result == dag
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [3, 12, 14]  # differences in blicked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_abc_1k_4_ok(showall):  # A->B->C 1k rows, Log-likelihood score
    dsc = '/discrete/tiny/abc.dsc'
    N = 1000
    params = {'score': 'loglik', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    assert dag.to_string() == '[A][B|A][C|A:B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [4]  # differences in blicked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_abc_3_1k_ok(showall):  # A->B->C 1k rows
    dsc = '/discrete/tiny/abc_3.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/abc_3_1k', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False, 'maxiter': 40})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_10_ok(showall):  # A->B<-C 10 rows
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_10', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A|C][B][C|B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [3]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_100_1_ok(showall):  # A->B<-C 100 rows
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_100', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B][C|B]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [12, 13]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_100_2_ok(showall):  # A->B<-C 100 rows, noinc=10
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_100', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False, 'noinc': 10})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B][C|B]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [12, 13]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_100_3_ok(showall):  # A->B<-C 100 rows, noinc=5
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_100', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False, 'noinc': 5})
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    assert dag.to_string() == '[A][B][C|B]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {('add', 'missing'): {('A', 'C'): (None, 7)},
            ('delete', 'missing'): {('A', 'B'): (None, 8),
                                    ('A', 'C'): (None, 12)},
            ('reverse', 'missing'): {('A', 'C'): (None, 9),
                                     ('C', 'B'): (None, 10),
                                     ('C', 'A'): (None, 11),
                                     ('B', 'C'): (None, 13)},
            ('stop', 'order'): {None: (7, 14)}} == major

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_1k_1_ok(showall):  # A->B<-C 1k rows
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_1k', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[A][B|A:C][C]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn

    # Extra entries in bnlearn tracas it doesn't terminatat correct point
    # duto resetting non-posivdelta count wrongly

    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {('stop', 'order'): {None: (18, 24)},
                     ('reverse', 'missing'): {('B', 'C'): (None, 18),
                                              ('B', 'A'): (None, 19),
                                              ('C', 'B'): (None, 23)},
                     ('delete', 'missing'): {('C', 'A'): (None, 20),
                                             ('A', 'B'): (None, 22)},
                     ('add', 'missing'): {('A', 'C'): (None, 21)}}
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_1k_2_ok(showall):  # A->B<-C 1k rows
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_1k', 'in': dsc}
    dag, trace = hc(data, context=context, params={'tabu': 10})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[A][B|A:C][C]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn

    # Extra entries in bnlearn tracas it doesn't terminatat correct point
    # duto resetting non-posivdelta count wrongly

    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}
    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_1k_3_ok(showall):  # A->B<-C 1k rows, BDeu score
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    params = {'score': 'bde', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))
    print('Blocked differences:')
    iter = 0
    for t, b in zip(trace.trace['blocked'], trace_bnlearn.trace['blocked']):
        if t != b:
            print(iter, t, b)
        iter += 1
    assert dag.to_string() == '[A][B|A:C][C]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn

    # Again bnlearn has extra tracentries

    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {('stop', 'order'): {None: (18, 24)},
                     ('reverse', 'missing'): {('B', 'C'): (None, 18),
                                              ('B', 'A'): (None, 19),
                                              ('C', 'B'): (None, 23)},
                     ('delete', 'missing'): {('C', 'A'): (None, 20),
                                             ('A', 'B'): (None, 22)},
                     ('add', 'missing'): {('A', 'C'): (None, 21)}}
    assert trace.result == dag_bnlearn

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_1k_4_ok(showall):  # A->B<-C 1k rows, BDeu score
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    params = {'score': 'bde', 'tabu': 10}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))
    print('Blocked differences:')
    iter = 0
    for t, b in zip(trace.trace['blocked'], trace_bnlearn.trace['blocked']):
        if t != b:
            print(iter, t, b)
        iter += 1
    assert dag.to_string() == '[A][B|A:C][C]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn

    # Again bnlearn has extra tracentries

    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}
    assert trace.result == dag_bnlearn

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_1k_5_ok(showall):  # A->B<-C 1k rows, BDS score
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    params = {'score': 'bds', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[A][B|A:C][C]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn

    # Again bnlearn has extra tracentries

    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {('stop', 'order'): {None: (18, 24)},
                     ('reverse', 'missing'): {('B', 'C'): (None, 18),
                                              ('B', 'A'): (None, 19),
                                              ('C', 'B'): (None, 23)},
                     ('delete', 'missing'): {('C', 'A'): (None, 20),
                                             ('A', 'B'): (None, 22)},
                     ('add', 'missing'): {('A', 'C'): (None, 21)}}
    assert trace.result == dag_bnlearn

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_1k_6_ok(showall):  # A->B<-C 1k rows, BDS score
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    params = {'score': 'bds', 'tabu': 10, 'bnlearn': True}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[A][B|A:C][C]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn

    # Again bnlearn has extra tracentries

    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}
    assert trace.result == dag_bnlearn

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_1k_7_ok(showall):  # A->B<-C 1k rows, Log-likelihood score
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    params = {'score': 'loglik', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[A][B|A][C|A:B]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [5, 13]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_10k_1_ok(showall):  # A->B<-C 10k rows
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_10k', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[A][B|A:C][C]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn

    # Extra entries in bnlearn tracas it doesn't terminatat correct point
    # duto resetting non-posivdelta count wrongly

    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {('stop', 'order'): {None: (18, 24)},
                     ('reverse', 'missing'): {('B', 'C'): (None, 18),
                                              ('B', 'A'): (None, 19),
                                              ('C', 'B'): (None, 23)},
                     ('delete', 'missing'): {('C', 'A'): (None, 20),
                                             ('A', 'B'): (None, 22)},
                     ('add', 'missing'): {('A', 'C'): (None, 21)}}

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_ab_cb_10k_2_ok(showall):  # A->B<-C 10k rows
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/ab_cb_10k', 'in': dsc}
    dag, trace = hc(data, context=context, params={'tabu': 10})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[A][B|A:C][C]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn

    # Extra entries in bnlearn tracas it doesn't terminatat correct point
    # duto resetting non-posivdelta count wrongly

    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


#   Learning and4_10: X1->X2->X4, X3->X2 - variation with N

def test_tabu_and4_10_1_ok(showall):  # X1->X2->X4, X3->X2 10 rows
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_10', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[X1][X2][X3][X4|X1:X2]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, _, _ = trace.diffs_from(trace_bnlearn, strict=False)

    # Extra tracentries in bnlearn

    assert major == {('delete', 'missing'): {('X1', 'X4'): (None, 16),
                                             ('X2', 'X4'): (None, 17)},
                     ('stop', 'order'): {None: (16, 18)}}

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_and4_10_2_ok(showall):  # X1->X2->X4, X3->X2 10 rows
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 10
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_10', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': True})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[X1][X2][X3][X4|X1:X2]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, _, _ = trace.diffs_from(trace_bnlearn, strict=False)

    # Extra tracentries in bnlearn

    assert major == {}

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_and4_10_3_ok(showall):  # X1->X2->X4, X3->X2 100 rows
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_100', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[X1][X2][X3|X2][X4|X2]'
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [12, 13]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_and4_10_4_ok(showall):  # X1->X2->X4, X3->X2 200 rows
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 200
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_200', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[X1][X2|X1][X3|X2][X4|X2]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [8, 9]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_and4_10_5_ok(showall):  # X1->X2->X4, X3->X2 1K rows
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_1K', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[X1|X2][X2|X4][X3|X2][X4]'
    assert dag.number_components() == 1

    # bnlearn and bnbench return different but equivalent DAGs

    assert PDAG.fromDAG(dag) == PDAG.fromDAG(dag_bnlearn)
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [4]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_and4_10_6_ok(showall):  # X1->X2->X4, X3->X2 1K rows, BDeu
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 1000
    params = {'score': 'bde', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_1K', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[X1][X2|X1][X3|X2][X4|X2]'
    assert dag.number_components() == 1

    # bnlearn and bnbench return different but equivalent DAGs

    assert PDAG.fromDAG(dag) == PDAG.fromDAG(dag_bnlearn)
    assert trace.result == dag
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [4, 10, 13]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_and4_10_7_ok(showall):  # X1->X2->X4, X3->X2 1K rows, BDS
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 1000
    params = {'score': 'bds', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_1K', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[X1][X2|X1][X3|X2][X4|X2]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [4, 10, 13]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_and4_10_8_ok(showall):  # X1->X2->X4, X3->X2 1K rows, Loglik
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 1000
    params = {'score': 'loglik', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/and4_10_1K', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.to_string() == '[X1][X2|X1][X3|X1:X2][X4|X1:X2:X3]'
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [7]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_cancer_1_ok(showall):  # Cancer 1K rows
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert '[Cancer][Dyspnoea|Cancer][Pollution][Smoker|Cancer][Xray|Cancer]' \
        == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [9, 15]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_cancer_2_ok(showall):  # Cancer 1K rows, BDeu score
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    params = {'score': 'bde', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[Cancer|Pollution:Smoker][Dyspnoea|Cancer][Pollution][Smoker]' +
            '[Xray|Cancer]') == dag.to_string()
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn

    # extra entries in bnlearn trace

    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {('add', 'missing'): {('Dyspnoea', 'Cancer'): (None, 26),
                                 ('Pollution', 'Cancer'): (None, 24)},
            ('delete', 'missing'): {('Cancer', 'Dyspnoea'): (None, 25),
                                    ('Pollution', 'Cancer'): (None, 27),
                                    ('Pollution', 'Smoker'): (None, 21)},
            ('reverse', 'missing'): {('Cancer', 'Smoker'): (None, 22),
                                     ('Cancer', 'Xray'): (None, 19),
                                     ('Dyspnoea', 'Cancer'): (None, 20),
                                     ('Smoker', 'Cancer'): (None, 28),
                                     ('Xray', 'Cancer'): (None, 23)},
            ('stop', 'order'): {None: (19, 30)}} == major

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_cancer_3_ok(showall):  # Cancer 1K rows, BDeu score
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    params = {'score': 'bde', 'tabu': 10}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[Cancer|Pollution:Smoker][Dyspnoea|Cancer][Pollution][Smoker]' +
            '[Xray|Cancer]') == dag.to_string()
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn

    # extra entries in bnlearn trace

    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_cancer_4_ok(showall):  # Cancer 1K rows, BDeu score, noinc=15
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    params = {'score': 'bde', 'tabu': 10, 'bnlearn': False, 'noinc': 15}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[Cancer|Pollution:Smoker][Dyspnoea|Cancer][Pollution][Smoker]' +
            '[Xray|Cancer]') == dag.to_string()
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn

    # noinc = 15 means have same number of entries in trace

    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}
    assert minor == [4, 19, 22, 25]

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_cancer_5_ok(showall):  # Cancer 1K rows, BDS score
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    params = {'score': 'bds', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[Cancer|Pollution:Smoker][Dyspnoea|Cancer][Pollution]' +
            '[Smoker][Xray|Cancer]') == dag.to_string()
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {('add', 'missing'): {('Dyspnoea', 'Cancer'): (None, 26),
                                 ('Pollution', 'Cancer'): (None, 24)},
           ('delete', 'missing'): {('Cancer', 'Dyspnoea'): (None, 25),
                                   ('Pollution', 'Cancer'): (None, 27),
                                   ('Pollution', 'Smoker'): (None, 21)},
           ('reverse', 'missing'): {('Cancer', 'Smoker'): (None, 22),
                                    ('Cancer', 'Xray'): (None, 19),
                                    ('Dyspnoea', 'Cancer'): (None, 20),
                                    ('Smoker', 'Cancer'): (None, 28),
                                    ('Xray', 'Cancer'): (None, 23)},
           ('stop', 'order'): {None: (19, 30)}} == major

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_cancer_6_ok(showall):  # Cancer 1K rows, BDS score
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    params = {'score': 'bds', 'tabu': 10}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[Cancer|Pollution:Smoker][Dyspnoea|Cancer][Pollution]' +
            '[Smoker][Xray|Cancer]') == dag.to_string()
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_cancer_7_ok(showall):  # Cancer 1K rows, Log-likelihood score
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    params = {'score': 'loglik', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/cancer_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               params=params, context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[Cancer][Dyspnoea|Cancer:Pollution][Pollution|Cancer]' +
            '[Smoker|Cancer:Dyspnoea:Pollution:Xray]' +
            '[Xray|Cancer:Dyspnoea:Pollution]') == dag.to_string()
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [12]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


#   Asia 8-nodmodel

def test_tabu_asia_1_ok(showall):  # Asia 500 rows
    dsc = '/discrete/small/asia.dsc'
    N = 500
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_500', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[asia][bronc|smoke][dysp|bronc:either][either|lung:tub][lung]' +
            '[smoke|lung][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {('reverse', 'missing'): {('smoke', 'tub'): (None, 21),
                                     ('lung', 'smoke'): (None, 22),
                                     ('smoke', 'bronc'): (None, 23),
                                     ('bronc', 'smoke'): (None, 26)},
            ('add', 'missing'): {('either', 'asia'): (None, 24)},
            ('delete', 'missing'): {('tub', 'smoke'): (None, 25),
                                    ('either', 'asia'): (None, 27)},
            ('stop', 'order'): {None: (21, 29)}} == major

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_asia_2_ok(showall):  # Asia 500 rows
    dsc = '/discrete/small/asia.dsc'
    N = 500
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_500', 'in': dsc}
    dag, trace = hc(data, context=context, params={'tabu': 10})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[asia][bronc|smoke][dysp|bronc:either][either|lung:tub][lung]' +
            '[smoke|lung][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_asia_3_ok(showall):  # Asia 1K rows
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[asia][bronc|smoke][dysp|bronc:either][either|lung:tub][lung]' +
            '[smoke|lung][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {('reverse', 'missing'): {('smoke', 'lung'): (None, 23),
                                     ('lung', 'smoke'): (None, 26),
                                     ('smoke', 'bronc'): (None, 27),
                                     ('bronc', 'smoke'): (None, 29)},
            ('add', 'missing'): {('smoke', 'tub'): (None, 24),
                                 ('either', 'asia'): (None, 28)},
            ('delete', 'missing'): {('either', 'asia'): (None, 25)},
            ('stop', 'order'): {None: (23, 30)}} == major

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_asia_4_ok(showall):  # Asia 1K rows
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, context=context, params={'tabu': 10})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[asia][bronc|smoke][dysp|bronc:either][either|lung:tub][lung]' +
            '[smoke|lung][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_asia_5_ok(showall):  # Asia 1K rows, BDeu score
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    params = {'score': 'bde', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context, params=params)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[asia][bronc|smoke][dysp|bronc:either][either|asia:lung:tub]' +
            '[lung][smoke|lung][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 1
    assert PDAG.fromDAG(dag) == PDAG.fromDAG(dag_bnlearn)
    assert trace.result == dag
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {('reverse', 'missing'): {('smoke', 'lung'): (None, 23),
                                     ('lung', 'smoke'): (None, 25)},
            ('delete', 'missing'): {('lung', 'asia'): (None, 24)},
            ('stop', 'order'): {None: (23, 26)}} == major

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_asia_6_ok(showall):  # Asia 1K rows, BDeu score
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    params = {'score': 'bde', 'tabu': 10}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context, params=params)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    # NB bnlearn & bnbenvh produce different orientation of smoke/lung even
    # though traces are the same - probably due to rounding differences when
    # determining highest scoring DAG.

    assert ('[asia][bronc|smoke][dysp|bronc:either][either|asia:lung:tub]' +
            '[lung][smoke|lung][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 1
    assert PDAG.fromDAG(dag) == PDAG.fromDAG(dag_bnlearn)
    assert trace.result == dag
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_asia_7_ok(showall):  # Asia 1K rows, BDS score
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    params = {'score': 'bds', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context, params=params)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[asia][bronc|smoke][dysp|bronc:either][either|lung:tub][lung]' +
            '[smoke|lung][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert PDAG.fromDAG(dag) == PDAG.fromDAG(dag_bnlearn)
    assert trace.result == dag
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {('add', 'missing'): {('lung', 'asia'): (None, 21)},
            ('delete', 'missing'): {('either', 'asia'): (None, 22),
                                    ('lung', 'asia'): (None, 24)},
            ('reverse', 'missing'): {('bronc', 'smoke'): (None, 23)},
            ('stop', 'order'): {None: (21, 25)}} == major

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_asia_8_ok(showall):  # Asia 1K rows, BDS score
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    params = {'score': 'bds', 'tabu': 10}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context, params=params)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    # NB bnlearn & bnbenvh produce different orientation of smoke/bronc even
    # though traces are the same - probably due to rounding differences when
    # determining highest scoring DAG.

    assert ('[asia][bronc|smoke][dysp|bronc:either][either|lung:tub]' +
            '[lung|smoke][smoke][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert PDAG.fromDAG(dag) == PDAG.fromDAG(dag_bnlearn)
    assert trace.result == dag
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {} == major

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_asia_9_ok(showall):  # Asia 1K rows, Loglik score
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    params = {'score': 'loglik', 'tabu': 10, 'bnlearn': False}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context, params=params)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[asia][bronc|asia:lung:tub]' +
            '[dysp|asia:bronc:either:lung:tub:xray][either|asia]' +
            '[lung|either][smoke|asia:bronc:dysp:lung:tub:xray]' +
            '[tub|either:lung][xray|asia:bronc:either]') == dag.to_string()
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [24]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


def test_tabu_asia_10_ok(showall):  # Asia 1K rows, Loglik score
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    params = {'score': 'loglik', 'tabu': 10}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/asia_1k', 'in': dsc}
    dag, trace = hc(data, params=params, context=context)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context, params=params)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert major == {}  # no major tracdifferences
    assert minor == [24]  # differences in blocked reporting

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


#   CHILD 20-nodBN

@pytest.mark.slow
def test_tabu_child_1_ok(showall):  # Child 1K rows
    dsc = '/discrete/medium/child.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/child_1k', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[Age|Disease][BirthAsphyxia][CO2|LungParench][CO2Report|CO2]' +
            '[CardiacMixing|Disease][ChestXray|LungFlow:LungParench]' +
            '[Disease|LungParench][DuctFlow|Disease]' +
            '[Grunting|LungParench:Sick][GruntingReport|Grunting]' +
            '[HypDistrib|CardiacMixing:DuctFlow][HypoxiaInO2|CardiacMixing]' +
            '[LVH|Disease][LVHreport|LVH]' +
            '[LowerBodyO2|HypDistrib:HypoxiaInO2][LungFlow|Disease]' +
            '[LungParench][RUQO2|HypoxiaInO2][Sick|Age]' +
            '[XrayReport|ChestXray]') == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {('reverse', 'missing'): {('Age', 'Sick'): (None, 39),
                                     ('Sick', 'Age'): (None, 41)},
            ('add', 'missing'): {('LVHreport', 'BirthAsphyxia'): (None, 40)},
            ('stop', 'order'): {None: (39, 42)}} == major
    assert [38] == minor

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


@pytest.mark.slow
def test_tabu_child_2_ok(showall):  # Child 1K rows
    dsc = '/discrete/medium/child.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/child_1k', 'in': dsc}
    dag, trace = hc(data, context=context, params={'tabu': 10})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[Age|Disease][BirthAsphyxia][CO2|LungParench][CO2Report|CO2]' +
            '[CardiacMixing|Disease][ChestXray|LungFlow:LungParench]' +
            '[Disease|LungParench][DuctFlow|Disease]' +
            '[Grunting|LungParench:Sick][GruntingReport|Grunting]' +
            '[HypDistrib|CardiacMixing:DuctFlow][HypoxiaInO2|CardiacMixing]' +
            '[LVH|Disease][LVHreport|LVH]' +
            '[LowerBodyO2|HypDistrib:HypoxiaInO2][LungFlow|Disease]' +
            '[LungParench][RUQO2|HypoxiaInO2][Sick|Age]' +
            '[XrayReport|ChestXray]') == dag.to_string()
    assert dag.number_components() == 2
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    major, minor, _ = trace.diffs_from(trace_bnlearn, strict=False)
    assert {} == major
    assert [38] == minor

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


@pytest.mark.slow
def test_tabu_child_3_ok(showall):  # Child 10K rows
    dsc = '/discrete/medium/child.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/child_10k', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tabu': 10, 'bnlearn': False})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[Age|Disease][BirthAsphyxia|Disease][CO2|LungParench]' +
            '[CO2Report|CO2][CardiacMixing][ChestXray|Disease]' +
            '[Disease|CardiacMixing][DuctFlow|Disease]' +
            '[Grunting|LungParench:Sick][GruntingReport|Grunting]' +
            '[HypDistrib|CardiacMixing:DuctFlow]' +
            '[HypoxiaInO2|CardiacMixing:LungParench][LVH|Disease]' +
            '[LVHreport|LVH][LowerBodyO2|HypDistrib:HypoxiaInO2]' +
            '[LungFlow|ChestXray:Disease][LungParench|ChestXray:LungFlow]' +
            '[RUQO2|HypoxiaInO2][Sick|Age:Disease]' +
            '[XrayReport|ChestXray]') == dag.to_string()
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


@pytest.mark.slow
def test_tabu_child_4_ok(showall):  # Child 10K rows
    dsc = '/discrete/medium/child.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/child_10k', 'in': dsc}
    dag, trace = hc(data, context=context, params={'tabu': 10})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    assert ('[Age|Disease][BirthAsphyxia|Disease][CO2|LungParench]' +
            '[CO2Report|CO2][CardiacMixing][ChestXray|Disease]' +
            '[Disease|CardiacMixing][DuctFlow|Disease]' +
            '[Grunting|LungParench:Sick][GruntingReport|Grunting]' +
            '[HypDistrib|CardiacMixing:DuctFlow]' +
            '[HypoxiaInO2|CardiacMixing:LungParench][LVH|Disease]' +
            '[LVHreport|LVH][LowerBodyO2|HypDistrib:HypoxiaInO2]' +
            '[LungFlow|ChestXray:Disease][LungParench|ChestXray:LungFlow]' +
            '[RUQO2|HypoxiaInO2][Sick|Age:Disease]' +
            '[XrayReport|ChestXray]') == dag.to_string()
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result == dag_bnlearn
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


@pytest.mark.slow
def test_tabu_child_5_ok(showall):  # Child 10K rows, BDeu score
    dsc = '/discrete/medium/child.dsc'
    N = 10000
    params = {'score': 'bic', 'tabu': 10}
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/child_10k', 'in': dsc}
    dag, trace = hc(data, context=context, params=params)
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    del params['tabu']
    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))
    assert dag.number_components() == 1
    assert dag == dag_bnlearn
    assert trace.result
    assert trace.diffs_from(trace_bnlearn, strict=False) is None

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


#   INSURANCE 27-node BN

@pytest.mark.slow
def test_tabu_insurance_1_ok(showall):  # Insuranc1K rows
    dsc = '/discrete/medium/insurance.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/insurance_1k', 'in': dsc}
    dag, trace = hc(data, context=context, params={'tabu': 10})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace_bnlearn,
                                                     dag_bnlearn))

    print(dag.to_string())
    print(dag_bnlearn.to_string())

    print(dag == dag_bnlearn)
    print(PDAG.fromDAG(dag) == PDAG.fromDAG(dag_bnlearn))
    print(dag.number_components())
    # assert dag.number_components() == 2
    # assert dag == dag_bnlearn
    # assert trace.result == dag_bnlearn
    print(trace.diffs_from(trace_bnlearn, strict=False))

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))


@pytest.mark.slow
def test_tabu_insurance_10k_ok(showall):  # Insuranc1K rows
    dsc = '/discrete/medium/insurance.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/insurance_10k', 'in': dsc}
    dag, trace = hc(data, context=context, params={'tabu': 10})
    print('\n\n{}\n\nbnbench produces:\n\n{}'.format(trace, dag))

    dag_bnlearn, trace_bnlearn = bnlearn_learn('tabu', Pandas(data),
                                               context=context)
    print('\n\n{}\n\nbnlearn produces:\n\n{}'.format(trace, dag))

    print(dag.to_string())
    print(dag_bnlearn.to_string())

    print(dag == dag_bnlearn)
    print(PDAG.fromDAG(dag) == PDAG.fromDAG(dag_bnlearn))
    print(dag.number_components())
    # assert dag.number_components() == 2
    # assert dag == dag_bnlearn
    # assert trace.result == dag_bnlearn
    print(trace.diffs_from(trace_bnlearn, strict=False))

    print('bnlearn was {} times faster'
          .format(round(trace.trace['time'][-1] /
                        (trace_bnlearn.trace['time'][-1] + 0.01), 2)))
