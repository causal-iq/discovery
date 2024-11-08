
# Tests of HC 'tree' search over different equivakent add sequences

import pytest

from fileio.common import TESTDATA_DIR
from core.bn import BN
from learn.hc import hc
from learn.knowledge import Knowledge
from learn.knowledge_rule import RuleSet


def test_hc_tree_type_error_1():  # tree parameter not tuple
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': 'invalid'})
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': None})
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': False})
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': 2})


def test_hc_tree_type_error_2():  # tree parameter wrong length
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': (1, )})
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': (1, 3)})
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': (1, -1, 4, 1)})


def test_hc_tree_type_error_3():  # tree tuple element types bad
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': (31.2, -1, 6)})
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False,
                         'tree': (True, -1, False)})
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False,
                         'tree': (1, 'bad', False)})
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False,
                         'tree': (2, True, False)})
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False,
                         'tree': (1, -1, [3])})
    with pytest.raises(TypeError):
        hc(data, params={'tabu': 10, 'bnlearn': False,
                         'tree': (2, -1, {False})})


def test_hc_tree_value_error_1():  # invalid depth value
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    context = {'id': 'test/hc_tree/ve_1', 'in': dsc}
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': (0, -1, 0)},
           context=context)
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': (11, -1, 0)},
           context=context)


def test_hc_tree_value_error_2():  # invalid width value
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    context = {'id': 'test/hc_tree/ve_2', 'in': dsc}
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': (2, -3, 0)},
           context=context)
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': (2, 101, 0)},
           context=context)


def test_hc_tree_value_error_3():  # invalid lookahead value
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    context = {'id': 'test/hc_tree/ve_2', 'in': dsc}
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': (2, -2, -2)},
           context=context)


def test_hc_tree_value_error_4():  # tree forbidden without context
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': (3, 8, 2)})


def test_hc_tree_value_error_5():  # tree cannot be specified with Knowledge
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 2, 'ref': bn, 'expertise': 1.0})
    data = bn.generate_cases(10)
    context = {'id': 'test/hc_tree/ve_4', 'in': dsc}
    with pytest.raises(ValueError):
        hc(data, params={'tabu': 10, 'bnlearn': False, 'tree': (1, -1, 0)},
           knowledge=know, context=context)


def test_hc_tree_ab_1_0_ok():  # HC, ab, tree=1,0
    dsc = '/discrete/tiny/ab.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/ab_1_0', 'in': dsc}
    dags, trace = hc(data, params={'tree': (1, 0, 0)}, context=context)


def test_hc_tree_ab_1_1_ok():  # HC, ab, tree=1,1
    dsc = '/discrete/tiny/ab.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/ab_1_0', 'in': dsc}
    dags, trace = hc(data, params={'tree': (1, 1, 0)}, context=context)


def test_hc_tree_ab_1_M1_ok():  # HC, ab, tree=1,-1
    dsc = '/discrete/tiny/ab.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/ab_1_m1', 'in': dsc}
    dags, trace = hc(data, params={'tree': (1, -1, 0)}, context=context)


def test_hc_tree_abc_1_0_ok():  # HC, abc, tree=1,0,0
    dsc = '/discrete/tiny/abc.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/abc_1', 'in': dsc}
    dags, trace = hc(data, params={'tree': (1, 0, 0)}, context=context)


def test_hc_tree_abc_1_1_ok():  # HC, abc, tree=1,1,0
    dsc = '/discrete/tiny/abc.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/abc_1', 'in': dsc}
    dags, trace = hc(data, params={'tree': (1, 1, 0)}, context=context)


def test_hc_tree_abc_1_M1_ok():  # HC, abc, tree=1,-1,0
    dsc = '/discrete/tiny/abc.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/abc_1', 'in': dsc}
    dags, trace = hc(data, params={'tree': (1, -1, 0)}, context=context)


def test_hc_tree_abc_2_0_ok():  # HC, abc, tree=2,0
    dsc = '/discrete/tiny/abc.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/abc_1', 'in': dsc}
    dags, trace = hc(data, params={'tree': (2, 0, 0)}, context=context)


def test_hc_tree_abc_2_1_ok():  # HC, abc, tree=2,-1
    dsc = '/discrete/tiny/abc.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/abc_2', 'in': dsc}
    dags, trace = hc(data, params={'tree': (2, 1, 0)}, context=context)


def test_hc_tree_abc_2_M1_ok():  # HC, abc, tree=2,-1
    dsc = '/discrete/tiny/abc.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/abc_2', 'in': dsc}
    dags, trace = hc(data, params={'tree': (2, -1, 0)}, context=context)


def test_hc_tree_abc_2_M2_ok():  # HC, abc, tree=2, -2
    dsc = '/discrete/tiny/abc.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/abc_2', 'in': dsc}
    dags, trace = hc(data, params={'tree': (2, -2, 0)}, context=context)


def test_tabu_tree_abc_1_0_ok():  # Tabu, abc, tree=1,0
    dsc = '/discrete/tiny/abc.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/abc_1_0', 'in': dsc}
    dags, trace = hc(data, params={'tree': (1, 0, 0), 'tabu': 10,
                                   'bnlearn': False}, context=context)


def test_tabu_tree_abc_2_0_ok():  # Tabu, abc, tree=2,0
    dsc = '/discrete/tiny/abc.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(100)
    context = {'id': 'test/hc_tree/abc_2_0', 'in': dsc}
    dags, trace = hc(data, params={'tree': (2, 0, 0), 'tabu': 10,
                                   'bnlearn': False}, context=context)


def test_hc_tree_cancer_1_0_ok():  # Tabu, abc, tree=2,0
    dsc = '/discrete/small/cancer.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10000)
    context = {'id': 'test/hc_tree/cancer_1_0', 'in': dsc}
    dags, trace = hc(data, params={'tree': (1, 0, 0), 'tabu': 10,
                                   'bnlearn': False}, context=context)


def test_hc_tree_asia_1_0_0_ok():  # Tabu, asia, tree=1,0
    dsc = '/discrete/small/asia.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10000)
    context = {'id': 'test/hc_tree/asia_1_0', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tree': (1, 0, 0), 'tabu': 10, 'bnlearn': False,
                            'stable': True})
    print(dag)
    print(trace)


def test_hc_tree_asia_1_5_0_ok():  # HC, abc, tree= 1,5
    dsc = '/discrete/small/asia.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10000)
    context = {'id': 'test/hc_tree/asia_1_5', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tree': (1, 5, 0), 'stable': True})
    print(dag)
    print(trace)


def test_hc_tree_asia_1_5_2_ok():  # HC, abc, tree= 1,5,2
    dsc = '/discrete/small/asia.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10000)
    context = {'id': 'test/hc_tree/asia_1_5_2', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tree': (1, 5, 2), 'stable': True})
    print(dag)
    print(trace)


@pytest.mark.slow
def test_hc_tree_sports_1_5_2_ok():  # HC, abc, tree= 1,5,2
    dsc = '/discrete/small/sports.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10000)
    context = {'id': 'test/hc_tree/sports', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tree': (1, 1, 30), 'tabu': 10, 'bnlearn': False,
                            'stable': True})
    print(dag)
    print(trace)


@pytest.mark.slow
def test_hc_tree_child_1_5_2_ok():  # HC, abc, tree= 1,5,2
    dsc = '/discrete/medium/child.dsc'
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(10000)
    context = {'id': 'test/hc_tree/child', 'in': dsc}
    dag, trace = hc(data, context=context,
                    params={'tree': (4, 0, -1), 'tabu': 10, 'bnlearn': False,
                            'stable': True})
    print(dag)
    print(trace)
