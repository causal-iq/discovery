
#   Test the hc hill-climbing structure learning with stop arcs

import pytest
from pandas import DataFrame, set_option

from fileio.common import TESTDATA_DIR
from core.bn import BN
from learn.hc import hc
from learn.knowledge import Knowledge, RuleSet


@pytest.fixture
def showall():
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)

# A->B learnt correctly with stop arcs

def test_hc_stop_ab_ok_1(showall):  # A->B 1K rows, stop A->B
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/ab_1', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('A', 'B'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B]'
    assert dag.number_components() == 1

    # 1st iteration tries A->B first but is blocked
    # stop iteration tries reversing B->A but is blocked

    assert (trace.trace['knowledge'] ==
            [None,
             ('stop_arc', True, 'stop_add', ('A', 'B')),
             ('stop_arc', True, 'stop_rev', ('B', 'A'))])


def test_hc_stop_ab_ok_2(showall):  # A->B 1K rows, stop B->A
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/ab_2', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('B', 'A'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1

    # 1st iteration tries A->B first which is OK
    # stop iteration tries reversing A->B but is blocked

    assert (trace.trace['knowledge'] ==
            [None, None, ('stop_arc', True, 'stop_rev', ('A', 'B'))])


def test_hc_stop_ab_ok_3(showall):  # A->B 1K rows, stop A->B & B->A
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/test_hc_stop_ab_ok_3', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('A', 'B'): True,
                                                 ('B', 'A'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B]'
    assert dag.number_components() == 2

    # stop iteration tries A->B, then B->A first which are both blocked

    assert trace.trace['knowledge'] == [None, ('stop_arc', True, 'stop_add',
                                               ('A', 'B'))]


# B->A learnt correctly with stop arcs

def test_hc_stop_ba_ok_1(showall):  # B->A 1K rows, stop A->B
    dsc = '/discrete/tiny/ba.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/ba_1', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('A', 'B'): False}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B]'
    assert dag.number_components() == 1

    # 1st iteration tries A->B first but is blocked
    # stop iteration tries reversing B->A but is blocked

    assert (trace.trace['knowledge'] ==
            [None,
             ('stop_arc', False, 'stop_add', ('A', 'B')),
             ('stop_arc', False, 'stop_rev', ('B', 'A'))])


def test_hc_stop_ba_ok_2(showall):  # B->A 1K rows, stop B->A
    dsc = '/discrete/tiny/ba.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/ba_2', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('B', 'A'): False}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1

    # 1st iteration tries A->B first which is OK
    # stop iteration tries reversing A->B but is blocked

    assert trace.trace['knowledge'] == [None, None, ('stop_arc', False,
                                                     'stop_rev', ('A', 'B'))]


def test_hc_stop_ba_ok_3(showall):  # B->A 1K rows, stop A->B & B->A
    dsc = '/discrete/tiny/ba.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/ba_3', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('A', 'B'): False,
                                                 ('B', 'A'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B]'
    assert dag.number_components() == 2

    # stop iteration tries A->B, then B->A first which are both blocked

    assert trace.trace['knowledge'] == [None, ('stop_arc', False,
                                               'stop_add', ('A', 'B'))]


# A->B->C

def test_hc_stop_abc_ok_1(showall):  # A->B->C 1K rows, stop A-->B
    dsc = '/discrete/tiny/abc.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/abc_1', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('A', 'B'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B][C|B]'
    assert dag.number_components() == 1

    # 1st iteration adds B->C before C->B
    # 2nd iteration blocks A->B, adding B->A
    # stop iteration blocks reverse of B->A

    assert ([None,
             None,
             ('stop_arc', True, 'stop_add', ('A', 'B')),
             ('stop_arc', True, 'stop_rev', ('B', 'A'))]
            == trace.trace['knowledge'])


def test_hc_stop_abc_ok_2(showall):  # A->B->C 1K rows, stop A->B & B->A
    dsc = '/discrete/tiny/abc.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/abc_2', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('A', 'B'): True,
                                                 ('B', 'A'): False}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|C][B][C|B]'
    assert dag.number_components() == 1

    # 1st iteration adds B->C before C->B
    # 2nd iteration blocks A->B & B->A, adds C->A instead
    # stop iteration blocks adding A->B or B->A

    assert ([None,
             None,
             ('stop_arc', True, 'stop_add', ('A', 'B')),
             ('stop_arc', False, 'stop_add', ('B', 'A'))]
            == trace.trace['knowledge'])


def test_hc_stop_abc_ok_3(showall):  # A->B->C 1K rows, stop A->B & B->C
    dsc = '/discrete/tiny/abc.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/abc_3', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('A', 'B'): True,
                                                 ('B', 'C'): False}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B|C][C]'
    assert dag.number_components() == 1

    # 1st iteration blocks B->C before C->B
    # 2nd B->A is highest scoring, nothing blocked
    # stop iteration blocks reversing C->B

    assert (trace.trace['knowledge'] ==
            [None,
             ('stop_arc', False, 'stop_add', ('B', 'C')),
             None,
             ('stop_arc', False, 'stop_rev', ('C', 'B'))])


# A-->B<--C tests

def test_hc_stop_ab_cb_ok_1(showall):  # stop A->B
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/ab_cb_1', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('A', 'B'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B][C|A:B]'
    assert dag.number_components() == 1
    assert (trace.trace['knowledge'] ==
            [None,
             None,  # B->C highest, not blocked
             None,  # A->C highest, not blocked
             ('stop_arc', True, 'stop_add', ('A', 'B')),  # add A->B blocked
             ('stop_arc', True, 'stop_rev', ('B', 'A'))])  # rev B->A blocked


def test_hc_stop_ab_cb_ok_2(showall):  # stop A->B, B->C
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/ab_cb_2', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('A', 'B'): False,
                                                 ('B', 'C'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B:C][B|C][C]'
    assert dag.number_components() == 1
    assert (trace.trace['knowledge'] ==
            [None,
             ('stop_arc', True, 'stop_add', ('B', 'C')),  # add B->C blocked
             ('stop_arc', False, 'stop_add', ('A', 'B')),  # add A->B blocked
             ('stop_arc', False, 'stop_rev', ('B', 'A')),  # rev B->A blocked
             ('stop_arc', False, 'stop_rev', ('B', 'A'))])  # rev B->A blocked


def test_hc_stop_ab_cb_ok_3(showall):  # stop A->B, B->C, C->B
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/ab_cb_3', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('A', 'B'): False,
                                                 ('B', 'C'): True,
                                                 ('C', 'B'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B:C][B][C]'
    assert dag.number_components() == 1
    assert (trace.trace['knowledge'] ==
            [None,
             ('stop_arc', True, 'stop_add', ('B', 'C')),  # add B->C blocked
             ('stop_arc', True, 'stop_add', ('B', 'C')),  # add B->C blocked
             ('stop_arc', True, 'stop_add', ('B', 'C'))])  # add B->C blocked


# X1->X2->X4, X3->X2

def test_hc_stop_and4_10_ok_1(showall):  # stop X1->X2
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/and4_10_1', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('X1', 'X2'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[X1|X2][X2][X3|X2][X4|X2]'
    assert dag.number_components() == 1
    assert (trace.trace['knowledge'] ==
            [None,
             None,  # X2->X4 not blocked
             None,  # X2->X3 not blocked
             ('stop_arc', True, 'stop_add', ('X1', 'X2')),  # add X1->X2 block
             ('stop_arc', True, 'stop_rev', ('X2', 'X1'))])  # rev X2->X1 block


def test_hc_stop_and4_10_ok_2(showall):  # stop X1->X2, X2->X3, X3->X2
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/and4_10_2', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('X1', 'X2'): True,
                                                 ('X2', 'X3'): True,
                                                 ('X3', 'X2'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[X1|X2][X2][X3][X4|X2]'
    assert dag.number_components() == 2
    assert (trace.trace['knowledge'] ==
            [None,
             None,  # X2->X4 not blocked
             ('stop_arc', True, 'stop_add', ('X2', 'X3')),  # add X2->X3 block
             ('stop_arc', True, 'stop_add', ('X2', 'X3'))])  # add X2->X3 block


# Cancer

def test_hc_stop_cancer_ok_1(showall):  # stop Cancer->Smoker
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/cancer_1', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('Cancer', 'Smoker'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert ('[Cancer|Smoker][Dyspnoea|Cancer][Pollution][Smoker]' +
            '[Xray|Cancer]' == dag.to_string())
    assert dag.number_components() == 2
    assert (trace.trace['knowledge'] ==
            [None,
             None,  # Cancer->Xray not blocked
             ('stop_arc', True, 'stop_add', ('Cancer', 'Smoker')),
             None,
             ('stop_arc', True, 'stop_rev', ('Smoker', 'Cancer'))])


def test_hc_stop_cancer_ok_2(showall):  # stop Cancer->Smoker, Cancer->Dyspnoea
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/cancer_2', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('Cancer', 'Smoker'): True,
                                                 ('Cancer',
                                                  'Dyspnoea'): False}})
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('Cancer', 'Smoker'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert ('[Cancer|Smoker][Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]'
            == dag.to_string())
    assert dag.number_components() == 2
    assert (trace.trace['knowledge'] ==
            [None, None,
             ('stop_arc', True, 'stop_add', ('Cancer', 'Smoker')),
             None,
             ('stop_arc', True, 'stop_rev', ('Smoker', 'Cancer'))])


#   Asia 8-node model

def test_hc_stop_asia_ok_1(showall):  # stop either->lung
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/asia_1', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('either', 'lung'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert ('[asia][bronc][dysp|bronc:either][either|lung:tub][lung|smoke]' +
            '[smoke|bronc][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert (trace.trace['knowledge'] ==
            [None, None,
             ('stop_arc', True, 'stop_add', ('either', 'lung')),
             None, None, None, None, None,
             ('stop_arc', True, 'stop_rev', ('lung', 'either'))])


def test_hc_stop_asia_ok_2(showall):  # stop either->lung, either->tub, b->s
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = bn.generate_cases(N)
    context = {'id': 'test/hc/stop/asia_2', 'in': dsc}
    know = Knowledge(RuleSet.STOP_ARC, {'stop': {('either', 'lung'): True,
                                                 ('either', 'tub'): True,
                                                 ('bronc', 'smoke'): True}})
    dag, trace = hc(data, context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert ('[asia][bronc|smoke][dysp|bronc:either][either|lung:tub][lung]' +
            '[smoke|lung][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert (trace.trace['knowledge'] ==
            [None, None,
             ('stop_arc', True, 'stop_add', ('either', 'lung')),
             None, None,
             ('stop_arc', True, 'stop_add', ('bronc', 'smoke')),
             None, None,
             ('stop_arc', True, 'stop_rev', ('smoke', 'bronc'))])
