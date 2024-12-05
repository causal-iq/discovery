
#   Test the hc hill-climbing structure learning with reqd arcs

import pytest
from pandas import set_option

from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from core.bn import BN, DAG
from learn.hc import hc
from learn.knowledge import Knowledge, RuleSet


@pytest.fixture
def showall():
    print()
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)


@pytest.fixture
def ab():  # Return 1K rows for A->B graph
    dsc = '/discrete/tiny/ab.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = Pandas(df=bn.generate_cases(N))
    return (bn.dag, data, dsc)


@pytest.fixture
def ba():  # Return 100 rows for B->A graph
    dsc = '/discrete/tiny/ba.dsc'
    N = 100
    bn = BN.read(TESTDATA_DIR + dsc)
    data = Pandas(df=bn.generate_cases(N))
    return (bn.dag, data, dsc)


@pytest.fixture
def abc():  # Return 10K rows for A->B->C graph
    dsc = '/discrete/tiny/abc.dsc'
    N = 10000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = Pandas(df=bn.generate_cases(N))
    return (bn.dag, data, dsc)


@pytest.fixture
def ab_cb():  # Return 1K rows for A->B<-C graph
    dsc = '/discrete/tiny/ab_cb.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = Pandas(df=bn.generate_cases(N))
    return (bn.dag, data, dsc)


@pytest.fixture
def and4_10():  # Return 1K rows for and4_10 graph
    dsc = '/discrete/tiny/and4_10.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = Pandas(df=bn.generate_cases(N))
    return (bn.dag, data, dsc)


@pytest.fixture
def cancer():  # Return 1K rows for cancer graph
    dsc = '/discrete/small/cancer.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = Pandas(df=bn.generate_cases(N))
    return (bn.dag, data, dsc)


@pytest.fixture
def asia():  # return 1K rows for asia graph
    dsc = '/discrete/small/asia.dsc'
    N = 1000
    bn = BN.read(TESTDATA_DIR + dsc)
    data = Pandas(df=bn.generate_cases(N))
    return (bn.dag, data, dsc)


def test_hc_reqd_value_error1(ab):  # initial contains unknown nodes
    wrong_dag = DAG(nodes=['B', 'C'], edges=[('B', '->', 'C')])
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('B', 'C'): True},
                                        'initial': wrong_dag})
    with pytest.raises(ValueError):
        hc(ab[1], knowledge=know)


# A->B 1k rows

def test_hc_reqd_ab_ok_1(showall, ab):  # require A->B
    context = {'id': 'test/hc/reqd/ab_1', 'in': ab[2]}
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('A', 'B'): True},
                                        'initial': ab[0]})
    dag, trace = hc(ab[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert (trace.trace['knowledge'] ==
            [None, ('reqd_arc', True, 'stop_rev', ('A', 'B'))])


def test_hc_reqd_ab_ok_2(showall, ab, ba):  # require B->A
    context = {'id': 'test/hc/reqd/ab_2', 'in': ab[2]}
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('B', 'A'): True},
                                        'initial': ba[0]})
    dag, trace = hc(ab[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B]'
    assert dag.number_components() == 1
    assert (trace.trace['knowledge'] ==
            [None, ('reqd_arc', True, 'stop_rev', ('B', 'A'))])


# B->A 100 rows

def test_hc_reqd_ba_ok_1(showall, ba):  # require B->A
    context = {'id': 'test/hc/reqd/ba_1', 'in': ba[2]}
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('B', 'A'): True},
                                        'initial': ba[0]})
    dag, trace = hc(ba[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B]'
    assert dag.number_components() == 1
    assert (trace.trace['knowledge'] ==
            [None, ('reqd_arc', True, 'stop_rev', ('B', 'A'))])


def test_hc_reqd_ba_ok_2(showall, ba):  # require A->B
    context = {'id': 'test/hc/reqd/ba_1', 'in': ba[2]}
    ab = DAG(nodes=['A', 'B'], edges=[('A', '->', 'B')])
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('A', 'B'): True},
                                        'initial': ab})
    dag, trace = hc(ba[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1
    assert (trace.trace['knowledge'] ==
            [None, ('reqd_arc', True, 'stop_rev', ('A', 'B'))])


# A->B->C 10K rows

def test_hc_reqd_abc_ok_1(showall, abc, ab):  # require A->B learns A->B->C
    context = {'id': 'test/hc/reqd/abc_1', 'in': abc[2]}
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('A', 'B'): True},
                                        'initial': ab[0]})
    dag, trace = hc(abc[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert ([None, None, ('reqd_arc', True, 'stop_rev', ('A', 'B'))]
            == trace.trace['knowledge'])


def test_hc_reqd_abc_ok_2(showall, abc, ba):  # require B->A learns A<-B->C
    context = {'id': 'test/hc/reqd/abc_2', 'in': abc[2]}
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('B', 'A'): False},
                                        'initial': ba[0]})
    dag, trace = hc(abc[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B][C|B]'
    assert dag.number_components() == 1
    assert ([None, None, ('reqd_arc', False, 'stop_rev', ('B', 'A'))]
            == trace.trace['knowledge'])


def test_hc_reqd_abc_ok_3(showall, abc):  # require B->C learns A->B->C
    context = {'id': 'test/hc/reqd/abc_3', 'in': abc[2]}
    bc = DAG(['B', 'C'], [('B', '->', 'C')])
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('B', 'C'): False},
                                        'initial': bc})
    dag, trace = hc(abc[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B|A][C|B]'
    assert dag.number_components() == 1
    assert [None, None, None] == trace.trace['knowledge']


def test_hc_reqd_abc_ok_4(showall, abc):  # require C->B learns A<-B<-C
    context = {'id': 'test/hc/reqd/abc_4', 'in': abc[2]}
    cb = DAG(['B', 'C'], [('C', '->', 'B')])
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('C', 'B'): False},
                                        'initial': cb})
    dag, trace = hc(abc[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B|C][C]'
    assert dag.number_components() == 1
    assert ([None, None, ('reqd_arc', False, 'stop_rev', ('C', 'B'))]
            == trace.trace['knowledge'])


def test_hc_reqd_abc_ok_5(showall, abc):  # require A->C learns A->B<-C<-A
    context = {'id': 'test/hc/reqd/abc_5', 'in': abc[2]}
    ac = DAG(['A', 'C'], [('A', '->', 'C')])
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('A', 'C'): False},
                                        'initial': ac})
    dag, trace = hc(abc[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B|A:C][C|A]'
    assert dag.number_components() == 1
    return
    assert ([None, None, None, ('reqd_arc', False, 'stop_rev', ('A', 'C'))]
            == trace.trace['knowledge'])


def test_hc_reqd_abc_ok_6(showall, abc, ab_cb):  # req A->B<-C => A->B<-C<-A
    context = {'id': 'test/hc/reqd/abc_6', 'in': abc[2]}
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('A', 'B'): True,
                                                 ('C', 'B'): False},
                                        'initial': ab_cb[0]})
    dag, trace = hc(abc[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B|A:C][C|A]'
    assert dag.number_components() == 1
    assert ([None,
            ('reqd_arc', True, 'stop_rev', ('A', 'B')),
            ('reqd_arc', True, 'stop_del', ('A', 'B'))]
            == trace.trace['knowledge'])


# A-->B<--C tests

def test_hc_reqd_ab_cb_ok_1(showall, ab_cb, ab):  # req A->B learns A->B<-C
    context = {'id': 'test/hc/reqd/test_hc_reqd_ab_cb_ok_1', 'in': ab_cb[2]}
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('A', 'B'): True},
                                        'initial': ab[0]})
    dag, trace = hc(ab_cb[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B|A:C][C]'
    assert dag.number_components() == 1
    assert ([None, None,
            ('reqd_arc', True, 'stop_rev', ('A', 'B'))]
            == trace.trace['knowledge'])


def test_hc_reqd_ab_cb_ok_2(showall, ab_cb, ba):  # req A<-B learns A<-B->C<-A
    context = {'id': 'test/hc/reqd/test_hc_reqd_ab_cb_ok_1', 'in': ab_cb[2]}
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('B', 'A'): True},
                                        'initial': ba[0]})
    dag, trace = hc(ab_cb[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B][C|A:B]'
    assert dag.number_components() == 1
    assert [None, None, None, None] == trace.trace['knowledge']


def test_hc_reqd_ab_cb_ok_3(showall, ab_cb):  # req A->B<-C learns A->B<-C
    context = {'id': 'test/hc/reqd/test_hc_reqd_ab_cb_ok_3', 'in': ab_cb[2]}
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('A', 'B'): True,
                                                 ('C', 'B'): True},
                                        'initial': ab_cb[0]})
    dag, trace = hc(ab_cb[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B|A:C][C]'
    assert dag.number_components() == 1
    assert ([None,
            ('reqd_arc', True, 'stop_rev', ('A', 'B'))]
            == trace.trace['knowledge'])


def test_hc_reqd_ab_cb_ok_4(showall, ab_cb):  # req A->C learns A->B->C<-A
    context = {'id': 'test/hc/reqd/test_hc_reqd_ab_cb_ok_3', 'in': ab_cb[2]}
    ac = DAG(['A', 'C'], [('A', '->', 'C')])
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('A', 'C'): False},
                                        'initial': ac})
    dag, trace = hc(ab_cb[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B|A][C|A:B]'
    assert dag.number_components() == 1
    assert [None, None, None, None] == trace.trace['knowledge']


# X1->X2->X4, X3->X2

def test_hc_reqd_and4_10_ok_1(showall, and4_10):  # req X4->X2
    context = {'id': 'test/hc/reqd/and4_10_1', 'in': and4_10[2]}
    x4x2 = DAG(['X2', 'X4'], [('X4', '->', 'X2')])
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('X4', 'X2'): True},
                                        'initial': x4x2})
    dag, trace = hc(and4_10[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[X1|X2][X2|X4][X3|X2][X4]'
    assert dag.number_components() == 1
    assert (trace.trace['knowledge'] ==
            [None, None,  None,
            ('reqd_arc', True, 'stop_rev', ('X4', 'X2'))])


def test_hc_reqd_and4_10_ok_2(showall, and4_10):  # req X1->X3
    context = {'id': 'test/hc/reqd/and4_10_2', 'in': and4_10[2]}
    x1x3 = DAG(['X1', 'X3'], [('X1', '->', 'X3')])
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('X1', 'X3'): True},
                                        'initial': x1x3})
    dag, trace = hc(and4_10[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[X1][X2|X1:X3][X3|X1][X4|X2]'
    assert dag.number_components() == 1
    assert (trace.trace['knowledge'] ==
            [None, None,  None, None,
            ('reqd_arc', True, 'stop_del', ('X1', 'X3'))])


# Cancer

def test_hc_reqd_cancer_ok_1(showall, cancer):  # req Smoker->Cancer
    context = {'id': 'test/hc/reqd/cancer_1', 'in': cancer[2]}
    sc = DAG(['Cancer', 'Smoker'], [('Smoker', '->', 'Cancer')])
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('Smoker', 'Cancer'): True},
                                        'initial': sc})
    dag, trace = hc(cancer[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert ('[Cancer|Smoker][Dyspnoea|Cancer][Pollution][Smoker]' +
            '[Xray|Cancer]' == dag.to_string())
    assert dag.number_components() == 2
    assert (trace.trace['knowledge'] ==
            [None, None, None,
             ('reqd_arc', True, 'stop_rev', ('Smoker', 'Cancer'))])


def test_hc_reqd_cancer_ok_2(showall, cancer):  # req Smoker->Cancer<-Pollution
    context = {'id': 'test/hc/reqd/cancer_2', 'in': cancer[2]}
    scp = DAG(['Cancer', 'Pollution', 'Smoker'],
              [('Smoker', '->', 'Cancer'),
               ('Pollution', '->', 'Cancer')])
    know = Knowledge(RuleSet.REQD_ARC,
                     {'reqd': {('Smoker', 'Cancer'): True,
                               ('Pollution', 'Cancer'): True},
                      'initial': scp})
    dag, trace = hc(cancer[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert ('[Cancer|Pollution:Smoker][Dyspnoea|Cancer][Pollution][Smoker]' +
            '[Xray|Cancer]' == dag.to_string())
    assert dag.number_components() == 1
    assert (trace.trace['knowledge'] ==
            [None, None,
             ('reqd_arc', True, 'stop_del', ('Pollution', 'Cancer')),
             ('reqd_arc', True, 'stop_del', ('Pollution', 'Cancer'))])


#   Asia 8-node model

def test_hc_reqd_asia_ok_1(showall, asia):  # stop either->lung
    context = {'id': 'test/hc/reqd/asia_1', 'in': asia[2]}
    le = DAG(['lung', 'either'], [('lung', '->', 'either')])
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('lung', 'either'): True},
                                        'initial': le})
    dag, trace = hc(asia[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert ('[asia][bronc][dysp|bronc:either][either|lung:tub][lung|smoke]' +
            '[smoke|bronc][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert (trace.trace['knowledge'] ==
            [None, None,  None, None, None, None, None, None])


def xtest_hc_reqd_asia_ok_2(showall, asia):  # stop either->lung
    context = {'id': 'test/hc/reqd/asia_1', 'in': asia[2]}
    le = DAG(['lung', 'either'], [('lung', '->', 'either')])
    know = Knowledge(RuleSet.REQD_ARC, {'reqd': {('lung', 'either'): True},
                                        'initial': le})
    dag, trace = hc(asia[1], context=context, knowledge=know)
    print('\n\n{}\n\nproduces:\n\n{}'.format(trace, dag))
    assert ('[asia][bronc][dysp|bronc:either][either|lung:tub][lung|smoke]' +
            '[smoke|bronc][tub][xray|either]') == dag.to_string()
    assert dag.number_components() == 2
    assert (trace.trace['knowledge'] ==
            [None, None,  None, None, None, None, None,
             ('reqd_arc', True, 'stop_rev', ('lung', 'either'))])
