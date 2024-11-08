
# Test the BNAnalysis module

import pytest
from pandas import set_option

from core.bn import BN
from core.metrics import values_same
from analysis.bn import DAGAnalysis, BNAnalysis
from fileio.common import TESTDATA_DIR


@pytest.fixture(scope='module')
def show_all():
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)


def test_dag_analysis_type_error_1():  # no arguments supplied
    with pytest.raises(TypeError):
        DAGAnalysis()


def test_dag_analysis_type_error_2():  # bad argument types supplied
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(TypeError):
        DAGAnalysis(None, bn.dag)
    with pytest.raises(TypeError):
        DAGAnalysis([bn.dag])
    with pytest.raises(TypeError):
        DAGAnalysis(bn)
    with pytest.raises(TypeError):
        DAGAnalysis(True)
    with pytest.raises(TypeError):
        DAGAnalysis({})


def test_bn_analysis_type_error_1():  # no arguments supplied
    with pytest.raises(TypeError):
        BNAnalysis()


def test_bn_analysis_type_error_2():  # bad argument types supplied
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(TypeError):
        BNAnalysis(None, bn)
    with pytest.raises(TypeError):
        BNAnalysis([bn])
    with pytest.raises(TypeError):
        BNAnalysis(True)
    with pytest.raises(TypeError):
        BNAnalysis({})


def test_bn_analysis_a_ok():  # node A only
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/a.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 1
    assert analysis.nodes['in'].mean() == 0
    assert analysis.nodes['in'].max() == 0
    assert analysis.nodes['deg'].mean() == 0
    assert analysis.nodes['deg'].max() == 0
    assert analysis.nodes['mb'].mean() == 0
    assert analysis.nodes['mb'].max() == 0
    assert analysis.nodes['card'].mean() == 2
    assert analysis.nodes['card'].max() == 2
    assert analysis.nodes['free'].mean() == 1
    assert analysis.nodes['free'].max() == 1
    assert values_same(analysis.nodes['k-l'].mean(), 0.1308120359)
    assert values_same(analysis.nodes['k-l'].max(), 0.1308120359)
    assert analysis.arcs is None


def test_bn_analysis_ab_ok():  # A->B
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 2
    assert analysis.nodes['in'].mean() == 0.5
    assert analysis.nodes['in'].max() == 1
    assert analysis.nodes['deg'].mean() == 1
    assert analysis.nodes['deg'].max() == 1
    assert analysis.nodes['mb'].mean() == 1
    assert analysis.nodes['mb'].max() == 1
    assert analysis.nodes['card'].mean() == 2
    assert analysis.nodes['card'].max() == 2
    assert values_same(analysis.nodes['free'].mean(), 1.5)
    assert analysis.nodes['free'].max() == 2
    assert values_same(analysis.nodes['k-l'].mean(), 0.08231705575)
    assert values_same(analysis.nodes['k-l'].max(), 0.1308120359)
    assert len(analysis.arcs) == 1
    assert analysis.arcs.loc[('A', 'B')].to_dict() == \
        {'reversible': True, 'aligned': True}


def test_bn_analysis_ba_ok():  # A<-B
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ba.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 2
    assert analysis.nodes['in'].mean() == 0.5
    assert analysis.nodes['in'].max() == 1
    assert analysis.nodes['deg'].mean() == 1
    assert analysis.nodes['deg'].max() == 1
    assert analysis.nodes['mb'].mean() == 1
    assert analysis.nodes['mb'].max() == 1
    assert analysis.nodes['card'].mean() == 2
    assert analysis.nodes['card'].max() == 2
    assert values_same(analysis.nodes['free'].mean(), 1.5)
    assert analysis.nodes['free'].max() == 2
    assert values_same(analysis.nodes['k-l'].mean(), 0.08231705575)
    assert values_same(analysis.nodes['k-l'].max(), 0.1308120359)
    assert len(analysis.arcs) == 1
    assert analysis.arcs.loc[('B', 'A')].to_dict() == \
        {'reversible': True, 'aligned': False}


def test_bn_analysis_a_b_ok():  # A B
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/a_b.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 2
    assert analysis.nodes['in'].mean() == 0
    assert analysis.nodes['in'].max() == 0
    assert analysis.nodes['deg'].mean() == 0
    assert analysis.nodes['deg'].max() == 0
    assert analysis.nodes['mb'].mean() == 0
    assert analysis.nodes['mb'].max() == 0
    assert analysis.nodes['card'].mean() == 2
    assert analysis.nodes['card'].max() == 2
    assert analysis.nodes['free'].mean() == 1
    assert analysis.nodes['free'].max() == 1
    assert values_same(analysis.nodes['k-l'].mean(), 0.06540601797)
    assert values_same(analysis.nodes['k-l'].max(), 0.1308120359)
    assert analysis.arcs is None


def test_bn_analysis_abc_ok():  # A->B->C
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 3
    assert analysis.nodes['in'].mean() == 2 / 3
    assert analysis.nodes['in'].max() == 1
    assert analysis.nodes['deg'].mean() == 4 / 3
    assert analysis.nodes['deg'].max() == 2
    assert analysis.nodes['mb'].mean() == 4 / 3
    assert analysis.nodes['mb'].max() == 2
    assert analysis.nodes['card'].mean() == 2
    assert analysis.nodes['card'].max() == 2
    assert values_same(analysis.nodes['free'].mean(), 5 / 3)
    assert analysis.nodes['free'].max() == 2
    assert values_same(analysis.nodes['k-l'].mean(), 0.1466767423)
    assert values_same(analysis.nodes['k-l'].max(), 0.2753961152)
    assert len(analysis.arcs) == 2
    assert analysis.arcs.loc[('A', 'B')].to_dict() == \
        {'reversible': True, 'aligned': True}
    assert analysis.arcs.loc[('B', 'C')].to_dict() == \
        {'reversible': True, 'aligned': True}


def test_bn_analysis_ab_cb_ok():  # A->B<-C
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_cb.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 3
    assert analysis.nodes['in'].mean() == 2 / 3
    assert analysis.nodes['in'].max() == 2
    assert analysis.nodes['deg'].mean() == 4 / 3
    assert analysis.nodes['deg'].max() == 2
    assert analysis.nodes['mb'].mean() == 2
    assert analysis.nodes['mb'].max() == 2
    assert analysis.nodes['card'].mean() == 2
    assert analysis.nodes['card'].max() == 2
    assert values_same(analysis.nodes['free'].mean(), 2)
    assert analysis.nodes['free'].max() == 4
    assert values_same(analysis.nodes['k-l'].mean(), 0.1216015206)
    assert values_same(analysis.nodes['k-l'].max(), 0.2515384911)
    assert len(analysis.arcs) == 2
    assert analysis.arcs.loc[('A', 'B')].to_dict() == \
        {'reversible': False, 'aligned': True}
    assert analysis.arcs.loc[('C', 'B')].to_dict() == \
        {'reversible': False, 'aligned': False}


def test_bn_analysis_ab_cb_unfaithful_ok():  # A->B<-C
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_cb_unfaithful.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 3
    assert analysis.nodes['in'].mean() == 2 / 3
    assert analysis.nodes['in'].max() == 2
    assert analysis.nodes['deg'].mean() == 4 / 3
    assert analysis.nodes['deg'].max() == 2
    assert analysis.nodes['mb'].mean() == 2
    assert analysis.nodes['mb'].max() == 2
    assert analysis.nodes['card'].mean() == 2
    assert analysis.nodes['card'].max() == 2
    assert values_same(analysis.nodes['free'].mean(), 2)
    assert analysis.nodes['free'].max() == 4
    assert values_same(analysis.nodes['k-l'].mean(), 0.1042868424)
    assert values_same(analysis.nodes['k-l'].max(), 0.1995944564)
    assert len(analysis.arcs) == 2
    assert analysis.arcs.loc[('A', 'B')].to_dict() == \
        {'reversible': False, 'aligned': True}
    assert analysis.arcs.loc[('C', 'B')].to_dict() == \
        {'reversible': False, 'aligned': False}


def test_bn_analysis_abc_dual_ok():  # A->B->C<-A
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc_dual.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 3
    assert analysis.nodes['in'].mean() == 1
    assert analysis.nodes['in'].max() == 2
    assert analysis.nodes['deg'].mean() == 2
    assert analysis.nodes['deg'].max() == 2
    assert analysis.nodes['mb'].mean() == 2
    assert analysis.nodes['mb'].max() == 2
    assert len(analysis.arcs) == 3
    assert analysis.arcs.loc[('A', 'B')].to_dict() == \
        {'reversible': True, 'aligned': True}
    assert analysis.arcs.loc[('B', 'C')].to_dict() == \
        {'reversible': True, 'aligned': True}
    assert analysis.arcs.loc[('A', 'C')].to_dict() == \
        {'reversible': True, 'aligned': True}


def test_bn_analysis_cancer_ok():  # 5 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 5
    assert analysis.nodes['in'].mean() == 0.8
    assert analysis.nodes['in'].max() == 2
    assert analysis.nodes['deg'].mean() == 1.6
    assert analysis.nodes['deg'].max() == 4
    assert analysis.nodes['mb'].mean() == 2
    assert analysis.nodes['mb'].max() == 4
    assert len(analysis.arcs) == 4
    assert analysis.arcs.to_dict('index') == \
        {('Pollution', 'Cancer'): {'reversible': False, 'aligned': False},
         ('Smoker', 'Cancer'): {'reversible': False, 'aligned': False},
         ('Cancer', 'Xray'): {'reversible': False, 'aligned': True},
         ('Cancer', 'Dyspnoea'): {'reversible': False, 'aligned': True}}


def test_bn_analysis_asia_ok():  # 8 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 8
    assert analysis.nodes['in'].mean() == 1
    assert analysis.nodes['in'].max() == 2
    assert analysis.nodes['deg'].mean() == 2
    assert analysis.nodes['deg'].max() == 4
    assert analysis.nodes['mb'].mean() == 2.5
    assert analysis.nodes['mb'].max() == 5
    assert len(analysis.arcs) == 8
    assert analysis.arcs.to_dict('index') == \
        {('asia', 'tub'): {'reversible': True, 'aligned': True},
         ('smoke', 'lung'): {'reversible': True, 'aligned': False},
         ('smoke', 'bronc'): {'reversible': True, 'aligned': False},
         ('lung', 'either'): {'reversible': False, 'aligned': False},
         ('tub', 'either'): {'reversible': False, 'aligned': False},
         ('either', 'xray'): {'reversible': False, 'aligned': True},
         ('bronc', 'dysp'): {'reversible': False, 'aligned': True},
         ('either', 'dysp'): {'reversible': False, 'aligned': False}}


def test_bn_analysis_sports_ok():  # 9 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 9
    assert values_same(analysis.nodes['in'].mean(), 1.67, sf=3)
    assert analysis.nodes['in'].max() == 2
    assert values_same(analysis.nodes['deg'].mean(), 3.33, sf=3)
    assert analysis.nodes['deg'].max() == 7
    assert values_same(analysis.nodes['mb'].mean(), 3.56, sf=3)
    assert analysis.nodes['mb'].max() == 7
    assert len(analysis.arcs) == 15
    assert analysis.arcs.to_dict('index') == \
        {('ATshotsOnTarget', 'ATgoals'): {'reversible': True,
                                          'aligned': False},
         ('RDlevel', 'ATgoals'): {'reversible': True, 'aligned': False},
         ('RDlevel', 'ATshots'): {'reversible': True, 'aligned': False},
         ('possession', 'ATshots'): {'reversible': True, 'aligned': False},
         ('ATshots', 'ATshotsOnTarget'): {'reversible': True, 'aligned': True},
         ('RDlevel', 'ATshotsOnTarget'): {'reversible': True,
                                          'aligned': False},
         ('ATgoals', 'HDA'): {'reversible': False, 'aligned': True},
         ('HTgoals', 'HDA'): {'reversible': False, 'aligned': False},
         ('HTshotOnTarget', 'HTgoals'): {'reversible': True, 'aligned': False},
         ('RDlevel', 'HTgoals'): {'reversible': True, 'aligned': False},
         ('HTshots', 'HTshotOnTarget'): {'reversible': True, 'aligned': False},
         ('RDlevel', 'HTshotOnTarget'): {'reversible': True, 'aligned': False},
         ('RDlevel', 'HTshots'): {'reversible': True, 'aligned': False},
         ('possession', 'HTshots'): {'reversible': True, 'aligned': False},
         ('RDlevel', 'possession'): {'reversible': True, 'aligned': True}}


def test_bn_analysis_sachs_ok():  # 11 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sachs.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 11
    assert values_same(analysis.nodes['in'].mean(), 17/11, sf=3)
    assert analysis.nodes['in'].max() == 3
    assert values_same(analysis.nodes['deg'].mean(), 3.09, sf=3)
    assert analysis.nodes['deg'].max() == 7
    assert values_same(analysis.nodes['mb'].mean(), 3.09, sf=3)
    assert analysis.nodes['mb'].max() == 7
    assert len(analysis.arcs) == 17
    assert analysis.arcs.to_dict('index') == \
        {('Erk', 'Akt'): {'reversible': True, 'aligned': False},
         ('PKA', 'Akt'): {'reversible': True, 'aligned': False},
         ('Mek', 'Erk'): {'reversible': True, 'aligned': False},
         ('PKA', 'Erk'): {'reversible': True, 'aligned': False},
         ('PKA', 'Jnk'): {'reversible': True, 'aligned': False},
         ('PKC', 'Jnk'): {'reversible': True, 'aligned': False},
         ('PKA', 'Mek'): {'reversible': True, 'aligned': False},
         ('PKC', 'Mek'): {'reversible': True, 'aligned': False},
         ('Raf', 'Mek'): {'reversible': True, 'aligned': False},
         ('PKA', 'P38'): {'reversible': True, 'aligned': False},
         ('PKC', 'P38'): {'reversible': True, 'aligned': False},
         ('PIP3', 'PIP2'): {'reversible': True, 'aligned': False},
         ('Plcg', 'PIP2'): {'reversible': True, 'aligned': False},
         ('Plcg', 'PIP3'): {'reversible': True, 'aligned': False},
         ('PKC', 'PKA'): {'reversible': True, 'aligned': False},
         ('PKA', 'Raf'): {'reversible': True, 'aligned': True},
         ('PKC', 'Raf'): {'reversible': True, 'aligned': True}}
    assert sum(analysis.arcs['reversible'] > 0) == 17
    assert sum(analysis.arcs['aligned'] > 0) == 2


def test_bn_analysis_child_ok():  # 20 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/child.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 20
    assert analysis.nodes['in'].mean() == 1.25
    assert analysis.nodes['in'].max() == 2
    assert analysis.nodes['deg'].mean() == 2.5  # BN Repo says 1.25 !!
    assert analysis.nodes['deg'].max() == 8
    assert analysis.nodes['mb'].mean() == 3
    assert analysis.nodes['mb'].max() == 8
    assert len(analysis.arcs) == 25
    assert sum(analysis.arcs['reversible'] > 0) == 12
    assert sum(analysis.arcs['aligned'] > 0) == 16


def test_bn_analysis_insurance_ok():  # 27 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 27
    assert values_same(analysis.nodes['in'].mean(), 1.93, sf=3)
    assert analysis.nodes['in'].max() == 3
    assert values_same(analysis.nodes['deg'].mean(), 3.85, sf=3)
    assert analysis.nodes['deg'].max() == 9
    assert values_same(analysis.nodes['mb'].mean(), 5.19, sf=3)
    assert analysis.nodes['mb'].max() == 10
    assert len(analysis.arcs) == 52
    assert sum(analysis.arcs['reversible'] > 0) == 18
    assert sum(analysis.arcs['aligned'] > 0) == 23


def test_bn_analysis_property_ok():  # 27 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/property.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 27
    assert values_same(analysis.nodes['in'].mean(), 1.15, sf=3)
    assert analysis.nodes['in'].max() == 3
    assert values_same(analysis.nodes['deg'].mean(), 2.30, sf=3)
    assert analysis.nodes['deg'].max() == 6
    assert values_same(analysis.nodes['mb'].mean(), 3.41, sf=3)
    assert analysis.nodes['mb'].max() == 10
    assert len(analysis.arcs) == 31
    assert sum(analysis.arcs['reversible'] > 0) == 3
    assert sum(analysis.arcs['aligned'] > 0) == 19


def test_bn_analysis_diarrhoea_ok(show_all):  # 28 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/diarrhoea.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 28
    assert values_same(analysis.nodes['in'].mean(), 2.43, sf=3)
    assert analysis.nodes['in'].max() == 8
    assert values_same(analysis.nodes['deg'].mean(), 4.86, sf=3)
    assert analysis.nodes['deg'].max() == 17
    assert values_same(analysis.nodes['mb'].mean(), 7.93, sf=3)
    assert analysis.nodes['mb'].max() == 20
    assert len(analysis.arcs) == 68
    assert sum(analysis.arcs['reversible'] > 0) == 28
    assert sum(analysis.arcs['aligned'] > 0) == 34


def test_bn_analysis_water_ok(show_all):  # 32 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/water.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 32
    assert values_same(analysis.nodes['in'].mean(), 2.06, sf=3)
    assert analysis.nodes['in'].max() == 5
    assert values_same(analysis.nodes['deg'].mean(), 4.12, sf=3)
    assert analysis.nodes['deg'].max() == 8
    assert values_same(analysis.nodes['mb'].mean(), 7.69, sf=3)
    assert analysis.nodes['mb'].max() == 13
    assert len(analysis.arcs) == 66
    assert sum(analysis.arcs['reversible'] > 0) == 6
    assert sum(analysis.arcs['aligned'] > 0) == 42


def test_bn_analysis_mildew_ok():  # 35 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/mildew.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 35
    assert values_same(analysis.nodes['in'].mean(), 1.31, sf=3)
    assert analysis.nodes['in'].max() == 3
    assert values_same(analysis.nodes['deg'].mean(), 2.63, sf=3)
    assert analysis.nodes['deg'].max() == 5
    assert values_same(analysis.nodes['mb'].mean(), 4.57, sf=3)
    assert analysis.nodes['mb'].max() == 9
    assert len(analysis.arcs) == 46
    assert sum(analysis.arcs['reversible'] > 0) == 0
    assert sum(analysis.arcs['aligned'] > 0) == 14


def test_bn_analysis_alarm_ok():  # 37 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/alarm.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 37
    assert values_same(analysis.nodes['in'].mean(), 1.24, sf=3)
    assert analysis.nodes['in'].max() == 4
    assert values_same(analysis.nodes['deg'].mean(), 2.49, sf=3)
    assert analysis.nodes['deg'].max() == 6
    assert values_same(analysis.nodes['mb'].mean(), 3.51, sf=3)
    assert analysis.nodes['mb'].max() == 8
    assert len(analysis.arcs) == 46
    assert sum(analysis.arcs['reversible'] > 0) == 4
    assert sum(analysis.arcs['aligned'] > 0) == 27


def test_bn_analysis_barley_ok(show_all):  # 48 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/barley.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 48
    assert values_same(analysis.nodes['in'].mean(), 1.75, sf=3)
    assert analysis.nodes['in'].max() == 4
    assert values_same(analysis.nodes['deg'].mean(), 3.50, sf=3)
    assert analysis.nodes['deg'].max() == 8
    assert values_same(analysis.nodes['mb'].mean(), 5.25, sf=3)
    assert analysis.nodes['mb'].max() == 13
    assert len(analysis.arcs) == 84
    assert sum(analysis.arcs['reversible'] > 0) == 9
    assert sum(analysis.arcs['aligned'] > 0) == 47


def test_bn_analysis_hailfinder_ok(show_all):  # 56 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/large/hailfinder.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 56
    assert values_same(analysis.nodes['in'].mean(), 1.18, sf=3)
    assert analysis.nodes['in'].max() == 4
    assert values_same(analysis.nodes['deg'].mean(), 2.36, sf=3)
    assert analysis.nodes['deg'].max() == 17
    assert values_same(analysis.nodes['mb'].mean(), 3.54, sf=3)
    assert analysis.nodes['mb'].max() == 17
    assert len(analysis.arcs) == 66
    assert sum(analysis.arcs['reversible'] > 0) == 17
    assert sum(analysis.arcs['aligned'] > 0) == 30


def test_bn_analysis_hepar2_ok(show_all):  # 70 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/large/hepar2.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 70
    assert values_same(analysis.nodes['in'].mean(), 1.76, sf=3)
    assert analysis.nodes['in'].max() == 6
    assert values_same(analysis.nodes['deg'].mean(), 3.51, sf=3)
    assert analysis.nodes['deg'].max() == 19
    assert values_same(analysis.nodes['mb'].mean(), 4.51, sf=3)
    assert analysis.nodes['mb'].max() == 26
    assert len(analysis.arcs) == 123
    assert sum(analysis.arcs['reversible'] > 0) == 9
    assert sum(analysis.arcs['aligned'] > 0) == 89


def test_bn_analysis_win95pts_ok(show_all):  # 76 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/large/win95pts.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 76
    assert values_same(analysis.nodes['in'].mean(), 1.47, sf=3)
    assert analysis.nodes['in'].max() == 7
    assert values_same(analysis.nodes['deg'].mean(), 2.95, sf=3)
    assert analysis.nodes['deg'].max() == 10
    assert values_same(analysis.nodes['mb'].mean(), 5.92, sf=3)
    assert analysis.nodes['mb'].max() == 29
    assert len(analysis.arcs) == 112
    assert sum(analysis.arcs['reversible'] > 0) == 12
    assert sum(analysis.arcs['aligned'] > 0) == 59


def test_bn_analysis_formed_ok(show_all):  # 88 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/large/formed.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 88
    assert values_same(analysis.nodes['in'].mean(), 1.57, sf=3)
    assert analysis.nodes['in'].max() == 6
    assert values_same(analysis.nodes['deg'].mean(), 3.14, sf=3)
    assert analysis.nodes['deg'].max() == 11
    assert values_same(analysis.nodes['mb'].mean(), 4.98, sf=3)
    assert analysis.nodes['mb'].max() == 22
    assert len(analysis.arcs) == 138
    assert sum(analysis.arcs['reversible'] > 0) == 24
    assert sum(analysis.arcs['aligned'] > 0) == 72


def test_bn_analysis_pathfinder_ok(show_all):  # 109 nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/verylarge/pathfinder.dsc')
    analysis = BNAnalysis(bn)
    print('\nNodes:\n{}\nEdges:\n{}'.format(analysis.nodes, analysis.arcs))
    assert len(analysis.nodes) == 109
    assert values_same(analysis.nodes['in'].mean(), 1.79, sf=3)
    assert analysis.nodes['in'].max() == 5
    assert values_same(analysis.nodes['deg'].mean(), 3.58, sf=3)
    assert analysis.nodes['deg'].max() == 106
    assert values_same(analysis.nodes['mb'].mean(), 3.82, sf=3)
    assert analysis.nodes['mb'].max() == 107
    assert len(analysis.arcs) == 195
    assert sum(analysis.arcs['reversible'] > 0) == 122
    assert sum(analysis.arcs['aligned'] > 0) == 43
