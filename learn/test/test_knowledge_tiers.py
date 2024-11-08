
# Test the Knowledge class - TIERS Ruleset

import pytest

from core.common import init_stable_random
from fileio.common import TESTDATA_DIR
from core.bn import BN
from learn.knowledge import Knowledge, Rule, RuleSet


@pytest.fixture
def abc():  # A --> B --> C BN
    return BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')


def test_tiers_type_error_1():  # nodes parameter has incorrect type
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.TIERS, params={'nodes': [3]})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.TIERS, params={'nodes': {'invalid'}})


def test_tiers_value_error_1():  # TIERS needs nodes parameter
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS, params={'limit': 4})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS, params={'reqd': {('A', 'B'): True}})


def test_tiers_value_error_2(abc):  # TIERS invalid nodes integer values
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS,
                  params={'nodes': 0, 'ref': abc})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS,
                  params={'nodes': 1, 'ref': abc})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS,
                  params={'nodes': 4, 'ref': abc})


def test_tiers_value_error_3(abc):  # TIERS invalid nodes float values
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS,
                  params={'nodes': -0.1, 'ref': abc})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS,
                  params={'nodes': 0.0, 'ref': abc})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.TIERS,
                  params={'nodes': 1.01, 'ref': abc})


def xtest_tiers_value_error_3():  # STOP_ARC stop frac missing ref
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': 0.1})


def xtest_tiers_value_error_4():  # STOP_ARC stop frac missing ref
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': 0.1, 'ref': ref},
                  sample=-1)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': 0.1, 'ref': ref},
                  sample=101)


def test_tiers_abc_1_ok(abc):  # Tiers knowledge - nodes=2, A->B->C
    tiers1 = Knowledge(rules=RuleSet.TIERS, params={'nodes': 2, 'ref': abc})

    assert tiers1.rules.rules == [Rule.TIERS]
    assert tiers1.ref == abc
    assert tiers1.limit is False
    assert tiers1.ignore == 0
    assert tiers1.expertise == 1.0
    assert tiers1.count == 0
    assert tiers1.label == ('Ruleset "Topological tiers" with '
                            + '1 prohibited and expertise 1.0')
    assert tiers1.stop == {('C', 'B'): (True, True)}
    assert tiers1.reqd == {}
    assert tiers1.event is None
    assert tiers1.event_delta is None
    assert tiers1.initial is None


def test_tiers_abc_2_ok(abc):  # Tiers knowledge - nodes=2, A->B->C, sample 1
    tiers1 = Knowledge(rules=RuleSet.TIERS, params={'nodes': 0.5, 'ref': abc},
                       sample=2)

    assert tiers1.rules.rules == [Rule.TIERS]
    assert tiers1.ref == abc
    assert tiers1.limit is False
    assert tiers1.ignore == 0
    assert tiers1.expertise == 1.0
    assert tiers1.count == 0
    assert tiers1.label == ('Ruleset "Topological tiers" with '
                            + '1 prohibited and expertise 1.0')
    assert tiers1.stop == {('B', 'A'): (True, True)}
    assert tiers1.reqd == {}
    assert tiers1.event is None
    assert tiers1.event_delta is None
    assert tiers1.initial is None


def test_tiers_abc_3_ok(abc):  # Tiers knowledge - nodes=0.5, A->B->C
    tiers1 = Knowledge(rules=RuleSet.TIERS, params={'nodes': 0.5, 'ref': abc})

    assert tiers1.rules.rules == [Rule.TIERS]
    assert tiers1.ref == abc
    assert tiers1.limit is False
    assert tiers1.ignore == 0
    assert tiers1.expertise == 1.0
    assert tiers1.count == 0
    assert tiers1.label == ('Ruleset "Topological tiers" with '
                            + '1 prohibited and expertise 1.0')
    assert tiers1.stop == {('C', 'B'): (True, True)}
    assert tiers1.reqd == {}
    assert tiers1.event is None
    assert tiers1.event_delta is None
    assert tiers1.initial is None


def test_tiers_cancer_1_ok():  # Cancer, nodes = 3
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=0,
                     params={'nodes': 3, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '3 prohibited and expertise 1.0')

    # 3 correct prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Cancer', 'Pollution'): (True, True),
                         ('Xray', 'Cancer'): (True, True),
                         ('Xray', 'Pollution'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_tiers_cancer_2_ok():  # Cancer, nodes = 0.60
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=0,
                     params={'nodes': 0.6, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '3 prohibited and expertise 1.0')

    # 3 correct prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Cancer', 'Pollution'): (True, True),
                         ('Xray', 'Cancer'): (True, True),
                         ('Xray', 'Pollution'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_tiers_cancer_3_ok():  # Cancer, nodes = 0.60, sample=1
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=1,
                     params={'nodes': 3, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '4 prohibited and expertise 1.0')

    # 4 correct prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Dyspnoea', 'Cancer'): (True, True),
                         ('Xray', 'Cancer'): (True, True),
                         ('Dyspnoea', 'Xray'): (True, True),
                         ('Xray', 'Dyspnoea'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_tiers_cancer_4_ok():  # Cancer, nodes = 0.25
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=1,
                     params={'nodes': 0.25, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '2 prohibited and expertise 1.0')

    # 2 correct prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Dyspnoea', 'Xray'): (True, True),
                         ('Xray', 'Dyspnoea'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_tiers_cancer_5_ok():  # Cancer, nodes = 1.0
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=1,
                     params={'nodes': 1.0, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '12 prohibited and expertise 1.0')

    # 12 correct prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Smoker', 'Pollution'): (True, True),
                         ('Pollution', 'Smoker'): (True, True),
                         ('Cancer', 'Pollution'): (True, True),
                         ('Cancer', 'Smoker'): (True, True),
                         ('Dyspnoea', 'Cancer'): (True, True),
                         ('Dyspnoea', 'Smoker'): (True, True),
                         ('Dyspnoea', 'Pollution'): (True, True),
                         ('Xray', 'Cancer'): (True, True),
                         ('Xray', 'Smoker'): (True, True),
                         ('Xray', 'Pollution'): (True, True),
                         ('Dyspnoea', 'Xray'): (True, True),
                         ('Xray', 'Dyspnoea'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


#   Cancer tiers but with imperfect expert

def test_tiers_cancer_6_ok():  # Cancer, nodes = 0.60, sample=1, exp=0.5

    init_stable_random()
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=1,
                     params={'nodes': 3, 'ref': ref, 'expertise': 0.5})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.5
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '4 prohibited and expertise 0.5')

    # 2 correct & 2 incorrect prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Cancer', 'Xray'): (False, True),
                         ('Cancer', 'Dyspnoea'): (False, True),
                         ('Xray', 'Cancer'): (True, True),
                         ('Xray', 'Dyspnoea'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_tiers_cancer_7_ok():  # Cancer, nodes = 0.60, sample=0, exp=0.5

    init_stable_random()
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=0,
                     params={'nodes': 3, 'ref': ref, 'expertise': 0.5})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.5
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '6 prohibited and expertise 0.5')

    # 4 correct & 2 incorrect prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Cancer', 'Xray'): (False, True),
                         ('Cancer', 'Pollution'): (True, True),
                         ('Xray', 'Cancer'): (True, True),
                         ('Xray', 'Pollution'): (True, True),
                         ('Pollution', 'Cancer'): (False, True),
                         ('Pollution', 'Xray'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_tiers_cancer_8_ok():  # Cancer, nodes = 0.60, sample=0, exp=0.8

    init_stable_random()
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=0,
                     params={'nodes': 3, 'ref': ref, 'expertise': 0.8})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.8
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '4 prohibited and expertise 0.8')

    # 3 correct & 1 incorrect prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Pollution', 'Xray'): (True, True),
                         ('Xray', 'Pollution'): (True, True),
                         ('Cancer', 'Pollution'): (True, True),
                         ('Cancer', 'Xray'): (False, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_tiers_cancer_9_ok():  # Cancer, nodes = 0.60, sample=1, exp=0.8

    init_stable_random()
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=1,
                     params={'nodes': 3, 'ref': ref, 'expertise': 0.8})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.8
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '4 prohibited and expertise 0.8')

    # 4 correct & 0 incorrect prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Xray', 'Cancer'): (True, True),
                         ('Dyspnoea', 'Cancer'): (True, True),
                         ('Xray', 'Dyspnoea'): (True, True),
                         ('Dyspnoea', 'Xray'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_tiers_cancer_10_ok():  # Cancer, nodes = 1.0, exp = 0.5
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=0,
                     params={'nodes': 1.0, 'ref': ref, 'expertise': 0.5})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.5
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '14 prohibited and expertise 0.5')

    # 11 correct & 3 incorrect prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Pollution', 'Cancer'): (False, True),
                         ('Pollution', 'Xray'): (True, True),
                         ('Cancer', 'Pollution'): (True, True),
                         ('Cancer', 'Xray'): (False, True),
                         ('Xray', 'Pollution'): (True, True),
                         ('Xray', 'Cancer'): (True, True),
                         ('Dyspnoea', 'Pollution'): (True, True),
                         ('Dyspnoea', 'Cancer'): (True, True),
                         ('Dyspnoea', 'Xray'): (True, True),
                         ('Smoker', 'Pollution'): (True, True),
                         ('Smoker', 'Cancer'): (False, True),
                         ('Smoker', 'Xray'): (True, True),
                         ('Dyspnoea', 'Smoker'): (True, True),
                         ('Smoker', 'Dyspnoea'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None

# Asia Tiers knowledge, perfect expert

def test_tiers_asia_1_ok():  # Asia, nodes = 0.50, sample = 1
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=1,
                     params={'nodes': 0.5, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert know.ref.dag.to_string() == (
        '[asia][bronc|smoke][dysp|bronc:either][either|lung:tub][lung|smoke]' +
        '[smoke][tub|asia][xray|either]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '7 prohibited and expertise 1.0')

    # 7 correct prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('either', 'bronc'): (True, True),
                         ('dysp', 'either'): (True, True),
                         ('dysp', 'bronc'): (True, True),
                         ('xray', 'either'): (True, True),
                         ('xray', 'bronc'): (True, True),
                         ('dysp', 'xray'): (True, True),
                         ('xray', 'dysp'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_tiers_asia_2_ok():  # Asia, nodes = 0.50, sample = 5
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=5,
                     params={'nodes': 0.5, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert know.ref.dag.to_string() == (
        '[asia][bronc|smoke][dysp|bronc:either][either|lung:tub][lung|smoke]' +
        '[smoke][tub|asia][xray|either]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '6 prohibited and expertise 1.0')

    # 7 correct prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('bronc', 'smoke'): (True, True),
                         ('either', 'smoke'): (True, True),
                         ('either', 'bronc'): (True, True),
                         ('dysp', 'either'): (True, True),
                         ('dysp', 'smoke'): (True, True),
                         ('dysp', 'bronc'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


# Asia, imperfect tiers knowledge

def test_tiers_asia_3_ok():  # Asia, nodes = 0.50, sample = 1, exp = 0.5
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=1,
                     params={'nodes': 0.5, 'ref': ref, 'expertise': 0.5})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert know.ref.dag.to_string() == (
        '[asia][bronc|smoke][dysp|bronc:either][either|lung:tub][lung|smoke]' +
        '[smoke][tub|asia][xray|either]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.5
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '7 prohibited and expertise 0.5')

    # 7 correct & 0 incorrect prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('xray', 'either'): (True, True),
                         ('bronc', 'either'): (True, True),
                         ('xray', 'bronc'): (True, True),
                         ('bronc', 'xray'): (True, True),
                         ('dysp', 'either'): (True, True),
                         ('dysp', 'xray'): (True, True),
                         ('dysp', 'bronc'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_tiers_asia_4_ok():  # Asia, nodes = 0.50, sample = 2, exp = 0.5
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.TIERS, sample=2,
                     params={'nodes': 0.5, 'ref': ref, 'expertise': 0.5})
    assert know.rules.rules == [Rule.TIERS]
    assert isinstance(know.ref, BN)
    assert know.ref.dag.to_string() == (
        '[asia][bronc|smoke][dysp|bronc:either][either|lung:tub][lung|smoke]' +
        '[smoke][tub|asia][xray|either]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.5
    assert know.count == 0
    assert know.label == ('Ruleset "Topological tiers" with '
                          + '8 prohibited and expertise 0.5')

    # 8 correct & 0 incorrect prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('dysp', 'smoke'): (True, True),
                         ('smoke', 'dysp'): (True, True),
                         ('bronc', 'dysp'): (False, True),
                         ('bronc', 'smoke'): (True, True),
                         ('tub', 'dysp'): (True, True),
                         ('tub', 'smoke'): (True, True),
                         ('bronc', 'tub'): (True, True),
                         ('tub', 'bronc'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None
