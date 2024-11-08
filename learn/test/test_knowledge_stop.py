
# Test the Knowledge class - STOP_ARC Ruleset

import pytest

from learn.knowledge import Knowledge, Rule, RuleSet, \
    KnowledgeOutcome, KnowledgeEvent
from learn.trace import Activity
from learn.dagchange import DAGChange, BestDAGChanges
from fileio.common import TESTDATA_DIR
from core.bn import BN


@pytest.fixture
def stop1():  # stop list with arc B --> C
    return Knowledge(rules=RuleSet.STOP_ARC,
                     params={'stop': {('B', 'C'): True}})


def test_knowledge_type_error_1():  # stop must be a dict/int/float
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': 'a'})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': (1, 2)})


def test_knowledge_type_error_2():  # stop: if dict must have tuple keys
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': {'A': [1, 2]}})


def test_knowledge_type_error_3():  # stop tuples must be len 2
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': {('A',): True}})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.STOP_ARC,
                  params={'stop': {('A', 'B', 'C'): True}})


def test_knowledge_type_error_4():  # stop tuples must contain strs
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': {(1, 2): True}})


def test_knowledge_type_error_5():  # stop tuples have boolean values
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': {('A', 'B'): 23}})


def test_knowledge_value_error_1():  # STOP_ARC needs stop parameter
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'limit': 4})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'reqd': {('A', 'B'): True}})


def test_knowledge_value_error_2():  # STOP_ARC invalid stop fraction
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': 0.0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': 1.0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': -0.1})


def test_knowledge_value_error_3():  # STOP_ARC stop frac missing ref
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': 0.1})


def test_knowledge_value_error_4():  # STOP_ARC stop frac missing ref
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': 0.1, 'ref': ref},
                  sample=-1)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.STOP_ARC, params={'stop': 0.1, 'ref': ref},
                  sample=101)


def test_knowledge_ok_1(stop1):  # STOP_ARC Knowledge
    assert stop1.rules.rules == [Rule.STOP_ARC]
    assert stop1.ref is None
    assert stop1.limit is False
    assert stop1.ignore == 0
    assert stop1.expertise == 1.0
    assert stop1.count == 0
    assert stop1.label == ('Ruleset "Prohibited arc" with '
                           + '1 prohibited and expertise 1.0')
    assert stop1.stop == {('B', 'C'): (True, True)}
    assert stop1.reqd == {}
    assert stop1.event is None
    assert stop1.event_delta is None
    assert stop1.initial is None


def test_knowledge_cancer_1_ok():  # STOP_ARC, stop 3
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.STOP_ARC,
                     params={'stop': 3, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.STOP_ARC]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '3 prohibited and expertise 1.0')

    # 3 correct prohibited arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Pollution', 'Xray'): (True, True),
                         ('Xray', 'Smoker'): (True, True),
                         ('Dyspnoea', 'Cancer'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_knowledge_cancer_2_ok():  # STOP_ARC, stop 0.19 (3/16 of non-arcs)
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.STOP_ARC,
                     params={'stop': 0.60, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.STOP_ARC]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '3 prohibited and expertise 1.0')

    # 3 correct prohibited arcs - same result using fraction as number

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Pollution', 'Xray'): (True, True),
                         ('Xray', 'Smoker'): (True, True),
                         ('Dyspnoea', 'Cancer'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_knowledge_cancer_3_ok():  # STOP_ARC, stop 3, sample 2
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.STOP_ARC, sample=2,
                     params={'stop': 3, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.STOP_ARC]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '3 prohibited and expertise 1.0')

    # 3 correct prohibited arcs - but offset gives different arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Dyspnoea', 'Cancer'): (True, True),
                         ('Dyspnoea', 'Pollution'): (True, True),
                         ('Smoker', 'Dyspnoea'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_knowledge_cancer_4_ok():  # STOP_ARC, stop 0.19, sample 2
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.STOP_ARC, sample=2,
                     params={'stop': 0.60, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.STOP_ARC]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '3 prohibited and expertise 1.0')

    # 3 correct prohibited arcs - but offset gives different arcs

    print('Prohibited arcs are: {}'.format(list(know.stop)))
    assert know.stop == {('Dyspnoea', 'Cancer'): (True, True),
                         ('Dyspnoea', 'Pollution'): (True, True),
                         ('Smoker', 'Dyspnoea'): (True, True)}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_knowledge_asia_1_ok():  # STOP_ARC, explicit stop list
    know = Knowledge(rules=RuleSet.STOP_ARC,
                     params={'stop': {('lung', 'asia'): True,
                                      ('xray', 'lung'): True,
                                      ('bronc', 'asia'): True,
                                      ('xray', 'either'): True}})
    assert know.rules.rules == [Rule.STOP_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '4 prohibited and expertise 1.0')
    assert know.stop == {('lung', 'asia'): (True, True),
                         ('xray', 'lung'): (True, True),
                         ('bronc', 'asia'): (True, True),
                         ('xray', 'either'): (True, True)}
    print('\nProhibited arcs are: {}'.format(list(know.stop.keys())))
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_knowledge_asia_2_ok():  # STOP_ARC, stop 4 correct
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.STOP_ARC,
                     params={'stop': 4, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.STOP_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '4 prohibited and expertise 1.0')
    assert know.stop == {('lung', 'asia'): (True, True),
                         ('xray', 'lung'): (True, True),
                         ('bronc', 'asia'): (True, True),
                         ('xray', 'either'): (True, True)}
    print('\nProhibited arcs are: {}'.format(list(know.stop.keys())))
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_knowledge_asia_3_ok():  # STOP_ARC, stop 0.0833 (1/12) correct
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.STOP_ARC,
                     params={'stop': 0.5, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.STOP_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '4 prohibited and expertise 1.0')
    assert know.stop == {('lung', 'asia'): (True, True),
                         ('xray', 'lung'): (True, True),
                         ('bronc', 'asia'): (True, True),
                         ('xray', 'either'): (True, True)}
    print('\nProhibited arcs are: {}'.format(list(know.stop.keys())))
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_stop_ok_1(stop1):  # stop add of specified arc

    # check initialisation is OK

    assert stop1.stop == {('B', 'C'): (True, True)}
    assert stop1.event is None
    assert stop1.event_delta is None

    # check blocked is OK

    result = stop1.blocked(DAGChange(Activity.ADD, ('B', 'C'), 1.0, {}))
    expected = KnowledgeEvent(Rule.STOP_ARC, True, KnowledgeOutcome.STOP_ADD,
                              ('B', 'C'))
    assert stop1.event == expected
    assert stop1.event_delta == 1.0
    assert result == expected

    # check hc_best returns correct event

    data = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc').generate_cases(10)
    parents = {'A': set(), 'B': {'A'}, 'C': {'B'}}
    best = BestDAGChanges()
    _best, event = stop1.hc_best(best, 6, data, parents)
    assert best == _best
    assert event == expected

    # check in-iteration event cleared

    assert stop1.event is None
    assert stop1.event_delta is None


def test_stop_ok_2(stop1):  # doesn't stop add of opposite arc
    result = stop1.blocked(DAGChange(Activity.ADD, ('C', 'B'), 1.0, {}))
    assert stop1.event is None
    assert stop1.event_delta is None
    assert result is None


def test_stop_ok_3(stop1):  # doesn't stop delete of arc
    result = stop1.blocked(DAGChange(Activity.DEL, ('B', 'C'), 1.0, {}))
    assert stop1.event is None
    assert stop1.event_delta is None
    assert result is None


def test_stop_ok_4(stop1):  # doesn't stop reverse of arc
    result = stop1.blocked(DAGChange(Activity.REV, ('B', 'C'), 1.0, {}))
    assert stop1.event is None
    assert stop1.event_delta is None
    assert result is None


def test_stop_ok_5(stop1):  # blocks reverse of opposite arc
    result = stop1.blocked(DAGChange(Activity.REV, ('C', 'B'), 1.0, {}))
    expected = KnowledgeEvent(Rule.STOP_ARC, True, KnowledgeOutcome.STOP_REV,
                              ('C', 'B'))
    assert stop1.event == expected
    assert stop1.event_delta == 1.0
    assert result == expected


def test_stop_ok_6(stop1):  # blocks add arc then reverse of opposite arc
    result = stop1.blocked(DAGChange(Activity.ADD, ('B', 'C'), 2.0, {}))
    expected = KnowledgeEvent(Rule.STOP_ARC, True, KnowledgeOutcome.STOP_ADD,
                              ('B', 'C'))
    print(result)
    print(expected)
    assert stop1.event == expected
    assert stop1.event_delta == 2.0
    assert result == expected

    # reverse of arc not blocked

    result = stop1.blocked(DAGChange(Activity.ADD, ('C', 'B'), 2.0, {}))
    assert stop1.event == expected
    assert stop1.event_delta == 2.0
    assert result is None

    # reverse of opposite arc blocked (but doesn't overwrite biggest block)

    result = stop1.blocked(DAGChange(Activity.REV, ('C', 'B'), 1.0, {}))
    expected2 = KnowledgeEvent(Rule.STOP_ARC, True, KnowledgeOutcome.STOP_REV,
                               ('C', 'B'))
    assert stop1.event == expected
    assert stop1.event_delta == 2.0
    assert result == expected2

    # reverse of opposite arc blocked (and overwrites biggest block)

    result = stop1.blocked(DAGChange(Activity.REV, ('C', 'B'), 3.0, {}))
    assert stop1.event == expected2
    assert stop1.event_delta == 3.0
    assert result == expected2
