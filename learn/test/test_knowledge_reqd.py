
# Test the Knowledge class

import pytest

from learn.knowledge import Knowledge, Rule, RuleSet, \
    KnowledgeOutcome, KnowledgeEvent
from learn.trace import Activity
from learn.dagchange import DAGChange, BestDAGChanges
from fileio.common import TESTDATA_DIR
from core.bn import BN
from core.graph import DAG


@pytest.fixture
def ab():  # return ab DAG
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    return ref.dag


@pytest.fixture
def reqd1():  # reqd list with arc B --> C (incorrect knowledge)
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    return Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': {('B', 'C'): False},
                             'initial': ref.dag})


@pytest.fixture
def abc1():  # data for A->B->C graph
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    parents = {'A': set(), 'B': {'A'}, 'C': {'B'}}
    return (ref.generate_cases(10), parents)


def test_reqd_type_error_1():  # initial value must be a DAG
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': {('A', 'B'): True},
                                                  'initial': 'invalid'})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': {('A', 'B'): True},
                                                  'initial': {'A': 'invalid'}})


def test_reqd_type_error_2():  # reqd must be a dict/int/float
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': 'a'})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': ('bad',)})


def test_reqd_type_error_3(ab):  # reqd: if dict, must have tuple keys
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': {'A': [1, 2]},
                                                  'initial': ab})


def test_reqd_type_error_4(ab):  # reqd tuples must be len 2
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': {('A',): True},
                                                  'initial': ab})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC,
                  params={'reqd': {('A', 'B', 'C'): True},
                          'initial': ab})


def test_reqd_type_error_5(ab):  # reqd tuples must contain strs
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': {(1, 2): True},
                                                  'initial': ab})


def test_reqd_type_error_6():  # reqd tuples have boolean values
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': {('A', 'B'): 23},
                                                  'initial': ab})


def test_reqd_type_error_7():  # sample must be an integer
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': 1, 'ref': ref},
                  sample='badtype')
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': 1, 'ref': ref},
                  sample=[4])


def test_reqd_value_error_1():  # REQD_ARC needs reqd parameter
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'limit': 4})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'stop': {('A', 'B'): True}})


def test_reqd_value_error_2():  # REQD_ARC needs initial parameter
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': {('A', 'B'): True}})


# For flexibility removed check that required arcs in initial graph

def xtest_reqd_value_error_3(ab):  # REQD_ARC not in initial
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC,
                  params={'reqd': {('B', 'A'): True}, 'initial': ab})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC,
                  params={'reqd': {('A', 'C'): True}, 'initial': ab})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC,
                  params={'reqd': {('A', 'A'): True}, 'initial': ab})


def test_reqd_value_error_4():  # REQD_ARC invalid reqd fraction
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': 0.0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': 1.0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': -0.1})


def test_reqd_value_error_5():  # REQD_ARC stop frac missing ref
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, params={'reqd': 0.1})


def test_reqd_value_error_6():  # REQD_ARC stop frac, initial attribute
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC,
                  params={'reqd': 0.1, 'ref': ref, 'initial': True})


def test_reqd_value_error_7():  # REQD_ARC stop frac, sample < 1. > 100
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, sample=-1,
                  params={'reqd': 0.5, 'ref': ref})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC, sample=101,
                  params={'reqd': 0.5, 'ref': ref})


def test_reqd_value_error_8():  # REQD_ARC with earlyok
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.REQD_ARC,
                  params={'reqd': 0.5, 'ref': ref, 'expertise': 0.5,
                          'earlyok': True})


def test_reqd_reqd1_1_ok(reqd1):  # REQD_ARC with specified arcs
    assert reqd1.rules.rules == [Rule.REQD_ARC]
    assert reqd1.ref is None
    assert reqd1.limit is False
    assert reqd1.ignore == 0
    assert reqd1.expertise == 1.0
    assert reqd1.count == 0
    assert reqd1.label == ('Ruleset "Required arc" with '
                           + '1 required and expertise 1.0')
    assert reqd1.stop == {}
    assert reqd1.reqd == {('B', 'C'): (False, True)}
    assert reqd1.event is None
    assert reqd1.event_delta is None
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    assert reqd1.initial == ref.dag


def test_reqd1_2_ok(reqd1, abc1):  # stop del of specified arc

    # check initialisation is OK

    assert reqd1.reqd == {('B', 'C'): (False, True)}
    assert reqd1.event is None
    assert reqd1.event_delta is None

    # check blocked is OK

    result = reqd1.blocked(DAGChange(Activity.DEL, ('B', 'C'), 1.0, {}))
    expected = KnowledgeEvent(Rule.REQD_ARC, False, KnowledgeOutcome.STOP_DEL,
                              ('B', 'C'))
    assert reqd1.event == expected
    assert reqd1.event_delta == 1.0
    assert result == expected

    # check hc_best returns correct event

    best = BestDAGChanges()
    _best, event = reqd1.hc_best(best, 6, abc1[0], abc1[1])
    assert best == _best
    assert event == expected

    # check in-iteration event cleared

    assert reqd1.event is None
    assert reqd1.event_delta is None


def test_reqd1_3_ok(reqd1, abc1):  # stop add of opposite of specified arc

    # check initialisation is OK

    assert reqd1.reqd == {('B', 'C'): (False, True)}
    assert reqd1.event is None
    assert reqd1.event_delta is None

    # check blocked is OK

    result = reqd1.blocked(DAGChange(Activity.ADD, ('C', 'B'), 1.0, {}))
    expected = KnowledgeEvent(Rule.REQD_ARC, False, KnowledgeOutcome.STOP_ADD,
                              ('C', 'B'))
    assert reqd1.event == expected
    assert reqd1.event_delta == 1.0
    assert result == expected

    # check hc_best returns correct event

    best = BestDAGChanges()
    _best, event = reqd1.hc_best(best, 6, abc1[0], abc1[1])
    assert best == _best
    assert event == expected

    # check in-iteration event cleared

    assert reqd1.event is None
    assert reqd1.event_delta is None


def test_reqd1_4_ok(reqd1):  # stops del of opposite arc (reverse needed)
    result = reqd1.blocked(DAGChange(Activity.DEL, ('C', 'B'), 1.0, {}))
    expected = KnowledgeEvent(Rule.REQD_ARC, False, KnowledgeOutcome.STOP_DEL,
                              ('C', 'B'))
    assert reqd1.event == expected
    assert reqd1.event_delta == 1.0
    assert result == expected


def test_reqd1_5_ok(reqd1):  # doesn't stop add of arc
    result = reqd1.blocked(DAGChange(Activity.ADD, ('B', 'C'), 1.0, {}))
    assert reqd1.event is None
    assert reqd1.event_delta is None
    assert result is None


def test_reqd1_6_ok(reqd1):  # doesn't stop reverse of opposite arc
    result = reqd1.blocked(DAGChange(Activity.REV, ('C', 'B'), 1.0, {}))
    assert reqd1.event is None
    assert reqd1.event_delta is None
    assert result is None


def test_reqd1_7_ok(reqd1):  # blocks reverse of arc
    result = reqd1.blocked(DAGChange(Activity.REV, ('B', 'C'), 1.0, {}))
    expected = KnowledgeEvent(Rule.REQD_ARC, False, KnowledgeOutcome.STOP_REV,
                              ('B', 'C'))
    assert reqd1.event == expected
    assert reqd1.event_delta == 1.0
    assert result == expected


def test_reqd1_8_ok(reqd1):  # blocks del arc then reverse of arc
    result = reqd1.blocked(DAGChange(Activity.DEL, ('B', 'C'), 2.0, {}))
    expected = KnowledgeEvent(Rule.REQD_ARC, False, KnowledgeOutcome.STOP_DEL,
                              ('B', 'C'))
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result == expected

    # reverse of opposite arc not blocked

    result = reqd1.blocked(DAGChange(Activity.REV, ('C', 'B'), 2.0, {}))
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result is None

    # reverse of arc blocked (but doesn't overwrite biggest block)

    result = reqd1.blocked(DAGChange(Activity.REV, ('B', 'C'), 1.0, {}))
    expected2 = KnowledgeEvent(Rule.REQD_ARC, False, KnowledgeOutcome.STOP_REV,
                               ('B', 'C'))
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result == expected2

    # reverse of arc blocked (and overwrites biggest block)

    result = reqd1.blocked(DAGChange(Activity.REV, ('B', 'C'), 2.001, {}))
    assert reqd1.event == expected2
    assert reqd1.event_delta == 2.001
    assert result == expected2


def test_reqd1_9_ok(reqd1):  # blocks del arc then add of reverse arc
    result = reqd1.blocked(DAGChange(Activity.DEL, ('B', 'C'), 2.0, {}))
    expected = KnowledgeEvent(Rule.REQD_ARC, False, KnowledgeOutcome.STOP_DEL,
                              ('B', 'C'))
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result == expected

    # reverse of opposite arc not blocked

    result = reqd1.blocked(DAGChange(Activity.REV, ('C', 'B'), 2.0, {}))
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result is None

    # add of opposite arc blocked (but doesn't overwrite biggest block)

    result = reqd1.blocked(DAGChange(Activity.ADD, ('C', 'B'), 1.0, {}))
    expected2 = KnowledgeEvent(Rule.REQD_ARC, False, KnowledgeOutcome.STOP_ADD,
                               ('C', 'B'))
    assert reqd1.event == expected
    assert reqd1.event_delta == 2.0
    assert result == expected2

    # reverse of arc blocked (and overwrites biggest block)

    result = reqd1.blocked(DAGChange(Activity.REV, ('B', 'C'), 2.001, {}))
    expected3 = KnowledgeEvent(Rule.REQD_ARC, False, KnowledgeOutcome.STOP_REV,
                               ('B', 'C'))
    assert reqd1.event == expected3
    assert reqd1.event_delta == 2.001
    assert result == expected3


# Cancer DAG

def test_reqd_cancer_1_ok():  # REQD_ARC, reqd 2
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 2, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '2 required and expertise 1.0')

    # 3 correct required arcs

    print('\nRequired arcs are: {}\n\nand initial graph:\n{}'
          .format(list(know.reqd), know.initial))
    assert know.reqd == {('Cancer', 'Xray'): (True, True),
                         ('Cancer', 'Dyspnoea'): (True, True)}
    assert know.stop == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Cancer', 'Dyspnoea', 'Xray'],
                               [('Cancer', '->', 'Xray'),
                                ('Cancer', '->', 'Dyspnoea')])


def test_reqd_cancer_2_ok():  # REQD_ARC, reqd 0.5
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 0.5, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '2 required and expertise 1.0')

    # 2 correct required arcs

    print('\nRequired arcs are: {}\n\nand initial graph:\n{}'
          .format(list(know.reqd), know.initial))
    assert know.reqd == {('Cancer', 'Xray'): (True, True),
                         ('Cancer', 'Dyspnoea'): (True, True)}
    assert know.stop == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Cancer', 'Dyspnoea', 'Xray'],
                               [('Cancer', '->', 'Xray'),
                                ('Cancer', '->', 'Dyspnoea')])


def test_reqd_cancer_3_ok():  # REQD_ARC, reqd 2, sample 1
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC, sample=1,
                     params={'reqd': 2, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '2 required and expertise 1.0')

    # different 2 required arcs

    print('\nRequired arcs are: {}\n\nand initial graph:\n{}'
          .format(list(know.reqd), know.initial))
    assert know.reqd == {('Cancer', 'Dyspnoea'): (True, True),
                         ('Pollution', 'Cancer'): (True, True)}
    assert know.stop == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Cancer', 'Dyspnoea', 'Pollution'],
                               [('Pollution', '->', 'Cancer'),
                                ('Cancer', '->', 'Dyspnoea')])


def test_reqd_cancer_4_ok():  # REQD_ARC, reqd 5, expetise 0.2
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 5, 'ref': ref, 'expertise': 0.2})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.2
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '5 required and expertise 0.2')

    # 5 required arcs - 4 wrong, 1 correct

    print('\nRequired arcs are: {}\n\nand initial graph:\n{}'
          .format(list(know.reqd), know.initial))
    assert know.reqd == {('Pollution', 'Xray'): (False, True),
                         ('Xray', 'Smoker'): (False, True),
                         ('Pollution', 'Cancer'): (True, True),
                         ('Dyspnoea', 'Smoker'): (False, True),
                         ('Pollution', 'Dyspnoea'): (False, True)}
    assert know.stop == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                                'Xray'],
                               [('Pollution', '->', 'Xray'),
                                ('Xray', '->', 'Smoker'),
                                ('Pollution', '->', 'Cancer'),
                                ('Dyspnoea', '->', 'Smoker'),
                                ('Pollution', '->', 'Dyspnoea')])


def test_reqd_cancer_5_ok():  # REQD_ARC, reqd 1.25, expetise 0.2
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 1.0, 'ref': ref, 'expertise': 0.2})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.2
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '5 required and expertise 0.2')

    # 5 required arcs - 4 wrong, 1 correct

    print('\nRequired arcs are: {}\n\nand initial graph:\n{}'
          .format(list(know.reqd), know.initial))
    assert know.reqd == {('Pollution', 'Xray'): (False, True),
                         ('Xray', 'Smoker'): (False, True),
                         ('Pollution', 'Cancer'): (True, True),
                         ('Dyspnoea', 'Smoker'): (False, True),
                         ('Pollution', 'Dyspnoea'): (False, True)}
    assert know.stop == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                                'Xray'],
                               [('Pollution', '->', 'Xray'),
                                ('Xray', '->', 'Smoker'),
                                ('Pollution', '->', 'Cancer'),
                                ('Dyspnoea', '->', 'Smoker'),
                                ('Pollution', '->', 'Dyspnoea')])


def test_reqd_cancer_6_ok():  # REQD_ARC, reqd 1.25, expertise 0.2
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 1.0, 'ref': ref, 'expertise': 0.2})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.2
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '5 required and expertise 0.2')

    # 5 required arcs - 4 wrong, 1 correct

    print('\nRequired arcs are: {}\n\nand initial graph:\n{}'
          .format(list(know.reqd), know.initial))
    assert know.reqd == {('Pollution', 'Xray'): (False, True),
                         ('Xray', 'Smoker'): (False, True),
                         ('Pollution', 'Cancer'): (True, True),
                         ('Dyspnoea', 'Smoker'): (False, True),
                         ('Pollution', 'Dyspnoea'): (False, True)}
    assert know.stop == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                                'Xray'],
                               [('Pollution', '->', 'Xray'),
                                ('Xray', '->', 'Smoker'),
                                ('Pollution', '->', 'Cancer'),
                                ('Dyspnoea', '->', 'Smoker'),
                                ('Pollution', '->', 'Dyspnoea')])


def test_reqd_cancer_7_ok():  # REQD_ARC with fraction 0.5
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 0.5, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '2 required and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {('Cancer', 'Dyspnoea'): (True, True),
                         ('Cancer', 'Xray'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    print('\nInitial DAG set to:\n{}'.format(know.initial))
    initial = DAG(['Cancer', 'Dyspnoea', 'Xray'],
                  [('Cancer', '->', 'Dyspnoea'),
                   ('Cancer', '->', 'Xray')])
    assert know.initial == initial


def test_reqd_cancer_8_ok():  # REQD_ARC with fraction 0.5, sample 1
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC, sample=1,
                     params={'reqd': 0.5, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '2 required and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {('Cancer', 'Dyspnoea'): (True, True),
                         ('Pollution', 'Cancer'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    print('\nInitial DAG set to:\n{}'.format(know.initial))
    initial = DAG(['Cancer', 'Dyspnoea', 'Pollution'],
                  [('Cancer', '->', 'Dyspnoea'),
                   ('Pollution', '->', 'Cancer')])
    assert know.initial == initial


def test_reqd_cancer_9_ok():  # REQD_ARC with fraction 0.5, expertise 0.5
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 0.5, 'ref': ref, 'expertise': 0.5})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.5
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '2 required and expertise 0.5')
    assert know.stop == {}
    assert know.reqd == {('Pollution', 'Xray'): (False, True),
                         ('Xray', 'Smoker'): (False, True)}
    assert know.event is None
    assert know.event_delta is None
    print('\nInitial DAG set to:\n{}'.format(know.initial))
    initial = DAG(['Pollution', 'Smoker', 'Xray'],
                  [('Xray', '->', 'Smoker'),
                   ('Pollution', '->', 'Xray')])
    assert know.initial == initial


def test_reqd_cancer_10_ok():  # REQD_ARC 0.5, expertise 0.5, sample 1
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC, sample=1,
                     params={'reqd': 0.5, 'ref': ref, 'expertise': 0.5})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.5
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '2 required and expertise 0.5')
    assert know.stop == {}
    assert know.reqd == {('Xray', 'Smoker'): (False, True),
                         ('Smoker', 'Cancer'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    print('\nInitial DAG set to:\n{}'.format(know.initial))
    initial = DAG(['Cancer', 'Smoker', 'Xray'],
                  [('Xray', '->', 'Smoker'),
                   ('Smoker', '->', 'Cancer')])
    assert know.initial == initial


def test_reqd_cancer_11_ok(reqd1):  # REQD_ARC with fraction 0.95
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 0.95, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '4 required and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {('Pollution', 'Cancer'): (True, True),
                         ('Smoker', 'Cancer'): (True, True),
                         ('Cancer', 'Dyspnoea'): (True, True),
                         ('Cancer', 'Xray'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    print('\nInitial DAG set to:\n{}'.format(know.initial))
    assert know.initial == ref.dag


def test_reqd_cancer_12_ok(reqd1):  # REQD_ARC 0.95, exp 0.05
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 0.80, 'ref': ref, 'expertise': 0.05})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.05
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '4 required and expertise 0.05')
    assert know.stop == {}
    assert know.reqd == {('Pollution', 'Xray'): (False, True),
                         ('Xray', 'Smoker'): (False, True),
                         ('Dyspnoea', 'Cancer'): (False, True),
                         ('Dyspnoea', 'Xray'): (False, True)}
    assert know.event is None
    assert know.event_delta is None
    print('\nInitial DAG set to:\n{}'.format(know.initial))
    initial = DAG(['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray'],
                  [('Pollution', '->', 'Xray'),
                   ('Xray', '->', 'Smoker'),
                   ('Dyspnoea', '->', 'Cancer'),
                   ('Dyspnoea', '->', 'Xray')])
    assert know.initial == initial


def test_reqd_cancer_13_ok(reqd1):  # REQD_ARC 0.40, suppress initial
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 0.40, 'ref': ref, 'expertise': 1.0,
                             'initial': False})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '2 required and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {('Cancer', 'Xray'): (True, True),
                         ('Cancer', 'Dyspnoea'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    print('\nInitial DAG set to:\n{}'.format(know.initial))
    assert know.initial is None


def test_reqd_cancer_14_ok(reqd1):  # REQD_ARC 0.40, specify initial
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    initial = DAG(['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray'],
                  [('Dyspnoea', '->', 'Cancer'),
                   ('Dyspnoea', '->', 'Xray'),
                   ('Pollution', '->', 'Xray'),
                   ('Xray', '->', 'Smoker')])
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 0.40, 'ref': ref, 'expertise': 1.0,
                             'initial': initial})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '2 required and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {('Cancer', 'Xray'): (True, True),
                         ('Cancer', 'Dyspnoea'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    print('\nInitial DAG set to:\n{}'.format(know.initial))
    assert know.initial == initial


def test_reqd_asia_1_ok():  # REQD_ARC, explicit list
    initial = DAG(['asia', 'bronc', 'dysp', 'either', 'tub'],
                  [('tub', '->', 'either'), ('either', '->', 'dysp'),
                   ('asia', '->', 'tub'), ('bronc', '->', 'dysp')])
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': {('tub', 'either'): True,
                                      ('either', 'dysp'): True,
                                      ('asia', 'tub'): True,
                                      ('bronc', 'dysp'): True},
                             'initial': initial})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '4 required and expertise 1.0')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('tub', 'either'): (True, True),
                         ('either', 'dysp'): (True, True),
                         ('asia', 'tub'): (True, True),
                         ('bronc', 'dysp'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['asia', 'bronc', 'dysp', 'either', 'tub'],
                               [('tub', '->', 'either'),
                                ('either', '->', 'dysp'),
                                ('asia', '->', 'tub'),
                                ('bronc', '->', 'dysp')])


def test_reqd_asia_2_ok():  # REQD_ARC, reqd 4 correct
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 4, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '4 required and expertise 1.0')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('tub', 'either'): (True, True),
                         ('either', 'dysp'): (True, True),
                         ('asia', 'tub'): (True, True),
                         ('bronc', 'dysp'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['asia', 'bronc', 'dysp', 'either', 'tub'],
                               [('tub', '->', 'either'),
                                ('either', '->', 'dysp'),
                                ('asia', '->', 'tub'),
                                ('bronc', '->', 'dysp')])


def test_reqd_asia_3_ok():  # REQD_ARC, reqd 0.5 correct
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 0.5, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '4 required and expertise 1.0')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('tub', 'either'): (True, True),
                         ('either', 'dysp'): (True, True),
                         ('asia', 'tub'): (True, True),
                         ('bronc', 'dysp'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['asia', 'bronc', 'dysp', 'either', 'tub'],
                               [('tub', '->', 'either'),
                                ('either', '->', 'dysp'),
                                ('asia', '->', 'tub'),
                                ('bronc', '->', 'dysp')])


def test_reqd_asia_4_ok():  # REQD_ARC, reqd 16, expertise 1/7
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 16, 'ref': ref, 'expertise': 1/7})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1/7
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '16 required and expertise 0.143')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('lung', 'asia'): (False, True),
                         ('xray', 'lung'): (False, True),
                         ('bronc', 'asia'): (False, True),
                         ('xray', 'either'): (False, True),
                         ('dysp', 'either'): (False, True),
                         ('either', 'bronc'): (False, True),
                         ('tub', 'lung'): (False, True),
                         ('dysp', 'xray'): (False, True),
                         ('tub', 'asia'): (False, True),
                         ('lung', 'bronc'): (False, True),
                         ('dysp', 'bronc'): (False, True),
                         ('tub', 'dysp'): (False, True),
                         ('either', 'lung'): (False, True),
                         ('smoke', 'lung'): (True, True),
                         ('smoke', 'xray'): (False, True),
                         ('smoke', 'either'): (False, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['asia', 'bronc', 'dysp', 'either', 'lung',
                                'smoke', 'tub', 'xray'],
                               [('lung', '->', 'asia'),
                                ('xray', '->', 'lung'),
                                ('bronc', '->', 'asia'),
                                ('xray', '->', 'either'),
                                ('dysp', '->', 'either'),
                                ('either', '->', 'bronc'),
                                ('tub', '->', 'lung'),
                                ('dysp', '->', 'xray'),
                                ('tub', '->', 'asia'),
                                ('lung', '->', 'bronc'),
                                ('dysp', '->', 'bronc'),
                                ('tub', '->', 'dysp'),
                                ('either', '->', 'lung'),
                                ('smoke', '->', 'lung'),
                                ('smoke', '->', 'xray'),
                                ('smoke', '->', 'either')])


def test_reqd_asia_5_ok():  # REQD_ARC, reqd 16, expertise unspecified
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 16, 'ref': ref})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1/7
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '16 required and expertise 0.143')
    print(know.reqd)
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('lung', 'asia'): (False, True),
                         ('xray', 'lung'): (False, True),
                         ('bronc', 'asia'): (False, True),
                         ('xray', 'either'): (False, True),
                         ('dysp', 'either'): (False, True),
                         ('either', 'bronc'): (False, True),
                         ('tub', 'lung'): (False, True),
                         ('dysp', 'xray'): (False, True),
                         ('tub', 'asia'): (False, True),
                         ('lung', 'bronc'): (False, True),
                         ('dysp', 'bronc'): (False, True),
                         ('tub', 'dysp'): (False, True),
                         ('either', 'lung'): (False, True),
                         ('smoke', 'lung'): (True, True),
                         ('smoke', 'xray'): (False, True),
                         ('smoke', 'either'): (False, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['asia', 'bronc', 'dysp', 'either', 'lung',
                                'smoke', 'tub', 'xray'],
                               [('lung', '->', 'asia'),
                                ('xray', '->', 'lung'),
                                ('bronc', '->', 'asia'),
                                ('xray', '->', 'either'),
                                ('dysp', '->', 'either'),
                                ('either', '->', 'bronc'),
                                ('tub', '->', 'lung'),
                                ('dysp', '->', 'xray'),
                                ('tub', '->', 'asia'),
                                ('lung', '->', 'bronc'),
                                ('dysp', '->', 'bronc'),
                                ('tub', '->', 'dysp'),
                                ('either', '->', 'lung'),
                                ('smoke', '->', 'lung'),
                                ('smoke', '->', 'xray'),
                                ('smoke', '->', 'either')])


def test_reqd_asia_6_ok():  # REQD_ARC, reqd 16 correct
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 16, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '8 required and expertise 1.0')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('tub', 'either'): (True, True),
                         ('either', 'dysp'): (True, True),
                         ('asia', 'tub'): (True, True),
                         ('bronc', 'dysp'): (True, True),
                         ('smoke', 'bronc'): (True, True),
                         ('lung', 'either'): (True, True),
                         ('either', 'xray'): (True, True),
                         ('smoke', 'lung'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['asia', 'bronc', 'dysp', 'either', 'lung',
                                'smoke', 'tub', 'xray'],
                               [('tub', '->', 'either'),
                                ('either', '->', 'dysp'),
                                ('asia', '->', 'tub'),
                                ('bronc', '->', 'dysp'),
                                ('smoke', '->', 'bronc'),
                                ('lung', '->', 'either'),
                                ('either', '->', 'xray'),
                                ('smoke', '->', 'lung')])


def test_reqd_insurance_1_ok():  # REQD_ARC 0.25 expertise 1.0
    ref = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 0.25, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '7 required and expertise 1.0')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('Theft', 'ThisCarCost'): (True, True),
                         ('Accident', 'ILiCost'): (True, True),
                         ('Accident', 'ThisCarDam'): (True, True),
                         ('VehicleYear', 'Airbag'): (True, True),
                         ('RiskAversion', 'MakeModel'): (True, True),
                         ('SeniorTrain', 'DrivingSkill'): (True, True),
                         ('Accident', 'MedCost'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Accident', 'Airbag', 'DrivingSkill',
                                'ILiCost', 'MakeModel', 'MedCost',
                                'RiskAversion', 'SeniorTrain', 'Theft',
                                'ThisCarCost', 'ThisCarDam', 'VehicleYear'],
                               [('Theft', '->', 'ThisCarCost'),
                                ('Accident', '->', 'ILiCost'),
                                ('Accident', '->', 'ThisCarDam'),
                                ('VehicleYear', '->', 'Airbag'),
                                ('RiskAversion', '->', 'MakeModel'),
                                ('SeniorTrain', '->', 'DrivingSkill'),
                                ('Accident', '->', 'MedCost')])


def test_reqd_insurance_2_ok():  # REQD_ARC, reqd 0.25 expertise 1.0, sample 1
    ref = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC, sample=1,
                     params={'reqd': 0.25, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '7 required and expertise 1.0')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('Accident', 'ILiCost'): (True, True),
                         ('SocioEcon', 'MakeModel'): (True, True),
                         ('DrivQuality', 'Accident'): (True, True),
                         ('MakeModel', 'Antilock'): (True, True),
                         ('RiskAversion', 'MakeModel'): (True, True),
                         ('RiskAversion', 'DrivQuality'): (True, True),
                         ('OtherCarCost', 'PropCost'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Accident', 'Antilock', 'DrivQuality',
                                'ILiCost', 'MakeModel', 'OtherCarCost',
                                'PropCost', 'RiskAversion', 'SocioEcon'],
                               [('Accident', '->', 'ILiCost'),
                                ('SocioEcon', '->', 'MakeModel'),
                                ('DrivQuality', '->', 'Accident'),
                                ('MakeModel', '->', 'Antilock'),
                                ('RiskAversion', '->', 'MakeModel'),
                                ('RiskAversion', '->', 'DrivQuality'),
                                ('OtherCarCost', '->', 'PropCost')])


def test_reqd_insurance_3_ok():  # REQD_ARC, reqd 0.25 expertise 1.0, sample 2
    ref = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC, sample=2,
                     params={'reqd': 0.25, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '7 required and expertise 1.0')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('Accident', 'ThisCarDam'): (True, True),
                         ('RuggedAuto', 'ThisCarDam'): (True, True),
                         ('CarValue', 'Theft'): (True, True),
                         ('SocioEcon', 'HomeBase'): (True, True),
                         ('SocioEcon', 'RiskAversion'): (True, True),
                         ('SeniorTrain', 'DrivingSkill'): (True, True),
                         ('DrivingSkill', 'DrivQuality'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Accident', 'CarValue', 'DrivQuality',
                                'DrivingSkill', 'HomeBase', 'RiskAversion',
                                'RuggedAuto', 'SeniorTrain', 'SocioEcon',
                                'Theft', 'ThisCarDam'],
                               [('Accident', '->', 'ThisCarDam'),
                                ('RuggedAuto', '->', 'ThisCarDam'),
                                ('CarValue', '->', 'Theft'),
                                ('SocioEcon', '->', 'HomeBase'),
                                ('SocioEcon', '->', 'RiskAversion'),
                                ('SeniorTrain', '->', 'DrivingSkill'),
                                ('DrivingSkill', '->', 'DrivQuality')])


def test_reqd_insurance_4_ok():  # REQD_ARC 0.25 expertise 0.70
    ref = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC,
                     params={'reqd': 0.25, 'ref': ref, 'expertise': 0.7})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.7
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '7 required and expertise 0.7')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('Theft', 'ThisCarCost'): (True, True),
                         ('ThisCarDam', 'MedCost'): (False, True),
                         ('Accident', 'ThisCarDam'): (True, True),
                         ('ThisCarDam', 'ILiCost'): (False, True),
                         ('DrivingSkill', 'DrivQuality'): (True, True),
                         ('Age', 'SeniorTrain'): (True, True),
                         ('SocioEcon', 'Airbag'): (False, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Accident', 'Age', 'Airbag',
                                'DrivQuality', 'DrivingSkill', 'ILiCost',
                                'MedCost', 'SeniorTrain', 'SocioEcon',
                                'Theft', 'ThisCarCost', 'ThisCarDam'],
                               [('Theft', '->', 'ThisCarCost'),
                                ('ThisCarDam', '->', 'MedCost'),
                                ('Accident', '->', 'ThisCarDam'),
                                ('ThisCarDam', '->', 'ILiCost'),
                                ('DrivingSkill', '->', 'DrivQuality'),
                                ('Age', '->', 'SeniorTrain'),
                                ('SocioEcon', '->', 'Airbag')])


def test_reqd_insurance_5_ok():  # REQD_ARC 0.25 expertise 0.70, sample 1
    ref = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC, sample=1,
                     params={'reqd': 0.25, 'ref': ref, 'expertise': 0.7})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.7
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '7 required and expertise 0.7')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('ThisCarDam', 'MedCost'): (False, True),
                         ('RiskAversion', 'MakeModel'): (True, True),
                         ('DrivQuality', 'Accident'): (True, True),
                         ('MakeModel', 'Antilock'): (True, True),
                         ('DrivingSkill', 'DrivQuality'): (True, True),
                         ('VehicleYear', 'Antilock'): (True, True),
                         ('PropCost', 'ThisCarCost'): (False, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Accident', 'Antilock', 'DrivQuality',
                                'DrivingSkill', 'MakeModel', 'MedCost',
                                'PropCost', 'RiskAversion', 'ThisCarCost',
                                'ThisCarDam', 'VehicleYear'],
                               [('ThisCarDam', '->', 'MedCost'),
                                ('RiskAversion', '->', 'MakeModel'),
                                ('DrivQuality', '->', 'Accident'),
                                ('MakeModel', '->', 'Antilock'),
                                ('DrivingSkill', '->', 'DrivQuality'),
                                ('VehicleYear', '->', 'Antilock'),
                                ('PropCost', '->', 'ThisCarCost')])


def test_reqd_insurance_6_ok():  # REQD_ARC 0.25 expertise 0.70, sample 2
    ref = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC, sample=2,
                     params={'reqd': 0.25, 'ref': ref, 'expertise': 0.7})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.7
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '7 required and expertise 0.7')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('Accident', 'ThisCarDam'): (True, True),
                         ('RuggedAuto', 'ThisCarDam'): (True, True),
                         ('CarValue', 'Theft'): (True, True),
                         ('SocioEcon', 'HomeBase'): (True, True),
                         ('SocioEcon', 'RiskAversion'): (True, True),
                         ('SeniorTrain', 'DrivingSkill'): (True, True),
                         ('DrivingSkill', 'DrivQuality'): (True, True)}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial == DAG(['Accident', 'CarValue', 'DrivQuality',
                                'DrivingSkill', 'HomeBase', 'RiskAversion',
                                'RuggedAuto', 'SeniorTrain', 'SocioEcon',
                                'Theft', 'ThisCarDam'],
                               [('Accident', '->', 'ThisCarDam'),
                                ('RuggedAuto', '->', 'ThisCarDam'),
                                ('CarValue', '->', 'Theft'),
                                ('SocioEcon', '->', 'HomeBase'),
                                ('SocioEcon', '->', 'RiskAversion'),
                                ('SeniorTrain', '->', 'DrivingSkill'),
                                ('DrivingSkill', '->', 'DrivQuality')])


def test_reqd_insurance_7_ok():  # REQD_ARC 0.25 expertise 0.30, sample 3
    ref = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    know = Knowledge(rules=RuleSet.REQD_ARC, sample=3,
                     params={'reqd': 0.25, 'ref': ref, 'expertise': 0.3})
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref == ref
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 0.3
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '7 required and expertise 0.3')
    print('\nRequired arcs are: {}\nand initial DAG:\n{}'
          .format(list(know.reqd.keys()), know.initial))
    assert know.stop == {}
    assert know.reqd == {('ThisCarDam', 'MakeModel'): (False, True),
                         ('OtherCar', 'ThisCarDam'): (False, True),
                         ('Antilock', 'Accident'): (True, True),
                         ('Age', 'GoodStudent'): (True, True),
                         ('DrivQuality', 'Theft'): (False, True),
                         ('GoodStudent', 'Accident'): (False, True),
                         ('HomeBase', 'OtherCar'): (False, True)}
    assert know.event is None
    assert know.event_delta is None
    # print(know.initial.nodes)
    # print([(a[0], '->', a[1]) for a in know.initial.edges])
    assert know.initial == DAG(['Accident', 'Age', 'Antilock',
                                'DrivQuality', 'GoodStudent', 'HomeBase',
                                'MakeModel', 'OtherCar', 'Theft',
                                'ThisCarDam'],
                               [('ThisCarDam', '->', 'MakeModel'),
                                ('OtherCar', '->', 'ThisCarDam'),
                                ('Antilock', '->', 'Accident'),
                                ('Age', '->', 'GoodStudent'),
                                ('DrivQuality', '->', 'Theft'),
                                ('GoodStudent', '->', 'Accident'),
                                ('HomeBase', '->', 'OtherCar')])
