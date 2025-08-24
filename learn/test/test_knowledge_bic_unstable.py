
# Test the Knowledge class using LL stability rule

import pytest

from learn.knowledge import Knowledge, Rule, RuleSet, \
    KnowledgeOutcome
from learn.trace import Activity
from learn.dagchange import DAGChange, BestDAGChanges
from fileio.common import TESTDATA_DIR
from fileio.numpy import NumPy
from core.bn import BN
from core.common import init_stable_random


@pytest.fixture
def know_abc_1():  # rule with limit of one
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    return Knowledge(rules=RuleSet.BIC_UNSTABLE,
                     params={'ref': ref, 'limit': 1})


@pytest.fixture
def know_abc_2():  # rule with limit of two, ignore first
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    return Knowledge(rules=RuleSet.BIC_UNSTABLE,
                     params={'ref': ref, 'limit': 2, 'ignore': 1})


@pytest.fixture
def know_abc_3():  # rule with limit of 3, expertise 0.5
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    return Knowledge(rules=RuleSet.BIC_UNSTABLE,
                     params={'ref': ref, 'limit': 3, 'expertise': 0.5})


@pytest.fixture
def know_abc_4():  # rule with limit of 1, ignore 1, expertise 0.8
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    return Knowledge(rules=RuleSet.BIC_UNSTABLE,
                     params={'ref': ref, 'limit': 1, 'ignore': 1,
                             'expertise': 0.8})


@pytest.fixture
def know_abc_5():  # rule with limit of 3, expertise 0.5
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    return Knowledge(rules=RuleSet.BIC_UNSTABLE,
                     params={'ref': ref, 'limit': 3, 'expertise': 0.5,
                             'partial': True})


@pytest.fixture
def abc1():  # data and parents for A->B->C graph
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    parents = {'A': set(), 'B': {'A'}, 'C': {'B'}}
    data = NumPy.from_df(df=ref.generate_cases(10), dstype='categorical',
                         keep_df=True)
    return (data, parents)


@pytest.fixture
def ab():  # return ab DAG
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    return ref.dag


@pytest.fixture
def asia1():  # Asia network with perfect partial expert
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = NumPy.from_df(df=ref.generate_cases(10), dstype='categorical',
                         keep_df=True)
    parents = {n: set() for n in ref.dag.nodes}
    know = Knowledge(rules=RuleSet.BIC_UNSTABLE,
                     params={'ref': ref, 'partial': True})
    return {'data': data, 'parents': parents, 'know': know}


# Test the Knowledge constructor

def test_bic_unstable_type_error_1():  # bad threshold type
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE,
                  params={'ref': ref, 'limit': 0.5, 'expertise': 1.0,
                          'threshold': 'bad'})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE,
                  params={'ref': ref, 'limit': 0.5, 'expertise': 1.0,
                          'threshold': 1})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE,
                  params={'ref': ref, 'limit': 0.5, 'expertise': 1.0,
                          'threshold': [2.0]})


def test_bic_unstable_value_error_1():  # threshold bad value
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE,
                  params={'ref': ref, 'limit': 0.5, 'expertise': 1.0,
                          'threshold': 2.0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE,
                  params={'ref': ref, 'limit': 0.5, 'expertise': 1.0,
                          'threshold': -0.01})


def test_bic_unstable_value_error_2():  # non-positive limit
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={'limit': 0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={'limit': -1})


def test_bic_unstable_value_error_3():  # limit is True
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={'limit': True})


def test_bic_unstable_value_error_4():  # limit is float, ref not specified
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={'limit': 0.2})


def test_bic_unstable_value_error_5():  # float limit not between 0 and 1
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE,
                  params={'limit': 0.0, 'ref': ref})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE,
                  params={'limit': 1.0, 'ref': ref})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE,
                  params={'limit': -0.1, 'ref': ref})


def test_bic_unstable_value_error_6():  # non-positive ignore
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={'ignore': -1})


def test_bic_unstable_value_error_7():  # out of range expertise
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={'expertise': -0.01})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={'expertise': 1.01})


def test_bic_unstable_value_error_8():  # BIC_UNSTABLE needs ref parameter
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={'limit': 4})


def test_bic_unstable_value_error_9():  # sample not allowed
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.BIC_UNSTABLE, params={'ref': ref, 'sample': 2})


def test_bic_unstable_1_ok():  # BIC_UNSTABLE ruleset, ref param
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    knowledge = Knowledge(rules=RuleSet.BIC_UNSTABLE, params={'ref': ref})
    assert knowledge.rules.rules == [Rule.BIC_UNSTABLE]
    assert knowledge.ref == ref
    assert knowledge.limit is False
    assert knowledge.threshold == 0.05
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == ('Ruleset "BIC unstable" with limit ' +
                               'False, threshold 0.05, partial False and ' +
                               'expertise 1.0')
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
    assert knowledge.event is None
    assert knowledge.event_delta is None
    assert knowledge.initial is None


def test_bic_unstable_2_ok():  # EQUIV_ADD ruleset, ref & limit params
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    knowledge = Knowledge(rules=RuleSet.BIC_UNSTABLE,
                          params={'ref': ref, 'limit': 10})
    assert knowledge.rules.rules == [Rule.BIC_UNSTABLE]
    assert knowledge.ref == ref
    assert knowledge.limit == 10
    assert knowledge.threshold == 0.05
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == ('Ruleset "BIC unstable" with limit ' +
                               '10, threshold 0.05, partial False and ' +
                               'expertise 1.0')
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
    assert knowledge.event is None
    assert knowledge.event_delta is None
    assert knowledge.initial is None


def test_bic_unstable_3_ok(know_abc_1):
    assert know_abc_1.rules.rules == [Rule.BIC_UNSTABLE]
    assert isinstance(know_abc_1.ref, BN)
    assert know_abc_1.ref.dag.to_string() == '[A][B|A][C|B]'
    assert know_abc_1.limit == 1
    assert know_abc_1.ignore == 0
    assert know_abc_1.threshold == 0.05
    assert know_abc_1.expertise == 1.0
    assert know_abc_1.partial is False
    assert know_abc_1.count == 0
    assert know_abc_1.label == ('Ruleset "BIC unstable" with limit ' +
                                '1, threshold 0.05, partial False and ' +
                                'expertise 1.0')
    assert know_abc_1.stop == {}
    assert know_abc_1.reqd == {}
    assert know_abc_1.event is None
    assert know_abc_1.event_delta is None
    assert know_abc_1.initial is None


def test_bic_unstable_4_ok(know_abc_2):
    assert know_abc_2.rules.rules == [Rule.BIC_UNSTABLE]
    assert isinstance(know_abc_2.ref, BN)
    assert know_abc_2.ref.dag.to_string() == '[A][B|A][C|B]'
    assert know_abc_2.limit == 2
    assert know_abc_2.ignore == 1
    assert know_abc_2.threshold == 0.05
    assert know_abc_2.expertise == 1.0
    assert know_abc_2.partial is False
    assert know_abc_2.count == 0
    assert know_abc_2.label == ('Ruleset "BIC unstable" with limit ' +
                                '2, threshold 0.05, partial False and ' +
                                'expertise 1.0')
    assert know_abc_2.stop == {}
    assert know_abc_2.reqd == {}
    assert know_abc_2.event is None
    assert know_abc_2.event_delta is None
    assert know_abc_2.initial is None


def test_bic_unstable_5_ok(know_abc_3):
    assert know_abc_3.rules.rules == [Rule.BIC_UNSTABLE]
    assert isinstance(know_abc_3.ref, BN)
    assert know_abc_3.ref.dag.to_string() == '[A][B|A][C|B]'
    assert know_abc_3.limit == 3
    assert know_abc_3.ignore == 0
    assert know_abc_3.threshold == 0.05
    assert know_abc_3.expertise == 0.5
    assert know_abc_3.partial is False
    assert know_abc_3.count == 0
    assert know_abc_3.label == ('Ruleset "BIC unstable" with limit ' +
                                '3, threshold 0.05, partial False and ' +
                                'expertise 0.5')
    assert know_abc_3.stop == {}
    assert know_abc_3.reqd == {}
    assert know_abc_3.event is None
    assert know_abc_3.event_delta is None
    assert know_abc_3.initial is None


def test_bic_unstable_6_ok(know_abc_4):
    assert know_abc_4.rules.rules == [Rule.BIC_UNSTABLE]
    assert isinstance(know_abc_4.ref, BN)
    assert know_abc_4.ref.dag.to_string() == '[A][B|A][C|B]'
    assert know_abc_4.limit == 1
    assert know_abc_4.ignore == 1
    assert know_abc_4.threshold == 0.05
    assert know_abc_4.expertise == 0.8
    assert know_abc_4.partial is False
    assert know_abc_4.count == 0
    assert know_abc_4.label == ('Ruleset "BIC unstable" with limit ' +
                                '1, threshold 0.05, partial False and ' +
                                'expertise 0.8')
    assert know_abc_4.stop == {}
    assert know_abc_4.reqd == {}
    assert know_abc_4.event is None
    assert know_abc_4.event_delta is None
    assert know_abc_4.initial is None


# Knowledge constructor with Cancer

def test_bic_unstable_7_ok():  # BIC_UNSTABLE, limit 0.2 --> 1
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.BIC_UNSTABLE,
                     params={'limit': 0.2, 'ref': ref})
    assert know.rules.rules == [Rule.BIC_UNSTABLE]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit == 1
    assert know.ignore == 0
    assert know.threshold == 0.05
    assert know.expertise == 1.0
    assert know.partial is False
    assert know.count == 0
    assert know.label == ('Ruleset "BIC unstable" with limit ' +
                          '1, threshold 0.05, partial False and ' +
                          'expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_bic_unstable_8_ok():  # BIC_UNSTABLE, limit 0.5 --> 2
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.BIC_UNSTABLE,
                     params={'limit': 0.5, 'ref': ref})
    assert know.rules.rules == [Rule.BIC_UNSTABLE]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit == 2
    assert know.ignore == 0
    assert know.threshold == 0.05
    assert know.expertise == 1.0
    assert know.partial is False
    assert know.count == 0
    assert know.label == ('Ruleset "BIC unstable" with limit ' +
                          '2, threshold 0.05, partial False and ' +
                          'expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_bic_unstable_9_ok():  # BIC_UNSTABLE, limit 0.05 --> 1
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.BIC_UNSTABLE,
                     params={'limit': 0.05, 'ref': ref})
    assert know.rules.rules == [Rule.BIC_UNSTABLE]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit == 1
    assert know.ignore == 0
    assert know.threshold == 0.05
    assert know.expertise == 1.0
    assert know.partial is False
    assert know.count == 0
    assert know.label == ('Ruleset "BIC unstable" with limit ' +
                          '1, threshold 0.05, partial False and ' +
                          'expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_bic_unstable_10_ok():  # BIC_UNSTABLE, limit 0.05 --> 1, threshold 0.1
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.BIC_UNSTABLE,
                     params={'limit': 0.05, 'ref': ref, 'threshold': 0.1})
    assert know.rules.rules == [Rule.BIC_UNSTABLE]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit == 1
    assert know.ignore == 0
    assert know.threshold == 0.1
    assert know.expertise == 1.0
    assert know.partial is False
    assert know.count == 0
    assert know.label == ('Ruleset "BIC unstable" with limit ' +
                          '1, threshold 0.1, partial False and ' +
                          'expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


# Knowledge constructor with asia

def test_bic_unstable_asia_1_ok():  # BIC_UNSTABLE Knowledge, 0.2 expertise
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.BIC_UNSTABLE,
                     params={'limit': 0.2, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.BIC_UNSTABLE]
    assert know.ref == ref
    assert know.limit == 2
    assert know.ignore == 0
    assert know.threshold == 0.05
    assert know.expertise == 1.0
    assert know.partial is False
    assert know.count == 0
    assert know.label == ('Ruleset "BIC unstable" with limit ' +
                          '2, threshold 0.05, partial False and ' +
                          'expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


# Test Knowledge.hc_best() method - cases match research log table 21/08/2023
# Here we have partial = False and expertise = 1.0

def test_hc_best_abc_1_ok(know_abc_1, abc1):  # allow add of true arc
    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    abc1[0].set_N(8)
    know_abc_1.threshold = 0.01  # lower threshold so bic_unstable rule 'fires'
    new_best, event = know_abc_1.hc_best(best=best, sf=6, data=abc1[0],
                                         parents=abc1[1])
    assert best == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_1.reqd == {('A', 'B'): (True, False)}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_2_ok(know_abc_1, abc1):  # stop add of misorientated arc
    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    abc1[0].set_N(4)
    new_best, event = know_abc_1.hc_best(best=best, sf=6, data=abc1[0],
                                         parents=abc1[1])

    # Algo trying to add B --> A, but expert correctly says should be A --> B

    assert BestDAGChanges(DAGChange(Activity.NONE, ('B', 'A'), 4.0, None),
                          best.second) == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.STOP_ADD
    assert know_abc_1.reqd == {('A', 'B'): (True, False)}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_3_ok(know_abc_1, abc1):  # stop add of extra edge
    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('A', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    abc1[0].set_N(7)
    new_best, event = know_abc_1.hc_best(best=best, sf=6, data=abc1[0],
                                         parents=abc1[1])

    # Algo trying to add A --> C but expert correctly says doesn't exist

    assert BestDAGChanges(DAGChange(Activity.NONE, ('A', 'C'), 4.0, None),
                          best.second) == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.EXT_ADD
    assert know_abc_1.reqd == {}
    assert know_abc_1.stop == {('A', 'C'): (True, False),
                               ('C', 'A'): (True, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_4_ok(know_abc_1, abc1):  # allow delete non-existing arc
    best = BestDAGChanges()
    best.top = DAGChange(Activity.DEL, ('A', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    abc1[0].set_N(7)
    new_best, event = know_abc_1.hc_best(best=best, sf=6, data=abc1[0],
                                         parents=abc1[1])

    # Algo trying to delete non-existent arc

    assert best == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_1.reqd == {}
    assert know_abc_1.stop == {('A', 'C'): (True, False),
                               ('C', 'A'): (True, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_5_ok(know_abc_1, abc1):  # delete correct arc
    best = BestDAGChanges()
    best.top = DAGChange(Activity.DEL, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('A', 'C'), 4.0, None)
    abc1[0].set_N(7)
    new_best, event = know_abc_1.hc_best(best=best, sf=6, data=abc1[0],
                                         parents=abc1[1])

    # Algo trying to delete true arc B --> C

    assert BestDAGChanges(DAGChange(Activity.NONE, ('B', 'C'), 4.0, None),
                          best.second) == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.STOP_DEL
    assert know_abc_1.reqd == {('B', 'C'): (True, False)}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_6_ok(know_abc_1, abc1):  # delete misorientated arc
    best = BestDAGChanges()
    best.top = DAGChange(Activity.DEL, ('B', 'A'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    abc1[0].set_N(7)
    new_best, event = know_abc_1.hc_best(best=best, sf=6, data=abc1[0],
                                         parents=abc1[1])

    # Algo trying to add A --> C but expert correctly says doesn't exist

    assert BestDAGChanges(DAGChange(Activity.NONE, ('B', 'A'), 4.0, None),
                          best.second) == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.STOP_DEL
    assert know_abc_1.reqd == {('A', 'B'): (True, False)}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_7_ok(know_abc_1, abc1):  # reverse correct arc
    best = BestDAGChanges()
    best.top = DAGChange(Activity.REV, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    abc1[0].set_N(7)
    new_best, event = know_abc_1.hc_best(best=best, sf=6, data=abc1[0],
                                         parents=abc1[1])

    # Algo trying to reverse A --> B

    assert BestDAGChanges(DAGChange(Activity.NONE, ('A', 'B'), 4.0, None),
                          best.second) == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.STOP_REV
    assert know_abc_1.reqd == {('A', 'B'): (True, False)}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_8_ok(know_abc_1, abc1):  # reverse misorientated arc
    best = BestDAGChanges()
    best.top = DAGChange(Activity.REV, ('B', 'A'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    abc1[0].set_N(7)
    new_best, event = know_abc_1.hc_best(best=best, sf=6, data=abc1[0],
                                         parents=abc1[1])

    # Algo trying to reverse B --> A

    assert best == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_1.reqd == {('A', 'B'): (True, False)}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_9_ok(know_abc_1, abc1):  # reverse non-existent edge
    best = BestDAGChanges()
    best.top = DAGChange(Activity.REV, ('A', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    abc1[0].set_N(7)
    new_best, event = know_abc_1.hc_best(best=best, sf=6, data=abc1[0],
                                         parents=abc1[1])

    # Algo trying to reverse B --> A

    assert BestDAGChanges(DAGChange(Activity.NONE, ('A', 'C'), 4.0, None),
                          best.second) == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.EXT_REV
    assert know_abc_1.reqd == {}
    assert know_abc_1.stop == {('A', 'C'): (True, False),
                               ('C', 'A'): (True, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_10_ok(know_abc_1, abc1):  # 10 rows not unbalanced
    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    new_best, event = know_abc_1.hc_best(best=best, sf=6, data=abc1[0],
                                         parents=abc1[1])
    assert best == new_best
    assert event is None
    assert know_abc_1.reqd == {}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


# Test Knowledge.hc_best() method - cases match research log table 21/08/2023
# Here we have partial = True and expertise = 1.0

def test_hc_best_part_1_ok(asia1):  # add of true arc

    init_stable_random()
    know = asia1['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('asia', 'tub'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('tub', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])

    # Perfect partial expert says proposed add is OK so get NO_OP

    assert best == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know.reqd == {('asia', 'tub'): (True, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


def test_hc_best_part_2_ok(asia1):  # add of misorientated arc

    init_stable_random()
    know = asia1['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('bronc', 'smoke'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('smoke', 'bronc'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])

    # Perfect partial expert says misorientated arc wrong so STOP_ADD

    assert new_best == \
        BestDAGChanges(DAGChange(Activity.NONE, ('bronc', 'smoke'), 4.0, None),
                       best.second)
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.STOP_ADD
    assert know.reqd == {('smoke', 'bronc'): (True, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


def test_hc_best_part_3_ok(asia1):  # add of non-existent arcs

    init_stable_random()
    know = asia1['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('asia', 'smoke'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('smoke', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])

    # Partial expert cannot return "non-existent" so randomly chooses
    # asia --> smoke, which is NO_OP

    assert new_best == best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know.reqd == {('asia', 'smoke'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # Partial expert cannot return "non-existent" so randomly chooses
    # lung --> tub, which is again NO_OP

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('tub', 'lung'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('lung', 'tub'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])
    assert new_best == best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know.reqd == {('asia', 'smoke'): (False, False),
                         ('tub', 'lung'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # Partial expert cannot return "non-existent" so randomly chooses
    # bronc--> either as correct which causes STOP_ADD

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('xray', 'smoke'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('smoke', 'xray'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])
    assert BestDAGChanges(DAGChange(Activity.NONE, ('xray', 'smoke'),
                                    4.0, None), best.second) == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.STOP_ADD
    assert know.reqd == {('asia', 'smoke'): (False, False),
                         ('tub', 'lung'): (False, False),
                         ('smoke', 'xray'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


def test_hc_best_part_4_ok(asia1):  # delete of true arc ignored

    init_stable_random()
    know = asia1['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.DEL, ('asia', 'tub'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('smoke', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])

    # Partial expert cannot return "non-existent" so randomly chooses
    # asia --> smoke, which is NO_OP

    assert new_best == best
    assert event is None


def test_hc_best_part_5_ok(asia1):  # delete of misorientated arc ignored

    init_stable_random()
    know = asia1['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.DEL, ('tub', 'asia'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('smoke', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])

    # Partial expert cannot return "non-existent" so randomly chooses
    # asia --> smoke, which is NO_OP

    assert new_best == best
    assert event is None


def test_hc_best_part_6_ok(asia1):  # delete of non-existent arc ignored

    init_stable_random()
    know = asia1['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.DEL, ('bronc', 'either'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('smoke', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])

    # Partial expert cannot return "non-existent" so randomly chooses
    # asia --> smoke, which is NO_OP

    assert new_best == best
    assert event is None


def test_hc_best_part_7_ok(asia1):  # reverse of true arc

    init_stable_random()
    know = asia1['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.REV, ('smoke', 'bronc'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('tub', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])

    # Perfect partial expert says proposed rev is wrong STOP_REV

    assert BestDAGChanges(DAGChange(Activity.NONE, ('smoke', 'bronc'),
                                    4.0, None), best.second) == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.STOP_REV
    assert know.reqd == {('smoke', 'bronc'): (True, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


def test_hc_best_part_8_ok(asia1):  # reverse of misorientated arc

    init_stable_random()
    know = asia1['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.REV, ('bronc', 'smoke'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('tub', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])

    # Perfect partial expert says proposed rev is OK so get NO_OP

    assert best == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know.reqd == {('smoke', 'bronc'): (True, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


def test_hc_best_part_9_ok(asia1):  # reverse of non-existent arcs

    init_stable_random()
    know = asia1['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.REV, ('asia', 'smoke'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('tub', 'bronc'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])

    # Partial expert cannot return "non-existent" so randomly chooses
    # asia --> smoke, which is STOP_REV

    assert BestDAGChanges(DAGChange(Activity.NONE, ('asia', 'smoke'),
                                    4.0, None), best.second) == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.STOP_REV
    assert know.reqd == {('asia', 'smoke'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # Partial expert cannot return "non-existent" so randomly chooses
    # tub --> lung, which is again STOP_REV

    best = BestDAGChanges()
    best.top = DAGChange(Activity.REV, ('tub', 'lung'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('tub', 'bronc'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])
    assert BestDAGChanges(DAGChange(Activity.NONE, ('tub', 'lung'),
                                    4.0, None), best.second) == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.STOP_REV
    assert know.reqd == {('asia', 'smoke'): (False, False),
                         ('tub', 'lung'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # Partial expert cannot return "non-existent" so randomly chooses
    # smoke --> xray as correct which causes NO_OP

    best = BestDAGChanges()
    best.top = DAGChange(Activity.REV, ('xray', 'smoke'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('tub', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'], asia1['parents'])
    assert best == new_best
    assert event.rule == Rule.BIC_UNSTABLE
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know.reqd == {('asia', 'smoke'): (False, False),
                         ('tub', 'lung'): (False, False),
                         ('smoke', 'xray'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))
