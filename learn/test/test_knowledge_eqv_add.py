
# Test the Knowledge class

import pytest

from learn.knowledge import Knowledge, Rule, RuleSet, \
    KnowledgeOutcome
from learn.trace import Activity
from learn.dagchange import DAGChange, BestDAGChanges
from fileio.common import TESTDATA_DIR
from core.bn import BN
from core.common import init_stable_random


@pytest.fixture
def know_abc_1():  # rule with limit of one
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    return Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'ref': ref, 'limit': 1})


@pytest.fixture
def know_abc_2():  # rule with limit of two, ignore first
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    return Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'ref': ref, 'limit': 2, 'ignore': 1})


@pytest.fixture
def know_abc_3():  # rule with limit of 3, expertise 0.5
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    return Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'ref': ref, 'limit': 3, 'expertise': 0.5})


@pytest.fixture
def know_abc_4():  # rule with limit of 1, ignore 1, expertise 0.8
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    return Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'ref': ref, 'limit': 1, 'ignore': 1,
                             'expertise': 0.8})


@pytest.fixture
def know_abc_5():  # rule with limit of 3, expertise 0.5
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    return Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'ref': ref, 'limit': 3, 'expertise': 0.5,
                             'partial': True})


@pytest.fixture
def asia1():  # Asia network with perfect partial expert
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = ref.generate_cases(10)
    parents = {n: set() for n in ref.dag.nodes}
    know = Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'ref': ref, 'partial': True})
    return {'data': data, 'parents': parents, 'know': know}


@pytest.fixture
def asia2():  # Asia network with imperfect partial expert
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = ref.generate_cases(10)
    parents = {n: set() for n in ref.dag.nodes}
    know = Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'ref': ref, 'partial': True,
                             'expertise': 0.5})
    return {'data': data, 'parents': parents, 'know': know}


@pytest.fixture
def abc1():  # data and parents for A->B->C graph
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    parents = {'A': set(), 'B': {'A'}, 'C': {'B'}}
    return (ref.generate_cases(10), parents)


@pytest.fixture
def ab():  # return ab DAG
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    return ref.dag


# Test the Knowledge constructor

def test_knowledge_type_error_1():  # no args
    with pytest.raises(TypeError):
        Knowledge()


def test_knowledge_type_error_2():  # unknown argument supplied
    with pytest.raises(TypeError):
        Knowledge(unknown=23)


def test_knowledge_type_error_3():  # ruleset bad type
    with pytest.raises(TypeError):
        Knowledge(rules=23)
    with pytest.raises(TypeError):
        Knowledge(rules='not a ruleset')
    with pytest.raises(TypeError):
        Knowledge(rules=Rule.EQUIV_ADD)
    with pytest.raises(TypeError):
        Knowledge(rules=[Rule.EQUIV_ADD])


def test_knowledge_type_error_4():  # params bad type
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params=3)


def test_knowledge_type_error_5():  # rules must be set if params not None
    with pytest.raises(TypeError):
        Knowledge(params={'limit': 3})


def test_knowledge_type_error_6():  # limit must be an int, float or None
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'limit': 'a'})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'limit': ref})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'limit': [2]})


def test_knowledge_type_error_7():  # ref must be a BN
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'ref': 'a'})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'ref': -1})


def test_knowledge_type_error_8():  # ignore must be an int
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'ignore': 'a'})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'ignore': -17.3})


def test_knowledge_type_error_9():  # expertise must be a float
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'expertise': 'a'})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'expertise': 0})


def test_knowledge_type_error_21():  # partial must be a bool
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'partial': 'a'})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'partial': 0})


def test_knowledge_type_error_22():  # earlyok must be a bool
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'earlyok': 'a'})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'earlyok': 0})


def test_knowledge_value_error_1():  # unknown parameters
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'unknown': 3})


def test_knowledge_value_error_2():  # non-positive limit
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'limit': 0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'limit': -1})


def test_knowledge_value_error_3():  # limit is True
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'limit': True})


def test_knowledge_value_error_4():  # limit is float, ref not specified
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'limit': 0.2})


def test_knowledge_value_error_5():  # float limit not between 0 and 1
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'limit': 0.0, 'ref': ref})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'limit': 1.0, 'ref': ref})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'limit': -0.1, 'ref': ref})


def test_knowledge_value_error_6():  # non-positive ignore
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'ignore': 0})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'ignore': -1})


def test_knowledge_value_error_7():  # out of range expertise
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'expertise': -0.01})
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'expertise': 1.01})


def test_knowledge_value_error_8():  # EQUIV_ADD needs ref parameter
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD)
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'limit': 4})


def test_knowledge_value_error_9():  # sample not allowed for equiv_add
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'ref': ref, 'sample': 2})


def test_knowledge_value_error_10():  # threshold not allowed for equiv_add
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'ref': ref,
                                                   'threshold': 0.1})


def test_knowledge_value_error_11():  # earlyok not allowed without expertise
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_ADD, params={'ref': ref,
                                                   'earlyok': True})


def test_knowledge_ok_1():  # EQUIV_ADD ruleset, ref param
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    knowledge = Knowledge(rules=RuleSet.EQUIV_ADD, params={'ref': ref})
    assert knowledge.rules.rules == [Rule.EQUIV_ADD]
    assert knowledge.ref == ref
    assert knowledge.limit is False
    assert knowledge.ignore == 0
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == ('Ruleset "Choose equivalent add" with limit ' +
                               'False, ignore 0, partial False and expertise' +
                               ' 1.0')
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
    assert knowledge.event is None
    assert knowledge.event_delta is None
    assert knowledge.initial is None


def test_knowledge_ok_2():  # EQUIV_ADD ruleset, ref & limit params
    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    knowledge = Knowledge(rules=RuleSet.EQUIV_ADD,
                          params={'ref': ref, 'limit': 10})
    assert knowledge.rules.rules == [Rule.EQUIV_ADD]
    assert knowledge.ref == ref
    assert knowledge.limit == 10
    assert knowledge.ignore == 0
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == ('Ruleset "Choose equivalent add" with limit ' +
                               '10, ignore 0, partial False and expertise' +
                               ' 1.0')
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
    assert knowledge.event is None
    assert knowledge.event_delta is None
    assert knowledge.initial is None


def test_knowledge_ok_3(know_abc_1):
    assert know_abc_1.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know_abc_1.ref, BN)
    assert know_abc_1.ref.dag.to_string() == '[A][B|A][C|B]'
    assert know_abc_1.limit == 1
    assert know_abc_1.ignore == 0
    assert know_abc_1.expertise == 1.0
    assert know_abc_1.partial is False
    assert know_abc_1.count == 0
    assert know_abc_1.label == ('Ruleset "Choose equivalent add" with limit ' +
                                '1, ignore 0, partial False and expertise' +
                                ' 1.0')
    assert know_abc_1.stop == {}
    assert know_abc_1.reqd == {}
    assert know_abc_1.event is None
    assert know_abc_1.event_delta is None
    assert know_abc_1.initial is None


def test_knowledge_ok_4(know_abc_2):
    assert know_abc_2.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know_abc_2.ref, BN)
    assert know_abc_2.ref.dag.to_string() == '[A][B|A][C|B]'
    assert know_abc_2.limit == 2
    assert know_abc_2.ignore == 1
    assert know_abc_2.expertise == 1.0
    assert know_abc_2.partial is False
    assert know_abc_2.count == 0
    assert know_abc_2.label == ('Ruleset "Choose equivalent add" with limit ' +
                                '2, ignore 1, partial False and expertise' +
                                ' 1.0')
    assert know_abc_2.stop == {}
    assert know_abc_2.reqd == {}
    assert know_abc_2.event is None
    assert know_abc_2.event_delta is None
    assert know_abc_2.initial is None


def test_knowledge_ok_5(know_abc_3):
    assert know_abc_3.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know_abc_3.ref, BN)
    assert know_abc_3.ref.dag.to_string() == '[A][B|A][C|B]'
    assert know_abc_3.limit == 3
    assert know_abc_3.ignore == 0
    assert know_abc_3.expertise == 0.5
    assert know_abc_3.partial is False
    assert know_abc_3.count == 0
    assert know_abc_3.label == ('Ruleset "Choose equivalent add" with limit ' +
                                '3, ignore 0, partial False and expertise' +
                                ' 0.5')
    assert know_abc_3.stop == {}
    assert know_abc_3.reqd == {}
    assert know_abc_3.event is None
    assert know_abc_3.event_delta is None
    assert know_abc_3.initial is None


def test_knowledge_ok_6(know_abc_4):
    assert know_abc_4.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know_abc_4.ref, BN)
    assert know_abc_4.ref.dag.to_string() == '[A][B|A][C|B]'
    assert know_abc_4.limit == 1
    assert know_abc_4.ignore == 1
    assert know_abc_4.expertise == 0.8
    assert know_abc_4.partial is False
    assert know_abc_4.count == 0
    assert know_abc_4.label == ('Ruleset "Choose equivalent add" with limit ' +
                                '1, ignore 1, partial False and expertise' +
                                ' 0.8')
    assert know_abc_4.stop == {}
    assert know_abc_4.reqd == {}
    assert know_abc_4.event is None
    assert know_abc_4.event_delta is None
    assert know_abc_4.initial is None


# Knowledge constructor with Cancer

def test_knowledge_ok_7():  # EQUIV_ADD, limit 0.2 --> 1
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'limit': 0.2, 'ref': ref})
    assert know.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit == 1
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.partial is False
    assert know.count == 0
    assert know.label == ('Ruleset "Choose equivalent add" with limit ' +
                          '1, ignore 0, partial False and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_knowledge_ok_8():  # EQUIV_ADD, limit 0.5 --> 2
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'limit': 0.5, 'ref': ref})
    assert know.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit == 2
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.partial is False
    assert know.count == 0
    assert know.label == ('Ruleset "Choose equivalent add" with limit ' +
                          '2, ignore 0, partial False and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_knowledge_ok_9():  # EQUIV_ADD, limit 0.05 --> 1
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'limit': 0.05, 'ref': ref})
    assert know.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit == 1
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.partial is False
    assert know.count == 0
    assert know.label == ('Ruleset "Choose equivalent add" with limit ' +
                          '1, ignore 0, partial False and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_knowledge_ok_10():  # EQUIV_ADD, limit 0.05 --> 1, partial T
    ref = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    know = Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'limit': 0.05, 'ref': ref, 'partial': True})
    assert know.rules.rules == [Rule.EQUIV_ADD]
    assert isinstance(know.ref, BN)
    assert (know.ref.dag.to_string() == '[Cancer|Pollution:Smoker]' +
            '[Dyspnoea|Cancer][Pollution][Smoker][Xray|Cancer]')
    assert know.limit == 1
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.partial is True
    assert know.count == 0
    assert know.label == ('Ruleset "Choose equivalent add" with limit ' +
                          '1, ignore 0, partial True and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


# Knowledge constructor with asia

def test_knowledge_asia_1_ok():  # EQUIV_ADD Knowledge, 0.2 expertise
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'limit': 0.2, 'ref': ref, 'expertise': 1.0})
    assert know.rules.rules == [Rule.EQUIV_ADD]
    assert know.ref == ref
    assert know.limit == 2
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.partial is False
    assert know.count == 0
    assert know.label == ('Ruleset "Choose equivalent add" with limit ' +
                          '2, ignore 0, partial False and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


def test_knowledge_asia_2_ok():  # EQUIV_ADD Knowledge, 0.2 expertise
    ref = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    know = Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'limit': 0.5, 'ref': ref, 'expertise': 0.8,
                             'earlyok': True})
    assert know.rules.rules == [Rule.EQUIV_ADD]
    assert know.ref == ref
    assert know.limit == 4
    assert know.ignore == 0
    assert know.expertise == 0.8
    assert know.earlyok is True
    assert know.partial is False
    assert know.count == 0
    assert know.label == ('Ruleset "Choose equivalent add" with limit ' +
                          '4, ignore 0, partial False and expertise 0.8')
    assert know.stop == {}
    assert know.reqd == {}
    assert know.event is None
    assert know.event_delta is None
    assert know.initial is None


# Test Knowledge.hc_best() method

def test_hc_best_abc_1_ok(know_abc_1, abc1):  # arcs not rev of each other
    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    new_best, event = know_abc_1.hc_best(best, 6, abc1[0], abc1[1])
    assert best == new_best
    assert event is None
    assert know_abc_1.reqd == {}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_2_ok(know_abc_1, abc1):  # scores not same - no swop
    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 5.0, None)
    new_best, event = know_abc_1.hc_best(best, 6, abc1[0], abc1[1])
    assert best == new_best
    assert event is None
    assert know_abc_1.reqd == {}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_3_ok(know_abc_1, abc1):  # not both adds - no swop

    init_stable_random()

    best = BestDAGChanges()
    best.top = DAGChange(Activity.DEL, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.REV, ('B', 'A'), 4.0, None)
    new_best, event = know_abc_1.hc_best(best, 6, abc1[0], abc1[1])
    assert best == new_best
    assert event is None
    assert know_abc_1.reqd == {}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_4_ok(know_abc_1, abc1):  # equiv_add but best OK => no_op

    init_stable_random()

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    new_best, event = know_abc_1.hc_best(best, 6, abc1[0], abc1[1])
    assert best == new_best
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_1.reqd == {('A', 'B'): (True, False)}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_5_ok(know_abc_1, abc1):  # equiv_add, 2nd correct => swop

    init_stable_random()

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    new_best, event = know_abc_1.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best.top == best.second
    assert new_best.second == best.top
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know_abc_1.reqd == {('A', 'B'): (True, False)}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_6_ok(know_abc_1, abc1):  # limit exceeded

    init_stable_random()

    # first request doesn't do anything since change was already correct

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0, None))
    new_best, event = know_abc_1.hc_best(best, 6, abc1[0], abc1[1])
    assert best == new_best
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_1.reqd == {('A', 'B'): (True, False)}
    print('Required arcs: {}'.format(know_abc_1.reqd))

    # second change would have resulted in a swap, but request limit reached

    best = BestDAGChanges(DAGChange(Activity.ADD, ('C', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_1.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_1.reqd == {('A', 'B'): (True, False)}
    assert know_abc_1.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_1.reqd, know_abc_1.stop))


def test_hc_best_abc_7_ok(know_abc_2, abc1):  # first request ignored

    init_stable_random()

    # first request ignored because of ignore = 1

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'A'), 4.0, None),
                          DAGChange(Activity.ADD, ('A', 'B'), 4.0, None))
    new_best, event = know_abc_2.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_2.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_2.reqd == {}
    assert know_abc_2.reqd == {}
    assert know_abc_2.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_2.reqd, know_abc_2.stop))

    # second change results in a swap

    best = BestDAGChanges(DAGChange(Activity.ADD, ('C', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_2.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know_abc_2.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know_abc_2.reqd == {('B', 'C'): (True, False)}
    assert know_abc_2.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_2.reqd, know_abc_2.stop))

    # third request accepted, but results in no change

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0, None))
    new_best, event = know_abc_2.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_2.count == 3
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_2.reqd == {('B', 'C'): (True, False),
                               ('A', 'B'): (True, False)}
    assert know_abc_2.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_2.reqd, know_abc_2.stop))

    # fourth request ignored as limit reached

    best = BestDAGChanges(DAGChange(Activity.ADD, ('C', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_2.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_2.count == 4
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_2.reqd == {('B', 'C'): (True, False),
                               ('A', 'B'): (True, False)}
    assert know_abc_2.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_2.reqd, know_abc_2.stop))


def test_hc_best_abc_8_ok(know_abc_3, abc1):  # limit 3, expertise 0.5

    init_stable_random()

    # first request erroneously says no arc, which leads to EXT_ADD

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    best.top.activity = Activity.NONE
    assert new_best == best
    assert know_abc_3.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.EXT_ADD
    assert know_abc_3.reqd == {}
    assert know_abc_3.stop == {('A', 'B'): (False, False),
                               ('B', 'A'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # second request correctly makes a swap

    best = BestDAGChanges(DAGChange(Activity.ADD, ('C', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know_abc_3.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know_abc_3.reqd == {('B', 'C'): (True, False)}
    assert know_abc_3.stop == {('A', 'B'): (False, False),
                               ('B', 'A'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # third request accepted, already incorrectly think A->B does not exist

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    best.top.activity = Activity.NONE
    assert best == new_best
    assert know_abc_3.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.EXT_ADD
    assert know_abc_3.reqd == {('B', 'C'): (True, False)}
    assert know_abc_3.stop == {('A', 'B'): (False, False),
                               ('B', 'A'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))


def test_hc_best_abc_9_ok(know_abc_3, abc1):  # limit 3, expertise 0.5

    init_stable_random()

    # first request erroneously says no arc, which leads to NO_OP

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    best.top.activity = Activity.NONE
    assert new_best == best
    assert know_abc_3.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.EXT_ADD
    assert know_abc_3.reqd == {}
    assert know_abc_3.stop == {('A', 'B'): (False, False),
                               ('B', 'A'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # second request correctly makes a swap

    best = BestDAGChanges(DAGChange(Activity.ADD, ('C', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know_abc_3.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know_abc_3.reqd == {('B', 'C'): (True, False)}
    assert know_abc_3.stop == {('A', 'B'): (False, False),
                               ('B', 'A'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # third change not equiv_add so ignored

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 2.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.top, best.second)
    assert know_abc_3.count == 2
    assert event is None
    assert know_abc_3.reqd == {('B', 'C'): (True, False)}
    assert know_abc_3.stop == {('A', 'B'): (False, False),
                               ('B', 'A'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))


def test_hc_best_abc_10_ok(know_abc_3, abc1):  # limit 3, expertise 0.5

    init_stable_random()

    # first change doesn't trigger equiv_add rule

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_3.count == 0
    assert event is None
    assert know_abc_3.reqd == {}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # first request erroneously says no arc, causing EXT_ADD

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'B'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    best.top.activity = Activity.NONE
    assert new_best == best
    assert know_abc_3.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.EXT_ADD
    assert know_abc_3.reqd == {}
    assert know_abc_3.stop == {('B', 'C'): (False, False),
                               ('C', 'B'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # third change doesn't trigger equiv_add rule

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0003, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_3.count == 1
    assert event is None
    assert know_abc_3.reqd == {}
    assert know_abc_3.stop == {('C', 'B'): (False, False),
                               ('B', 'C'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # second request accepted, knowledge cache says C->B does not exist

    best = BestDAGChanges(DAGChange(Activity.ADD, ('C', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    best.top.activity = Activity.NONE
    assert new_best == best
    assert know_abc_3.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.EXT_ADD
    assert know_abc_3.reqd == {}
    assert know_abc_3.stop == {('C', 'B'): (False, False),
                               ('B', 'C'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))


def test_hc_best_abc_11_ok(know_abc_3, abc1):  # limit 3, expertise 0.5

    init_stable_random()

    # first change doesn't trigger equiv_add rule

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_3.count == 0
    assert event is None
    assert know_abc_3.reqd == {}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # first request erroneously says no arc, causing EXT_ADD

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'B'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    best.top.activity = Activity.NONE
    assert new_best == best
    assert know_abc_3.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.EXT_ADD
    assert know_abc_3.reqd == {}
    assert know_abc_3.stop == {('C', 'B'): (False, False),
                               ('B', 'C'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # third change doesn't trigger equiv_add rule

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0003, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_3.count == 1
    assert event is None
    assert know_abc_3.reqd == {}
    assert know_abc_3.stop == {('C', 'B'): (False, False),
                               ('B', 'C'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # second request accepted, and correctly makes swap

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'A'), 4.0, None),
                          DAGChange(Activity.ADD, ('A', 'B'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know_abc_3.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know_abc_3.reqd == {('A', 'B'): (True, False)}
    assert know_abc_3.stop == {('C', 'B'): (False, False),
                               ('B', 'C'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # third request accepted, cache correctly says A->B exists, so swap OK

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'A'), 4.0, None),
                          DAGChange(Activity.ADD, ('A', 'B'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know_abc_3.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know_abc_3.reqd == {('A', 'B'): (True, False)}
    assert know_abc_3.stop == {('C', 'B'): (False, False),
                               ('B', 'C'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))


def test_hc_best_abc_12_ok(know_abc_4, abc1):  # lim 1, igno 1, expertise 0.8

    init_stable_random()

    # first request ignored because of ignore param

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0, None))
    new_best, event = know_abc_4.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_4.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_4.reqd == {}
    assert know_abc_4.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_4.reqd, know_abc_4.stop))

    # second request correctly makes no swap

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'B'), 4.0, None))
    new_best, event = know_abc_4.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_4.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_4.reqd == {('B', 'C'): (True, False)}
    assert know_abc_4.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_4.reqd, know_abc_4.stop))

    # third request ignored as limit reached

    best = BestDAGChanges(DAGChange(Activity.ADD, ('C', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_4.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_4.count == 3
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_4.reqd == {('B', 'C'): (True, False)}
    assert know_abc_4.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_4.reqd, know_abc_4.stop))


def test_hc_best_abc_13_ok(know_abc_3, abc1):  # limit 3, expert 0.5, offset 2

    init_stable_random(offset=2)

    # first change doesn't trigger equiv_add rule

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best is best
    assert know_abc_3.count == 0
    assert event is None
    assert know_abc_3.reqd == {}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # first request correctly makes no swap

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'B'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_3.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_3.reqd == {('B', 'C'): (True, False)}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # third change doesn't trigger equiv_add rule

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0003, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_3.count == 1
    assert event is None
    assert know_abc_3.reqd == {('B', 'C'): (True, False)}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # second request accepted, and correctly does swap

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'A'), 4.0, None),
                          DAGChange(Activity.ADD, ('A', 'B'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know_abc_3.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know_abc_3.reqd == {('B', 'C'): (True, False),
                               ('A', 'B'): (True, False)}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # third request accepted, cache correctly says A->B exists, NO_OP

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_3.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_3.reqd == {('B', 'C'): (True, False),
                               ('A', 'B'): (True, False)}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))


def test_hc_best_abc_14_ok(know_abc_3, abc1):  # lim 3, expert 0.5, offset 2

    init_stable_random(offset=2)

    # first change doesn't trigger equiv_add rule

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best is best
    assert know_abc_3.count == 0
    assert event is None
    assert know_abc_3.reqd == {}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # first request correctly makes no swap

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'B'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_3.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_3.reqd == {('B', 'C'): (True, False)}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # third change doesn't trigger equiv_add rule

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0003, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_3.count == 1
    assert event is None
    assert know_abc_3.reqd == {('B', 'C'): (True, False)}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # second request accepted, and correctly makes swap_best

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'A'), 4.0, None),
                          DAGChange(Activity.ADD, ('A', 'B'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know_abc_3.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know_abc_3.reqd == {('A', 'B'): (True, False),
                               ('B', 'C'): (True, False)}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))

    # third request accepted, incorrectly says unknown edge correct

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'A'), 4.0, None))
    new_best, event = know_abc_3.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_3.count == 3
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_3.reqd == {('A', 'B'): (True, False),
                               ('B', 'C'): (True, False),
                               ('A', 'C'): (False, False)}
    assert know_abc_3.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_3.reqd, know_abc_3.stop))


def test_hc_best_abc_15_ok(know_abc_5, abc1):  # lim 3, expert 0.5, offset 2

    init_stable_random(offset=2)

    # first change doesn't trigger equiv_add rule

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'C'), 4.0, None))
    new_best, event = know_abc_5.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best is best
    assert know_abc_5.count == 0
    assert event is None
    assert know_abc_5.reqd == {}
    assert know_abc_5.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_5.reqd, know_abc_5.stop))

    # first request correctly makes no swap

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'B'), 4.0, None))
    new_best, event = know_abc_5.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_5.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_5.reqd == {('B', 'C'): (True, False)}
    assert know_abc_5.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_5.reqd, know_abc_5.stop))

    # third change doesn't trigger equiv_add rule

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'B'), 4.0, None),
                          DAGChange(Activity.ADD, ('B', 'A'), 4.0003, None))
    new_best, event = know_abc_5.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_5.count == 1
    assert event is None
    assert know_abc_5.reqd == {('B', 'C'): (True, False)}
    assert know_abc_5.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_5.reqd, know_abc_5.stop))

    # second request accepted, and correctly makes swap_best

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'A'), 4.0, None),
                          DAGChange(Activity.ADD, ('A', 'B'), 4.0, None))
    new_best, event = know_abc_5.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know_abc_5.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know_abc_5.reqd == {('A', 'B'): (True, False),
                               ('B', 'C'): (True, False)}
    assert know_abc_5.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_5.reqd, know_abc_5.stop))

    # third request accepted, reference graph indicates no arc, but because
    # partial is true we just return arc that algorithm was going to add

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'A'), 4.0, None))
    new_best, event = know_abc_5.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know_abc_5.count == 3
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know_abc_5.reqd == {('A', 'B'): (True, False),
                               ('B', 'C'): (True, False),
                               ('A', 'C'): (False, False)}
    assert know_abc_5.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know_abc_5.reqd, know_abc_5.stop))


def test_hc_best_abc_16_ok(abc1):  # lim 4, expert 0.5, get FTF sequence

    init_stable_random()

    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    know = Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'ref': ref, 'limit': 3, 'expertise': 0.5,
                             'partial': False})

    # first request incorrectly does EXT_ADD

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'B'), 4.0, None))
    new_best, event = know.hc_best(best, 6, abc1[0], abc1[1])
    best.top.activity = Activity.NONE
    assert new_best == best
    assert know.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.EXT_ADD
    assert know.reqd == {}
    assert know.stop == {('B', 'C'): (False, False),
                         ('C', 'B'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # second request correctly swaps B --> A

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'A'), 4.0, None),
                          DAGChange(Activity.ADD, ('A', 'B'), 4.0, None))
    new_best, event = know.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know.reqd == {('A', 'B'): (True, False)}
    assert know.stop == {('B', 'C'): (False, False),
                         ('C', 'B'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # third request incorrectly does a swap best

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'A'), 4.0, None))
    new_best, event = know.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know.count == 3
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know.reqd == {('A', 'B'): (True, False),
                         ('C', 'A'): (False, False)}
    assert know.stop == {('B', 'C'): (False, False),
                         ('C', 'B'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


def test_hc_best_abc_17_ok(abc1):  # lim 4, expert 0.5, earlyok=True

    init_stable_random()

    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    know = Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'ref': ref, 'limit': 3, 'expertise': 0.5,
                             'earlyok': True})

    # first request correctly does no_op

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'B'), 4.0, None))
    new_best, event = know.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know.reqd == {('B', 'C'): (True, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # second request correctly swaps B --> A

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'A'), 4.0, None),
                          DAGChange(Activity.ADD, ('A', 'B'), 4.0, None))
    new_best, event = know.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know.reqd == {('A', 'B'): (True, False),
                         ('B', 'C'): (True, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # third request incorrectly does a no_op

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'A'), 4.0, None))
    new_best, event = know.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == best
    assert know.count == 3
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know.reqd == {('A', 'B'): (True, False),
                         ('B', 'C'): (True, False),
                         ('A', 'C'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


def test_hc_best_abc_18_ok(abc1):  # lim 4, expert 0.5, get FTF sequence

    init_stable_random()

    ref = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    know = Knowledge(rules=RuleSet.EQUIV_ADD,
                     params={'ref': ref, 'limit': 3, 'expertise': 0.5,
                             'partial': False})

    # first request incorrectly does EXT_ADD

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'B'), 4.0, None))
    new_best, event = know.hc_best(best, 6, abc1[0], abc1[1])
    best.top.activity = Activity.NONE
    assert new_best == best
    assert know.count == 1
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.EXT_ADD
    assert know.reqd == {}
    assert know.stop == {('B', 'C'): (False, False),
                         ('C', 'B'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # second request correctly swaps B --> A

    best = BestDAGChanges(DAGChange(Activity.ADD, ('B', 'A'), 4.0, None),
                          DAGChange(Activity.ADD, ('A', 'B'), 4.0, None))
    new_best, event = know.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know.count == 2
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know.reqd == {('A', 'B'): (True, False)}
    assert know.stop == {('B', 'C'): (False, False),
                         ('C', 'B'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # third request incorrectly does a swap best

    best = BestDAGChanges(DAGChange(Activity.ADD, ('A', 'C'), 4.0, None),
                          DAGChange(Activity.ADD, ('C', 'A'), 4.0, None))
    new_best, event = know.hc_best(best, 6, abc1[0], abc1[1])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert know.count == 3
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know.reqd == {('A', 'B'): (True, False),
                         ('C', 'A'): (False, False)}
    assert know.stop == {('B', 'C'): (False, False),
                         ('C', 'B'): (False, False)}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


# Tests with a partial perfect expert

def test_hc_best_part_1_ok(asia1):  # add of true arc

    init_stable_random()
    know = asia1['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('asia', 'tub'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('tub', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'],
                                   asia1['parents'])

    # Perfect partial expert says proposed add is OK so get NO_OP

    assert best == new_best
    assert event.rule == Rule.EQUIV_ADD
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
    best.top = DAGChange(Activity.ADD, ('tub', 'asia'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('asia', 'tub'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'],
                                   asia1['parents'])

    # Partial expert swaps misoriented add so get SWAP_BEST

    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know.reqd == {('asia', 'tub'): (True, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


def test_hc_best_part_3_ok(asia1):  # add of non-existent arcs

    init_stable_random()
    know = asia1['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('asia', 'smoke'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('smoke', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'],
                                   asia1['parents'])

    # Partial expert cannot return "non-existent" so randomly chooses
    # asia --> smoke, which is NO_OP

    assert new_best == best
    assert event.rule == Rule.EQUIV_ADD
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
    new_best, event = know.hc_best(best, 6, asia1['data'],
                                   asia1['parents'])
    assert new_best == best
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know.reqd == {('asia', 'smoke'): (False, False),
                         ('tub', 'lung'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # Partial expert cannot return "non-existent" so randomly chooses
    # bronc--> either, which is SWAP_BEST this time

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('either', 'bronc'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('bronc', 'either'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia1['data'],
                                   asia1['parents'])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know.reqd == {('asia', 'smoke'): (False, False),
                         ('tub', 'lung'): (False, False),
                         ('bronc', 'either'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


# Tests with a partial imperfect expert

def test_hc_best_part_4_ok(asia2):  # add of true arc

    init_stable_random()
    know = asia2['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('asia', 'tub'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('tub', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia2['data'],
                                   asia2['parents'])

    # Imerfect partial expert incorrectly says proposed add is wrong so
    # get SWAP_BEST

    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know.reqd == {('tub', 'asia'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


def test_hc_best_part_5_ok(asia2):  # add of misorientated arc

    init_stable_random()
    know = asia2['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('tub', 'asia'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('asia', 'tub'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia2['data'],
                                   asia2['parents'])

    # Imperfect partial expert incorrectly says misorientated arc is
    # correct so get NO_OP

    assert new_best == best
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know.reqd == {('tub', 'asia'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))


def test_hc_best_part_6_ok(asia2):  # add of non-existent arcs

    init_stable_random()
    know = asia2['know']

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('asia', 'smoke'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('smoke', 'asia'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia2['data'],
                                   asia2['parents'])

    # Partial expert cannot return "non-existent" so randomly chooses
    # asia --> smoke, which is NO_OP

    assert new_best == best
    assert event.rule == Rule.EQUIV_ADD
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
    new_best, event = know.hc_best(best, 6, asia2['data'],
                                   asia2['parents'])
    assert new_best == best
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert know.reqd == {('asia', 'smoke'): (False, False),
                         ('tub', 'lung'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))

    # Partial expert cannot return "non-existent" so randomly chooses
    # bronc--> either, which is SWAP_BEST this time

    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('either', 'bronc'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('bronc', 'either'), 4.0, None)
    new_best, event = know.hc_best(best, 6, asia2['data'],
                                   asia2['parents'])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert know.reqd == {('asia', 'smoke'): (False, False),
                         ('tub', 'lung'): (False, False),
                         ('bronc', 'either'): (False, False)}
    assert know.stop == {}
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(know.reqd, know.stop))
