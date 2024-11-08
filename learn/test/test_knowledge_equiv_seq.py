
# Test the EQUIV_SEQ Knowledge rule

import pytest

from learn.knowledge import Knowledge, Rule, RuleSet, \
    KnowledgeOutcome
from learn.trace import Activity
from learn.dagchange import DAGChange, BestDAGChanges
from fileio.common import TESTDATA_DIR
from core.bn import BN


@pytest.fixture
def abc():  # BN, data and parents for A->B->C graph
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    parents = {'A': set(), 'B': {'A'}, 'C': {'B'}}
    return {'bn': bn, 'pa': parents, 'da': bn.generate_cases(10)}


# Test the Knowledge constructor

def test_equiv_seq_type_error_1():  # sequence parameter not tuple
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={'sequence': 3})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={'sequence': None})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={'sequence': True})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={'sequence': [True]})


def test_equiv_seq_type_error_2():  # sequence parameter has no elements
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={'sequence': tuple()})


def test_equiv_seq_type_error_3():  # sequence parameter not tuple of bools
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ, params={'sequence': tuple([1])})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ,
                  params={'sequence': tuple([1, True])})


def test_equiv_seq_type_error_4():  # pause parameter is not a bool
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ,
                  params={'sequence': tuple([True]), 'pause': 1})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ,
                  params={'sequence': tuple([True]), 'pause': [False]})
    with pytest.raises(TypeError):
        Knowledge(rules=RuleSet.EQUIV_SEQ,
                  params={'sequence': tuple([True]), 'pause': None})


def test_equiv_seq_value_error_1():  # no sequence parameter specified
    with pytest.raises(ValueError):
        Knowledge(rules=RuleSet.EQUIV_SEQ)


def test_equiv_seq_1_ok():  # EQUIV_SEQ ruleset, 1 element sequence
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': tuple([True])})
    assert knowledge.rules.rules == [Rule.EQUIV_SEQ]
    assert knowledge.sequence == (True, )
    assert knowledge.pause is False
    assert knowledge.limit is False
    assert knowledge.ignore == 0
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == ('Ruleset "Equivalent add sequence" ' +
                               'with sequence of length 1 then no pause')
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
    assert knowledge.event is None
    assert knowledge.event_delta is None
    assert knowledge.initial is None


def test_equiv_seq_2_ok():  # EQUIV_SEQ ruleset, 2 element sequence
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (True, False), 'pause': True})
    assert knowledge.rules.rules == [Rule.EQUIV_SEQ]
    assert knowledge.sequence == (True, False)
    assert knowledge.pause is True
    assert knowledge.limit is False
    assert knowledge.ignore == 0
    assert knowledge.expertise == 1.0
    assert knowledge.partial is False
    assert knowledge.count == 0
    assert knowledge.label == ('Ruleset "Equivalent add sequence" ' +
                               'with sequence of length 2 then pause')
    assert knowledge.stop == {}
    assert knowledge.reqd == {}
    assert knowledge.event is None
    assert knowledge.event_delta is None
    assert knowledge.initial is None


# Test Knowledge.hc_best() method

def test_hc_best_abc_1_ok(abc):  # doesn't trigger EQUIV_SEQ: not opposites
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (True,)})
    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert best == new_best
    assert event is None
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 0
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(knowledge.reqd, knowledge.stop))


def test_hc_best_abc_2_ok(abc):  # doesn't trigger EQUIV_SEQ: diff scores
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (True,)})
    best = BestDAGChanges()
    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.01, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert best == new_best
    assert event is None
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 0
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(knowledge.reqd, knowledge.stop))


def test_hc_best_abc_3_ok(abc):  # doesn't trigger EQUIV_SEQ: not adds
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (True,)})
    best = BestDAGChanges()
    best.top = DAGChange(Activity.REV, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.DEL, ('B', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert best == new_best
    assert event is None
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 0
    print('\nRequired arcs: {}\nprohibited arcs: {}'
          .format(knowledge.reqd, knowledge.stop))


def test_hc_best_abc_4_ok(abc):  # EQUIV_SEQ: (False,)
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (False,)})
    best = BestDAGChanges()

    # Sequence element #1 is False so get NO_OP event

    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('A', 'B')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for first change: {}'.format(event))

    # Beyond sequence so no event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for second change: {}'.format(event))


def test_hc_best_abc_5_ok(abc):  # EQUIV_SEQ: (True,)
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (True,)})
    best = BestDAGChanges()

    # Sequence element #1 is False so get NO_OP event

    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert event.arc == ('A', 'B')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for first change: {}'.format(event))

    # Beyond sequence so no event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for second change: {}'.format(event))


def test_hc_best_abc_6_ok(abc):  # EQUIV_SEQ: (False,)
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (False,)})
    best = BestDAGChanges()

    # Sequence element #1 is False so get NO_OP event

    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('A', 'B')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for first change: {}'.format(event))

    # Beyond sequence so no event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for second change: {}'.format(event))


def test_hc_best_abc_7_ok(abc):  # EQUIV_SEQ: (False, False)
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (False, False)})
    best = BestDAGChanges()

    # Sequence element #1 is False so get NO_OP event

    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('A', 'B')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for first change: {}'.format(event))

    # Sequence element #2 is False so get NO_OP event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
    print('Knowledge event for second change: {}'.format(event))

    # Beyond sequence so no event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
    print('Knowledge event for third change: {}'.format(event))


def test_hc_best_abc_8_ok(abc):  # EQUIV_SEQ: (False, True)
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (False, True)})
    best = BestDAGChanges()

    # Sequence element #1 is False so get NO_OP event

    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('A', 'B')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for first change: {}'.format(event))

    # Not an equiv_add change so no event

    best.top = DAGChange(Activity.ADD, ('A', 'C'), 4.2, None)
    best.second = DAGChange(Activity.ADD, ('C', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == best
    assert event is None
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for second change: {}'.format(event))

    # Sequence element #2 is True so get SWAP_BEST event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
    print('Knowledge event for third change: {}'.format(event))

    # Beyond sequence so no event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
    print('Knowledge event for fourth change: {}'.format(event))


def test_hc_best_abc_9_ok(abc):  # EQUIV_SEQ: (True, False)
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (True, False)})
    best = BestDAGChanges()

    # Sequence element #1 is True so get SWAP_BEST event

    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert event.arc == ('A', 'B')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for first change: {}'.format(event))

    # Not an equiv_add change so no event

    best.top = DAGChange(Activity.ADD, ('A', 'C'), 4.2, None)
    best.second = DAGChange(Activity.ADD, ('C', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == best
    assert event is None
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for second change: {}'.format(event))

    # Sequence element #2 is True so get SWAP_BEST event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
    print('Knowledge event for third change: {}'.format(event))

    # Beyond sequence so no event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
    print('Knowledge event for fourth change: {}'.format(event))


def test_hc_best_abc_10_ok(abc):  # EQUIV_SEQ: (True, True)
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (True, True)})
    best = BestDAGChanges()

    # Sequence element #1 is True so get SWAP_BEST event

    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert event.arc == ('A', 'B')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for first change: {}'.format(event))

    # Not an equiv_add change so no event

    best.top = DAGChange(Activity.ADD, ('A', 'C'), 4.2, None)
    best.second = DAGChange(Activity.ADD, ('C', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == best
    assert event is None
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for second change: {}'.format(event))

    # Sequence element #2 is True so get SWAP_BEST event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
    print('Knowledge event for third change: {}'.format(event))

    # Beyond sequence so no event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert best == new_best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
    print('Knowledge event for fourth change: {}'.format(event))


def test_hc_best_abc_11_ok(abc):  # EQUIV_SEQ: (True,) then pause
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (True,), 'pause': True})
    best = BestDAGChanges()

    # Sequence element #1 is False so get NO_OP event

    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert event.arc == ('A', 'B')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for first change: {}'.format(event))

    # Beyond sequence so no event, but PAUSE status set

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best.top == DAGChange(Activity.PAUSE, best.top.arc,
                                     best.top.delta, best.top.counts)
    assert new_best.second == best.second
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for second change: {}'.format(event))


def test_hc_best_abc_13_ok(abc):  # EQUIV_SEQ: (True, True), then pause
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (True, True), 'pause': True})
    best = BestDAGChanges()

    # Sequence element #1 is True so get SWAP_BEST event

    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert event.arc == ('A', 'B')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for first change: {}'.format(event))

    # Not an equiv_add change so no event

    best.top = DAGChange(Activity.ADD, ('A', 'C'), 4.2, None)
    best.second = DAGChange(Activity.ADD, ('C', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == best
    assert event is None
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for second change: {}'.format(event))

    # Sequence element #2 is True so get SWAP_BEST event

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
    print('Knowledge event for third change: {}'.format(event))

    # Beyond sequence so no event, but PAUSE set

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best.top == DAGChange(Activity.PAUSE, best.top.arc,
                                     best.top.delta, best.top.counts)
    assert new_best.second == best.second
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
    print('Knowledge event for fourth change: {}'.format(event))


# Test re-setting the EQUIV_SEQ decision sequence

def test_set_sequence_type_error_1():  # too few arguments
    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (True, ), 'pause': True})
    with pytest.raises(TypeError):
        know.set_sequence()
    with pytest.raises(TypeError):
        know.set_sequence(sequence=(False,))
    with pytest.raises(TypeError):
        know.set_sequence(pause=True)


def test_set_sequence_type_error_2():  # sequence must be tuple of bools
    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (True, ), 'pause': True})
    with pytest.raises(TypeError):
        know.set_sequence(sequence=True, pause=True)
    with pytest.raises(TypeError):
        know.set_sequence(sequence=[True], pause=True)
    with pytest.raises(TypeError):
        know.set_sequence(sequence=tuple(), pause=True)


def test_set_sequence_type_error_3():  # pause must be a bool
    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (True, ), 'pause': True})
    with pytest.raises(TypeError):
        know.set_sequence(sequence=(True, ), pause='bad')
    with pytest.raises(TypeError):
        know.set_sequence(sequence=(True, ), pause=(True, ))


def test_set_sequence_value_error_1():  # only for EQUIV_SEQ rukes
    know = Knowledge(rules=RuleSet.STOP_ARC,
                     params={'stop': {('B', 'C'): True}})
    with pytest.raises(ValueError):
        know.set_sequence((True, False), True)


def test_set_sequence_1_ok():  # check modifies test_set_sequence_value_error_1
    know = Knowledge(rules=RuleSet.EQUIV_SEQ, params={'sequence': (True, )})
    assert know.sequence == (True, )
    assert know.pause is False

    know.set_sequence((True, False), True)
    assert know.sequence == (True, False)
    assert know.pause is True


def test_set_sequence_2_ok(abc):  # EQUIV_SEQ: (True,), pause. restart
    knowledge = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (True,), 'pause': True})
    best = BestDAGChanges()

    # Sequence element #1 is True so get SWAP_BEST event

    best.top = DAGChange(Activity.ADD, ('A', 'B'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('B', 'A'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == BestDAGChanges(best.second, best.top)
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST
    assert event.arc == ('A', 'B')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('\n\nKnowledge event for first change: {}'.format(event))

    # Beyond sequence so no event, but PAUSE set

    best.top = DAGChange(Activity.ADD, ('B', 'C'), 4.0, None)
    best.second = DAGChange(Activity.ADD, ('C', 'B'), 4.0, None)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best.top == DAGChange(Activity.PAUSE, best.top.arc,
                                     best.top.delta, best.top.counts)
    assert new_best.second == best.second
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 1
    print('Knowledge event for second change: {}'.format(event))

    # Extend sequence and re-attempt second change

    knowledge.set_sequence((True, False), True)
    new_best, event = knowledge.hc_best(best, 6, abc['da'], abc['pa'])
    assert new_best == best
    assert event.rule == Rule.EQUIV_SEQ
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP
    assert event.arc == ('B', 'C')
    assert knowledge.reqd == {}
    assert knowledge.stop == {}
    assert knowledge.count == 2
    print('Knowledge event for repeat second change: {}'.format(event))
