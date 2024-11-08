
# Test the Knowledge class

import pytest

from learn.knowledge import Rule, KnowledgeEvent, KnowledgeOutcome


def test_knowledge_event_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        KnowledgeEvent()


def test_knowledge_event_type_error_2():  # 1 argument
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.STOP_ARC)
    with pytest.raises(TypeError):
        KnowledgeEvent(rule=Rule.STOP_ARC)
    with pytest.raises(TypeError):
        KnowledgeEvent(correct=False)
    with pytest.raises(TypeError):
        KnowledgeEvent(outcome=KnowledgeOutcome.SWAP_BEST)


def test_knowledge_event_type_error_3():  # 2 arguments
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.STOP_ARC, True)
    with pytest.raises(TypeError):
        KnowledgeEvent(rule=Rule.STOP_ARC, correct=False)
    with pytest.raises(TypeError):
        KnowledgeEvent(correct=True, outcome=KnowledgeOutcome.SWAP_BEST)


def test_knowledge_event_type_error_4():  # rule is wrong type
    with pytest.raises(TypeError):
        KnowledgeEvent(True, True, KnowledgeOutcome.SWAP_BEST)
    with pytest.raises(TypeError):
        KnowledgeEvent(32, True, KnowledgeOutcome.SWAP_BEST)


def test_knowledge_event_type_error_5():  # correct is wrong type
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.STOP_ARC, 37, KnowledgeOutcome.SWAP_BEST)
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.STOP_ARC, [21.2], KnowledgeOutcome.SWAP_BEST)


def test_knowledge_event_type_error_6():  # outcome is wrong type
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.EQUIV_ADD, True, Rule.STOP_ARC)
    with pytest.raises(TypeError):
        KnowledgeEvent(Rule.STOP_ARC, True, {KnowledgeOutcome.SWAP_BEST})


def test_knowledge_event_value_error_1():  # correct ==> NO_OP outcome
    with pytest.raises(ValueError):
        KnowledgeEvent(Rule.EQUIV_ADD, None, KnowledgeOutcome.SWAP_BEST)


def test_knowledge_event_ok_1():  # EQUIV_ADD, knowledge true, NO_OP
    event = KnowledgeEvent(Rule.EQUIV_ADD, True, KnowledgeOutcome.NO_OP)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.NO_OP


def test_knowledge_event_ok_2():  # EQUIV_ADD, knowledge true, SWAP_BEST
    event = KnowledgeEvent(Rule.EQUIV_ADD, True, KnowledgeOutcome.SWAP_BEST)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is True
    assert event.outcome == KnowledgeOutcome.SWAP_BEST


def test_knowledge_event_ok_3():  # EQUIV_ADD, knowledge false, NO_OP
    event = KnowledgeEvent(Rule.EQUIV_ADD, False, KnowledgeOutcome.NO_OP)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.NO_OP


def test_knowledge_event_ok_4():  # EQUIV_ADD, knowledge false, SWAP_BEST
    event = KnowledgeEvent(Rule.EQUIV_ADD, False, KnowledgeOutcome.SWAP_BEST)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is False
    assert event.outcome == KnowledgeOutcome.SWAP_BEST


def test_knowledge_event_ok_5():  # EQUIV_ADD, knowledge None, SWAP_BEST
    event = KnowledgeEvent(Rule.EQUIV_ADD, None, KnowledgeOutcome.NO_OP)
    assert event.rule == Rule.EQUIV_ADD
    assert event.correct is None
    assert event.outcome == KnowledgeOutcome.NO_OP
