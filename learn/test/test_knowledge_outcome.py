
# Test the Knowledge Outcome class

import pytest

from learn.knowledge import KnowledgeOutcome


# Test the KnowledgeOutcome class

def test_outcome_attribute_error_1():  # unknown outcome name
    with pytest.raises(AttributeError):
        KnowledgeOutcome.UNKNOWN


def test_outcome_attribute_error_2():  # unknown outcome attribute
    with pytest.raises(AttributeError):
        KnowledgeOutcome.NO_OP.unknown


def test_outcome_attribute_error_3():  # value attribute is read-only
    with pytest.raises(AttributeError):
        KnowledgeOutcome.NO_OP.value = 'read only'


def test_outcome_attribute_error_4():  # label attribute is read-only
    with pytest.raises(AttributeError):
        KnowledgeOutcome.SWAP_BEST.label = 'read only'


def test_outcome_strings_ok():
    assert str(KnowledgeOutcome.NO_OP) == 'no_op'
    assert str(KnowledgeOutcome.SWAP_BEST) == 'swap_best'


def test_outcome_labels_ok():
    assert KnowledgeOutcome.NO_OP.label == 'No operation'
    assert KnowledgeOutcome.SWAP_BEST.label == 'Swap best and 2nd best'


def test_outcome_values_ok():
    assert KnowledgeOutcome.NO_OP.value == 'no_op'
    assert KnowledgeOutcome.SWAP_BEST.value == 'swap_best'


def test_outcome_by_value_ok():
    assert KnowledgeOutcome('no_op') == KnowledgeOutcome.NO_OP
    assert KnowledgeOutcome('swap_best') == KnowledgeOutcome.SWAP_BEST
