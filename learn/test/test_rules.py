
# Test the Rules and Ruleset classes

import pytest

from learn.knowledge import Rule, RuleSet
from learn.trace import Activity


# Test the Rule class

def test_rule_attribute_error_1():  # unknown rule name
    with pytest.raises(AttributeError):
        Rule.UNKNOWN


def test_rule_attribute_error_2():  # unknown rule attribute
    with pytest.raises(AttributeError):
        Rule.EQUIV_ADD.unknown


def test_rule_attribute_error_3():  # value attribute is read-only
    with pytest.raises(AttributeError):
        Rule.EQUIV_ADD.value = 'not allowed'


def test_rule_attribute_error_4():  # label attribute is read-only
    with pytest.raises(AttributeError):
        Rule.EQUIV_ADD.label = 'not allowed'


def test_rule_strings_ok():
    assert str(Rule.EQUIV_ADD) == 'equiv_add'
    assert str(Rule.STOP_ARC) == 'stop_arc'


def test_rule_labels_ok():
    assert Rule.EQUIV_ADD.label == 'Choose equivalent add'
    assert Rule.STOP_ARC.label == 'Prohibited arc'


def test_rule_values_ok():
    assert Rule.EQUIV_ADD.value == 'equiv_add'
    assert Rule.STOP_ARC.value == 'stop_arc'


def test_rule_by_value_ok():
    assert Rule('equiv_add') == Rule.EQUIV_ADD
    assert Rule('stop_arc') == Rule.STOP_ARC


# Test the RuleSet class


def test_ruleset_attribute_error_1():  # unknown ruleset name
    with pytest.raises(AttributeError):
        RuleSet.UNKNOWN


def test_ruleset_attribute_error_2():  # unknown ruleset attribute
    with pytest.raises(AttributeError):
        RuleSet.EQUIV_ADD.unknown


def test_ruleset_attribute_error_3():  # value attribute is read-only
    with pytest.raises(AttributeError):
        RuleSet.EQUIV_ADD.value = 'read only'


def test_ruleset_attribute_error_4():  # label attribute is read-only
    with pytest.raises(AttributeError):
        RuleSet.EQUIV_ADD.label = 'read only'


def test_ruleset_attribute_error_5():  # rules attribute is read-only
    with pytest.raises(AttributeError):
        RuleSet.EQUIV_ADD.rules = 'read only'


def test_ruleset_strings_ok():
    assert str(RuleSet.EQUIV_ADD) == 'equiv_add'


def test_ruleset_labels_ok():
    assert RuleSet.EQUIV_ADD.label == 'Choose equivalent add'


def test_ruleset_values_ok():
    assert RuleSet.EQUIV_ADD.value == 'equiv_add'


def test_ruleset_rules_ok():
    assert RuleSet.EQUIV_ADD.rules == [Rule.EQUIV_ADD]


def test_ruleset_by_value_ok():
    assert RuleSet('equiv_add') == RuleSet.EQUIV_ADD
