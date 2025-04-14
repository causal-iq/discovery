# Test function which tests for correct_p_values

import pytest

from analysis.statistics import correct_p_values
from core.metrics import dicts_same


def test_correct_p_values_type_error_1_():  # no argument
    with pytest.raises(TypeError):
        correct_p_values()


def test_correct_p_values_type_error_2_():  # bad argument type
    with pytest.raises(TypeError):
        correct_p_values(10)
    with pytest.raises(TypeError):
        correct_p_values({'method1': 'invalid', 'method2': 0.05})


def test_correct_p_values_value_error_1_():  # invalid significance level
    with pytest.raises(ValueError):
        correct_p_values({'method1': 0.01}, significance_level=1.5)
    with pytest.raises(ValueError):
        correct_p_values({'method1': 0.01}, significance_level=-0.1)


def test_correct_p_values_value_error_2_():  # unsupported correction method
    with pytest.raises(ValueError):
        correct_p_values({'method1': 0.01}, correction='unsupported')


def test_correct_p_values_value_error_3_():  # empty p_values dictionary
    p_values = {}
    with pytest.raises(ValueError):
        correct_p_values(p_values, correction='bonferroni',
                         significance_level=0.05)


def test_correct_p_values_1_ok():  # no correction
    p_values = {'method1': 0.01, 'method2': 0.03, 'method3': 0.2}
    result = correct_p_values(p_values, correction=None,
                              significance_level=0.05)
    print('\nCorrected p-values (no correction):\n{}'.format(result))
    assert dicts_same(dict1=result, sf=6, strict=True,
                      dict2={'method1': 0.01, 'method2': 0.03})


def test_correct_p_values_2_ok():  # Bonferroni correction
    p_values = {'method1': 0.01, 'method2': 0.03, 'method3': 0.2}
    result = correct_p_values(p_values, correction='bonferroni',
                              significance_level=0.05)
    print('\nCorrected p-values (Bonferroni):\n{}'.format(result))
    assert dicts_same(dict1=result, sf=6, strict=True,
                      dict2={'method1': 0.03})


def test_correct_p_values_3_ok():  # Benjamini-Hochberg correction
    p_values = {'method1': 0.01, 'method2': 0.03, 'method3': 0.2}
    result = correct_p_values(p_values, correction='fdr_bh',
                              significance_level=0.05)
    print('\nCorrected p-values (Benjamini-Hochberg):\n{}'.format(result))
    assert dicts_same(dict1=result, sf=6, strict=True,
                      dict2={'method1': 0.03, 'method2': 0.045})


def test_correct_p_values_4_ok():  # Holm correction
    p_values = {'method1': 0.01, 'method2': 0.03, 'method3': 0.2}
    result = correct_p_values(p_values, correction='holm',
                              significance_level=0.05)
    print('\nCorrected p-values (Holm):\n{}'.format(result))
    assert dicts_same(dict1=result, sf=6, strict=True,
                      dict2={'method1': 0.03})


def test_correct_p_values_single_p_value():  # single p-value in p_values
    p_values = {'method1': 0.01}
    result = correct_p_values(p_values, correction='bonferroni',
                              significance_level=0.05)
    print('\nCorrected p-values (single p-value):\n{}'.format(result))
    assert dicts_same(dict1=result, sf=6, strict=True, dict2={'method1': 0.01})


def test_correct_p_values_all_above_threshold():  # all p-values > threshold
    p_values = {'method1': 0.2, 'method2': 0.3, 'method3': 0.4}
    result = correct_p_values(p_values, correction='bonferroni',
                              significance_level=0.05)
    print('\nCorrected p-values (all above threshold):\n{}'.format(result))
    assert result == {}


def test_correct_p_values_large_input():  # large number of p-values
    p_values = {f'method{i}': 0.0001 * (i + 1) for i in range(100)}
    result = correct_p_values(p_values, correction='fdr_bh',
                              significance_level=0.05)
    print('\nCorrected p-values (large input):\n{}'.format(result))
    assert len(result) > 0  # Ensure some methods are significant


def test_correct_p_values_no_significant():  # no significant p-values
    p_values = {'method1': 0.2, 'method2': 0.3, 'method3': 0.4}
    result = correct_p_values(p_values, correction=None,
                              significance_level=0.05)
    print('\nCorrected p-values (no significant):\n{}'.format(result))
    assert result == {}
