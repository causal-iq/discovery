
# Test function which tests for normality

import pytest
from numpy.random import normal, seed, standard_t, lognormal, uniform
from pandas import DataFrame

from analysis.statistics import normality
from core.metrics import dicts_same


def test_normality_type_error_1_():  # no argument
    with pytest.raises(TypeError):
        normality()


def test_normality_type_error_2_():  # bad argument type
    with pytest.raises(TypeError):
        normality(10)
    with pytest.raises(TypeError):
        normality([10, 11, 12.0])


def test_normality_type_error_3_():  # non-numeric values
    with pytest.raises(TypeError):
        normality(DataFrame({'sample': ['A', 1, 2]}))


def test_normality_value_error_1_():  # sample size too small
    with pytest.raises(ValueError):
        normality(DataFrame({'sample': [1.0, 2.2]}))


def test_normality_1_ok():  # single Gaussian distribution n = 10
    seed(0)
    samples = DataFrame({'s1': normal(50, 5, 10)})

    result = normality(samples)
    print('\nNormality result for normal(50, 5, 10):\n{}'.format(result))

    result = result.loc['s1'].to_dict()
    assert result["D'Agostino"] is None
    assert dicts_same(dict1=result, sf=6, strict=False,
                      dict2={'Shapiro-Wilk': 0.835664,
                             'Kolmogorov-Smirnov': 0.969202,
                             'Anderson-Darling': 0.859577})


def test_normality_2_ok():  # single Gaussian distribution n = 50
    seed(0)
    samples = DataFrame({'s1': normal(0, 5, 50)})

    result = normality(samples)
    print('\nNormality result for normal(0, 5, 50):\n{}'.format(result))

    result = result.loc['s1'].to_dict()
    assert dicts_same(dict1=result, sf=6, strict=True,
                      dict2={'Shapiro-Wilk': 0.876542,
                             'D\'Agostino': 0.772765,
                             'Kolmogorov-Smirnov': 0.995039,
                             'Anderson-Darling': 0.962989})


def test_normality_3_ok():  # single Gaussian distribution n = 10k
    seed(0)
    samples = DataFrame({'s1': normal(0, 0.1, 10000)})

    result = normality(samples)
    print('\nNormality result for normal(0, 0.1, 10000):\n{}'.format(result))

    result = result.loc['s1'].to_dict()
    assert result['Shapiro-Wilk'] is None
    assert dicts_same(dict1=result, sf=6, strict=False,
                      dict2={'D\'Agostino': 0.460549,
                             'Kolmogorov-Smirnov': 0.741393,
                             'Anderson-Darling': 0.549481})


def test_normality_4_ok():  # single Gaussian distribution n = 100k
    seed(0)
    samples = DataFrame({'s1': normal(0, 1, 100000)})

    result = normality(samples)
    print('\nNormality result for normal(0, 1, 100000):\n{}'.format(result))

    result = result.loc['s1'].to_dict()
    assert result['Shapiro-Wilk'] is None
    assert dicts_same(dict1=result, sf=6, strict=False,
                      dict2={'D\'Agostino': 0.0739616,
                             'Kolmogorov-Smirnov': 0.913895,
                             'Anderson-Darling': 0.428378})


def test_normality_5_ok():  # single T-distribution df=15, n=5000
    seed(0)
    samples = DataFrame({'s1': standard_t(15, 5000)})

    result = normality(samples)
    print('\nNormality result for standard_t(15, 5000):\n{}'.format(result))

    result = result.loc['s1'].to_dict()
    assert dicts_same(dict1=result, sf=6, strict=False,
                      dict2={'Shapiro-Wilk': 0.000324347,
                             'D\'Agostino': 0.000114900,
                             'Kolmogorov-Smirnov': 0.287993,
                             'Anderson-Darling': 0.000198548})


def test_normality_6_ok():  # single lognormal, sigma=0.2
    seed(0)
    samples = DataFrame({'s1': lognormal(mean=0, sigma=0.2, size=120)})

    result = normality(samples)
    print('\nNormality result for lognormal(0, 0.2, 120):\n{}'.format(result))

    result = result.loc['s1'].to_dict()
    assert dicts_same(dict1=result, sf=6, strict=False,
                      dict2={'Shapiro-Wilk': 0.01057169,
                             'D\'Agostino':  0.0605383,
                             'Kolmogorov-Smirnov': 0.396460,
                             'Anderson-Darling': 0.0140862})


def test_normality_7_ok():  # single uniform, 100
    seed(0)
    samples = DataFrame({'s1': uniform(size=100)})

    result = normality(samples)
    print('\nNormality result for uniform(100)):\n{}'.format(result))

    result = result.loc['s1'].to_dict()
    assert dicts_same(dict1=result, sf=6, strict=False,
                      dict2={'Shapiro-Wilk': 0.000974977,
                             'D\'Agostino':  8.63224e-07,
                             'Kolmogorov-Smirnov': 0.386708,
                             'Anderson-Darling': 0.00343314})
