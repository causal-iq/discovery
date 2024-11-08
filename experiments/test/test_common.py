
# Test modules in experiments.ommon

import pytest

from learn.knowledge import RuleSet
from experiments.common import series_props, series_comparator, Package, \
    Ordering, Algorithm, sample_sizes, to_int


def _p(d):
    return '\n'.join(['  {}: {}'.format(k, v) for k, v in d.items()])

# Test series_props()


def test_common_series_props_type_error_1():  # no arguments specified
    with pytest.raises(TypeError):
        series_props()


def test_common_series_props_value_error_1():  # unknown series
    with pytest.raises(ValueError):
        series_props('UNKNOWN')


def test_common_series_props_ok_1():
    s = 'HC/STD'
    p = series_props(s)
    assert p == {'package': Package.BNBENCH,
                 'algorithm': Algorithm.HC,
                 'datagen': 'v1',
                 'ordering': Ordering.STANDARD,
                 'params': {'score': 'bic', 'base': 'e', 'k': 1},
                 'knowledge': False,
                 'kparams': None,
                 'randomise': False
                 }
    print('\n\nSeries "{}" has properties:\n{}'.format(s, _p(p)))


def test_common_series_props_ok_2():
    s = 'HC/KS_L16'
    p = series_props(s)
    assert p == {'package': Package.BNBENCH,
                 'algorithm': Algorithm.HC,
                 'datagen': 'v1',
                 'ordering': Ordering.STANDARD,
                 'params': {'score': 'bic', 'base': 'e', 'k': 1},
                 'knowledge': RuleSet.EQUIV_ADD,
                 'kparams': {'limit': 16, 'ignore': 0, 'expertise': 1.0},
                 'randomise': False
                 }
    print('\nSeries "{}" has properties:\n{}'.format(s, _p(p)))

# test series_comparator


def test_common_series_comparator_value_error_1():  # unknown series
    with pytest.raises(ValueError):
        series_comparator('UNKNOWN')


def test_common_series_comparator_ok_1():
    s = 'HC/BAD'
    c = series_comparator(s, as_dict=True)
    print(c)
    assert c == {'package': 'BNBENCH',
                 'algorithm': 'HC',
                 'ordering': 'worst',
                 'score': 'bic',
                 'test': None,
                 'k': 1,
                 'iss': None,
                 'alpha': None,
                 'knowledge': False,
                 'limit': None,
                 'ignore': None,
                 'expertise': None,
                 'reqd': None,
                 'threshold': None,
                 'earlyok': None,
                 'partial': None,
                 'nodes': None,
                 'stop': None}
    print('\nSeries "{}" has comparator:\n{}'.format(s, _p(c)))

    c = series_comparator(s)
    assert c == ('BNBENCH', 'HC', 'worst', 'bic', None, 1, None, None, False,
                 None, None, None, None, None, None, None, None, None)
    print('\nSeries "{}" has comparator key:\n{}'.format(s, c))


def test_common_series_comparator_ok_2():
    s = 'HC/KS_L16'
    c = series_comparator(s, as_dict=True)
    assert c == {'package': 'BNBENCH',
                 'algorithm': 'HC',
                 'ordering': 'alphabetic',
                 'score': 'bic',
                 'test': None,
                 'k': 1,
                 'iss': None,
                 'alpha': None,
                 'knowledge': 'equiv_add',
                 'limit': 16,
                 'ignore': 0,
                 'expertise': 1.0,
                 'reqd': None,
                 'threshold': None,
                 'earlyok': None,
                 'partial': None,
                 'nodes': None,
                 'stop': None}
    print('\nSeries "{}" has comparator:\n{}'.format(s, _p(c)))

    c = series_comparator(s)
    assert c == ('BNBENCH', 'HC', 'alphabetic', 'bic', None, 1, None, None,
                 'equiv_add', 16, 0, 1.0, None, None, None, None, None, None)
    print('\nSeries "{}" has comparator key:\n{}'.format(s, c))


def test_common_sample_sizes_type_error_1():  # arg not a string
    with pytest.raises(TypeError):
        sample_sizes(17)
    with pytest.raises(TypeError):
        sample_sizes(True)
    with pytest.raises(TypeError):
        sample_sizes(14.7)
    with pytest.raises(TypeError):
        sample_sizes((1, 10))


def test_common_sample_sizes_type_error_2():  # non-ints in string
    with pytest.raises(TypeError):
        sample_sizes('nonint-10')
    with pytest.raises(TypeError):
        sample_sizes('True')
    with pytest.raises(TypeError):
        sample_sizes('A,B')
    with pytest.raises(TypeError):
        sample_sizes('1,None')
    with pytest.raises(TypeError):
        sample_sizes('1-A')
    with pytest.raises(TypeError):
        sample_sizes('1,10')
    with pytest.raises(TypeError):
        sample_sizes('-1-100')


def test_common_sample_sizes_type_error_3():  # non-integer mantissa
    with pytest.raises(TypeError):
        sample_sizes('10-20;A')
    with pytest.raises(TypeError):
        sample_sizes('10-20;1,2.1')
    with pytest.raises(TypeError):
        sample_sizes('10-20;3,4.0')


def test_common_sample_sizes_type_error_4():  # non-positive mantissa
    with pytest.raises(TypeError):
        sample_sizes('10-20;-2,-1')
    with pytest.raises(TypeError):
        sample_sizes('10-20;-5,-2,-1')


def test_common_sample_sizes_type_error_5():  # non-integer sub-samples
    with pytest.raises(TypeError):
        sample_sizes('10-20;1;A')
    with pytest.raises(TypeError):
        sample_sizes('10-20;2,5;1,2.1')
    with pytest.raises(TypeError):
        sample_sizes('10-20;4;3,4.0')
    with pytest.raises(TypeError):
        sample_sizes('10-20;4,8;3,4')


def test_common_sample_sizes_type_error_6():  # negative sub-samples
    with pytest.raises(TypeError):
        sample_sizes('10-20;1;-2--1')
    with pytest.raises(TypeError):
        sample_sizes('10-20;2,5;-1-2')


def test_common_sample_sizes_value_error_1():  # invalid min, max
    with pytest.raises(ValueError):
        sample_sizes('0-10')
    with pytest.raises(ValueError):
        sample_sizes('100-10')
    with pytest.raises(ValueError):
        sample_sizes('100-10')
    with pytest.raises(ValueError):
        sample_sizes('1M-1K')


def test_common_sample_sizes_value_error_2():  # bad mantissa values
    with pytest.raises(ValueError):
        sample_sizes('10-20;0')
    with pytest.raises(ValueError):
        sample_sizes('10-20;12')
    with pytest.raises(ValueError):
        sample_sizes('10-20;9,10')


def test_common_sample_sizes_value_error_3():  # non-increasing mantissa
    with pytest.raises(ValueError):
        sample_sizes('10-20;1,1')
    with pytest.raises(ValueError):
        sample_sizes('10-20;3,2,4')


def test_common_sample_sizes_value_error_4():  # non-increasing sub-samples
    with pytest.raises(ValueError):
        sample_sizes('10-20;1;2-2')
    with pytest.raises(ValueError):
        sample_sizes('10-20;1,5;2-1')


def test_common_sample_sizes_1_ok():  # default samples
    assert [10, 20, 40, 50, 80,
            100, 200, 400, 500, 800,
            1000, 2000, 4000, 5000, 8000,
            10000, 20000, 40000, 50000, 80000,
            100000, 200000, 400000, 500000, 800000,
            1000000] == sample_sizes()[0]


def test_common_sample_sizes_2_ok():  # single value
    assert [10] == sample_sizes('10')[0]


def test_common_sample_sizes_3_ok():  # small values
    assert [4, 5] == sample_sizes('3-6')[0]


def test_common_sample_sizes_4_ok():  # big values
    assert [100000000, 200000000, 400000000, 500000000, 800000000,
            1000000000] == sample_sizes('100m-1g')[0]


def test_common_sample_sizes_5_ok():  # empty set
    assert [] == sample_sizes('7')[0]


def test_common_sample_sizes_6_ok():  # empty set
    assert [] == sample_sizes('6-7')[0]


def test_common_sample_sizes_7_ok():  # different mantissa lists
    assert [30] == sample_sizes('10-100;3')[0]
    assert [30, 300] == sample_sizes('10-1K;3')[0]
    assert [30, 60] == sample_sizes('10-100;3,6')[0]
    assert [10, 100] == sample_sizes('10-100;1')[0]
    assert [10, 50, 100] == sample_sizes('10-100;1,5')[0]
    assert [1000, 10000, 100000] == sample_sizes('1K-100K;1')[0]
    assert [1000, 5000, 10000, 50000, 100000] == sample_sizes('1K-100K;1,5')[0]


def test_common_sample_sizes_8_ok():  # empty mantissa
    assert [10, 20, 40, 50, 80, 100, 200] == sample_sizes('10-200;')[0]


def test_common_sample_sizes_9_ok():  # subsample range 0 -> 3
    Ns, Ss = sample_sizes('10-2K;1,2;0-3')
    assert Ns == [10, 20, 100, 200, 1000, 2000]
    assert Ss == (0, 3)


def test_common_sample_sizes_10_ok():  # subsample range 1 value
    Ns, Ss = sample_sizes('10-2K;1,2;0')
    assert Ns == [10, 20, 100, 200, 1000, 2000]
    assert Ss == (0, 0)


def test_common_sample_sizes_11_ok():  # subsample range 1
    Ns, Ss = sample_sizes('10-2K;1,2;5-12')
    assert Ns == [10, 20, 100, 200, 1000, 2000]
    assert Ss == (5, 12)


def test_common_to_int_type_error_1():  # no argument
    with pytest.raises(TypeError):
        to_int()


def test_common_to_int_type_error_2():  # bad arg types
    with pytest.raises(TypeError):
        to_int(12)
    with pytest.raises(TypeError):
        to_int(3.5)
    with pytest.raises(TypeError):
        to_int(['bad arg type'])


def test_common_to_int_ok_1():  # simple integers
    assert to_int('1') == 1
    assert to_int('445') == 445
    assert to_int('0') == 0
    assert to_int('10000000') == 10000000


def test_common_to_int_ok_2():  # multipliers
    assert to_int('1G') == 10**9
    assert to_int('445M') == 445 * 10**6
    assert to_int('33K') == 33000
    assert to_int('174k') == 174000
    assert to_int('1000000G') == 10**15


def test_common_to_int_ok_3():  # invalid
    assert to_int('?') is None
    assert to_int('10.7') is None
    assert to_int('-3') is None
    assert to_int('+4') is None
    assert to_int('10T') is None
    assert to_int('10**4') is None
