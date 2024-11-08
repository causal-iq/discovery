
#   Test adjmat and environment methods in core/common.py

from pytest import raises
from pandas import DataFrame
from os import remove, utime
from os.path import exists
from time import time
import pytest

from core.common import adjmat, environment, EnumWithAttrs, \
    generate_stable_random, stable_random, init_stable_random, \
    RandomIntegers, STABLE_RANDOM_FILE, rndsf

from fileio.common import EXPTS_DIR, TESTDATA_DIR


class ExampleEnumWithAttrs(EnumWithAttrs):
    VALUE1 = 'val_1', 'Value 1'
    VALUE2 = 'val_2', 'Value 2'


@pytest.fixture
def test_sequence():  # obtains sequence of randos in test file
    file_name = TESTDATA_DIR + '/experiments' + STABLE_RANDOM_FILE
    with open(file_name, 'r') as file:
        sequence = file.readlines()
    return [float(i.strip()) for i in sequence]


def test_common_adjmat_type_error_1():  # arg is not a dict
    with raises(TypeError):
        adjmat()
    with raises(TypeError):
        adjmat(3)
    with raises(TypeError):
        adjmat('invalid')
    with raises(TypeError):
        adjmat([2, 3, 4])
    with raises(TypeError):
        adjmat([[0, 0], [0, 0]])


def test_common_adjmat_type_error_2():  # arg is not a dict of lists
    with raises(TypeError):
        adjmat({'A': 'should be a list'})
    with raises(TypeError):
        adjmat({'A': [1], 'B': 'should be a list'})


def test_common_adjmat_type_error_3():  # arg is not a dict of lists of ints
    with raises(TypeError):
        adjmat({'A': ['should be int']})


def test_common_adjmat_value_error_1():  # list lengths must match num cols
    with raises(ValueError):
        adjmat({'A': [0, 1]})
    with raises(ValueError):
        adjmat({'A': [0], 'B': [1]})


def test_common_adjmat_value_error_2():  # must be valid ints
    with raises(ValueError):
        adjmat({'A': [7]})
    with raises(ValueError):
        adjmat({'A': [-1]})
    with raises(ValueError):
        adjmat({'A': [0, 1], 'B': [-99, 0]})


def test_common_adjmat_1x1_ok_1():  # arg is not a dict of lists of ints
    expected = (DataFrame({'': ['A'], 'A': [0]}).set_index('')
                .astype(dtype='int8'))
    print(expected)
    result = adjmat({'A': [0]})
    assert isinstance(result, DataFrame)
    assert result.equals(expected)


def test_common_adjmat_2x2_ok_1():  # arg is not a dict of lists of ints
    expected = (DataFrame({'': ['A', 'B'], 'A': [0, 1], 'B': [0, 0]})
                .set_index('').astype(dtype='int8'))
    result = adjmat({'A': [0, 1], 'B': [0, 0]})
    assert isinstance(result, DataFrame)
    assert result.equals(expected)


def test_common_environment_ok_1():  # environment.json not present
    if exists(EXPTS_DIR + '/environment.json'):
        remove(EXPTS_DIR + '/environment.json')
    env = environment()
    assert set(env.keys()) == {'cpu', 'os', 'python', 'ram'}


def test_common_environment_ok_2():  # use environment.json - should be fast
    env = environment()
    assert set(env.keys()) == {'cpu', 'os', 'python', 'ram'}


def test_common_environment_ok_3():  # rewrite stale environment.json
    overaday = time() - 2 * 24 * 3600
    if exists(EXPTS_DIR + '/environment.json'):
        utime(EXPTS_DIR + '/environment.json', times=(overaday, overaday))
    env = environment()
    assert set(env.keys()) == {'cpu', 'os', 'python', 'ram'}

# test EnumWithAttrs class


def test_enumwithattrs_attribute_error_1():  # unknown rule name
    with pytest.raises(AttributeError):
        ExampleEnumWithAttrs.UNKNOWN


def test_enumwithattrs_attribute_error_2():  # unknown rule attribute
    with pytest.raises(AttributeError):
        ExampleEnumWithAttrs.VALUE1.unknown


def test_enumwithattrs_attribute_error_3():  # value attribute is read-only
    with pytest.raises(AttributeError):
        ExampleEnumWithAttrs.VALUE1.value = 'not allowed'


def test_enumwithattrs_attribute_error_4():  # label attribute is read-only
    with pytest.raises(AttributeError):
        ExampleEnumWithAttrs.VALUE1.label = 'not allowed'


def test_enumwithattrs_strings_ok():
    assert str(ExampleEnumWithAttrs.VALUE1) == 'val_1'
    assert str(ExampleEnumWithAttrs.VALUE2) == 'val_2'


def test_enumwithattrs_labels_ok():
    assert ExampleEnumWithAttrs.VALUE1.label == 'Value 1'
    assert ExampleEnumWithAttrs.VALUE2.label == 'Value 2'


def test_enumwithattrs_values_ok():
    assert ExampleEnumWithAttrs.VALUE1.value == 'val_1'
    assert ExampleEnumWithAttrs.VALUE2.value == 'val_2'

# test stable random number sequences


def test_stable_random_1_ok():  # generate and fetch test sequence

    # Generate 1000 random numbers and write to file

    N = 1000
    path = TESTDATA_DIR + '/tmp'
    sequence = generate_stable_random(N, path)

    # get random numbers one at a time checking they are the ones generated

    init_stable_random()
    for i in range(N):
        assert stable_random(path) == sequence[i]

    # attempt to get more than generated raises exception

    with pytest.raises(StopIteration):
        stable_random(path)


def test_stable_random_2_ok(test_sequence):  # check init_stable_random resets

    N = 5
    path = TESTDATA_DIR + '/experiments'

    # get random numbers one at a time checking they are the ones generated

    init_stable_random()
    for i in range(N):
        assert stable_random(path) == test_sequence[i]

    # reset random number cache and check get same sequence again

    init_stable_random()
    for i in range(N):
        assert stable_random(path) == test_sequence[i]

    # attempt to get more than generated raises exception

    with pytest.raises(StopIteration):
        stable_random(path)


def test_stable_random_3_ok(test_sequence):  # check diff seq offset 1

    N = 5
    path = TESTDATA_DIR + '/experiments'

    # retrieve and check sequence .. offset is 0

    init_stable_random()
    for i in range(N):
        assert stable_random(path) == test_sequence[i]
    with pytest.raises(StopIteration):
        stable_random(path)

    # reset random number cache with offset 1

    init_stable_random(offset=1)
    assert stable_random(path) == test_sequence[1]
    assert stable_random(path) == test_sequence[3]
    assert stable_random(path) == test_sequence[4]
    assert stable_random(path) == test_sequence[0]
    assert stable_random(path) == test_sequence[2]
    with pytest.raises(StopIteration):
        stable_random(path)


def test_stable_random_4_ok(test_sequence):  # check diff seq offset 2

    path = TESTDATA_DIR + '/experiments'
    init_stable_random(offset=2)
    assert stable_random(path) == test_sequence[2]
    assert stable_random(path) == test_sequence[4]
    assert stable_random(path) == test_sequence[0]
    assert stable_random(path) == test_sequence[3]
    assert stable_random(path) == test_sequence[1]
    with pytest.raises(StopIteration):
        stable_random(path)


def test_stable_random_5_ok(test_sequence):  # check diff seq with offset 3

    path = TESTDATA_DIR + '/experiments'
    init_stable_random(offset=3)
    assert stable_random(path) == test_sequence[3]
    assert stable_random(path) == test_sequence[0]
    assert stable_random(path) == test_sequence[4]
    assert stable_random(path) == test_sequence[2]
    assert stable_random(path) == test_sequence[1]
    with pytest.raises(StopIteration):
        stable_random(path)


def test_stable_random_6_ok(test_sequence):  # check diff seq with offset 4

    path = TESTDATA_DIR + '/experiments'
    init_stable_random(offset=4)
    assert stable_random(path) == test_sequence[4]
    assert stable_random(path) == test_sequence[2]
    assert stable_random(path) == test_sequence[0]
    assert stable_random(path) == test_sequence[1]
    assert stable_random(path) == test_sequence[3]
    with pytest.raises(StopIteration):
        stable_random(path)

# test RandomIntegers


def test_common_random_integers_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        RandomIntegers()


def test_common_random_integers_type_error_2():  # non-integer n
    with pytest.raises(TypeError):
        RandomIntegers([4])
    with pytest.raises(TypeError):
        RandomIntegers('invalid')


def test_common_random_integers_type_error_3():  # non-integer subsample
    with pytest.raises(TypeError):
        RandomIntegers(5, [4])
    with pytest.raises(TypeError):
        RandomIntegers(10, 'invalid')


def test_common_random_integers_type_error_4():  # non-string path
    with pytest.raises(TypeError):
        RandomIntegers(5, path=10)
    with pytest.raises(TypeError):
        RandomIntegers(10, path=['invalid'])


def test_common_random_integers_value_error_1():  # invalid value for n
    with pytest.raises(ValueError):
        RandomIntegers(0)
    with pytest.raises(ValueError):
        RandomIntegers(-1)
    with pytest.raises(ValueError):
        RandomIntegers(1001)


def test_common_random_integers_value_error_2():  # invalid value for subsample
    with pytest.raises(ValueError):
        RandomIntegers(10, -1)
    with pytest.raises(ValueError):
        RandomIntegers(10, 1001)


def test_common_random_integers_filenotfound_error_1():  # bad path
    init_stable_random()
    with pytest.raises(FileNotFoundError):
        iter = RandomIntegers(10, path='non-existent')
        next(iter)


def test_common_random_integers_1_ok():  # 5 repeatable, subsample 0
    for j in range(10):
        assert [i for i in RandomIntegers(5)] == [2, 4, 0, 3, 1]


def test_common_random_integers_2_ok():  # 5 repeatable, subsample 1
    for j in range(10):
        assert [i for i in RandomIntegers(5, 1)] == [4, 1, 0, 2, 3]


def test_common_random_integers_3_ok():  # 5 repeatable, subsample 2
    for j in range(10):
        assert [i for i in RandomIntegers(5, 2)] == [0, 1, 3, 4, 2]


def test_common_random_integers_4_ok():  # 5 repeatable, subsample 3
    for j in range(10):
        assert [i for i in RandomIntegers(5, 3)] == [4, 2, 0, 1, 3]


def test_common_random_integers_5_ok():  # 5 repeatable, subsample 4
    for j in range(10):
        assert [i for i in RandomIntegers(5, 4)] == [1, 0, 2, 4, 3]


def test_common_random_integers_6_ok():  # check distribution for two ints
    dist = {}
    for subsample in range(1000):
        order = tuple([i for i in RandomIntegers(2, subsample)])
        if order not in dist:
            dist[order] = 0
        dist[order] += 1

    # distribution of combinations approximately uniform

    assert dist[(0, 1)] == 507
    assert dist[(1, 0)] == 493


def test_common_random_integers_7_ok():  # check distribution for three ints
    dist = {}
    for subsample in range(1000):
        order = tuple([i for i in RandomIntegers(3, subsample)])
        if order not in dist:
            dist[order] = 0
        dist[order] += 1

    # distribution of combinations approximately uniform

    assert dist[(0, 1, 2)] == 187
    assert dist[(0, 2, 1)] == 152
    assert dist[(1, 0, 2)] == 175
    assert dist[(1, 2, 0)] == 146
    assert dist[(2, 0, 1)] == 174
    assert dist[(2, 1, 0)] == 166


def test_common_random_integers_8_ok():  # first 600 8 integer sequences unique
    sequences = set()
    for subsample in range(600):
        order = tuple([i for i in RandomIntegers(8, subsample)])
        # print(order)
        assert order not in sequences
        sequences.add(order)


# test rndsf function which rounds to specified number of sf

def test_rndsf_type_error_1():  # no args
    with pytest.raises(TypeError):
        rndsf()


def test_rndsf_type_error_2():  # bad number type
    with pytest.raises(TypeError):
        rndsf('A', 2)
    with pytest.raises(TypeError):
        rndsf(None, 2)
    with pytest.raises(TypeError):
        rndsf(True, 2)
    with pytest.raises(TypeError):
        rndsf({1}, 2)


def test_rndsf_type_error_3():  # bad sf type
    with pytest.raises(TypeError):
        rndsf(3.12)
    with pytest.raises(TypeError):
        rndsf(3.12, None)
    with pytest.raises(TypeError):
        rndsf(3.12, 2.1)
    with pytest.raises(TypeError):
        rndsf(3.12, True)
    with pytest.raises(TypeError):
        rndsf(3.12, (1,))


def test_rndsf_type_error_4():  # bad min_val type
    with pytest.raises(TypeError):
        rndsf(3.12, 2, 1)
    with pytest.raises(TypeError):
        rndsf(3.12, 2, 0)
    with pytest.raises(TypeError):
        rndsf(3.12, 2, True)
    with pytest.raises(TypeError):
        rndsf(3.12, 2, [0.1001])


def test_rndsf_value_error_1():  # bad sf value
    with pytest.raises(ValueError):
        rndsf(3.12, 1)
    with pytest.raises(ValueError):
        rndsf(3.12, 0)
    with pytest.raises(ValueError):
        rndsf(3.12, -1)
    with pytest.raises(ValueError):
        rndsf(3.12, 11)


def test_rndsf_value_error_2():  # bad zero value
    with pytest.raises(ValueError):
        rndsf(3.12, 2, zero=0.2)
    with pytest.raises(ValueError):
        rndsf(3.12, 2, zero=-1.0)
    with pytest.raises(ValueError):
        rndsf(3.12, 2, zero=1E-21)


def test_rndsf_1_ok():  # values below implicit min_val returned as 0.0
    assert rndsf(0.001, 2) == '0.0'
    assert rndsf(1E-5, 4) == '0.0'
    assert rndsf(-1E-11, 10) == '0.0'
    assert rndsf(0.0, 10) == '0.0'
    assert rndsf(-0.0, 10) == '0.0'
    assert rndsf(0, 10) == '0.0'
    assert rndsf(-0, 10) == '0.0'


def test_rndsf_2_ok():  # fractions rounded to 2 s.f
    assert rndsf(0.334, 2) == '0.33'
    assert rndsf(-0.334, 2) == '-0.33'
    assert rndsf(0.337, 2) == '0.34'
    assert rndsf(0.00337, 2, zero=1E-5) == '0.0034'
    assert rndsf(-0.00337, 2, zero=1E-5) == '-0.0034'
    assert rndsf(0.00337001, 2, zero=1E-5) == '0.0034'
    assert rndsf(1E-12, 2, zero=1E-12) == '0.000000000001'
    assert rndsf(-2.1E-12, 2, zero=1E-12) == '-0.0000000000021'


def test_rndsf_3_ok():  # integers rounded to 2 s.f
    assert rndsf(103, 2) == '100.0'
    assert rndsf(-103, 2) == '-100.0'
    assert rndsf(130.17, 2) == '130.0'
    assert rndsf(-130.17, 2) == '-130.0'
    assert rndsf(1056.88, 3) == '1060.0'
    assert rndsf(-1056.88, 3) == '-1060.0'
    assert rndsf(1, 2) == '1.0'
    assert rndsf(1.0, 2) == '1.0'
    assert rndsf(-1, 2) == '-1.0'
    assert rndsf(-1.0, 2) == '-1.0'


def test_rndsf_4_ok():  # some big rounded to 2 s.f
    assert rndsf(1E15, 2) == '1000000000000000.0'
    assert rndsf(-1E15, 2) == '-1000000000000000.0'
    assert rndsf(1.08E15, 2) == '1100000000000000.0'
    assert rndsf(-1.08E15, 2) == '-1100000000000000.0'
    assert rndsf(1.44E15, 2) == '1400000000000000.0'
    assert rndsf(-1.44E15, 2) == '-1400000000000000.0'
    assert rndsf(01234567.89, 2) == '1200000.0'
    assert rndsf(-01234567.89, 2) == '-1200000.0'


def test_rndsf_5_ok():  # integers rounded to 10 s.f
    assert rndsf(1, 10) == '1.0'
    assert rndsf(-1, 10) == '-1.0'
    assert rndsf(12.345678954, 10) == '12.34567895'
    assert rndsf(-12.345678954, 10) == '-12.34567895'
    assert rndsf(12.345678955, 10) == '12.34567896'
    assert rndsf(-12.345678955, 10) == '-12.34567896'
