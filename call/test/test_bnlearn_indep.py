
# Tests calling bnlearn CI tests

import pytest
from pandas import DataFrame

from core.metrics import dicts_same
from call.bnlearn import bnlearn_indep
from fileio.common import FileFormatError, TESTDATA_DIR


def test_bnlearn_indep_type_error_1():  # bad primary arg types
    with pytest.raises(TypeError):
        bnlearn_indep()
    with pytest.raises(TypeError):
        bnlearn_indep(6, 'a')
    with pytest.raises(TypeError):
        bnlearn_indep('A', 'B', 6,
                      DataFrame({'A': ['1', '0'], 'B': ['1', '0']}), 'mi')
    with pytest.raises(TypeError):
        bnlearn_indep('A', 'B', None, {'A': ['1', '0'], 'B': ['1', '0']}, 'mi')


def test_bnlearn_indep_type_error_2():  # bad types in z list
    lizards_data = TESTDATA_DIR + '/simple/lizards.csv'
    with pytest.raises(TypeError):
        bnlearn_indep('A', 'B', ['C', True],
                      DataFrame({'A': ['1', '0'], 'B': ['1', '0'],
                                 'C': ['2', '3']}), 'mi')
    with pytest.raises(TypeError):
        bnlearn_indep('Diameter', 'Height', [10, 'Species'], lizards_data,
                      ['mi'])


def test_bnlearn_indep_type_error_3():  # bad types in types list
    lizards_data = TESTDATA_DIR + '/simple/lizards.csv'
    with pytest.raises(TypeError):
        bnlearn_indep('Diameter', 'Height', ['Species'], lizards_data,
                      ['mi', 3.5])
    with pytest.raises(TypeError):
        bnlearn_indep('Diameter', 'Height', ['Species'], lizards_data,
                      ['x2', ['mi']])


def test_bnlearn_indep_file_error_1():  # non-existent file for data
    with pytest.raises(FileNotFoundError):
        bnlearn_indep('Diameter', 'Height', ['Species'], 'nonexistent.txt')


def test_bnlearn_indep_file_error_2():  # binary file for data
    with pytest.raises(FileFormatError):
        bnlearn_indep('Diameter', 'Height', ['Species'],
                      TESTDATA_DIR + '/misc/null.sys')


def test_bnlearn_indep_value_error_1():  # variable name duplicated
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', ['Diameter'],
                      TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        bnlearn_indep('Height', 'Height', ['Diameter'],
                      TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', ['Species', 'Species'],
                      TESTDATA_DIR + '/simple/lizards.csv')


def test_bnlearn_indep_value_error_2():  # variable names not in data
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', ['Unknown'],
                      TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', ['Species', 'Unknown'],
                      TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        bnlearn_indep('Unknown', 'Height', ['Species'],
                      TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Unknown', ['Species'],
                      TESTDATA_DIR + '/simple/lizards.csv')


def test_bnlearn_indep_value_error_3():  # duplicate tests specified
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', None,
                      TESTDATA_DIR + '/simple/lizards.csv', ['mi', 'mi'])
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', None,
                      TESTDATA_DIR + '/simple/lizards.csv', ['mi', 'x2', 'mi'])


def test_bnlearn_indep_value_error_4():  # empty list of tests specified
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', None,
                      TESTDATA_DIR + '/simple/lizards.csv', [])


def test_bnlearn_indep_value_error_5():  # unsupported test specified
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', None,
                      TESTDATA_DIR + '/simple/lizards.csv',
                      ['mi', 'unsupported'])
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', None,
                      TESTDATA_DIR + '/simple/lizards.csv', 'unsupported')


def test_bnlearn_indep_a_b_ok():  # A, B unconnected
    data = DataFrame({'A': ['1', '0'], 'B': ['1', '0']})
    value = bnlearn_indep('A', 'B', None, data, types=['mi', 'x2'])
    print(value)


# URL of bnlearn CI lizard tests is https://www.bnlearn.com/examples/ci.test/

def test_bnlearn_indep_lizards_ok1():  # file of bnlearn lizards sample dataset
    tests = bnlearn_indep('Height', 'Diameter', 'Species',
                          TESTDATA_DIR + '/simple/lizards.csv',
                          types=['mi', 'x2'])
    print(tests['mi'].to_dict())
    assert dicts_same({'statistic': 2.0256, 'df': 2, 'p_value': 0.3632},
                      tests['mi'].to_dict(), sf=4)


def test_bnlearn_indep_lizards_ok2():  # file of bnlearn lizards sample dataset
    tests = bnlearn_indep('Species', 'Diameter', 'Height',
                          TESTDATA_DIR + '/simple/lizards.csv', types='mi')
    print(tests['mi'].to_dict())
    assert dicts_same({'statistic': 14.024, 'df': 2, 'p_value': 0.0009009},
                      tests['mi'].to_dict(), sf=4)
