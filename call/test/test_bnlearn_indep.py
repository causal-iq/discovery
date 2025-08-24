
# Tests calling bnlearn CI tests

import pytest
from pandas import DataFrame

from core.metrics import dicts_same
from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_indep
from fileio.common import FileFormatError, TESTDATA_DIR


# --- Failure cases

# bad primary arg types
def test_bnlearn_indep_type_error_1():
    with pytest.raises(TypeError):
        bnlearn_indep()
    with pytest.raises(TypeError):
        bnlearn_indep(6, 'a')
    with pytest.raises(TypeError):
        bnlearn_indep('A', 'B', 6,
                      DataFrame({'A': ['1', '0'], 'B': ['1', '0']}), 'mi')
    with pytest.raises(TypeError):
        bnlearn_indep('A', 'B', None, {'A': ['1', '0'], 'B': ['1', '0']}, 'mi')


# bad types in z list
def test_bnlearn_indep_type_error_2():
    lizards_data = TESTDATA_DIR + '/simple/lizards.csv'
    with pytest.raises(TypeError):
        bnlearn_indep('A', 'B', ['C', True],
                      DataFrame({'A': ['1', '0'], 'B': ['1', '0'],
                                 'C': ['2', '3']}), 'mi')
    with pytest.raises(TypeError):
        bnlearn_indep('Diameter', 'Height', [10, 'Species'], lizards_data,
                      ['mi'])


# bad types in types list
def test_bnlearn_indep_type_error_3():
    lizards_data = TESTDATA_DIR + '/simple/lizards.csv'
    with pytest.raises(TypeError):
        bnlearn_indep('Diameter', 'Height', ['Species'], lizards_data,
                      ['mi', 3.5])
    with pytest.raises(TypeError):
        bnlearn_indep('Diameter', 'Height', ['Species'], lizards_data,
                      ['x2', ['mi']])


# non-existent file for data
def test_bnlearn_indep_file_error_1():
    with pytest.raises(FileNotFoundError):
        bnlearn_indep('Diameter', 'Height', ['Species'], 'nonexistent.txt')


# binary file for data
def test_bnlearn_indep_file_error_2():
    with pytest.raises(FileFormatError):
        bnlearn_indep('Diameter', 'Height', ['Species'],
                      TESTDATA_DIR + '/misc/null.sys')


# variable name duplicated
def test_bnlearn_indep_value_error_1():
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', ['Diameter'],
                      TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        bnlearn_indep('Height', 'Height', ['Diameter'],
                      TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', ['Species', 'Species'],
                      TESTDATA_DIR + '/simple/lizards.csv')


# variable names not in data
def test_bnlearn_indep_value_error_2():
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


# duplicate tests specified
def test_bnlearn_indep_value_error_3():
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', None,
                      TESTDATA_DIR + '/simple/lizards.csv', ['mi', 'mi'])
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', None,
                      TESTDATA_DIR + '/simple/lizards.csv', ['mi', 'x2', 'mi'])


# empty list of tests specified
def test_bnlearn_indep_value_error_4():
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', None,
                      TESTDATA_DIR + '/simple/lizards.csv', [])


# unsupported test specified
def test_bnlearn_indep_value_error_5():
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', None,
                      TESTDATA_DIR + '/simple/lizards.csv',
                      ['mi', 'unsupported'])
    with pytest.raises(ValueError):
        bnlearn_indep('Diameter', 'Height', None,
                      TESTDATA_DIR + '/simple/lizards.csv', 'unsupported')


# --- Successful independence tests

# A, B unconnected
@requires_r_and_bnlearn
def test_bnlearn_indep_a_b_ok():
    data = DataFrame({'A': ['1', '0'], 'B': ['1', '0']})
    value = bnlearn_indep('A', 'B', None, data, types=['mi', 'x2'])
    print(value)


# URL of bnlearn CI lizard tests is https://www.bnlearn.com/examples/ci.test/
# file of bnlearn lizards sample dataset
@requires_r_and_bnlearn
def test_bnlearn_indep_lizards_ok1():
    tests = bnlearn_indep('Height', 'Diameter', 'Species',
                          TESTDATA_DIR + '/simple/lizards.csv',
                          types=['mi', 'x2'])
    print(tests['mi'].to_dict())
    assert dicts_same({'statistic': 2.0256, 'df': 2, 'p_value': 0.3632},
                      tests['mi'].to_dict(), sf=4)


# file of bnlearn lizards sample dataset
@requires_r_and_bnlearn
def test_bnlearn_indep_lizards_ok2():
    tests = bnlearn_indep('Species', 'Diameter', 'Height',
                          TESTDATA_DIR + '/simple/lizards.csv', types='mi')
    print(tests['mi'].to_dict())
    assert dicts_same({'statistic': 14.024, 'df': 2, 'p_value': 0.0009009},
                      tests['mi'].to_dict(), sf=4)
