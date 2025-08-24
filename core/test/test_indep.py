
# Test probability independence testing code

import pytest
from pandas import DataFrame

from core.indep import indep, check_test_params
from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_indep
from core.bn import BN
from core.metrics import dicts_same, values_same
from fileio.common import FileFormatError, TESTDATA_DIR
from fileio.pandas import Pandas
from fileio.bayesys import read as read_dag

TYPES = ['x2', 'mi']


# --- Failure cases

# bad primary arg types
def test_indep_type_error_1():
    bn_cancer = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    with pytest.raises(TypeError):
        indep()
    with pytest.raises(TypeError):
        indep(6, 'a')
    with pytest.raises(TypeError):
        indep('A', 'B', 6, DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
              types='mi')
    with pytest.raises(TypeError):
        indep('A', 'B', None, {'A': ['1', '0'], 'B': ['1', '0']},
              types='mi')
    with pytest.raises(TypeError):
        indep('A', 'B', None, DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
              bn=False, types='mi')
    with pytest.raises(TypeError):
        indep('A', 'B', None, None, bn=False, types='mi')
    with pytest.raises(TypeError):
        indep('A', 'B', None, None, bn=bn_cancer, N='badtype', types='mi')
    with pytest.raises(TypeError):
        indep('A', 'B', None, None, bn=bn_cancer, N=True, types='mi')
    with pytest.raises(TypeError):
        indep('A', 'B', None, DataFrame({'A': ['1', '0'], 'B': ['1', '0']}),
              bn=bn_cancer, types='mi')


# bad types in z list
def test_indep_type_error_2():
    lizards_data = TESTDATA_DIR + '/simple/lizards.csv'
    with pytest.raises(TypeError):
        indep('A', 'B', ['C', True],
              DataFrame({'A': ['1', '0'], 'B': ['1', '0'],
                         'C': ['2', '3']}), types='mi')
    with pytest.raises(TypeError):
        indep('Diameter', 'Height', [10, 'Species'], lizards_data,
              types=['mi'])


# bad types in types list
def test_indep_type_error_3():
    lizards_data = TESTDATA_DIR + '/simple/lizards.csv'
    with pytest.raises(TypeError):
        indep('Diameter', 'Height', ['Species'], lizards_data,
              types=['mi', 3.5])
    with pytest.raises(TypeError):
        indep('Diameter', 'Height', ['Species'], lizards_data,
              types=['x2', ['mi']])


# non-existent file for data
def test_indep_file_error_1():
    with pytest.raises(FileNotFoundError):
        indep('Diameter', 'Height', ['Species'], 'nonexistent.txt')


# binary file for data
def test_indep_file_error_2():
    with pytest.raises(FileFormatError):
        indep('Diameter', 'Height', ['Species'],
              TESTDATA_DIR + '/misc/null.sys')


# variable name duplicated
def test_indep_value_error_1():
    with pytest.raises(ValueError):
        indep('Diameter', 'Height', ['Diameter'],
              TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        indep('Height', 'Height', ['Diameter'],
              TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        indep('Diameter', 'Height', ['Species', 'Species'],
              TESTDATA_DIR + '/simple/lizards.csv')


# variable names not in data
def test_indep_value_error_2():
    with pytest.raises(ValueError):
        indep('Diameter', 'Height', ['Unknown'],
              TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        indep('Diameter', 'Height', ['Species', 'Unknown'],
              TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        indep('Unknown', 'Height', ['Species'],
              TESTDATA_DIR + '/simple/lizards.csv')
    with pytest.raises(ValueError):
        indep('Diameter', 'Unknown', ['Species'],
              TESTDATA_DIR + '/simple/lizards.csv')


# variable names not in BN
def test_indep_value_error_3():
    bn_cancer = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    with pytest.raises(ValueError):
        indep('Smoker', 'Pollution', ['Unknown'], data=None, bn=bn_cancer)
    with pytest.raises(ValueError):
        indep('Xray', 'Cancer', ['Smoker', 'Unknown'], data=None, bn=bn_cancer)
    with pytest.raises(ValueError):
        indep('Unknown', 'Smoker', ['Pollution'], data=None, bn=bn_cancer)
    with pytest.raises(ValueError):
        indep('Cancer', 'Unknown', ['Pollution'], data=None, bn=bn_cancer)


# Sample size is negative
def test_indep_value_error_4():
    bn_cancer = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    with pytest.raises(ValueError):
        indep('Smoker', 'Pollution', ['Cancer'], data=None, N=-1, bn=bn_cancer)
    with pytest.raises(ValueError):
        indep('Smoker', 'Pollution', ['Cancer'], data=None, N=-3, bn=bn_cancer)


# duplicate tests specified
def test_indep_value_error_5():
    with pytest.raises(ValueError):
        indep('Diameter', 'Height', None, TESTDATA_DIR + '/simple/lizards.csv',
              types=['mi', 'mi'])
    with pytest.raises(ValueError):
        indep('Diameter', 'Height', None, TESTDATA_DIR + '/simple/lizards.csv',
              types=['mi', 'x2', 'mi'])


# empty list of tests specified
def test_indep_value_error_6():
    with pytest.raises(ValueError):
        indep('Diameter', 'Height', None, TESTDATA_DIR + '/simple/lizards.csv',
              types=[])


# unsupported test specified
def test_indep_value_error_7():
    with pytest.raises(ValueError):
        indep('Diameter', 'Height', None, TESTDATA_DIR + '/simple/lizards.csv',
              types=['mi', 'unsupported'])
    with pytest.raises(ValueError):
        indep('Diameter', 'Height', None, TESTDATA_DIR + '/simple/lizards.csv',
              types='unsupported')


# --- Successful independence tests

# A, B deterministic, 2 cases
@requires_r_and_bnlearn
def test_indep_a_b_ok1():
    data = DataFrame({'A': ['1', '0'], 'B': ['1', '0']})
    test = indep('A', 'B', None, data, types=TYPES)
    print('\nIndependence tests for 2 deterministic cases:\n{}'.format(test))
    bnlearn = bnlearn_indep('A', 'B', None, data, types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# A, B deterministic, 10 cases
@requires_r_and_bnlearn
def test_indep_a_b_ok2():
    data = DataFrame({'A': ['1', '1', '1', '1', '1', '0', '0', '0', '0', '0'],
                      'B': ['0', '0', '0', '0', '0', '1', '1', '1', '1', '1']})
    test = indep('A', 'B', None, data, types=TYPES)
    print('\nIndependence tests for 10 deterministic cases:\n{}'.format(test))
    bnlearn = bnlearn_indep('A', 'B', None, data, types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# A->B check data, cpt, bnlearn all give same p-value
@requires_r_and_bnlearn
def test_indep_ab_ok1():

    ab = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')  # get A-->B BN
    N = 1000
    data = Pandas(df=ab.generate_cases(N))

    ab = BN.fit(ab.dag, data)  # re-fit BN so CPTs match data

    # Check CI test results from data and CPT parameters match

    dep_data = indep('A', 'B', None, data.sample, types=TYPES)
    dep_cpt = indep('A', 'B', None, None, ab, N, types=TYPES)
    for type in TYPES:
        assert dicts_same(dep_data[type].to_dict(), dep_cpt[type].to_dict())

    # Check bnlearn gives (approximately) same results too

    dep_bnlearn = bnlearn_indep('A', 'B', None, data.sample, types=TYPES)
    for type in TYPES:
        assert dicts_same(dep_data[type].to_dict(),
                          dep_bnlearn[type].to_dict(), sf=4)


# A->B->C - check A, B dependencies match
@requires_r_and_bnlearn
def test_indep_abc_1_ok():

    abc = BN.read(TESTDATA_DIR + '/dsc/abc.dsc')  # get A-->B BN
    N = 1000
    data = Pandas(df=abc.generate_cases(N))
    print(abc.global_distribution())

    abc = BN.fit(abc.dag, data)  # re-fit BN so CPTs match data

    # Check CI test results from data and CPT parameters match

    dep_data = indep('A', 'B', None, data.sample, types=TYPES)
    dep_cpt = indep('A', 'B', None, None, abc, N, types=TYPES)
    for type in TYPES:
        assert dicts_same(dep_data[type].to_dict(), dep_cpt[type].to_dict())

    # Check bnlearn gives (approximately) same results too

    dep_bnlearn = bnlearn_indep('A', 'B', None, data.sample, types=TYPES)
    for type in TYPES:
        assert dicts_same(dep_data[type].to_dict(),
                          dep_bnlearn[type].to_dict(), sf=4)


# A->B->C - check B, C dependencies match
@requires_r_and_bnlearn
def test_indep_abc_2_ok():

    abc = BN.read(TESTDATA_DIR + '/dsc/abc.dsc')  # get A-->B-->C BN
    N = 1000
    data = Pandas(df=abc.generate_cases(N))
    print(abc.global_distribution())

    abc = BN.fit(abc.dag, data)  # re-fit BN so CPTs match data

    # Check CI test results from data and CPT parameters match

    dep_data = indep('B', 'C', None, data.sample, types=TYPES)
    dep_cpt = indep('B', 'C', None, None, abc, N, types=TYPES)
    for type in TYPES:
        assert dicts_same(dep_data[type].to_dict(), dep_cpt[type].to_dict())

    # Check bnlearn gives same statistic value
    # (p-values differ because underflow happening in Python code)

    dep_bnlearn = bnlearn_indep('B', 'C', None, data.sample, types=TYPES)
    # print(dep_data)
    # print(dep_bnlearn)
    for type in TYPES:
        assert values_same(dep_data[type].to_dict()['statistic'],
                           dep_bnlearn[type].to_dict()['statistic'], sf=4)


# A->B->C - check A, C dependencies match
@requires_r_and_bnlearn
def test_indep_abc_ok3():

    abc = BN.read(TESTDATA_DIR + '/dsc/abc.dsc')  # get A-->B-->C BN
    N = 10000
    data = Pandas(df=abc.generate_cases(N))

    abc = BN.fit(abc.dag, data)  # re-fit BN so CPTs match data

    # Check CI test p-values from data and CPT parameters match and show
    #  . Note the test statistics are not compared because they are
    # not expected to be exactly the same as the CPT parameters won't be 'nice'
    # fractions and so a huge sample size is needed for the data and CPT
    # derived frequencies to match exactly.

    dep_data = indep('A', 'C', None, data.sample, types=TYPES)
    dep_cpt = indep('A', 'C', None, None, abc, N, types=TYPES)
    for type in TYPES:
        assert values_same(dep_data[type].to_dict()['p_value'], 0.0)
        assert values_same(dep_cpt[type].to_dict()['p_value'], 0.0)

    # Check bnlearn gives same statistic value
    # (p-values differ because underflow happening in Python code)

    dep_bnlearn = bnlearn_indep('A', 'C', None, data.sample, types=TYPES)
    for type in TYPES:
        assert values_same(dep_data[type].to_dict()['statistic'],
                           dep_bnlearn[type].to_dict()['statistic'])


# A->B->C - check A, C given B independent
@requires_r_and_bnlearn
def test_indep_abc_ok4():

    abc = BN.read(TESTDATA_DIR + '/dsc/abc.dsc')  # get A-->B-->C BN
    N = 10000
    data = abc.generate_cases(N)  # generate data for 1000 cases

    dep_data = indep('A', 'C', 'B', data, types=TYPES)
    dep_bnlearn = bnlearn_indep('A', 'C', 'B', data, types=TYPES)
    for type in TYPES:
        assert values_same(dep_data[type].to_dict()['statistic'],
                           dep_bnlearn[type].to_dict()['statistic'])


# Dependence in Lizards
@requires_r_and_bnlearn
def test_bnlearn_indep_lizards_1_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/lizards.csv',
                       dstype='categorical')

    # check CI statistics from this code and bnlearn the same
    # Note they DO NOT cause rejection of independence

    dep_data = indep('Height', 'Diameter', None, data.sample, types=TYPES)
    print('\nLizards - Height, Diameter CI stats from real data:\n{}\n'
          .format(dep_data))
    dep_bnlearn = bnlearn_indep('Height', 'Diameter', None, data.sample,
                                types=TYPES)
    for t in TYPES:
        assert dicts_same(dep_bnlearn[t].to_dict(), dep_data[t].to_dict())

    # check CI test statistics from CPTs learnt from data do show independence
    # at large sample sizes

    dag = read_dag(TESTDATA_DIR + '/bayesys/lizards.csv')
    bn = BN.fit(dag, data)
    dep_cpt = indep('Height', 'Diameter', None, None, bn=bn, types=TYPES)
    print('\nLizards - Height Species CI stats from CPTs with N=10**9:\n{}\n'
          .format(dep_cpt))
    for type in TYPES:
        assert values_same(dep_cpt[type].to_dict()['p_value'], 0)


# Conditional Independence in Lizards
@requires_r_and_bnlearn
def test_bnlearn_indep_lizards_2_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/lizards.csv',
                       dstype='categorical')

    # check CI statistics from this code and bnlearn the same
    # Data does show independence

    dep_data = indep('Height', 'Diameter', 'Species', data.sample, types=TYPES)
    print('\nLizards - Height, Diameter | Species CI stats from data:\n{}\n'
          .format(dep_data))
    dep_bnlearn = bnlearn_indep('Height', 'Diameter', 'Species', data.sample,
                                types=TYPES)
    for type in TYPES:
        assert dicts_same(dep_bnlearn[type].to_dict(),
                          dep_data[type].to_dict())

    # check CI test statistics from CPTs learnt from data show independence

    dag = read_dag(TESTDATA_DIR + '/bayesys/lizards.csv')
    bn = BN.fit(dag, data)
    dep_cpt = indep('Height', 'Diameter', 'Species', None, bn=bn, types=TYPES)
    print('\nLizards - Height, Diameter | Species CI stats from CPTs:\n{}\n'
          .format(dep_cpt))
    for type in TYPES:
        assert values_same(dep_cpt[type].to_dict()['p_value'], 1)


# Conditional Dependence in Lizards
@requires_r_and_bnlearn
def test_bnlearn_indep_lizards_ok3():
    data = Pandas.read(TESTDATA_DIR + '/simple/lizards.csv',
                       dstype='categorical')

    # check CI statistics from this code and bnlearn the same

    dep_data = indep('Species', 'Diameter', 'Height', data.sample, types=TYPES)
    print('\nLizards - Species, Diameter | Height CI stats from data:\n{}\n'
          .format(dep_data))
    dep_bnlearn = bnlearn_indep('Species', 'Diameter', 'Height', data.sample,
                                types=TYPES)
    for type in TYPES:
        assert dicts_same(dep_bnlearn[type].to_dict(),
                          dep_data[type].to_dict())

    # check CI test statistics from CPTs learnt from data show dependence

    dag = read_dag(TESTDATA_DIR + '/bayesys/lizards.csv')
    bn = BN.fit(dag, data)
    dep_cpt = indep('Species', 'Diameter', 'Height', None, bn=bn, types=TYPES)
    print('\nLizards - Species, Diameter | Heights CI stats from CPTs:\n{}\n'
          .format(dep_cpt))
    for type in TYPES:
        assert values_same(dep_cpt[type].to_dict()['p_value'], 0)


# indep in cancer BN
@requires_r_and_bnlearn
def test_bnlearn_indep_cancer_ok1():
    cancer = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    data = Pandas(df=cancer.generate_cases(1000))
    test = indep('Pollution', 'Smoker', None, data.sample, types=TYPES)
    print('\nCancer - Pollution, Smoker:\n{}'.format(test))
    bnlearn = bnlearn_indep('Pollution', 'Smoker', None, data.sample,
                            types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# dependence in cancer BN
@requires_r_and_bnlearn
def test_bnlearn_indep_cancer_ok2():
    cancer = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    data = Pandas(df=cancer.generate_cases(1000))
    test = indep('Smoker', 'Cancer', None, data.sample, types=TYPES)
    print('\nCancer - Smoker, Cancer:\n{}'.format(test))
    bnlearn = bnlearn_indep('Smoker', 'Cancer', None, data.sample, types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# dependence in cancer BN
@requires_r_and_bnlearn
def test_bnlearn_indep_cancer_ok3():
    cancer = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    data = Pandas(df=cancer.generate_cases(1000))
    test = indep('Pollution', 'Cancer', None, data.sample, types=TYPES)
    print('\nCancer - Pollution, Cancer:\n{}'.format(test))
    bnlearn = bnlearn_indep('Pollution', 'Cancer', None, data.sample,
                            types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# cond. indep in cancer BN, cond set = 1
@requires_r_and_bnlearn
def test_bnlearn_indep_cancer_ok4():
    cancer = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    data = Pandas(df=cancer.generate_cases(1000))
    test = indep('Xray', 'Smoker', 'Cancer', data.sample, types=TYPES)
    print('\nCancer - Xray, Smoker | Cancer:\n{}'.format(test))
    bnlearn = bnlearn_indep('Xray', 'Smoker', 'Cancer', data.sample,
                            types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# cond. indep in cancer BN, cond set = 2
@requires_r_and_bnlearn
def test_bnlearn_indep_cancer_ok5():
    cancer = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    data = Pandas(df=cancer.generate_cases(5000))
    cancer = BN.fit(cancer.dag, data)
    test = indep('Xray', 'Smoker', ['Cancer', 'Pollution'], data.sample,
                 types=TYPES)
    print('\nCancer - Xray, Smoker | Cancer, Pollution:\n{}'.format(test))
    bnlearn = bnlearn_indep('Xray', 'Smoker', ['Cancer', 'Pollution'],
                            data.sample, types=TYPES)
    print(indep('Xray', 'Smoker', ['Cancer', 'Pollution'], None, cancer, 5000,
                types=TYPES))
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# cond. dependence in cancer BN
@requires_r_and_bnlearn
def test_bnlearn_indep_cancer_ok6():
    cancer = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    data = cancer.generate_cases(1000)
    test = indep('Smoker', 'Pollution', 'Cancer', data, types=TYPES)
    print('\nCancer - Smoker, Pollution | Cancer:\n{}'.format(test))
    bnlearn = bnlearn_indep('Smoker', 'Pollution', 'Cancer', data, types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# --- Independence tests with continuous data

# X1 and Y2 are unconditionally indep.
@requires_r_and_bnlearn
def test_bnlearn_indep_xyzw_ok1():
    data = Pandas.read(TESTDATA_DIR + '/simple/xyzw.csv',
                       dstype='categorical').df
    print(data.value_counts())
    test = indep('X1', 'Y2', None, data, types=TYPES)
    print('\nXYZW - X1, Y2:\n{}'.format(test))
    bnlearn = bnlearn_indep('X1', 'Y2', None, data, types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# X1 and W4 are dependent - 2x2 table
@requires_r_and_bnlearn
def test_bnlearn_indep_xyzw_ok2():
    data = Pandas.read(TESTDATA_DIR + '/simple/xyzw.csv',
                       dstype='categorical').df
    test = indep('X1', 'W4', None, data, types=TYPES)
    print('\nXYZW - X1, W4:\n{}'.format(test))
    bnlearn = bnlearn_indep('X1', 'W4', None, data, types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# X1 and Z3 are dependent - 2x3 table
@requires_r_and_bnlearn
def test_bnlearn_indep_xyzw_ok3():
    data = Pandas.read(TESTDATA_DIR + '/simple/xyzw.csv',
                       dstype='categorical').df
    test = indep('X1', 'Z3', None, data, types=TYPES)
    print('\nXYZW - X1, Z3:\n{}'.format(test))
    bnlearn = bnlearn_indep('X1', 'Z3', None, data, types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# single conditioning variable
@requires_r_and_bnlearn
def test_bnlearn_indep_xyzw_ok4():
    data = Pandas.read(TESTDATA_DIR + '/simple/xyzw.csv',
                       dstype='categorical').df
    test = indep('X1', 'Y2', 'Z3', data, types=TYPES)
    print('\nXYZW - X1, Y2 | Z3:\n{}'.format(test))
    bnlearn = bnlearn_indep('X1', 'Y2', 'Z3', data, types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# two conditioning variables
@requires_r_and_bnlearn
def test_bnlearn_indep_xyzw_ok5():
    data = Pandas.read(TESTDATA_DIR + '/simple/xyzw.csv',
                       dstype='categorical').df
    test = indep('X1', 'Y2', ['Z3', 'W4'], data, types=TYPES)
    print('\nXYZW - X1, Y2 | Z3, W4:\n{}'.format(test))
    bnlearn = bnlearn_indep('X1', 'Y2', ['Z3', 'W4'], data, types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# generate zero row in a contingency table
@requires_r_and_bnlearn
def test_bnlearn_indep_xyzw_ok6():
    data = Pandas.read(TESTDATA_DIR + '/simple/xyzw.csv',
                       dstype='categorical').df
    indep('W4', 'Z3', ['X1', 'Y2'], data, types='x2')
    test = indep('Z3', 'W4', ['X1', 'Y2'], data, types=TYPES)
    print('\nXYZW - Z3, W4 | X1, Y2:\n{}'.format(test))
    bnlearn = bnlearn_indep('Z3', 'W4', ['X1', 'Y2'], data, types=TYPES)
    for type in TYPES:
        assert dicts_same(bnlearn[type].to_dict(), test[type].to_dict())


# --- Test check_test_params

# alpha not a float
def test_check_test_params_type_error_1():
    with pytest.raises(TypeError):
        check_test_params({'alpha': 1})
    with pytest.raises(TypeError):
        check_test_params({'alpha': 'wrong type'})


# alpha out of range
def test_check_test_params_value_error_1():
    with pytest.raises(ValueError):
        check_test_params({'alpha': 1E-20})
    with pytest.raises(ValueError):
        check_test_params({'alpha': 1.0})
    with pytest.raises(ValueError):
        check_test_params({'alpha': 0.0})


# sets defaults
def test_check_test_params_ok_1():
    assert check_test_params({}) == {'alpha': 0.05}


# accepts valid values
def test_check_test_params_ok_2():
    assert check_test_params({'alpha': 0.01}) == {'alpha': 0.01}
    assert check_test_params({'alpha': 0.1}) == {'alpha': 0.1}
    assert check_test_params({'alpha': 0.999}) == {'alpha': 0.999}
    assert check_test_params({'alpha': 1E-2}) == {'alpha': 1E-2}
    assert check_test_params({'alpha': 1E-3}) == {'alpha': 1E-3}
