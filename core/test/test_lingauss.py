
# Unit test Linear Gaussian Distribution

import pytest
import numpy as np
from pandas import DataFrame

from core.lingauss import LinGauss
from fileio.pandas import Pandas
from core.metrics import values_same


@pytest.fixture(scope="function")  # simple lg specification
def lg():
    return {'coeffs': {'A': 1.1, 'B': -0.25},
            'mean': 2.2,
            'sd': 0.34}


@pytest.fixture(scope="module")  # generate data used to test .fit()
def data():
    N = 10000000
    np.random.seed(42)
    t = 3.0 + 1.0 * np.random.randn(N)
    u = 1.0 + 3.0 * np.random.randn(N) - 4.0 * t
    v = 0.5 + 0.5 * np.random.randn(N) + 1.0 * t - 0.1 * u
    return Pandas(DataFrame({'T': t, 'U': u, 'V': v}))


# Test the LinGauss() constructor

def test_lingauss_type_error_1():  # no aruments
    with pytest.raises(TypeError):
        LinGauss()


def test_lingauss_type_error_2(lg):  # argument is not a dictionary
    with pytest.raises(TypeError):
        LinGauss(12)
    with pytest.raises(TypeError):
        LinGauss(True)
    with pytest.raises(TypeError):
        LinGauss([lg])


def test_lingauss_type_error_3(lg):  # lg has missing/extra keys
    lg.update({'extra': 23.2})
    with pytest.raises(TypeError):
        LinGauss(lg)
    lg.pop('extra')
    lg.pop('sd')
    with pytest.raises(TypeError):
        LinGauss(lg)
    lg.update({'invalid': 2.0})
    with pytest.raises(TypeError):
        LinGauss(lg)


def test_lingauss_type_error_4(lg):  # lg values have incorrect types
    lg.update({'coeffs': 23.1})
    with pytest.raises(TypeError):
        LinGauss(lg)
    lg.update({'coeffs': {'A': 3.0}, 'mean': 2})
    with pytest.raises(TypeError):
        LinGauss(lg)
    lg.update({'mean': -3.4, 'sd': True})
    with pytest.raises(TypeError):
        LinGauss(lg)


def test_lingauss_type_error_5(lg):  # coeffs have wrong key types
    lg.update({'coeffs': {1: 2.5}})
    with pytest.raises(TypeError):
        LinGauss(lg)
    lg.update({'coeffs': {'good': 2.7, True: 1.0}})
    with pytest.raises(TypeError):
        LinGauss(lg)


def test_lingauss_type_error_6(lg):  # coeffs have wrong value types
    lg.update({'coeffs': {'ok': 2.5, 'invalid': 3}})
    with pytest.raises(TypeError):
        LinGauss(lg)
    lg.update({'coeffs': {'ok': 2.5, 'invalid': True}})
    with pytest.raises(TypeError):
        LinGauss(lg)
    lg.update({'coeffs': {'ok': 2.5, 'invalid': (2.0, )}})
    with pytest.raises(TypeError):
        LinGauss(lg)
    lg.update({'coeffs': {'ok': 2.5, 'invalid': True}})
    with pytest.raises(TypeError):
        LinGauss(lg)
    lg.update({'coeffs': {'ok': 2.5, 'invalid': True}})
    with pytest.raises(TypeError):
        LinGauss(lg)


def test_lingauss_value_error_1(lg):  # SD is negative
    lg.update({'sd': -0.1})
    with pytest.raises(ValueError):
        LinGauss(lg)
    lg.update({'sd': -1.0E-20})
    with pytest.raises(ValueError):
        LinGauss(lg)


def test_lingauss_lg_1_ok(lg):  # lg fixture OK
    cnd = LinGauss(lg)
    assert isinstance(cnd, LinGauss)
    assert cnd.coeffs == {'A': 1.1, 'B': -0.25}
    assert cnd.mean == 2.2
    assert cnd.sd == 0.34
    assert '{}'.format(cnd) == '1.1*A-0.25*B+Normal(2.2,0.34)'
    print('\nLinGauss: {}'.format(cnd))


def test_lingauss_lg_2_ok(lg):  # leading negative coefficient
    lg.update({'coeffs': {'A': -0.02, 'B': -0.25}})
    cnd = LinGauss(lg)
    assert isinstance(cnd, LinGauss)
    assert cnd.coeffs == {'A': -0.02, 'B': -0.25}
    assert cnd.mean == 2.2
    assert cnd.sd == 0.34
    assert '{}'.format(cnd) == '-0.02*A-0.25*B+Normal(2.2,0.34)'
    print('\nLinGauss: {}'.format(cnd))


def test_lingauss_lg_3_ok(lg):  # v small coeffs dropped in string
    lg.update({'coeffs': {'A': -1E-11, 'B': -0.25}})
    cnd = LinGauss(lg)
    assert isinstance(cnd, LinGauss)
    assert cnd.coeffs == {'A': -1E-11, 'B': -0.25}
    assert cnd.mean == 2.2
    assert cnd.sd == 0.34
    assert '{}'.format(cnd) == '-0.25*B+Normal(2.2,0.34)'
    print('\nLinGauss: {}'.format(cnd))


def test_lingauss_lg_4_ok(lg):  # rounding to 10 s.f.. string
    lg.update({'coeffs': {'A': 1.0123456789, 'B': -0.00987654321}})
    cnd = LinGauss(lg)
    assert isinstance(cnd, LinGauss)
    assert cnd.coeffs == {'A': 1.0123456789, 'B': -0.00987654321}
    assert cnd.mean == 2.2
    assert cnd.sd == 0.34
    assert '{}'.format(cnd) == '1.012345679*A-0.00987654321*B+Normal(2.2,0.34)'
    print('\nLinGauss: {}'.format(cnd))


def test_lingauss_lg_5_ok(lg):  # orphan node OK
    lg.update({'coeffs': {}})
    cnd = LinGauss(lg)
    assert isinstance(cnd, LinGauss)
    assert cnd.coeffs == {}
    assert cnd.mean == 2.2
    assert cnd.sd == 0.34
    assert '{}'.format(cnd) == 'Normal(2.2,0.34)'
    print('\nLinGauss: {}'.format(cnd))


# Test equality

def test_eq_1_ok(lg):  # LinGauss equals itself
    lg = LinGauss(lg)
    assert lg == lg


def test_eq_2_ok(lg):  # LinGauss equals identical LinGauss
    lg1 = LinGauss(lg)
    lg2 = LinGauss(lg)
    assert id(lg1) != id(lg2)
    assert lg1 == lg2


def test_eq_3_ok(lg):  # LinGauss equals very similar LinGauss
    lg1 = LinGauss(lg)
    lg.update({'mean': 2.2000000001})
    lg2 = LinGauss(lg)
    assert id(lg1) != id(lg2)
    assert lg1 == lg2
    lg.update({'mean': 2.2, 'sd': 0.33999999999})
    lg3 = LinGauss(lg)
    assert id(lg1) != id(lg3)
    assert lg1 == lg3
    lg.update({'coeffs': {'A': 1.1000000001, 'B': -0.24999999996}})
    lg4 = LinGauss(lg)
    assert id(lg1) != id(lg4)
    assert lg1 == lg4


# Test inequality

def test_ne_ok_1(lg):  # not equal to non LinGauss objects
    lg = LinGauss(lg)
    assert lg != 1
    assert lg != {'mean': 0.0, 'sd': 1.0, 'coeffs': {}}
    assert lg is not None
    assert lg is not True
    assert lg is not False


def test_ne_ok_2(lg):  # not equal if mean different
    lg1 = LinGauss(lg)
    lg.update({'mean': 2.199999999})
    lg2 = LinGauss(lg)
    assert id(lg1) != id(lg2)
    assert lg1 != lg2


def test_ne_ok_3(lg):  # not equal if sd different
    lg1 = LinGauss(lg)
    lg.update({'sd': 0.3400000001})
    lg2 = LinGauss(lg)
    assert id(lg1) != id(lg2)
    assert lg1 != lg2


def test_ne_ok_4(lg):  # not equal if coeff key different
    lg1 = LinGauss(lg)
    lg.update({'coeffs': {'AA': 1.1, 'B': -0.25}})
    lg2 = LinGauss(lg)
    assert id(lg1) != id(lg2)
    assert lg1 != lg2


def test_ne_ok_5(lg):  # not equal if missing coeff
    lg1 = LinGauss(lg)
    lg.update({'coeffs': {'B': -0.25}})
    lg2 = LinGauss(lg)
    assert id(lg1) != id(lg2)
    assert lg1 != lg2


def test_ne_ok_6(lg):  # not equal if extra coeff
    lg1 = LinGauss(lg)
    lg.update({'coeffs': {'A': 1.1, 'B': -0.25, 'C': 3.9}})
    lg2 = LinGauss(lg)
    assert id(lg1) != id(lg2)
    assert lg1 != lg2


def test_ne_ok_7(lg):  # not equal if coeff value different
    lg1 = LinGauss(lg)
    lg.update({'coeffs': {'A': 1.1, 'B': -0.2499999999}})
    lg2 = LinGauss(lg)
    assert id(lg1) != id(lg2)
    assert lg1 != lg2


# Test Data fitting

def test_fit_type_error_1():  # no parameters specified
    with pytest.raises(TypeError):
        LinGauss.fit()


def test_fit_type_error_2(data):  # node is not a string
    with pytest.raises(TypeError):
        LinGauss.fit(None, None, data)
    with pytest.raises(TypeError):
        LinGauss.fit(12, None, data)
    with pytest.raises(TypeError):
        LinGauss.fit(['T'], None, data)


def test_fit_type_error_3(data):  # parents is not None or tuple of strings
    with pytest.raises(TypeError):
        LinGauss.fit('T', False, data)
    with pytest.raises(TypeError):
        LinGauss.fit('U', ['T'], data)
    with pytest.raises(TypeError):
        LinGauss.fit('U', 'T', data)
    with pytest.raises(TypeError):
        LinGauss.fit('U', tuple(), data)
    with pytest.raises(TypeError):
        LinGauss.fit('U', (1, 2), data)


def test_fit_type_error_4(data):  # data is not Data type
    with pytest.raises(TypeError):
        LinGauss.fit('U', ('T', ), True)
    with pytest.raises(TypeError):
        LinGauss.fit('U', ('T', ), data.sample)


def test_fit_value_error_1(data):  # node is one of parents
    with pytest.raises(ValueError):
        LinGauss.fit('U', ('T', 'U'), data)


def test_fit_value_error_2(data):  # node or parent is not in data
    with pytest.raises(ValueError):
        LinGauss.fit('invalid', ('T', 'U'), data)
    with pytest.raises(ValueError):
        LinGauss.fit('T', ('invalid', 'U'), data)


def test_fit_1_ok(data):  # Check fitted values for T (orphan) node
    lg, _ = LinGauss.fit('T', None, data)

    # t = 3.0 + 1.0 * np.random.randn(N)

    assert values_same(lg[1]['mean'], 3.00, sf=4)
    assert values_same(lg[1]['sd'], 1.00, sf=5)
    assert lg[1]['coeffs'] == {}
    print('\nT: mean: {:.4f}, s.d.: {:.4f}'
          .format(lg[1]['mean'], lg[1]['sd']))


def test_fit_2_ok(data):  # Check fitted values for U with parent T
    lg, _ = LinGauss.fit('U', ('T', ), data)

    # u = 1.0 + 3.0 * np.random.randn(N) - 4.0 * t

    assert values_same(lg[1]['mean'], 1.000, sf=4)
    assert values_same(lg[1]['sd'], 3.000, sf=4)
    assert set(lg[1]['coeffs']) == {'T'}
    assert values_same(lg[1]['coeffs']['T'], -4.0, sf=4)
    print('\nU: mean: {:.4f}, s.d.: {:.4f}, T: {:.4f}'
          .format(lg[1]['mean'], lg[1]['sd'], lg[1]['coeffs']['T']))


def test_fit_3_ok(data):  # Check fitted values for V with parents T & U
    lg, _ = LinGauss.fit('V', ('T', 'U'), data)

    # v = 0.5 + 0.5 * np.random.randn(N) + 1.0 * t - 0.1 * u

    assert values_same(lg[1]['mean'], 0.50, sf=2)
    assert values_same(lg[1]['sd'], 0.500, sf=3)
    assert set(lg[1]['coeffs']) == {'T', 'U'}
    assert values_same(lg[1]['coeffs']['T'], 1.0, sf=3)
    assert values_same(lg[1]['coeffs']['U'], -0.1, sf=4)
    print('\nU: mean: {:.4f}, s.d.: {:.4f}, T: {:.4f}, U: {:.4f}'
          .format(lg[1]['mean'], lg[1]['sd'], lg[1]['coeffs']['T'],
                  lg[1]['coeffs']['U']))


# test to_spec function including name mapping

def test_to_spec_type_error_1(lg):  # no arguments
    with pytest.raises(TypeError):
        LinGauss(lg).to_spec()


def test_to_spec_type_error_2(lg):  # name_map not a dictionary
    with pytest.raises(TypeError):
        LinGauss(lg).to_spec(False)
    with pytest.raises(TypeError):
        LinGauss(lg).to_spec(None)
    with pytest.raises(TypeError):
        LinGauss(lg).to_spec(1)
    with pytest.raises(TypeError):
        LinGauss(lg).to_spec(23.2)
    with pytest.raises(TypeError):
        LinGauss(lg).to_spec(['A'])


def test_to_spec_type_error_3(lg):  # name_map keys not strings
    with pytest.raises(TypeError):
        LinGauss(lg).to_spec({1: 'A', 'B': 'S'})


def test_to_spec_type_error_4(lg):  # name_map values not strings
    with pytest.raises(TypeError):
        LinGauss(lg).to_spec({'A': 'A', 'B': 0.05})


def test_to_spec_value_error_1(lg):  # name_map doesn't include all coeff keys
    with pytest.raises(ValueError):
        LinGauss(lg).to_spec({'A': 'X', 'C': 'Y'})


def test_to_spec_1_ok(lg):  # names remaining the same
    lg1 = LinGauss(lg)
    name_map = {n: n for n in lg1.coeffs}
    spec = lg1.to_spec(name_map)
    print('\n\nSpec is {}'.format(spec))
    assert spec == {'coeffs': {'A': 1.1, 'B': -0.25},
                    'mean': 2.2,
                    'sd': 0.34}


def test_to_spec_2_ok(lg):  # mapping names
    lg1 = LinGauss(lg)
    name_map = {'A': 'AA', 'B': 'BB'}
    spec = lg1.to_spec(name_map)
    print('\n\nSpec is {}'.format(spec))
    assert spec == {'coeffs': {'AA': 1.1, 'BB': -0.25},
                    'mean': 2.2,
                    'sd': 0.34}
