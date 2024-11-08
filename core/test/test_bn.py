
import pytest

from core.bn import BN
from core.cpt import CPT
from core.lingauss import LinGauss
from core.metrics import values_same
import testdata.example_dags as dag
import testdata.example_bns as bn
from fileio.common import TESTDATA_DIR, EXPTS_DIR
from fileio.pandas import Pandas


@pytest.fixture(scope="module")  # simple Data object
def data():
    return Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', dstype='categorical')


def test_bn_type_error_1_():  # bad argument types
    with pytest.raises(TypeError):
        BN()
    with pytest.raises(TypeError):
        BN(32)
    with pytest.raises(TypeError):
        BN('not', 'right')
    with pytest.raises(TypeError):
        BN(dag.ab(), 'not right')


def test_bn_type_error_2_():  # bad type within nodes
    with pytest.raises(TypeError):
        BN([1], [])


def test_bn_type_error_3_():  # bad type for cpts arg
    with pytest.raises(TypeError):
        BN(['A', 'B'], [('A', '->', 'B')], [])


def test_bn_type_error_4_():  # bad type within cpts values
    with pytest.raises(TypeError):
        BN(['A', 'B'], [('A', '->', 'B')], {'A': 3, 'B': 4})


def test_bn_value_error_1_():  # nodes in DAG and cpts don't match
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'C': (CPT, [({'A': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'A': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.ab(), cpts)


def test_bn_value_error_2_():  # cpts keys don't match DAG nodes
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'C': (CPT, [({'A': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'A': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.ab(), cpts)


def test_bn_value_error_3_():  # cpt keys vary
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, [({'A': '0'}, {'0': 0.2, '1': 0.8}),
                        ({}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.ab(), cpts)


def test_bn_value_error_4_():  # cpt pmf keys vary
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, [({'A': '0'}, {'0': 0.2, '2': 0.8}),
                        ({'A': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.ab(), cpts)


def test_bn_value_error_5_():  # cpt pmf keys vary
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, [({'A': '0'}, {'0': 0.2}),
                        ({'A': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.ab(), cpts)


def test_bn_value_error_6_():  # parents in CPT key don't match DAG
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, [({'C': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'C': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.ab(), cpts)


def test_bn_value_error_7_():  # parents in CPT key don't match DAG
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, [({'B': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'B': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.ab(), cpts)


def test_bn_value_error_8_():  # parents in CPT key don't match DAG
    cpts = {'B': (CPT, {'0': 0.25, '1': 0.75}),
            'A': (CPT, [({'B': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'B': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.ab(), cpts)


def test_bn_value_error_9_():  # parents in CPT key don't match DAG
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, [({'A': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'A': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.a_b(), cpts)


def test_bn_value_error_10_():  # parents in CPT key don't match DAG
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, [({'B': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'B': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.a_b(), cpts)


def test_bn_value_error_11_():  # parents in CPT key don't match DAG
    cpts = {'B': (CPT, [({'A': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'A': '1'}, {'0': 0.7, '1': 0.3})]),
            'A': (CPT, [({'B': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'B': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.ab(), cpts)


def test_bn_value_error_12_():  # parents in CPT key don't match DAG
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, [({'A': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'A': '1'}, {'0': 0.7, '1': 0.3})]),
            'C': (CPT, {'0': 0.25, '1': 0.75})}
    with pytest.raises(ValueError):
        BN(dag.abc(), cpts)


def test_bn_value_error_13_():  # parents in CPT key don't match DAG
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, [({'A': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'A': '1'}, {'0': 0.7, '1': 0.3})]),
            'C': (CPT, [({'A': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'A': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.abc(), cpts)


def test_bn_value_error_14_():  # parents in CPT key don't match DAG
    cpts = {'A': (CPT, [({'B': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'B': '1'}, {'0': 0.7, '1': 0.3})]),
            'B': (CPT, {'0': 0.25, '1': 0.75}),
            'C': (CPT, [({'B': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'B': '1'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.abc(), cpts)


def test_bn_value_error_15_():  # values in CPT key don't match parent values
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, [({'A': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'A': '1'}, {'0': 0.7, '1': 0.3})]),
            'C': (CPT, [({'B': '2'}, {'0': 0.2, '1': 0.8}),
                        ({'B': '3'}, {'0': 0.7, '1': 0.3})])}
    with pytest.raises(ValueError):
        BN(dag.abc(), cpts)


def test_bn_value_error_16_():  # values in CPT key don't match parent values
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, {'0': 0.40, '1': 0.60}),
            'C': (CPT, [({'A': '0', 'B': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'A': '0', 'B': '2'}, {'0': 0.7, '1': 0.3}),
                        ({'A': '1', 'B': '0'}, {'0': 0.1, '1': 0.9}),
                        ({'A': '1', 'B': '2'}, {'0': 0.4, '1': 0.6})])}
    with pytest.raises(ValueError):
        BN(dag.ac_bc(), cpts)


def test_bn_value_error_17_():  # values in CPT key don't match parent values
    cpts = {'A': (CPT, {'2': 0.25, '3': 0.75}),
            'B': (CPT, {'0': 0.40, '1': 0.60}),
            'C': (CPT, [({'A': '0', 'B': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'A': '0', 'B': '1'}, {'0': 0.7, '1': 0.3}),
                        ({'A': '1', 'B': '0'}, {'0': 0.1, '1': 0.9}),
                        ({'A': '1', 'B': '1'}, {'0': 0.4, '1': 0.6})])}
    with pytest.raises(ValueError):
        BN(dag.ac_bc(), cpts)


def test_bn_value_error_18_():  # not all parental combos in CPT
    cpts = {'A': (CPT, {'0': 0.25, '1': 0.75}),
            'B': (CPT, {'0': 0.40, '1': 0.60}),
            'C': (CPT, [({'A': '0', 'B': '0'}, {'0': 0.2, '1': 0.8}),
                        ({'A': '0', 'B': '1'}, {'0': 0.7, '1': 0.3}),
                        ({'A': '1', 'B': '1'}, {'0': 0.4, '1': 0.6})])}
    with pytest.raises(ValueError):
        BN(dag.ac_bc(), cpts)


def test_bn_value_error_19_():  # nodes in DAG and LinGauss don't match
    lgs = {'A': (LinGauss, {'coeffs': {}, 'mean': 0.0, 'sd': 1.0}),
           'B': (LinGauss, {'coeffs': {}, 'mean': 0.0, 'sd': 1.0})}
    with pytest.raises(ValueError):
        BN(dag.ab(), lgs)


def test_bn_value_error_20_():  # nodes in DAG and LinGauss don't match
    lgs = {'A': (LinGauss, {'coeffs': {}, 'mean': 0.0, 'sd': 1.0}),
           'B': (LinGauss, {'coeffs': {'A': 1.0}, 'mean': 0.0, 'sd': 1.0})}
    with pytest.raises(ValueError):
        BN(dag.a_b(), lgs)


def test_bn_value_error_21_():  # nodes in DAG and LinGauss don't match
    lgs = {'A': (LinGauss, {'coeffs': {}, 'mean': 0.0, 'sd': 1.0}),
           'B': (LinGauss, {'coeffs': {'C': 1.0}, 'mean': 0.0, 'sd': 1.0})}
    with pytest.raises(ValueError):
        BN(dag.ab(), lgs)


def test_bn_fit_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        BN.fit()


def test_bn_fit_type_error_2(data):  # invalid DAG argument
    with pytest.raises(TypeError):
        BN.fit(1, data)
    with pytest.raises(TypeError):
        BN.fit([dag.ab()], data)
    with pytest.raises(TypeError):
        BN.fit(['A', 'B'], data)


def test_bn_fit_value_error_1(data):  # column mismatch
    with pytest.raises(ValueError):
        BN.fit(dag.a(), data)


def test_bn_empty_ok():
    bn = BN(dag.empty(), {})
    assert isinstance(bn, BN)


# Create categorical BNs with constructor

def test_bn_a_1_ok():  # A
    bn.a(bn.a())


def test_bn_ab_1_ok():
    bn.ab(bn.ab())


# Create continuous BNs with constructor

def test_bn_x_1_ok():  # X
    bn.x(bn.x())


def test_bn_xy_1_ok():  # X --> Y
    bn.xy(bn.xy())


def test_bn_x_y_1_ok():  # X   Y
    bn.x_y(bn.x_y())


def test_bn_xyz_1_ok():  # X --> Y --> Z
    bn.xyz(bn.xyz())


def test_bn_xy_zy_1_ok():  # X --> Y <-- Z
    bn.xy_zy(bn.xy_zy())


# Check fitting BN to categorical data

def test_bn_fit_ab_3_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/ab_3.csv', dstype='categorical')
    bn = BN.fit(dag.ab(), data)
    dag.ab(bn.dag)  # check DAG in BN is as expected
    assert bn.cnds['A'] == CPT({'1': 0.666666667, '0': 0.333333333})
    assert bn.cnds['B'] == CPT([({'A': '0'}, {'0': 1.0, '1': 0.0}),
                                ({'A': '1'}, {'0': 0.5, '1': 0.5})])
    assert bn.free_params == 3
    assert bn.estimated_pmfs == {}


def test_bn_fit_ac_bc_4_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/abc_4.csv',
                       dstype='categorical')
    bn = BN.fit(dag.ac_bc(), data)
    dag.ac_bc(bn.dag)  # check DAG in BN is as expected
    assert bn.cnds['A'] == CPT({'1': 0.75, '0': 0.25})
    assert bn.cnds['B'] == CPT({'1': 0.75, '0': 0.25})
    assert bn.cnds['C'] == \
        CPT([({'A': '0', 'B': '0'}, {'0': 0.75, '1': 0.25}),
             ({'A': '0', 'B': '1'}, {'0': 1.0, '1': 0.0}),
             ({'A': '1', 'B': '0'}, {'0': 1.0, '1': 0.0}),
             ({'A': '1', 'B': '1'}, {'0': 0.5, '1': 0.5})])
    assert bn.free_params == 6
    assert bn.estimated_pmfs == {'C': 1}


def test_bn_fit_ac_bc_5_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/abc_5.csv',
                       dstype='categorical')
    bn = BN.fit(dag.ac_bc(), data)
    dag.ac_bc(bn.dag)  # check DAG in BN is as expected
    assert bn.cnds['A'] == CPT({'1': 0.6, '0': 0.4})
    assert bn.cnds['B'] == CPT({'1': 0.6, '0': 0.4})
    assert bn.cnds['C'] == \
        CPT([({'A': '0', 'B': '0'}, {'0': 1.0, '1': 0.0}),
             ({'A': '0', 'B': '1'}, {'0': 1.0, '1': 0.0}),
             ({'A': '1', 'B': '0'}, {'0': 1.0, '1': 0.0}),
             ({'A': '1', 'B': '1'}, {'0': 0.5, '1': 0.5})])
    assert bn.free_params == 6
    assert bn.estimated_pmfs == {}


# Check fitting BN to Gaussian data

def test_bn_fit_xy_1_ok():  # X --> Y, 10K

    # X = Normal(2.0,1.0)
    # Y = 1.5*X+Normal(0.5,0.5)

    bn = BN.read(TESTDATA_DIR + '/xdsl/xy.xdsl')
    data = Pandas(df=bn.generate_cases(10000))

    bn = BN.fit(bn.dag, data)

    assert values_same(bn.cnds['X'].mean, 2.000, sf=4)
    assert values_same(bn.cnds['X'].sd, 1.00, sf=2)
    assert bn.cnds['X'].coeffs == {}

    assert values_same(bn.cnds['Y'].mean, 0.5, sf=1)
    assert values_same(bn.cnds['Y'].sd, 0.50, sf=2)
    assert set(bn.cnds['Y'].coeffs) == {'X'}
    assert values_same(bn.cnds['Y'].coeffs['X'], 1.50, sf=3)


def test_bn_fit_xyz_1_ok():  # X --> Y --> Z, 10K

    # X = Normal(0.0,1.0)
    # Y = 1.5*X+Normal(0.5,0.5)
    # Z = -2.0*Y+Normal(-2.0,0.2)

    bn = BN.read(TESTDATA_DIR + '/xdsl/xyz.xdsl')
    data = Pandas(df=bn.generate_cases(10000))

    bn = BN.fit(bn.dag, data)
    print('\nX = {}'.format(bn.cnds['X']))
    print('Y = {}'.format(bn.cnds['Y']))
    print('Z = {}'.format(bn.cnds['Z']))

    assert round(bn.cnds['X'].mean, 3) == 0.000
    assert values_same(bn.cnds['X'].sd, 1.00, sf=2)
    assert bn.cnds['X'].coeffs == {}

    assert values_same(bn.cnds['Y'].mean, 0.5, sf=2)
    assert values_same(bn.cnds['Y'].sd, 0.50, sf=2)
    assert set(bn.cnds['Y'].coeffs) == {'X'}
    assert values_same(bn.cnds['Y'].coeffs['X'], 1.50, sf=2)

    assert values_same(bn.cnds['Z'].mean, -2.00, sf=3)
    assert values_same(bn.cnds['Z'].sd, 0.200, sf=3)
    assert set(bn.cnds['Z'].coeffs) == {'Y'}
    assert values_same(bn.cnds['Z'].coeffs['Y'], -2.000, sf=4)


def test_bn_fit_xy_zy_1_ok():  # X --> Y --> Z, 100K

    # X = Normal(0.0, 1.0)
    # Y = 1.5*X - 2.2*Z + Normal(0.5, 0.5)
    # Z = Normal(-2.0, 0.2)

    bn = BN.read(TESTDATA_DIR + '/xdsl/xy_zy.xdsl')
    data = Pandas(df=bn.generate_cases(100000))

    bn = BN.fit(bn.dag, data)
    print('\nX = {}'.format(bn.cnds['X']))
    print('Y = {}'.format(bn.cnds['Y']))
    print('Z = {}'.format(bn.cnds['Z']))

    assert round(bn.cnds['X'].mean, 2) == 0.00
    assert values_same(bn.cnds['X'].sd, 1.00, sf=2)
    assert bn.cnds['X'].coeffs == {}

    assert values_same(bn.cnds['Y'].mean, 0.5, sf=1)
    assert values_same(bn.cnds['Y'].sd, 0.50, sf=2)
    assert set(bn.cnds['Y'].coeffs) == {'X', 'Z'}
    assert values_same(bn.cnds['Y'].coeffs['X'], 1.50, sf=3)
    assert values_same(bn.cnds['Y'].coeffs['Z'], -2.2, sf=2)

    assert values_same(bn.cnds['Z'].mean, -2.000, sf=4)
    assert values_same(bn.cnds['Z'].sd, 0.200, sf=3)
    assert bn.cnds['Z'].coeffs == {}


def test_bn_fit_sachs_c_ok():  # Fitting sachs to real cont data

    bn = BN.read(TESTDATA_DIR + '/discrete/small/sachs.dsc')
    print(bn.dag)
    data = Pandas.read(EXPTS_DIR + '/realdata/sachs_2005_cont.data.gz',
                       dstype='continuous')
    print()
    print(data.df)
    print()

    fitted = BN.fit(bn.dag, data)
    print('\nLearnt functions are:\n')
    for node, cnd in fitted.cnds.items():
        print('{}: {}'.format(node, cnd))

    # fitted.write(EXPTS_DIR + '/bn/xdsl/sachs_c.xdsl')

    ref = BN.read(EXPTS_DIR + '/bn/xdsl/sachs_c.xdsl')
    assert ref == fitted


def test_bn_fit_covid_c_ok():  # Fitting covid to real cont data

    bn = BN.read(TESTDATA_DIR + '/discrete/medium/covid.dsc')
    print(bn.dag)
    data = Pandas.read(EXPTS_DIR + '/realdata/covid_cont.data.gz',
                       dstype='continuous')
    print()
    print(data.df)
    print()

    fitted = BN.fit(bn.dag, data)
    print('\nLearnt functions are:\n')
    for node, cnd in fitted.cnds.items():
        print('{}: {}'.format(node, cnd))

    fitted.write(EXPTS_DIR + '/bn/xdsl/covid_c.xdsl')

    # ref = BN.read(EXPTS_DIR + '/bn/xdsl/covid_c.xdsl')
    # assert ref == fitted
