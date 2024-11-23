
import pytest
from random import random
from os import remove

from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from core.graph import DAG
from core.bn import BN
from core.metrics import values_same
import testdata.example_bns as ex_bn


@pytest.fixture(scope="function")  # temp file, automatically removed
def bn():
    dag = DAG(['N1', 'N2'], [('N1', '->', 'N2')])
    data = Pandas.read(TESTDATA_DIR + '/simple/heckerman.csv',
                       dstype='categorical')
    return BN.fit(dag, data)


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.csv'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


def test_bn_generate_cases_type_error(bn):  # bad arg types
    with pytest.raises(TypeError):
        bn.generate_cases()
    with pytest.raises(TypeError):
        bn.generate_cases(3.7)
    with pytest.raises(TypeError):
        bn.generate_cases('a lot')
    with pytest.raises(TypeError):
        bn.generate_cases(True)
    with pytest.raises(TypeError):
        bn.generate_cases(4, 4)


def test_bn_generate_cases_value_error1(bn):  # bad number of cases
    with pytest.raises(ValueError):
        bn.generate_cases(0)
    with pytest.raises(ValueError):
        bn.generate_cases(-1)
    with pytest.raises(ValueError):
        bn.generate_cases(100000001)


def test_bn_generate_cases_value_error2(bn):  # non-existent directory
    with pytest.raises(FileNotFoundError):
        bn.generate_cases(2, TESTDATA_DIR + '/nonexistent/bad.csv')


# Test generating cases for discrete networks

def test_bn_generate_cases_heckerman_demo_1(bn):  # OK with heckerman N1 --> N2
    data = bn.generate_cases(10000)
    summary = data.groupby(['N1', 'N2']) \
        .size().reset_index().rename(columns={0: 'count'}) \
        .sort_values(bn.dag.nodes, ignore_index=True)

    assert len(summary.columns) == 3
    assert len(summary.index) == 4

    assert dict(summary.iloc[0]) == {'N1': '1', 'N2': '1', 'count': 2434}
    assert dict(summary.iloc[1]) == {'N1': '1', 'N2': '2', 'count': 1678}
    assert dict(summary.iloc[2]) == {'N1': '2', 'N2': '1', 'count': 2517}
    assert dict(summary.iloc[3]) == {'N1': '2', 'N2': '2', 'count': 3371}

    print('\n10,000 cases for N1 --> N2 Heckerman BN:')
    print(summary.to_string(index=False))


def test_bn_generate_cases_heckerman_demo_2():  # OK with unconnected graph?
    data = Pandas.read(TESTDATA_DIR + '/simple/heckerman.csv',
                       dstype='categorical')
    bn = BN.fit(DAG(['N1', 'N2'], []), data)
    data = bn.generate_cases(10000)
    summary = data.groupby(['N1', 'N2']) \
        .size().reset_index().rename(columns={0: 'count'}) \
        .sort_values(bn.dag.nodes, ignore_index=True)

    assert len(summary.columns) == 3
    assert len(summary.index) == 4

    assert dict(summary.iloc[0]) == {'N1': '1', 'N2': '1', 'count': 2062}
    assert dict(summary.iloc[1]) == {'N1': '1', 'N2': '2', 'count': 2050}
    assert dict(summary.iloc[2]) == {'N1': '2', 'N2': '1', 'count': 2934}
    assert dict(summary.iloc[3]) == {'N1': '2', 'N2': '2', 'count': 2954}

    print('\n10,000 cases for N1, N2 Heckerman BN:')
    print(summary.to_string(index=False))


def test_bn_generate_cases_cancer_demo():  # OK Cancer BN
    bn = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    data = bn.generate_cases(10000)
    summary = data.groupby(bn.dag.nodes) \
        .size().reset_index().rename(columns={0: 'count'}) \
        .sort_values(['count'] + bn.dag.nodes, ignore_index=True,
                     ascending=[False] + [True] * len(bn.dag.nodes))

    assert len(summary.columns) == 6
    assert len(summary.index) == 32

    assert dict(summary.iloc[0]) == \
        {'Cancer': 'False', 'Dyspnoea': 'False', 'Pollution': 'low',
         'Smoker': 'False', 'Xray': 'negative', 'count': 3533}
    assert dict(summary.iloc[9]) == \
        {'Cancer': 'False', 'Dyspnoea': 'False', 'Pollution': 'high',
         'Smoker': 'True', 'Xray': 'negative', 'count': 162}
    assert dict(summary.iloc[-1]) == \
        {'Cancer': 'True', 'Dyspnoea': 'True', 'Pollution': 'low',
         'Smoker': 'False', 'Xray': 'negative', 'count': 0}

    print('\n10,000 cases for Cancer BN:')
    print(summary.to_string(index=False))


def test_bn_generate_cases_cancer_write_ok(tmpfile):  # OK writing Cancer BN
    bn = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    cases = bn.generate_cases(10000, tmpfile)
    check = Pandas.read(tmpfile, dstype='categorical').df
    assert check.equals(cases)


# Check generating continuous data

def test_bn_generate_cases_x_10k_ok():
    data = ex_bn.x().generate_cases(10000)
    mean = data['X'].mean().item()
    sd = data['X'].std().item()
    assert values_same(mean, -0.002599, sf=4)
    assert values_same(sd, 1.003916, sf=7)

    # X defined to have mean 0 and SD 1.0

    print('\n\n10K of X has mean {:.6f} and S.D. {:.6f}'.format(mean, sd))


def test_bn_generate_cases_x_y_10k_ok():
    data = ex_bn.x_y().generate_cases(10000)
    mean = data['X'].mean().item()
    sd = data['X'].std().item()

    # X defined to have mean 0.2, SD 0.1

    assert values_same(mean, 0.199981, sf=6)
    assert values_same(sd, 0.099736, sf=5)
    print('\n\nX Y disconnected:')
    print('10K of X has mean {:.6f} and S.D. {:.6f}'.format(mean, sd))
    mean = data['Y'].mean().item()
    sd = data['Y'].std().item()

    # X defined to have mean -4.0, SD 2.0

    assert values_same(mean, -4.034616, sf=7)
    assert values_same(sd, 2.008486, sf=7)
    print('10K of Y has mean {:.6f} and S.D. {:.6f}'.format(mean, sd))


def test_bn_generate_cases_xy_10k_ok():
    data = ex_bn.xy().generate_cases(10000)
    mean = data['X'].mean().item()
    sd = data['X'].std().item()

    # X defined to have mean 2.0, SD 1.0

    assert values_same(mean, 1.999813, sf=7)
    assert values_same(sd, 0.997355, sf=6)
    print('\n\nX --> Y:')
    print('10K of X has mean {:.6f} and S.D. {:.6f}'.format(mean, sd))
    mean = data['Y'].mean().item()
    sd = data['Y'].std().item()

    # Y = 1.5 * X + Normal(0.5, 0.5) ==> mean = 1.5 * 2 + 0.5 = 3.5
    #                                ==> sd = sqrt(1.5**2 +0.5**2) = 1.581139

    assert values_same(mean, 3.491065, sf=7)
    assert values_same(sd, 1.576990, sf=7)
    print('10K of Y has mean {:.6f} and S.D. {:.6f}'.format(mean, sd))


def test_bn_generate_cases_xyz_10k_ok():
    data = ex_bn.xyz().generate_cases(10000)

    # X defined to have mean 0.0, SD 5.0

    mean = data['X'].mean().item()
    sd = data['X'].std().item()
    assert values_same(mean, -0.002438, sf=4)
    assert values_same(sd, 4.98589, sf=6)
    print('\n\nX --> Y --> Z:')
    print('10K of X has mean {:.6f} and S.D. {:.6f}'.format(mean, sd))

    # Y = Normal(-0.7, 0.5) -1.2 * X ==> mean = -0.7
    #     ==> sd = sqrt(1.2**2x5**2 +0.5**2) = 6.020797

    mean = data['Y'].mean().item()
    sd = data['Y'].std().item()
    assert values_same(mean, -0.699809, sf=6)
    assert values_same(sd, 5.998931, sf=7)
    print('10K of Y has mean {:.6f} and S.D. {:.6f}'.format(mean, sd))

    # Z = Normal(0.03, 0.05) + 0.3 * Y ==> mean = 0.03 - 0.3 * 0.7 = -0.18
    #     ==> sd = sqrt([0.3*6.020797]**2 + 0.05**2) = 1.806931

    mean = data['Z'].mean().item()
    sd = data['Z'].std().item()
    assert values_same(mean, -0.180399, sf=6)
    assert values_same(sd, 1.800566, sf=7)
    print('10K of Z has mean {:.6f} and S.D. {:.6f}'.format(mean, sd))


def test_bn_generate_cases_xy_zy_10k_ok():
    data = ex_bn.xy_zy().generate_cases(10000)

    # X defined to have mean -3.07, SD 0.45

    mean = data['X'].mean().item()
    sd = data['X'].std().item()
    assert values_same(mean, -3.070220, sf=7)
    assert values_same(sd, 0.448731, sf=6)
    print('\n\nX --> Y <-- Z:')
    print('10K of X has mean {:.6f} and S.D. {:.6f}'.format(mean, sd))

    # Z = Normal(6.2, 1.4)

    mean = data['Z'].mean().item()
    sd = data['Z'].std().item()
    assert values_same(mean, 6.192341, sf=7)
    assert values_same(sd, 1.410168, sf=7)
    print('10K of Z has mean {:.6f} and S.D. {:.6f}'.format(mean, sd))

    # Y = 1.2 * X - 0.4 * Z + Normal(-2.7, 1.5)
    # mean = 1.2 * - 3.07 - 0.4 * 6.2 - 2.7 = -8.864
    # sd = sqrt([1.2*0.45]**2 + [0.4*1.4]**2 + 1.5**2) = 1.6897

    mean = data['Y'].mean().item()
    sd = data['Y'].std().item()
    assert values_same(mean, -8.874899, sf=6)
    assert values_same(sd, 1.684662, sf=7)
    print('10K of Y has mean {:.6f} and S.D. {:.6f}'.format(mean, sd))
