
#   Test BN.remove_single_valued() which removes single-valued variables
#   from a BN.

import pytest
from pandas import DataFrame

from core.bn import BN
from core.graph import DAG
from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
import testdata.example_dags as ex_dag


def test_bn_remove_single_valued_type_error_1():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    with pytest.raises(TypeError):
        bn.remove_single_valued()
    with pytest.raises(TypeError):
        bn.remove_single_valued(52.1)
    with pytest.raises(TypeError):
        bn.remove_single_valued({'A': ['0'], 'B': ['1']})


def test_bn_remove_single_valued_value_error_1():  # one row only
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = DataFrame({'A': ['0'], 'B': ['1'], 'C': ['1']}, dtype='category')
    with pytest.raises(ValueError):
        bn.remove_single_valued(data)


def test_bn_remove_single_valued_value_error_2():  # all vars single-valued
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = DataFrame({'A': ['0', '0'], 'B': ['1', '1'], 'C': ['1', '1']},
                     dtype='category')
    with pytest.raises(ValueError):
        bn.remove_single_valued(data)


def test_bn_remove_single_valued_value_error_3():  # one var multi-valued
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = DataFrame({'A': ['0', '0'], 'B': ['0', '1'], 'C': ['1', '1']},
                     dtype='category')
    with pytest.raises(ValueError):
        bn.remove_single_valued(data)


def test_bn_remove_single_valued_abc_ok_1():  # no variables need removing
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = DataFrame({'A': ['0', '1'], 'B': ['1', '0'], 'C': ['1', '0']},
                     dtype='category')
    new_bn, new_data, removed = bn.remove_single_valued(data)
    assert removed == []
    assert bn == new_bn
    assert not len(new_data.compare(data).columns)


def test_bn_remove_single_valued_abc_ok_2():  # one value needs removing
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = DataFrame({'A': ['0', '1'], 'B': ['1', '0'], 'C': ['1', '1']},
                     dtype='category')
    new_bn, new_data, removed = bn.remove_single_valued(data)
    assert removed == ['C']
    expected_data = DataFrame({'A': ['0', '1'], 'B': ['1', '0']},
                              dtype='category')
    assert not len(new_data.compare(expected_data).columns)
    expected_data = Pandas(df=expected_data)
    print(type(expected_data))
    return
    expected_bn = BN.fit(ex_dag.ab(), expected_data)
    assert expected_bn == new_bn


def test_bn_remove_single_valued_abc_ok_3():  # one value needs removing
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = DataFrame({'A': ['0', '0'], 'B': ['1', '0'], 'C': ['1', '0']},
                     dtype='category')
    new_bn, new_data, removed = bn.remove_single_valued(data)
    assert removed == ['A']
    expected_data = DataFrame({'B': ['1', '0'], 'C': ['1', '0']},
                              dtype='category')
    assert not len(new_data.compare(expected_data).columns)
    expected_data = Pandas(df=expected_data)
    expected_bn = BN.fit(DAG(['B', 'C'], [('B', '->', 'C')]), expected_data)
    assert expected_bn == new_bn


def test_bn_remove_single_valued_cancer_ok_1():  # no variables need removing
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = bn.generate_cases(100)
    new_bn, new_data, removed = bn.remove_single_valued(data)
    assert removed == []
    assert bn == new_bn
    assert not len(new_data.compare(data).columns)


def test_bn_remove_single_valued_cancer_ok_2():  # Xray need removing
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = bn.generate_cases(100).assign(Xray='negative')
    new_bn, new_data, removed = bn.remove_single_valued(data)
    assert removed == ['Xray']
    expected_data = data.drop(labels=['Xray'], axis='columns').copy()
    assert not len(new_data.compare(expected_data).columns)
    expected_data = Pandas(df=expected_data)
    expected_bn = BN.fit(DAG(['Smoker', 'Cancer', 'Pollution', 'Dyspnoea'],
                             [('Smoker', '->', 'Cancer'),
                              ('Pollution', '->', 'Cancer'),
                              ('Cancer', '->', 'Dyspnoea')]),
                         expected_data)
    assert expected_bn == new_bn


def test_bn_remove_single_valued_cancer_ok_3():  # Xray, Smoker need removing
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = bn.generate_cases(100).assign(Xray='negative', Smoker='False')
    new_bn, new_data, removed = bn.remove_single_valued(data)
    assert removed == ['Smoker', 'Xray']
    expected_data = data.drop(labels=['Xray', 'Smoker'], axis='columns').copy()
    assert not len(new_data.compare(expected_data).columns)
    expected_data = Pandas(df=expected_data)
    expected_bn = BN.fit(DAG(['Cancer', 'Pollution', 'Dyspnoea'],
                             [('Pollution', '->', 'Cancer'),
                              ('Cancer', '->', 'Dyspnoea')]),
                         expected_data)
    assert expected_bn == new_bn


@pytest.mark.slow
def test_bn_remove_single_valued_pathfinder_1K_ok():  # Pathfinder, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/verylarge/pathfinder.dsc')
    data = bn.generate_cases(1000)
    _, _, removed = bn.remove_single_valued(data)
    assert removed == ['F13', 'F15', 'F27', 'F69', 'F72', 'F75']


@pytest.mark.slow
def test_bn_remove_single_valued_pathfinder_2K_ok():  # Pathfinder, 2K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/verylarge/pathfinder.dsc')
    data = bn.generate_cases(2000)
    _, _, removed = bn.remove_single_valued(data)
    assert removed == ['F15', 'F27', 'F72', 'F75']
