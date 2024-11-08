
import pytest

from core.metrics import values_same
from fileio.compare import compare_all
from fileio.common import TESTDATA_DIR


def test_core_metrics_compare_all_type_error():
    with pytest.raises(TypeError):
        compare_all()
    with pytest.raises(TypeError):
        compare_all(1)
    with pytest.raises(TypeError):
        compare_all('wrong', 3.7)
    with pytest.raises(TypeError):
        compare_all('wrong', metric='f1-b', bayesys=2)


def test_core_metrics_compare_all_value_error1():
    with pytest.raises(FileNotFoundError):  # non-existent directory
        compare_all('doesnotexist', metric='f1-b', bayesys='v1.5+')
    with pytest.raises(FileNotFoundError):  # dir argument is a file
        compare_all(TESTDATA_DIR + '/dhs/d8atr/d8atr_fges.csv', metric='f1-b',
                    bayesys='v1.5+')


def test_core_metrics_compare_all_value_error2():
    with pytest.raises(ValueError):  # bad bayesys value
        compare_all(TESTDATA_DIR + '/dhs/d7a', metric='f1-b',
                    bayesys='unsupported')


def test_core_metrics_compare_all_value_error3():
    with pytest.raises(ValueError):  # bad metric value
        compare_all(TESTDATA_DIR + '/dhs/d7a', metric='unsupported',
                    bayesys='v1.5+')


def test_core_metrics_compare_all_d7a_no_know():
    assert values_same(compare_all(TESTDATA_DIR + '/dhs/d7a', metric='bsf',
                                   bayesys='v1.5+'), 0.356, sf=3) is True


def test_core_metrics_compare_all_d7at():
    assert values_same(compare_all(TESTDATA_DIR + '/dhs/d7at', metric='bsf',
                                   bayesys='v1.5+'), 0.403, sf=3) is True


def test_core_metrics_compare_all_d8atr():
    assert values_same(compare_all(TESTDATA_DIR + '/dhs/d8atr', metric='bsf',
                                   bayesys='v1.5+'), 0.508, sf=3) is True


def test_core_metrics_compare_all_d7a_no_know_old():
    assert values_same(compare_all(TESTDATA_DIR + '/dhs/d7a', metric='bsf',
                                   bayesys='v1.3'), 0.330, sf=3) is True


def test_core_metrics_compare_all_d7at_old():
    assert values_same(compare_all(TESTDATA_DIR + '/dhs/d7at', metric='bsf',
                                   bayesys='v1.3'), 0.374, sf=3) is True


def test_core_metrics_compare_all_d8atr_old():
    assert values_same(compare_all(TESTDATA_DIR + '/dhs/d8atr', metric='bsf',
                                   bayesys='v1.3'), 0.488, sf=3) is True
