
import pytest

from learn.pc import sgs_skeleton
from fileio.common import FileFormatError, TESTDATA_DIR
from fileio.pandas import Pandas
from core.bn import BN


def test_sgs_skeleton_type_error1():  # data not a dataframe or string
    with pytest.raises(TypeError):
        sgs_skeleton(True)
    with pytest.raises(TypeError):
        sgs_skeleton(-1)
    with pytest.raises(TypeError):
        sgs_skeleton([2, 1, 3])


def test_sgs_skeleton_type_error2():  # bn arg is not a BN object
    with pytest.raises(TypeError):
        sgs_skeleton(None, True)


def test_sgs_skeleton_type_error3():  # both data and BN specified
    with pytest.raises(TypeError):
        sgs_skeleton(True, True)


def test_sgs_skeleton_type_error4():  # invalid sample size type
    with pytest.raises(TypeError):
        sgs_skeleton(None, TESTDATA_DIR + '/cancer/cancer.dsc', False)
    with pytest.raises(TypeError):
        sgs_skeleton(None, TESTDATA_DIR + '/cancer/cancer.dsc', -3.0)
    with pytest.raises(TypeError):
        sgs_skeleton(None, TESTDATA_DIR + '/cancer/cancer.dsc', 'wrongtype')


def test_sgs_skeleton_file_not_found_error():  # nonexistent file for data/bn
    with pytest.raises(FileNotFoundError):
        sgs_skeleton('nonexistent.txt')
    with pytest.raises(FileNotFoundError):
        sgs_skeleton(None, 'nonexistent.dsc')


def test_sgs_skeleton_file_format_error():  # binary file for data or bn
    with pytest.raises(FileFormatError):
        sgs_skeleton(TESTDATA_DIR + '/misc/null.sys')
    with pytest.raises(FileFormatError):
        sgs_skeleton(None, TESTDATA_DIR + '/misc/null.sys.dsc')


def test_sgs_skeleton_data_ok1():  # generate data for A->B->C chain
    bn = BN.read(TESTDATA_DIR + '/dsc/abc.dsc')
    data = bn.generate_cases(3)
    dependency_model = sgs_skeleton(data)
    for edge, sepset in dependency_model.items():
        print(edge, sepset)
    print()


def test_sgs_skeleton_bn_ok1():  # valid input bn
    bn = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    print(bn.global_distribution())
    data = Pandas(df=bn.generate_cases(10000))
    bn2 = BN.fit(bn.dag, data)
    print(bn2.global_distribution())
    dependency_model = sgs_skeleton(data.sample)
    for edge, sepset in dependency_model.items():
        print(edge, sepset)
    print()
