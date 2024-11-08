
#   Test of dsc module to read and write DSC format BN definitions

from random import random
from os import remove
import pytest

from fileio.common import FileFormatError, TESTDATA_DIR
from core.bn import BN


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.dsc'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


def test_fileio_dsc_type_error1():  # missing input argument
    with pytest.raises(TypeError):
        BN.read()


def test_fileio_dsc_type_error2():  # incorrect argument types
    with pytest.raises(TypeError):
        BN.read(1)
    with pytest.raises(TypeError):
        BN.read(0.7)
    with pytest.raises(TypeError):
        BN.read(False)


def test_fileio_dsc_type_error3():  # fail on non-existent file
    with pytest.raises(FileNotFoundError):
        BN.read('doesnotexist.dsc')


def test_fileio_dsc_binaryfile():  # fail on binary file
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/misc/null.sys.dsc')


def test_fileio_dsc_emptyfile():  # fail on empty file
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/empty.dsc')


def test_fileio_dsc_ab_format_error1():  # fail on network line
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_network1.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_network2.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_missing_network.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_duplicate_network.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_missplaced_network.dsc')


def test_fileio_dsc_ab_format_error2():  # fail on no node section
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_only_network.dsc')


def test_fileio_dsc_ab_format_error3():  # fail on node section
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_node1.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_duplicate_node.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_type1.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_type2.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_type3.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_node_value1.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_duplicate_node_value.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_node_bad_num_values.dsc')


def test_fileio_dsc_ab_format_error4():  # fail on node probability errors
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_prob_node.dsc')


def test_fileio_dsc_ab_format_error5():  # node errors
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_prob_cond_node1.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_prob_cond_node2.dsc')


def test_fileio_dsc_ab_format_error6():  # prob errors
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_prob1.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_prob2.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_prob3.dsc')


def test_fileio_dsc_ab_format_error7():  # prob errors
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_cond_prob1.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_cond_prob2.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_cond_prob3.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_cond_prob4.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_cond_prob5.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_cond_prob6.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_cond_prob7.dsc')
    with pytest.raises(FileFormatError):
        BN.read(TESTDATA_DIR + '/dsc/ab_bad_cond_prob8.dsc')


def test_fileio_dsc_ab_ok():
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    assert isinstance(bn, BN)
    assert bn.free_params == 3


def test_fileio_dsc_cancer():
    bn = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    assert isinstance(bn, BN)
    assert bn.free_params == 10


def test_fileio_dsc_asia():
    bn = BN.read(TESTDATA_DIR + '/asia/asia.dsc')
    assert isinstance(bn, BN)
    assert bn.free_params == 18


def test_fileio_dsc_alarm():
    bn = BN.read(TESTDATA_DIR + '/alarm/alarm.dsc')
    assert isinstance(bn, BN)
    assert bn.free_params == 509


def test_fileio_dsc_pathfinder():
    bn = BN.read(TESTDATA_DIR + '/pathfinder/pathfinder.dsc')
    assert isinstance(bn, BN)
    assert bn.free_params == 72079


def test_fileio_dsc_write_not_found():  # fail on write to non-existent path
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    with pytest.raises(FileNotFoundError):
        bn.write(TESTDATA_DIR + '/nonexistent/ab.dsc')


def test_fileio_dsc_write_ab_ok(tmpfile):  # successfully writes ab file
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    bn.write(tmpfile)
    bn_check = BN.read(tmpfile)
    assert bn == bn_check


def test_fileio_dsc_write_cancer_ok(tmpfile):  # write cancer file ok
    bn = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')
    bn.write(tmpfile)
    bn_check = BN.read(tmpfile)
    assert bn == bn_check


def test_fileio_dsc_write_asia_ok(tmpfile):  # write asia file ok
    bn = BN.read(TESTDATA_DIR + '/asia/asia.dsc')
    bn.write(tmpfile)
    bn_check = BN.read(tmpfile)
    assert bn == bn_check


def test_fileio_dsc_write_alarm_ok(tmpfile):  # write alarm file ok
    bn = BN.read(TESTDATA_DIR + '/alarm/alarm.dsc')
    bn.write(tmpfile)
    bn_check = BN.read(tmpfile)
    assert bn == bn_check


def test_fileio_dsc_write_pathfinder_ok(tmpfile):  # write pathfinder file ok
    bn = BN.read(TESTDATA_DIR + '/pathfinder/pathfinder.dsc')
    bn.write(tmpfile)
    bn_check = BN.read(tmpfile)
    assert bn == bn_check
