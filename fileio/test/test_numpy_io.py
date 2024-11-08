
# Test the read and write functions of NumPy concrete implementation of Data

import pytest

from fileio.common import TESTDATA_DIR, FileFormatError
from fileio.numpy import NumPy

AB_3 = TESTDATA_DIR + '/simple/ab_3.csv'
PQ_3 = TESTDATA_DIR + '/simple/pq_3.csv'
YESNO_4 = TESTDATA_DIR + '/simple/yesno_4.csv'


def test_read_type_error_1_():  # no arguments provided
    with pytest.raises(TypeError):
        NumPy.read()


def test_read_type_error_2_():  # filename bad arg type
    with pytest.raises(TypeError):
        NumPy.read(None, dstype='continuous')
    with pytest.raises(TypeError):
        NumPy.read(True, dstype='continuous')
    with pytest.raises(TypeError):
        NumPy.read(1, dstype='categorical')
    with pytest.raises(TypeError):
        NumPy.read({AB_3}, dstype='categorical')


def test_read_type_error_3_():  # dstype bad type
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype='invalid')
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype=True)
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype={'continuous'})
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype=6)


def test_read_type_error_4_():  # N bad type
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype='categorical', N=True)
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype='categorical', N=1.2)
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype='categorical', N=(3,))


def test_read_filenotfound_error_1_():  # non-existent file
    with pytest.raises(FileNotFoundError):
        NumPy.read(TESTDATA_DIR + '/simple/nonexistent.csv',
                   dstype='categorical')
    with pytest.raises(FileNotFoundError):
        NumPy.read(TESTDATA_DIR + '/nonexistent/ab_3.csv',
                   dstype='categorical')


def test_read_value_error_1_():  # mixed datasets not supported yet
    with pytest.raises(ValueError):
        NumPy.read(AB_3, dstype='mixed')


def test_read_value_error_2_():  # N bad value
    with pytest.raises(ValueError):
        NumPy.read(AB_3, dstype='categorical', N=1)
    with pytest.raises(ValueError):
        NumPy.read(AB_3, dstype='categorical', N=-1)
    with pytest.raises(ValueError):
        NumPy.read(AB_3, dstype='categorical', N=0)


def test_read_value_error_3_():  # N more than number of rows in file
    with pytest.raises(ValueError):
        NumPy.read(AB_3, dstype='categorical', N=4)


def test_read_value_error_5_():  # File only contains one column
    with pytest.raises(ValueError):
        NumPy.read(TESTDATA_DIR + '/simple/a_2.csv', dstype='categorical')


def test_read_value_error_6_():  # File only contains one row
    with pytest.raises(ValueError):
        NumPy.read(TESTDATA_DIR + '/simple/ab_1.csv', dstype='categorical')


def test_read_value_error_7_():  # file categorical, dstype cont
    with pytest.raises(ValueError):
        NumPy.read(PQ_3, dstype='continuous')


def test_read_fileformat_error_1_():  # an empty plain file
    with pytest.raises(FileFormatError):
        NumPy.read(TESTDATA_DIR + '/misc/empty.txt', dstype='categorical')


def test_read_fileformat_error_2_():  # an empty compressed file
    with pytest.raises(FileFormatError):
        NumPy.read(TESTDATA_DIR + '/misc/empty.pkl.gz', dstype='categorical')


def test_read_fileformat_error_3_():  # reading a binary file
    with pytest.raises(FileFormatError):
        NumPy.read(TESTDATA_DIR + '/misc/null.sys', dstype='categorical')


def test_read_ab_ok_1_():
    data = NumPy.read(AB_3, dstype='categorical')
    assert isinstance(data, NumPy)


def test_read_yesno_ok_1_():
    data = NumPy.read(YESNO_4, dstype='categorical')
    assert isinstance(data, NumPy)


def test_read_asia_ok_1_():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                      dstype='categorical')
    assert isinstance(data, NumPy)
