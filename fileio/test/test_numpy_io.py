
# Test the read and write functions of NumPy concrete implementation of Data

import pytest
from pandas import DataFrame
from numpy import array
from os import remove
from random import random

from fileio.common import TESTDATA_DIR, FileFormatError
from fileio.numpy import NumPy

AB_3 = TESTDATA_DIR + '/simple/ab_3.csv'
PQ_3 = TESTDATA_DIR + '/simple/pq_3.csv'
YESNO_4 = TESTDATA_DIR + '/simple/yesno_4.csv'


@pytest.fixture(scope="module")  # AB, 2 rows
def ab2():
    df = DataFrame({'A': ['0', '1'], 'B': ['1', '1']}, dtype='category')
    return NumPy.from_df(df=df, dstype='categorical', keep_df=True)


@pytest.fixture(scope="module")  # continuous XY, 2 rows
def xy2():
    data = array([[1.04, -.00348], [132, 0.0000453]], dtype='float32')
    dstype = 'continuous'
    col_values = {'X': None, 'Y': None}
    return NumPy(data, dstype, col_values)


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.csv'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpgzfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.csv.gz'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


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


def test_read_asia_ok_1_():  # Asia, 1K rows
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/asia.data.gz',
                      dstype='categorical')
    assert isinstance(data, NumPy)
    assert data.N == 1000


def test_read_xyz10_ok_1_():  # XYZ, 1K rows
    data = NumPy.read(TESTDATA_DIR + '/simple/xyz_10.csv',
                      dstype='continuous')
    assert isinstance(data, NumPy)
    assert data.N == 10


# Pandas write tests

def test_write_type_error_1():  # write with no arguments
    with pytest.raises(TypeError):
        NumPy.write()


def test_write_type_error_2():  # write with bad filename type
    with pytest.raises(TypeError):
        NumPy.write(3)
    with pytest.raises(TypeError):
        NumPy.write(None, compress=True)
    with pytest.raises(TypeError):
        NumPy.write([42])
    with pytest.raises(TypeError):
        NumPy.write('invalid')


def test_write_type_error_3(ab2):  # write with bad compress type
    with pytest.raises(TypeError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', compress=None)
    with pytest.raises(TypeError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_createt.csv', compress=1)


def test_write_type_error_4(ab2):  # write with bad sf type
    with pytest.raises(TypeError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=None)
    with pytest.raises(TypeError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf='one')
    with pytest.raises(TypeError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=(1,))
    with pytest.raises(TypeError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=0.2)
    with pytest.raises(TypeError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=True)


def test_write_type_error_5(ab2):  # write with bad zero type
    with pytest.raises(TypeError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=2, zero=False)
    with pytest.raises(TypeError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=2, zero=1)


def test_write_type_error_6(ab2):  # write with bad preserve type
    with pytest.raises(TypeError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', preserve=1)


def test_write_value_error_1(ab2):  # sf not between 2 and 10
    with pytest.raises(ValueError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=1)
    with pytest.raises(ValueError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=0)
    with pytest.raises(ValueError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=-1)
    with pytest.raises(ValueError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=11)


def test_write_value_error_2(ab2):  # zero not between 1E-20 and 1E-1
    with pytest.raises(ValueError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=2, zero=0.2)
    with pytest.raises(ValueError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=2, zero=-0.01)
    with pytest.raises(ValueError):
        ab2.write(TESTDATA_DIR + '/tmp/wont_create.csv', sf=2, zero=1E-21)


def test_write_filenotfound_error_1(ab2):  # bad directory
    with pytest.raises(FileNotFoundError):
        ab2.write(TESTDATA_DIR + '/nonexistent/bad.csv')


def test_write_ab2_1_ok(ab2, tmpfile):  # write a non-compressed file OK
    ab2.write(tmpfile)
    check = NumPy.read(tmpfile, dstype='categorical')
    assert check.as_df().to_dict() == ab2.as_df().to_dict()


def test_write_ab2_2_ok(ab2, tmpgzfile):  # write a compressed file OK
    ab2.write(tmpgzfile, compress=True)
    check = NumPy.read(tmpgzfile, dstype='categorical')
    assert check.as_df().to_dict() == ab2.as_df().to_dict()


def test_write_xy2_1_ok(xy2, tmpgzfile):  # write a cont file OK
    xy2.write(tmpgzfile, compress=True)
    check = NumPy.read(tmpgzfile, dstype='continuous')
    assert check.as_df().to_dict() == xy2.as_df().to_dict()


def test_numpy_xy2_2_ok(xy2, tmpgzfile):  # check round to 2 s.f., zero 0.01
    xy2.write(tmpgzfile, compress=True, sf=2)
    check = NumPy.read(tmpgzfile, dstype='continuous')
    assert (check.data == array([[1.0, 0.0], [130.0, 0.0]])).all().all()


def test_numpy_xy2_3_ok(xy2, tmpfile):  # check rounding to 2 s.f., zero 1E-5
    xy2.write(tmpfile, compress=False, sf=2, zero=1E-5)
    check = NumPy.read(tmpfile, dstype='continuous')
    assert (check.data == array([[1.0, -0.0035], [130.0, 0.000045]],
                                dtype='float32')).all().all()


def test_write_xy2_4_ok(xy2, tmpfile):  # check rounding to 3 s.f., zero 1E-6
    xy2.write(tmpfile, compress=False, sf=3, zero=1E-6)
    check = NumPy.read(tmpfile, dstype='continuous')
    assert (check.data == array([[1.04, -0.00348], [132.0, 0.0000453]],
                                dtype='float32')).all().all()


def test_write_xy3_1_ok(tmpfile):  # check rounding to 7 s.f.,
    data = array([[1.04, 132], [-.00348066, 0.00045], [1E-5, 43.12345]],
                 dtype='float32')
    data = NumPy(data, 'continuous', {'X': None, 'Y': None})
    data.write(tmpfile, compress=False, sf=3, zero=1E-6)
    check = NumPy.read(tmpfile, dstype='continuous')
    assert (check.data == array([[1.04, 132.0],
                                 [-0.00348, 0.00045],
                                 [1e-5, 43.1]],
                                dtype='float32')).all().all()
