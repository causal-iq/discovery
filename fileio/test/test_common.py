
import pytest

from fileio.common import is_valid_path, TESTDATA_DIR


def test_fileio_common_is_valid_path_type_error():
    with pytest.raises(TypeError):
        is_valid_path()
    with pytest.raises(TypeError):
        is_valid_path(None)
    with pytest.raises(TypeError):
        is_valid_path(-3)
    with pytest.raises(TypeError):
        is_valid_path(['file.txt'], False)
    with pytest.raises(TypeError):
        is_valid_path('file', is_file=3)


def test_fileio_common_is_valid_path_value_error1():
    with pytest.raises(FileNotFoundError):
        is_valid_path('doesnotexist')


def test_fileio_common_is_valid_path_value_error2():
    with pytest.raises(FileNotFoundError):
        is_valid_path(TESTDATA_DIR + '/misc/empty.txt', is_file=False)
    with pytest.raises(FileNotFoundError):
        is_valid_path(TESTDATA_DIR + '/misc', is_file=True)


def test_fileio_common_is_valid_path_file_ok():
    assert is_valid_path(TESTDATA_DIR + '/misc/empty.txt') is True


def test_fileio_common_is_valid_path_dir_ok():
    assert is_valid_path(TESTDATA_DIR + '/misc', is_file=False) is True
