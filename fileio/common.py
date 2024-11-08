#
#   Common file i/o Functions
#

from os.path import isfile, isdir
from strenum import StrEnum


TESTDATA_DIR = 'testdata'  # Directory where noisy test data files reside
EXPTS_DIR = 'experiments'  # Directory for experimental data and results


class DatasetType(StrEnum):
    CATEGORICAL = 'categorical'  # all categorical variables
    CONTINUOUS = 'continuous'  # all float variables
    MIXED = 'mixed'  # mixed categorical, float or numeric


class VariableType(StrEnum):
    INT16 = 'int16'
    INT32 = 'int32'
    INT64 = 'int64'
    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    CATEGORY = 'category'


class FileFormatError(Exception):
    pass


def is_valid_path(path, is_file=True):
    """
        Checks path is a string and it exists

        :param str path: full path name of file
        :param bool is_file: should path be a file (otherwise a directory)

        :raises TypeError: with bad arg types
        :raises FileNotFoundError: if path is not found

        :returns: True if path is valid and exists
        :rtype: boolean
    """
    if not isinstance(path, str) or not isinstance(is_file, bool):
        raise TypeError('is_valid_path() bad arg types')

    if (is_file and not isfile(path)) or (not is_file and not isdir(path)):
        raise FileNotFoundError('path {} not found'.format(path))

    return True
