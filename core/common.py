#
#   Classes used across core package
#

from enum import Enum
from numpy import log, log2, log10
from math import floor
from numpy.random import default_rng
from pandas import DataFrame
from pathlib import Path
from json import dump, load
from time import time
from platform import uname
from cpuinfo import get_cpu_info
from psutil import virtual_memory

from fileio.common import EXPTS_DIR

SOFTWARE_VERSION = 210

BAYESYS_VERSIONS = ['v1.3', 'v1.5+']


class EdgeMark(Enum):  # supported "ends" of an edge
    NONE = 0
    LINE = 1
    ARROW = 2
    CIRCLE = 3


class EdgeType(Enum):  # supported edge types and their symbols
    NONE = (0, EdgeMark.NONE, EdgeMark.NONE, '')
    DIRECTED = (1, EdgeMark.LINE, EdgeMark.ARROW, '->')
    UNDIRECTED = (2, EdgeMark.LINE, EdgeMark.LINE, '-')
    BIDIRECTED = (3, EdgeMark.ARROW, EdgeMark.ARROW, '<->')
    SEMIDIRECTED = (4, EdgeMark.CIRCLE, EdgeMark.ARROW, 'o->')
    NONDIRECTED = (5, EdgeMark.CIRCLE, EdgeMark.CIRCLE, 'o-o')
    SEMIUNDIRECTED = (6, EdgeMark.CIRCLE, EdgeMark.LINE, 'o-')


class Randomise(Enum):  # supported experiment randomisations
    ORDER = 'order'  # randomise order of variables in dataset
    NAMES = 'names'  # randomise variable names
    KNOWLEDGE = 'knowledge'  # randomise knowledge
    ROWS = 'rows'  # randomise row order in dataset
    SAMPLE = 'sample'  # randomise the sample of rows in dataset


_rng = default_rng(1)  # numpy random number generator

STABLE_RANDOM_FILE = '/stable_random.dat'
_stable_random = None  # sequence of random numbers read from disk file
_stable_random_offset = 0  # shift which allows different sequences of randos


class EnumWithAttrs(Enum):
    """
        Base class for an Enumeration with values exposed as read-only
        attributes.

        Values of this enum should set by e.g. VALUE1 = value1, label1
        This base class has a read-only value and label. Sub-classes can be
        extended to include more attributes by following the pattern for
        label here, and sub-classing the __init__ with more arguments.
    """
    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, label: str):
        self._label_ = label

    def __str__(self):
        return self.value

    @property
    def label(self):
        return self._label_


def rndsf(x, sf, zero=None):
    """
        Round a number to specified number of significant figures.

        :param float/int x: number to round
        :param int sf: number of significant values required
        :param float/None zero: abs values below this treated as zero,
                                None ==> set to 10**(-sf)

        :raises TypeError: if bad arg types
        :raises ValueError: if bad arg values

        :return str: rounded value as a float expressed as a string
    """
    if (not isinstance(x, (float, int)) or isinstance(x, bool)
            or not isinstance(sf, int) or isinstance(sf, bool)
            or (zero is not None and not isinstance(zero, float))):
        raise TypeError("rndsf bad arg types")

    zero = zero if zero is not None else 10 ** (- sf)
    if sf < 2 or sf > 10 or zero < 10 ** -20 or zero > 0.1:
        raise ValueError("rndsf bad arg values")
    if -zero < x and x < zero:
        return '0.0'

    exp = int(floor(log10(abs(x))))
    x = round(x, sf - exp - 1)
    str = "{:.{}f}".format(x, max(1, sf - exp - 1))
    str = str if str.endswith('.0') else str.rstrip('0')
    str = str + '0' if str.endswith('.') else str
    return str


def ln(x, base='e'):
    """
        Return logarithm to specified base

        :param float x: number to obtain logarithm of
        :param int/str base: base to use - 2, 10 or 'e'

        :raises TypeError: for bad argument types
        :raises ValueError: for bad argument values

        :returns float: logarithm(x) to specified base
    """
    if not isinstance(base, str) and not isinstance(base, int) \
            or isinstance(base, bool):
        raise TypeError('ln bad argument type')

    if base not in [2, 10, 'e']:
        raise ValueError('ln bad argument value')

    return log2(x) if base == 2 else (log10(x) if base == 10 else log(x))


def random_generator():
    """
        Generate a random number in the range 0.0 to 1.0 inclusive.
    """
    global _rng
    return _rng


def set_random_seed(seed=None):
    """
        Set the seed of the random number generator

        :param int/None seed: seed for pseudo-random or None for truly random

        :raises TypeError: if seed not an int or None
    """
    if seed is not None and not isinstance(seed, int):
        raise TypeError('set_random_seed called with bad arg type')

    global _rng
    _rng = default_rng(seed)


def generate_stable_random(N, path=EXPTS_DIR):
    """
        Generates a sequence of random numbers and saves them to a file so
        that the same random sequence can be used for repeatability.

        :param int N: number of random numbers to generate
        :param str path: location of saved random sequence file

        :returns list: of random numbers generated and saved.
    """
    random_sequence = [_rng.random() for i in range(N)]

    with open(path + STABLE_RANDOM_FILE, 'w') as file:
        file.writelines(['{}\n'.format(n) for n in random_sequence])

    return random_sequence


def stable_random(path=EXPTS_DIR):
    """
        Returns next random number in stable sequence.

        :param str path: location of saved random sequence file

        :raises StopIteration: when no more numbers available

        :returns float: next random number from stable sequence
    """
    global _stable_random, _stable_random_offset

    if _stable_random is None:
        try:
            with open(path + STABLE_RANDOM_FILE, 'r') as file:
                _stable_random = file.readlines()
                _stable_random = [float(i.strip()) for i in _stable_random]

        except EnvironmentError:
            raise FileNotFoundError('Unable to open random numbers file')

    if not len(_stable_random):
        raise StopIteration('No more stable random numbers')

    pop_idx = _stable_random_offset % len(_stable_random)
    rando = _stable_random.pop(pop_idx)
    if len(_stable_random) and _stable_random_offset != 0:
        shift = round(rando * len(_stable_random)) + _stable_random_offset
        shift = shift % len(_stable_random)
        _stable_random = _stable_random[shift:] + _stable_random[:shift]
    # print('Rando is {:.3f}, len is {}'.format(rando, len(_stable_random)))

    return rando


def init_stable_random(offset=0):
    """
        Sets the stable random offset so that different stable sequences can
        be generated from the same file of random numbers. Also clears the
        cache of stable random numbers.

        :param int offset: offset which generates different sequences
    """
    global _stable_random, _stable_random_offset
    _stable_random_offset = offset
    _stable_random = None


def adjmat(columns):
    """
        Create an adjacency matrix with specified entries

        :param dict columns: data for matrix specified by column

        :raises TypeError: if arg types incorrect
        :raises ValueError: if values specified are invalid

        :returns DataFrame: the adjacency matrix
    """
    if not isinstance(columns, dict) \
            or not all([isinstance(c, list) for c in columns.values()]) \
            or not all([isinstance(e, int)
                        for c in columns.values() for e in c]):
        raise TypeError('adjmat called with bad arg type')

    if not all([len(c) == len(columns) for c in columns.values()]):
        raise ValueError('some columns wrong length for adjmat')

    valid = [e.value[0] for e in EdgeType]  # valid edge integer codes
    if not all([e in valid for c in columns.values() for e in c]):
        raise ValueError('invalid integer values for adjmat')

    adjmat = DataFrame(columns, dtype='int8')
    adjmat[''] = list(adjmat.columns)
    return adjmat.set_index('')


def environment():
    """
        Obtain details of the hardware and software environment that the
        software is running under. For efficiency, this is obtained from
        a file "environment.json" in EXPTS_DIR if one modified in the last
        24 hours is available, otherwise the OS (registry etc.) is queried
        and a new version of "environment.json" created.

        :returns dict: of environment information
    """
    envfile_name = EXPTS_DIR + '/environment.json'
    envfile = Path(envfile_name)
    if not envfile.exists() or time() - envfile.stat().st_mtime > 24 * 3600:
        env = {'os': uname().system + ' v' + uname().version,
               'cpu': get_cpu_info()['brand_raw'],
               'python': get_cpu_info()['python_version'],
               'ram': round(virtual_memory().total / (1024 * 1024 * 1024))}
        with open(envfile_name, "w") as file:
            dump(env, file)
    else:
        with open(envfile_name, 'r') as file:
            env = load(file)

    return env


class RandomIntegers():
    """
        Iterable producing repeateable sequences of random integers for all
        integers in a range from 0 to specified n.

        :param int n: maximum integer in sequence
        :param int subsample: unique integer id for each sequence
        :param str path: path to stable random numbers file

        :raises TypeError: if bad arg type
        :raises ValueError: if bad arg value
        :raises FileNotFoundError: if path points to non-existent location
    """
    def __init__(self, n, subsample=0, path=EXPTS_DIR):

        if (not isinstance(n, int) or not isinstance(subsample, int)
                or not isinstance(path, str)):
            raise TypeError('RandomIntegers() bad arg type')

        if n < 1 or n > 1000 or subsample < 0 or subsample > 999:
            raise ValueError('RandomIntegers() bad arg value')

        self._n = n
        self._subsample = subsample
        self._path = path
        self._available = [i for i in range(self._n)]

    def __iter__(self):
        """
            Initialises the iterator.

            :returns RandomIntegers: the initialised iterator
        """
        init_stable_random(offset=self._subsample)
        return self

    def __next__(self):
        """
            Return next integer in random sequence.

            :raises StopIteration: when all values have been returned

            :returns int: next integer in random sequence
        """
        if not len(self._available):
            raise StopIteration()

        pop_idx = round((len(self._available)
                        * stable_random(path=self._path)) - 0.5)
        return self._available.pop(pop_idx)
