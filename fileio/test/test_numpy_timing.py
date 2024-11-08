
# Does some NumPy benchmark timings

import pytest
from numpy import unique, arange, prod as npprod, array, sum as npsum, prod, \
    bincount
from numpy.random import choice

from fileio.common import EXPTS_DIR
from fileio.numpy import NumPy
from fileio.pandas import Pandas
from core.timing import Timing


@pytest.fixture(scope="module")  # covid, 1M rows
def data():
    N = 1000000
    Timing.on(True)
    start = Timing.now()
    pandas = Pandas.read(EXPTS_DIR + '/datasets/covid.data.gz',
                         dstype='categorical', N=N)
    Timing.record('panda_read', N, start)

    start = Timing.now()
    data = NumPy.from_df(pandas.as_df(), dstype='categorical', keep_df=True)
    Timing.record('np_fromdf', N, start)
    return data


@pytest.mark.slow
def test_numpy_covid_1_ok(data):  # Simple np.unique

    for j in range(5):
        for n in range(1, 8):
            cols = choice(arange(0, data.data.shape[1]), size=n, replace=False)

            start = Timing.now()
            combos, counts = unique(data.data[:, cols], axis=0,
                                    return_counts=True)
            Timing.record('np.unique', n, start)

    print(Timing)


@pytest.mark.slow
def test_unique_covid_2_ok(data):  # using hashes

    for j in range(5):
        for n in range(1, 8):
            cols = choice(arange(0, data.data.shape[1]), size=n, replace=False)

            # Compute unique 'hash' for each combo - each variable has 4 values

            start0 = Timing.now()
            multipliers = 4 ** arange(n)
            hash_values = data.data[:, cols] @ multipliers
            Timing.record('hash_vals', n, start0)

            # Count unique integers, returning index to these integers

            start = Timing.now()
            uniques, indices, counts = unique(hash_values, return_index=True,
                                              return_counts=True)
            Timing.record('uniq_hash', n, start)
            print('Cols: {}, {} unique'.format(cols, len(uniques)))
            print(indices[:5])

            # Code to recover original rows from hashes via indices ...

            Timing.record('marginals', n, start0)

    print(Timing)


@pytest.mark.slow
def test_unique_covid_3_ok(data):  # using binning

    for j in range(5):
        for n in range(1, 8):
            cols = choice(arange(0, data.data.shape[1]), size=n, replace=False)

            # Compute unique 'hash' for each combo - each variable has 4 values

            start0 = Timing.now()

            # Create vector of number of values in each bin, bin_sizes

            # bin_sizes = [4] * n

            # Vector of multipliers for values in each bin

            # bin_mult = array([npprod(bin_sizes[i+1:]) for i in range(n)])
            multipliers = 4 ** arange(n)

            # create vector of hash values for each row using bin_mult

            # hashes = npsum(data.sample[:, cols] * bin_mult, axis=1)
            hashes = data.sample[:, cols] @ multipliers

            # count each integer value using bin counts

            start = Timing.now()
            counts = bincount(hashes, minlength=prod(multipliers))
            Timing.record('bincount', n, start)

            Timing.record('marginals', n, start0)

    print(Timing)


@pytest.mark.slow
def test_set_N_hailfinder_1_ok():  # Hailfinder, N=1M, timings
    N = 100000
    Timing.on(True)
    start = Timing.now()
    pandas = Pandas.read(EXPTS_DIR + '/datasets/hailfinder.data.gz',
                         dstype='categorical', N=N)
    Timing.record('panda_read', N, start)

    start = Timing.now()
    df = pandas.as_df()
    Timing.record('panda_asdf', N, start)

    start = Timing.now()
    data = NumPy.from_df(df=df, dstype='categorical', keep_df=False)
    Timing.record('np_fromdf', N, start)

    # Time set_N without re-ordering

    for n in range(100):
        start = Timing.now()
        data.set_N(data.N - 1)
        Timing.record('np_setN_1', N, start)

    # Time set_N with re-ordering

    for n in range(100):
        start = Timing.now()
        data.set_N(data.N - 1, seed=n)
        Timing.record('np_setN_2', N, start)

    # Time set_N reverting to original dataset

    for n in range(100):
        start = Timing.now()
        data.set_N(N)
        Timing.record('np_setN_3', N, start)

    print(Timing)
