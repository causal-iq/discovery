
# Does some NumPy benchmark timings

import pytest
from numpy import unique, arange
from numpy.random import choice

from fileio.common import EXPTS_DIR
from fileio.numpy import NumPy
from fileio.pandas import Pandas
from core.timing import Timing
from learn.hc import hc


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


def do_expt(network, N, id, params=None):
    """
        Do an individual timings experiment.

        :param str network: network to use
        :param int N: sample size
        :param int id: experiment id

        :returns dict: of requests counts {size: count}
    """
    Timing.on(True)

    dstype = 'continuous' if network.endswith('_c') else 'categorical'
    data = Pandas.read(EXPTS_DIR + '/datasets/' + network + '.data.gz',
                       dstype=dstype, N=N)
    data = NumPy.from_df(data.df, dstype='categorical', keep_df=False)

    context = {'id': 'timings/{}_{}'.format(network, id), 'in': network}

    start = Timing.now()
    dag, trace = hc(data=data, params=params, context=context)
    Timing.record('learning', N, start)

    print('\n\n{}\n\n{}\n'.format(dag, trace))

    print(Timing)

    return (Timing.times)


def test_numpy_tabu_asia_1_timings():  # Tabu, Asia, 1K
    timing = do_expt(network='asia', N=1000, id=1, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


def test_numpy_tabu_asia_2_timings():  # Tabu-Stable, Asia, 1K
    timing = do_expt(network='asia', N=1000, id=2,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


@pytest.mark.slow
def test_numpy_tabu_asia_3_timings():  # Tabu, Asia, 1M
    timing = do_expt(network='asia', N=1000000, id=3, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


@pytest.mark.slow
def test_numpy_tabu_asia_4_timings():  # Tabu-Stable, Asia, 1M
    timing = do_expt(network='asia', N=1000000, id=4,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


@pytest.mark.slow
def test_numpy_tabu_covid_1_timings():  # Tabu, Covid, 1K
    timing = do_expt(network='covid', N=1000, id=1, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 17
    assert timing['marginals'][2]['count'] == 272


@pytest.mark.slow
def test_numpy_tabu_covid_2_timings():  # Tabu-Stable, Covid, 1K
    timing = do_expt(network='covid', N=1000, id=2,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 17
    assert timing['marginals'][2]['count'] == 272


@pytest.mark.slow
def test_numpy_tabu_covid_3_timings():  # Tabu, Covid, 1M
    timing = do_expt(network='covid', N=1000000, id=3, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 17
    assert timing['marginals'][2]['count'] == 272


@pytest.mark.slow
def test_numpy_tabu_covid_4_timings():  # Tabu-Stable, Covid, 1M
    timing = do_expt(network='covid', N=1000000, id=4,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 17
    assert timing['marginals'][2]['count'] == 272


@pytest.mark.slow
def test_numpy_tabu_diarrhoea_1_timings():  # Tabu, diarrhoea, 1K
    timing = do_expt(network='diarrhoea', N=1000, id=1, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 28
    assert timing['marginals'][2]['count'] == 756


@pytest.mark.slow
def test_numpy_tabu_diarrhoea_2_timings():  # Tabu-Stable, diarrhoea, 1K
    timing = do_expt(network='diarrhoea', N=1000, id=2,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 28
    assert timing['marginals'][2]['count'] == 756


@pytest.mark.slow
def test_numpy_tabu_diarrhoea_3_timings():  # Tabu, diarrhoea, 1M
    timing = do_expt(network='diarrhoea', N=1000000, id=3, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 28
    assert timing['marginals'][2]['count'] == 756


@pytest.mark.slow
def test_numpy_tabu_diarrhoea_4_timings():  # Tabu-Stable, diarrhoea, 1M
    timing = do_expt(network='diarrhoea', N=1000000, id=4,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 28
    assert timing['marginals'][2]['count'] == 756


@pytest.mark.slow
def xtest_numpy_covid_1_ok(data):  # Simple np.unique

    for j in range(5):
        for n in range(1, 8):
            cols = choice(arange(0, data.data.shape[1]), size=n, replace=False)

            start = Timing.now()
            combos, counts = unique(data.data[:, cols], axis=0,
                                    return_counts=True)
            Timing.record('np.unique', n, start)

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
