
# Does some Pandas benchmark timings

import pytest

from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from learn.hc import hc
from core.timing import Timing


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

    context = {'id': 'timings/{}_{}'.format(network, id), 'in': network}

    start = Timing.now()
    dag, trace = hc(data=data, params=params, context=context)
    Timing.record('learning', N, start)

    print('\n\n{}\n\n{}\n'.format(dag, trace))

    print(Timing)

    return (Timing.times)


def test_pandas_tabu_asia_1_timings():  # Tabu, Asia, 1K
    timing = do_expt(network='asia', N=1000, id=1, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


def test_pandas_tabu_asia_2_timings():  # Tabu-Stable, Asia, 1K
    timing = do_expt(network='asia', N=1000, id=2,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


@pytest.mark.slow
def test_pandas_tabu_asia_3_timings():  # Tabu, Asia, 1M
    timing = do_expt(network='asia', N=1000000, id=3, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


@pytest.mark.slow
def test_pandas_tabu_asia_4_timings():  # Tabu-Stable, Asia, 1M
    timing = do_expt(network='asia', N=1000000, id=4,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


def test_pandas_hc_asia_5_timings():  # HC, Asia, 1K, maxiter=1
    timing = do_expt(network='asia', N=1000, id=5, params={'maxiter': 1})
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


def test_pandas_hc_asia_6_timings():  # HC, Asia, 1M, maxiter=1
    timing = do_expt(network='asia', N=1000000, id=6, params={'maxiter': 1})
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


def test_pandas_hc_asia_7_timings():  # HC, Asia, 1K
    timing = do_expt(network='asia', N=1000, id=7)
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


def test_pandas_hc_asia_8_timings():  # HC, Asia, 1M
    timing = do_expt(network='asia', N=1000000, id=8)
    assert timing['marginals'][1]['count'] == 8
    assert timing['marginals'][2]['count'] == 56


@pytest.mark.slow
def test_pandas_tabu_covid_1_timings():  # Tabu, Covid, 1K
    timing = do_expt(network='covid', N=1000, id=1, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 17
    assert timing['marginals'][2]['count'] == 272


@pytest.mark.slow
def test_pandas_tabu_covid_2_timings():  # Tabu-Stable, Covid, 1K
    timing = do_expt(network='covid', N=1000, id=2,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 17
    assert timing['marginals'][2]['count'] == 272


@pytest.mark.slow
def test_pandas_tabu_covid_3_timings():  # Tabu, Covid, 1M
    timing = do_expt(network='covid', N=1000000, id=3, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 17
    assert timing['marginals'][2]['count'] == 272


@pytest.mark.slow
def test_pandas_tabu_covid_4_timings():  # Tabu-Stable, Covid, 1M
    timing = do_expt(network='covid', N=1000000, id=4,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 17
    assert timing['marginals'][2]['count'] == 272


@pytest.mark.slow
def test_pandas_tabu_diarrhoea_1_timings():  # Tabu, diarrhoea, 1K
    timing = do_expt(network='diarrhoea', N=1000, id=1, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 28
    assert timing['marginals'][2]['count'] == 756


@pytest.mark.slow
def test_pandas_tabu_diarrhoea_2_timings():  # Tabu-Stable, diarrhoea, 1K
    timing = do_expt(network='diarrhoea', N=1000, id=2,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 28
    assert timing['marginals'][2]['count'] == 756


@pytest.mark.slow
def test_pandas_tabu_diarrhoea_3_timings():  # Tabu, diarrhoea, 1M
    timing = do_expt(network='diarrhoea', N=1000000, id=3, params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 28
    assert timing['marginals'][2]['count'] == 756


@pytest.mark.slow
def test_pandas_tabu_diarrhoea_4_timings():  # Tabu-Stable, diarrhoea, 1M
    timing = do_expt(network='diarrhoea', N=1000000, id=4,
                     params={'tabu': 10, 'stable': 'score+'})
    assert timing['marginals'][1]['count'] == 28
    assert timing['marginals'][2]['count'] == 756


@pytest.mark.slow
def test_pandas_tabu_hailfinder_3_timings():  # Tabu, hailfinder, 1M
    timing = do_expt(network='hailfinder', N=1000000, id=3,
                     params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 56
    assert timing['marginals'][2]['count'] == 3080


# Gaussian network timings

def test_pandas_tabu_sachs_c_1_timings():  # Tabu, sachs_c, 1K
    do_expt(network='sachs_c', N=1000, id=1, params={'tabu': 10})


def test_pandas_tabu_sachs_c_2_timings():  # Tabu-Stable, sachs_c, 1K
    do_expt(network='sachs_c', N=1000, id=2,
            params={'tabu': 10, 'stable': 'score+'})


@pytest.mark.slow
def test_pandas_tabu_sachs_c_3_timings():  # Tabu, sachs_c, 1M
    do_expt(network='sachs_c', N=1000000, id=3, params={'tabu': 10})


@pytest.mark.slow
def test_pandas_tabu_sachs_c_4_timings():  # Tabu, sachs_c, 1M
    do_expt(network='sachs_c', N=1000000, id=4,
            params={'tabu': 10, 'stable': 'score+'})


def test_pandas_tabu_covid_c_1_timings():  # Tabu, covid_c, 1K
    do_expt(network='covid_c', N=1000, id=1, params={'tabu': 10})


@pytest.mark.slow
def test_pandas_tabu_covid_c_3_timings():  # Tabu, covid_c, 1M
    do_expt(network='covid_c', N=1000000, id=3, params={'tabu': 10})


@pytest.mark.slow
def test_pandas_tabu_covid_c_4_timings():  # Tabu-Stable, covid_c, 1M
    do_expt(network='covid_c', N=1000000, id=4,
            params={'tabu': 10, 'stable': 'score+'})


# read and set_N timings

@pytest.mark.slow
def test_set_N_hailfinder_1_ok():  # Hailfinder, N=100K, timings
    N = 100000
    Timing.on(True)
    start = Timing.now()
    data = Pandas.read(EXPTS_DIR + '/datasets/hailfinder.data.gz',
                       dstype='categorical', N=N)
    Timing.record('panda_read', data.N, start)

    # Time set_N without re-ordering

    for n in range(10):
        start = Timing.now()
        data.set_N(data.N - 1)
        Timing.record('np_setN_1', N, start)

    # Time set_N with re-ordering

    for n in range(10):
        start = Timing.now()
        data.set_N(data.N - 1, seed=n)
        Timing.record('np_setN_2', N, start)

    # Time set_N reverting to original dataset

    for n in range(10):
        start = Timing.now()
        data.set_N(data.N)
        Timing.record('np_setN_3', N, start)

    print(Timing)
