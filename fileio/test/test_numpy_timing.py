
# Does some NumPy benchmark timings

import pytest

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
def test_numpy_tabu_mildew_3_timings():  # Tabu, mildew, 1M
    timing = do_expt(network='mildew', N=1000000, id=3,
                     params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 35
    assert timing['marginals'][2]['count'] == 1190


@pytest.mark.slow
def test_numpy_tabu_barley_3_timings():  # Tabu, barley, 1M
    timing = do_expt(network='barley', N=1000000, id=3,
                     params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 48
    assert timing['marginals'][2]['count'] == 2256


@pytest.mark.slow
def test_numpy_tabu_hailfinder_3_timings():  # Tabu, hailfinder, 1M
    timing = do_expt(network='hailfinder', N=1000000, id=3,
                     params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 56
    assert timing['marginals'][2]['count'] == 3080


@pytest.mark.slow
def test_numpy_tabu_formed_3_timings():  # Tabu, formed, 1M
    timing = do_expt(network='formed', N=1000000, id=3,
                     params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 88
    assert timing['marginals'][2]['count'] == 7656


@pytest.mark.slow
def test_numpy_tabu_pathfinder_3_timings():  # Tabu, pathfinder, 1M
    timing = do_expt(network='pathfinder', N=1000000, id=3,
                     params={'tabu': 10})
    assert timing['marginals'][1]['count'] == 109
    assert timing['marginals'][2]['count'] == 11772


# this test is very slow - disable it

@pytest.mark.slow
def xtest_numpy_tabu_gaming_3_timings():  # Tabu, gaming, 1M
    timing = do_expt(network='gaming', N=1000000, id=3,
                     params={'tabu': 10})
    # assert timing['marginals'][1]['count'] == 109
    # assert timing['marginals'][2]['count'] == 11772
