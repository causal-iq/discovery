
# system testing of impact analysis

import pytest

from fileio.common import TESTDATA_DIR
from experiments.impact_analysis import impact_analysis


def systest_impact_analysis_type_error_1():
    with pytest.raises(TypeError):
        impact_analysis()


def systest_impact_analysis_know_asia_1():  # N=10, no, 4, 16 know requests
    series_arg = ['HC/ORDER/KL16', 'HC/ORDER/KL4', 'HC/ORDER/BASE']
    networks = ['asia']
    metrics_arg = ['f1']
    N_range = (10, 10)
    expts_dir = TESTDATA_DIR + '/experiments'
    data, props, stats = impact_analysis(series_arg, networks, metrics_arg,
                                         N_range, {}, {'series': 'unused'},
                                         expts_dir)

    data = data.to_dict(orient='records')
    assert len(data) == 20
    assert data[0] == {'subplot': 'unused', 'x_val': 'limit = 16',
                       'y_val': 0.615}  # F1 is 0.615 vs 0.000
    assert data[12] == {'subplot': 'unused', 'x_val': 'limit = 4',
                        'y_val': 0.307}  # F1 is 0.615 vs 0.308
    assert list(stats.keys()) == ['unused']
    assert list(stats['unused'].keys()) == ['limit = 16', 'limit = 4']
    assert (stats['unused']['limit = 4'] ==
            {'min': 0.000, 'max': 0.615, 'mean': 0.461, 'std': 0.205,
             'count': 10})
    assert (stats['unused']['limit = 16'] ==
            {'min': 0.000, 'max': 0.615, 'mean': 0.461, 'std': 0.205,
             'count': 10})


def systest_impact_analysis_know_asia_2():  # N=20, no, 4, 16 know requests
    series_arg = ['HC/ORDER/KL16', 'HC/ORDER/KL4', 'HC/ORDER/BASE']
    networks = ['asia']
    metrics_arg = ['f1']
    N_range = (20, 20)
    expts_dir = TESTDATA_DIR + '/experiments'
    data, props, stats = impact_analysis(series_arg, networks, metrics_arg,
                                         N_range, {}, {'series': 'unused'},
                                         expts_dir)

    data = data.to_dict(orient='records')
    assert len(data) == 20
    assert data[0] == {'subplot': 'unused', 'x_val': 'limit = 16',
                       'y_val': 0.667}  # F1 is 0.667 vs 0.000
    assert data[14] == {'subplot': 'unused', 'x_val': 'limit = 4',
                        'y_val': 0.500}  # F1 is 0.667 vs 0.100
    assert list(stats.keys()) == ['unused']
    assert list(stats['unused'].keys()) == ['limit = 16', 'limit = 4']
    assert (stats['unused']['limit = 4'] ==
            {'min': 0.000, 'max': 0.667, 'mean': 0.500, 'std': 0.222,
             'count': 10})
    assert (stats['unused']['limit = 16'] ==
            {'min': 0.000, 'max': 0.667, 'mean': 0.500, 'std': 0.222,
             'count': 10})


def systest_impact_analysis_know_asia_3():  # N=10&20, no, 4, 16 know requests
    series_arg = ['HC/ORDER/KL16', 'HC/ORDER/KL4', 'HC/ORDER/BASE']
    networks = ['asia']
    metrics_arg = ['f1']
    N_range = (10, 20)
    expts_dir = TESTDATA_DIR + '/experiments'
    data, props, stats = impact_analysis(series_arg, networks, metrics_arg,
                                         N_range, {}, {'series': 'unused'},
                                         expts_dir)

    data = data.to_dict(orient='records')
    assert len(data) == 40
    assert data[17] == {'subplot': 'unused', 'x_val': 'limit = 16',
                        'y_val': 0.334}  # F1 is 0.667 vs 0.334
    assert data[28] == {'subplot': 'unused', 'x_val': 'limit = 4',
                        'y_val': 0.615}  # F1 is 0.667 vs 0.0.52
    assert list(stats.keys()) == ['unused']
    assert list(stats['unused'].keys()) == ['limit = 16', 'limit = 4']
    assert (stats['unused']['limit = 4'] ==
            {'min': 0.000, 'max': 0.667, 'mean': 0.481, 'std': 0.209,
             'count': 20})
    assert (stats['unused']['limit = 16'] ==
            {'min': 0.000, 'max': 0.667, 'mean': 0.481, 'std': 0.209,
             'count': 20})


def systest_impact_analysis_know_asia_4():  # N=100, expert=no,1.0,.67,.50
    series_arg = ['HC/ORDER/KL0', 'HC/ORDER/KE67', 'HC/ORDER/KE50',
                  'HC/ORDER/BASE']
    networks = ['asia']
    metrics_arg = ['f1']
    N_range = (100, 100)
    expts_dir = TESTDATA_DIR + '/experiments'
    data, props, stats = impact_analysis(series_arg, networks, metrics_arg,
                                         N_range, {}, {'series': 'unused'},
                                         expts_dir)

    data = data.to_dict(orient='records')
    assert len(data) == 30
    assert data[0] == {'subplot': 'unused', 'x_val': 'expertise = 1.0',
                       'y_val': 0.857}  # F1 is 0.857 vs 0.000
    assert data[18] == {'subplot': 'unused', 'x_val': 'expertise = 0.67',
                        'y_val': 0.714}  # F1 is 0.667 vs 0.0.52
    assert list(stats.keys()) == ['unused']
    assert list(stats['unused'].keys()) == ['expertise = 0.5',
                                            'expertise = 0.67',
                                            'expertise = 1.0']
    assert (stats['unused']['expertise = 0.5'] ==
            {'min': -0.857, 'max': 0.571, 'mean': 0.129, 'std': 0.406,
             'count': 10})
    assert (stats['unused']['expertise = 0.67'] ==
            {'min': -0.857, 'max': 0.714, 'mean': 0.285, 'std': 0.457,
             'count': 10})
    assert (stats['unused']['expertise = 1.0'] ==
            {'min': 0.000, 'max': 0.857, 'mean': 0.657, 'std': 0.287,
             'count': 10})
