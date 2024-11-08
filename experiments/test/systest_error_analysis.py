
# system testing of error analysis

import pytest
from random import random
from os import remove

from fileio.common import TESTDATA_DIR
from experiments.error_analysis import error_analysis


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.dsc'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


def systest_error_analysis_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        error_analysis()


def systest_error_analysis_type_error_2():  # insufficient arguments
    series = ['HC/KW_L1', 'HC/BAD']
    networks = ['asia']
    with pytest.raises(TypeError):
        error_analysis(series)
    with pytest.raises(TypeError):
        error_analysis(series, networks)


def systest_error_analysis_type_error_3():  # bad series type
    networks = ['asia']
    with pytest.raises(TypeError):
        error_analysis('HC/KW_L1', networks, None)
    with pytest.raises(TypeError):
        error_analysis(32, networks, None)
    with pytest.raises(TypeError):
        error_analysis([1, 'HC/KW_L1'], networks, None)


def systest_error_analysis_type_error_4():  # bad network type
    series = ['HC/KW_L1', 'HC/BAD']
    with pytest.raises(TypeError):
        error_analysis(series, 'asia', None)
    with pytest.raises(TypeError):
        error_analysis(series, 32, None)
    with pytest.raises(TypeError):
        error_analysis(series, [1, 'asia'], None)


def systest_error_analysis_type_error_5():  # bad N_range type
    series = ['HC/KW_L1', 'HC/BAD']
    networks = ['asia']
    with pytest.raises(TypeError):
        error_analysis(series, networks, 6)
    with pytest.raises(TypeError):
        error_analysis(series, networks, (10,))
    with pytest.raises(TypeError):
        error_analysis(series, networks, 'invalid')


def systest_error_analysis_type_error_6():  # bad params type
    series = ['HC/STD']
    networks = ['asia']
    N_range = (10, 20)
    with pytest.raises(TypeError):
        error_analysis(series, networks, N_range, 32)
    with pytest.raises(TypeError):
        error_analysis(series, networks, N_range, 3.4)
    with pytest.raises(TypeError):
        error_analysis(series, networks, N_range, 'wrong type')


def systest_error_analysis_value_error_1():  # bad params keys
    series = ['HC/STD']
    networks = ['asia']
    Ns = [10, 20]
    with pytest.raises(ValueError):
        error_analysis(series, networks, Ns, {})
    with pytest.raises(ValueError):
        error_analysis(series, networks, Ns, {'unknown': 1})
    with pytest.raises(ValueError):
        error_analysis(series, networks, Ns, {'criterion': 'eqv',
                                              'unknown': 1})
    with pytest.raises(ValueError):
        error_analysis(series, networks, Ns, {'metric': 'ok'})


def systest_error_analysis_value_error_2():  # bad params values
    series = ['HC/STD']
    networks = ['asia']
    Ns = [10, 20]
    with pytest.raises(ValueError):
        error_analysis(series, networks, Ns, {'criterion': 'unknown',
                                              'metric': 'ok'})
    with pytest.raises(ValueError):
        error_analysis(series, networks, Ns, {'criterion': 'eqv',
                                              'metric': 'unknown'})


def systest_error_analysis_asia_ok_1():  # Asia 10 rows, eqv crit, ok metric
    series = ['HC/STD']
    networks = ['asia']
    Ns = [10]
    root_dir = TESTDATA_DIR + '/experiments'
    ct = error_analysis(series, networks, Ns, {'criterion': 'eqv',
                                               'metric': 'ok'}, root_dir)
    assert ct == {'err': {'eqv': 1, 'non': 2}, 'ok': {'eqv': 2, 'non': 0}}


def systest_error_analysis_asia_ok_2():  # Asia N=10, eqv crit, ok-eqv metric
    series = ['HC/STD']
    networks = ['asia']
    Ns = [10]
    root_dir = TESTDATA_DIR + '/experiments'
    ct = error_analysis(series, networks, Ns,
                        {'criterion': 'eqv', 'metric': 'ok-eqv'}, root_dir)
    assert ct == {'err': {'eqv': 1, 'non': 2}, 'ok': {'eqv': 2, 'non': 0}}


def systest_error_analysis_asia_ok_3():  # Asia N=10, mi crit, ok metric
    series = ['HC/STD']
    networks = ['asia']
    Ns = [10]
    root_dir = TESTDATA_DIR + '/experiments'
    ct = error_analysis(series, networks, Ns, {'criterion': 'mi',
                                               'metric': 'ok'}, root_dir)
    assert ct == {'err': {'eqv-hi': 1, 'eqv-sim': 0, 'non-hi': 2},
                  'ok': {'eqv-hi': 1, 'eqv-sim': 1, 'non-hi': 0}}


def systest_error_analysis_asia_ok_4():  # Asia N=10, mi crit, ok metric
    series = ['HC/STD']
    networks = ['asia']
    Ns = [10]
    root_dir = TESTDATA_DIR + '/experiments'
    ct = error_analysis(series, networks, Ns,
                        {'criterion': 'mi', 'metric': 'status'}, root_dir)

    assert ct == {'ext': {'eqv-hi': 0, 'eqv-sim': 0, 'non-hi': 1},
                  'ok': {'eqv-hi': 1, 'eqv-sim': 1, 'non-hi': 0},
                  'rev': {'eqv-hi': 1, 'eqv-sim': 0, 'non-hi': 1}}


def systest_error_analysis_asia_ok_5():  # Asia N=10, lt5 crit, ok metric
    series = ['HC/STD']
    networks = ['asia']
    Ns = [10]
    root_dir = TESTDATA_DIR + '/experiments'
    ct = error_analysis(series, networks, Ns, {'criterion': 'lt5',
                                               'metric': 'ok'}, root_dir)
    assert ct == {'err': {'1/2 --> 1': 3}, 'ok': {'1/2 --> 1': 2}}


def systest_error_analysis_asia_ok_6():  # Asia N=1K, eqv crit, ok metric
    series = ['HC/STD']
    networks = ['asia']
    Ns = [1000]
    root_dir = TESTDATA_DIR + '/experiments'
    ct = error_analysis(series, networks, Ns, {'criterion': 'eqv',
                                               'metric': 'ok'}, root_dir)
    print(ct)
    assert ct == {'err': {'eqv': 4, 'non': 3}, 'ok': {'eqv': 2, 'non': 0}}


def systest_error_analysis_asia_ok_7():  # Asia N=1K, eqv crit, ok metric
    series = ['HC/STD']
    networks = ['asia']
    Ns = [1000]
    root_dir = TESTDATA_DIR + '/experiments'
    ct = error_analysis(series, networks, Ns,
                        {'criterion': 'lt5', 'metric': 'status'}, root_dir)
    print(ct)
    assert ct == \
        {'eqv': {'0': 1, '1/16 --> 1/8': 1, '1/4 --> 1/2': 0},
         'ext': {'0': 1, '1/16 --> 1/8': 0, '1/4 --> 1/2': 1},
         'ok': {'0': 1, '1/16 --> 1/8': 1, '1/4 --> 1/2': 0},
         'rev': {'0': 1, '1/16 --> 1/8': 2, '1/4 --> 1/2': 0}}
