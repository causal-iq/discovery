
# system testing of summary analysis

import pytest
from math import isnan

from experiments.run_analysis import run_analysis
from fileio.common import TESTDATA_DIR

ROOT_DIR = TESTDATA_DIR + '/experiments'


def systest_summary_analysis_type_error_1():  # missing args argument
    with pytest.raises(TypeError):
        run_analysis()


def systest_summary_analysis_no_args_1_ok():  # missing mandatory argument
    assert run_analysis({}) is None
    assert run_analysis({'action': 'summary'}) is None


def systest_summary_analysis_bad_arg_1_ok():  # invalid argument values
    assert run_analysis({'action': 'invalid'}) is None
    assert run_analysis({'action': 'summary', 'series': 'invalid'}) is None
    assert run_analysis({'action': 'summary',
                         'series': 'TABU/BASE', 'N': 'invalid'}) is None
    assert run_analysis({'action': 'summary', 'series': 'TABU/BASE', 'N': '1k',
                         'maxtime': 'invalid'}) is None
    assert run_analysis({'action': 'summary', 'series': 'TABU/BASE', 'N': '1k',
                         'maxtime': '10000'}) is None


def systest_summary_analysis_base2_1_ok():  # single series & metric
    means, _ = run_analysis({'action': 'summary', 'series': 'TABU/BASE2',
                             'N': '10;;0-2', 'params': 'summ',
                             'networks': 'asia', 'metrics': 'f1-e'}, ROOT_DIR)
    assert means.to_dict('index') == {'TABU/BASE2': {'f1-e': 0.154}}


def systest_summary_analysis_base2_2_ok():  # single series, multi metrics
    means, _ = run_analysis({'action': 'summary', 'series': 'TABU/BASE2',
                             'N': '1K', 'params': 'summ',
                             'networks': 'asia',
                             'metrics': 'f1-e,f1-e-std,expts,score'}, ROOT_DIR)
    assert means.to_dict('index') == \
        {'TABU/BASE2': {'f1-e': 0.747, 'f1-e-std': 0.3222, 'expts': 3.0,
                        'score': -2.274}}


def systest_summary_analysis_base2_3_ok():  # multi series, single metrics
    means, _ = run_analysis({'action': 'summary',
                             'series': 'TABU/BASE2,TABU/STD',
                             'N': '100;;0-1', 'params': 'summ',
                             'networks': 'asia', 'metrics': 'time'}, ROOT_DIR)
    assert means.to_dict('index') == \
        {'TABU/BASE2': {'time': 1.3},
         'TABU/STD': {'time': 6.9}}


def systest_summary_analysis_base2_4_ok():  # multi series, multi metrics
    means = run_analysis({'action': 'summary',
                          'series': 'TABU/BASE2,TABU/STD',
                          'N': '1k;;0-2', 'params': 'summ',
                          'networks': 'asia,sachs',
                          'metrics': 'f1-e,f1-e-std,time'},
                         ROOT_DIR)[0].to_dict('index')
    assert isnan(means['TABU/STD'].pop('f1-e-std'))  # can't compare nans
    assert means == \
        {'TABU/BASE2': {'f1-e': 0.7204, 'f1-e-std': 0.2306, 'time': 3.1},
         'TABU/STD': {'f1-e': 0.733, 'time': 19.1}}
