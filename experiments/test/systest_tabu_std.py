
# system testing of run_learn structure learning entry point
# - checking that traces are repeatable for TABU/STD learning

import pytest

from run_learn import run_learn
from fileio.common import TESTDATA_DIR

Ns = [10, 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000]
# Ns = [10, 20]


def systest_run_learn_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        run_learn()


def systest_run_learn_type_error_2():  # args is not a dict
    with pytest.raises(TypeError):
        run_learn(16)
    with pytest.raises(TypeError):
        run_learn('bad type')
    with pytest.raises(TypeError):
        run_learn(37.2)
    with pytest.raises(TypeError):
        run_learn(['args'])


def systest_run_learn_type_error_3():  # root_dir is not a string
    with pytest.raises(TypeError):
        run_learn({}, 16)
    with pytest.raises(TypeError):
        run_learn({}, {})
    with pytest.raises(TypeError):
        run_learn({}, 37.2)
    with pytest.raises(TypeError):
        run_learn({}, ['args'])


def systest_run_learn_no_args_1_ok():  # parameters empty
    assert run_learn({}) is None


def systest_run_learn_no_args_2_ok():  # missing mandatory args
    assert run_learn({'action': None}) is None
    assert run_learn({'series': None}) is None


def systest_run_learn_no_args_3():  # missing mandatory args
    assert run_learn({'action': None, 'series': None}) is None


def systest_run_learn_tabu_std_asia_ok_10():  # TABU/STD asia N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'asia', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_asia_ok_20():  # TABU/STD asia N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'asia', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_asia_ok_40():  # TABU/STD asia N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'asia', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_asia_ok_50():  # TABU/STD asia N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'asia', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_asia_ok_80():  # TABU/STD asia N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'asia', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_asia_ok_100():  # TABU/STD asia N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'asia', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_asia_ok_200():  # TABU/STD asia N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'asia', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_asia_ok_400():  # TABU/STD asia N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'asia', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_asia_ok_500():  # TABU/STD asia N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'asia', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_asia_ok_800():  # TABU/STD asia N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'asia', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_asia_ok_1k():  # TABU/STD asia N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'asia', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sports_ok_10():  # TABU/STD sports N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sports', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sports_ok_20():  # TABU/STD sports N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sports', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sports_ok_40():  # TABU/STD sports N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sports', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sports_ok_50():  # TABU/STD sports N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sports', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sports_ok_80():  # TABU/STD sports N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sports', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sports_ok_100():  # TABU/STD sports N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sports', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sports_ok_200():  # TABU/STD sports N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sports', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sports_ok_400():  # TABU/STD sports N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sports', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sports_ok_500():  # TABU/STD sports N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sports', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sports_ok_800():  # TABU/STD sports N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sports', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sports_ok_1k():  # TABU/STD sports N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sports', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sachs_ok_10():  # TABU/STD sachs N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sachs', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sachs_ok_20():  # TABU/STD sachs N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sachs', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sachs_ok_40():  # TABU/STD sachs N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sachs', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sachs_ok_50():  # TABU/STD sachs N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sachs', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sachs_ok_80():  # TABU/STD sachs N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sachs', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sachs_ok_100():  # TABU/STD sachs N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sachs', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sachs_ok_200():  # TABU/STD sachs N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sachs', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sachs_ok_400():  # TABU/STD sachs N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sachs', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sachs_ok_500():  # TABU/STD sachs N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sachs', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sachs_ok_800():  # TABU/STD sachs N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sachs', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_sachs_ok_1k():  # TABU/STD sachs N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'sachs', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_child_ok_10():  # TABU/STD child N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'child', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_child_ok_20():  # TABU/STD child N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'child', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_child_ok_40():  # TABU/STD child N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'child', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_child_ok_50():  # TABU/STD child N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'child', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_child_ok_80():  # TABU/STD child N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'child', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_child_ok_100():  # TABU/STD child N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'child', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_child_ok_200():  # TABU/STD child N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'child', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_child_ok_400():  # TABU/STD child N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'child', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_child_ok_500():  # TABU/STD child N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'child', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_child_ok_800():  # TABU/STD child N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'child', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_child_ok_1k():  # TABU/STD child N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'child', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_insurance_ok_10():  # TABU/STD insurance N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'insurance', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_insurance_ok_20():  # TABU/STD insurance N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'insurance', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_insurance_ok_40():  # TABU/STD insurance N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'insurance', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_insurance_ok_50():  # TABU/STD insurance N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'insurance', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_insurance_ok_80():  # TABU/STD insurance N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'insurance', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_insurance_ok_100():  # TABU/STD insurance N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'insurance', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_insurance_ok_200():  # TABU/STD insurance N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'insurance', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_insurance_ok_400():  # TABU/STD insurance N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'insurance', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


# This test gives very minor difference, so remove test for now

@pytest.mark.slow
def xsystest_run_learn_tabu_std_insurance_ok_500():  # TABU/STD insurance N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'insurance', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_insurance_ok_800():  # TABU/STD insurance N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'insurance', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_insurance_ok_1k():  # TABU/STD insurance N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'insurance', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_property_ok_10():  # TABU/STD property N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'property', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_property_ok_20():  # TABU/STD property N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'property', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_property_ok_40():  # TABU/STD property N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'property', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_property_ok_50():  # TABU/STD property N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'property', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_property_ok_80():  # TABU/STD property N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'property', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_property_ok_100():  # TABU/STD property N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'property', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_property_ok_200():  # TABU/STD property N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'property', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_property_ok_400():  # TABU/STD property N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'property', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_property_ok_500():  # TABU/STD property N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'property', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_property_ok_800():  # TABU/STD property N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'property', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_property_ok_1k():  # TABU/STD property N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'property', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_diarrhoea_ok_10():  # TABU/STD diarrhoea N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'diarrhoea', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_diarrhoea_ok_20():  # TABU/STD diarrhoea N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'diarrhoea', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_diarrhoea_ok_40():  # TABU/STD diarrhoea N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'diarrhoea', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_diarrhoea_ok_50():  # TABU/STD diarrhoea N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'diarrhoea', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_diarrhoea_ok_80():  # TABU/STD diarrhoea N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'diarrhoea', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_diarrhoea_ok_100():  # TABU/STD diarrhoea N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'diarrhoea', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_diarrhoea_ok_200():  # TABU/STD diarrhoea N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'diarrhoea', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_diarrhoea_ok_400():  # TABU/STD diarrhoea N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'diarrhoea', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_diarrhoea_ok_500():  # TABU/STD diarrhoea N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'diarrhoea', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_diarrhoea_ok_800():  # TABU/STD diarrhoea N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'diarrhoea', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_diarrhoea_ok_1k():  # TABU/STD diarrhoea N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'diarrhoea', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_water_ok_10():  # TABU/STD water N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'water', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_water_ok_20():  # TABU/STD water N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'water', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_water_ok_40():  # TABU/STD water N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'water', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_water_ok_50():  # TABU/STD water N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'water', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_water_ok_80():  # TABU/STD water N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'water', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_water_ok_100():  # TABU/STD water N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'water', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_water_ok_200():  # TABU/STD water N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'water', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_water_ok_400():  # TABU/STD water N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'water', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_water_ok_500():  # TABU/STD water N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'water', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_water_ok_800():  # TABU/STD water N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'water', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_water_ok_1k():  # TABU/STD water N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'water', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_mildew_ok_10():  # TABU/STD mildew N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'mildew', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_mildew_ok_20():  # TABU/STD mildew N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'mildew', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_mildew_ok_40():  # TABU/STD mildew N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'mildew', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_mildew_ok_50():  # TABU/STD mildew N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'mildew', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_mildew_ok_80():  # TABU/STD mildew N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'mildew', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_mildew_ok_100():  # TABU/STD mildew N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'mildew', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_mildew_ok_200():  # TABU/STD mildew N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'mildew', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_mildew_ok_400():  # TABU/STD mildew N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'mildew', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_mildew_ok_500():  # TABU/STD mildew N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'mildew', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_mildew_ok_800():  # TABU/STD mildew N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'mildew', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_mildew_ok_1k():  # TABU/STD mildew N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'mildew', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_alarm_ok_10():  # TABU/STD alarm N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'alarm', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_alarm_ok_20():  # TABU/STD alarm N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'alarm', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_alarm_ok_40():  # TABU/STD alarm N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'alarm', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_alarm_ok_50():  # TABU/STD alarm N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'alarm', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_alarm_ok_80():  # TABU/STD alarm N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'alarm', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_alarm_ok_100():  # TABU/STD alarm N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'alarm', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_alarm_ok_200():  # TABU/STD alarm N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'alarm', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_alarm_ok_400():  # TABU/STD alarm N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'alarm', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_alarm_ok_500():  # TABU/STD alarm N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'alarm', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_alarm_ok_800():  # TABU/STD alarm N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'alarm', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_alarm_ok_1k():  # TABU/STD alarm N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'alarm', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_barley_ok_10():  # TABU/STD barley N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'barley', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_barley_ok_20():  # TABU/STD barley N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'barley', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_barley_ok_40():  # TABU/STD barley N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'barley', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_barley_ok_50():  # TABU/STD barley N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'barley', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_barley_ok_80():  # TABU/STD barley N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'barley', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_barley_ok_100():  # TABU/STD barley N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'barley', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_barley_ok_200():  # TABU/STD barley N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'barley', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_barley_ok_400():  # TABU/STD barley N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'barley', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_barley_ok_500():  # TABU/STD barley N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'barley', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_barley_ok_800():  # TABU/STD barley N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'barley', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_barley_ok_1k():  # TABU/STD barley N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'barley', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_barley_ok_1():  # TABU/STD barley N=10->50
    for N in Ns:
        if N > 50:
            continue
        assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                          'networks': 'barley', 'N': '{}'.format(N),
                          'nodes': None}, TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_barley_ok_2():  # TABU/STD barley N=80->1K
    for N in Ns:
        if N <= 50:
            continue
        assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                          'networks': 'barley', 'N': '{}'.format(N),
                          'nodes': None}, TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_hailfinder_ok_10():  # TABU/STD hailfinder N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hailfinder', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_hailfinder_ok_20():  # TABU/STD hailfinder N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hailfinder', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hailfinder_ok_40():  # TABU/STD hailfinder N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hailfinder', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hailfinder_ok_50():  # TABU/STD hailfinder N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hailfinder', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hailfinder_ok_80():  # TABU/STD hailfinder N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hailfinder', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hailfinder_ok_100():  # TABU/STD hailf N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hailfinder', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hailfinder_ok_200():  # TABU/STD hailf N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hailfinder', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hailfinder_ok_400():  # TABU/STD hailf N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hailfinder', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hailfinder_ok_500():  # TABU/STD hailf N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hailfinder', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hailfinder_ok_800():  # TABU/STD hailf N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hailfinder', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hailfinder_ok_1k():  # TABU/STD hailfinder N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hailfinder', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_hepar2_ok_10():  # TABU/STD hepar2 N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hepar2', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_hepar2_ok_20():  # TABU/STD hepar2 N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hepar2', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hepar2_ok_40():  # TABU/STD hepar2 N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hepar2', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hepar2_ok_50():  # TABU/STD hepar2 N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hepar2', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hepar2_ok_80():  # TABU/STD hepar2 N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hepar2', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hepar2_ok_100():  # TABU/STD hepar2 N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hepar2', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hepar2_ok_200():  # TABU/STD hepar2 N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hepar2', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hepar2_ok_400():  # TABU/STD hepar2 N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hepar2', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hepar2_ok_500():  # TABU/STD hepar2 N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hepar2', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hepar2_ok_800():  # TABU/STD hepar2 N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hepar2', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hepar2_ok_1k():  # TABU/STD hepar2 N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'hepar2', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_hepar2_ok_1():  # TABU/STD hepar2 N=10->50
    for N in Ns:
        if N > 50:
            continue
        assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                          'networks': 'hepar2', 'N': '{}'.format(N),
                          'nodes': None}, TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_win95pts_ok_10():  # TABU/STD win95pts N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'win95pts', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_win95pts_ok_20():  # TABU/STD win95pts N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'win95pts', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_win95pts_ok_40():  # TABU/STD win95pts N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'win95pts', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_win95pts_ok_50():  # TABU/STD win95pts N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'win95pts', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_win95pts_ok_80():  # TABU/STD win95pts N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'win95pts', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_win95pts_ok_100():  # TABU/STD win95pts N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'win95pts', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_win95pts_ok_200():  # TABU/STD win95pts N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'win95pts', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_win95pts_ok_400():  # TABU/STD win95pts N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'win95pts', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_win95pts_ok_500():  # TABU/STD win95pts N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'win95pts', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_win95pts_ok_800():  # TABU/STD win95pts N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'win95pts', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_win95pts_ok_1k():  # TABU/STD win95pts N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'win95pts', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_formed_ok_10():  # TABU/STD formed N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'formed', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_formed_ok_20():  # TABU/STD formed N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'formed', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_formed_ok_40():  # TABU/STD formed N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'formed', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_formed_ok_50():  # TABU/STD formed N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'formed', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_formed_ok_80():  # TABU/STD formed N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'formed', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_formed_ok_100():  # TABU/STD formed N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'formed', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_formed_ok_200():  # TABU/STD formed N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'formed', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_formed_ok_400():  # TABU/STD formed N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'formed', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


# slight random difference in mean_N count

@pytest.mark.slow
def xsystest_run_learn_tabu_std_formed_ok_500():  # TABU/STD formed N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'formed', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_formed_ok_800():  # TABU/STD formed N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'formed', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


# seems to be inconsistent how mean_N computed - remove test for now

@pytest.mark.slow
def xsystest_run_learn_tabu_std_formed_ok_1k():  # TABU/STD formed N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'formed', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_pathfinder_ok_10():  # TABU/STD pathfinder N=10
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'pathfinder', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_tabu_std_pathfinder_ok_20():  # TABU/STD pathfinder N=20
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'pathfinder', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_pathfinder_ok_40():  # TABU/STD pathfinder N=40
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'pathfinder', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_pathfinder_ok_50():  # TABU/STD pathfinder N=50
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'pathfinder', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_pathfinder_ok_80():  # TABU/STD pathfinder N=80
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'pathfinder', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_pathfinder_ok_100():  # TABU/STD pathf N=100
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'pathfinder', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_pathfinder_ok_200():  # TABU/STD pathf N=200
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'pathfinder', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_pathfinder_ok_400():  # TABU/STD pathf N=400
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'pathfinder', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_pathfinder_ok_500():  # TABU/STD pathf N=500
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'pathfinder', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_pathfinder_ok_800():  # TABU/STD pathf N=800
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'pathfinder', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_tabu_std_pathfinder_ok_1k():  # TABU/STD pathfinder N=1k
    assert run_learn({'action': 'compare', 'series': 'TABU/STD',
                      'networks': 'pathfinder', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')
