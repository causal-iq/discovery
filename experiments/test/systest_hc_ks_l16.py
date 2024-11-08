
# system testing of run_learn structure learning entry point
# - checking that traces are repeatable for HC/KS_L16 learning

import pytest

from run_learn import run_learn
from fileio.common import TESTDATA_DIR


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


def systest_run_learn_no_arg_1_ok():  # parameters empty
    assert run_learn({}) is None


def systest_run_learn_no_arg_2_ok():  # missing mandatory args
    assert run_learn({'action': None}) is None
    assert run_learn({'series': None}) is None


def systest_run_learn_no_arg_3_ok():  # missing mandatory args
    assert run_learn({'action': None, 'series': None}) is None


def systest_run_learn_hc_ks_l16_asia_10_ok():  # HC/KS_L16 asia N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'asia', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_asia_20_ok():  # HC/KS_L16 asia N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'asia', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_asia_40_ok():  # HC/KS_L16 asia N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'asia', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_asia_50_ok():  # HC/KS_L16 asia N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'asia', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_asia_80_ok():  # HC/KS_L16 asia N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'asia', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_asia_100_ok():  # HC/KS_L16 asia N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'asia', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_asia_200_ok():  # HC/KS_L16 asia N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'asia', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_asia_400_ok():  # HC/KS_L16 asia N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'asia', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_asia_500_ok():  # HC/KS_L16 asia N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'asia', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_asia_800_ok():  # HC/KS_L16 asia N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'asia', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_asia_1k_ok():  # HC/KS_L16 asia N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'asia', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sports_10_ok():  # HC/KS_L16 sports N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sports', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sports_20_ok():  # HC/KS_L16 sports N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sports', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sports_40_ok():  # HC/KS_L16 sports N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sports', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sports_50_ok():  # HC/KS_L16 sports N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sports', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sports_80_ok():  # HC/KS_L16 sports N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sports', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sports_100_ok():  # HC/KS_L16 sports N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sports', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sports_200_ok():  # HC/KS_L16 sports N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sports', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sports_400_ok():  # HC/KS_L16 sports N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sports', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sports_500_ok():  # HC/KS_L16 sports N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sports', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sports_800_ok():  # HC/KS_L16 sports N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sports', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sports_1k_ok():  # HC/KS_L16 sports N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sports', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sachs_10_ok():  # HC/KS_L16 sachs N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sachs', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sachs_20_ok():  # HC/KS_L16 sachs N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sachs', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sachs_40_ok():  # HC/KS_L16 sachs N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sachs', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sachs_50_ok():  # HC/KS_L16 sachs N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sachs', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sachs_80_ok():  # HC/KS_L16 sachs N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sachs', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sachs_100_ok():  # HC/KS_L16 sachs N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sachs', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sachs_200_ok():  # HC/KS_L16 sachs N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sachs', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sachs_400_ok():  # HC/KS_L16 sachs N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sachs', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sachs_500_ok():  # HC/KS_L16 sachs N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sachs', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sachs_800_ok():  # HC/KS_L16 sachs N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sachs', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_sachs_1k_ok():  # HC/KS_L16 sachs N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'sachs', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_child_10_ok():  # HC/KS_L16 child N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'child', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_child_20_ok():  # HC/KS_L16 child N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'child', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_child_40_ok():  # HC/KS_L16 child N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'child', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_child_50_ok():  # HC/KS_L16 child N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'child', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_child_80_ok():  # HC/KS_L16 child N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'child', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_child_100_ok():  # HC/KS_L16 child N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'child', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_child_200_ok():  # HC/KS_L16 child N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'child', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_child_400_ok():  # HC/KS_L16 child N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'child', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_child_500_ok():  # HC/KS_L16 child N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'child', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_child_800_ok():  # HC/KS_L16 child N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'child', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_child_1k_ok():  # HC/KS_L16 child N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'child', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_insurance_10_ok():  # HC/KS_L16 insurance N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'insurance', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_insurance_20_ok():  # HC/KS_L16 insurance N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'insurance', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_insurance_40_ok():  # HC/KS_L16 insurance N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'insurance', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_l16_insurance_50_ok():  # HC/KS_L16 insurance N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'insurance', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_insurance_80_ok():  # HC/KS_L16 insurance N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'insurance', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_insurance_100_ok():  # HC/KS_L16 insur N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'insurance', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_insurance_200_ok():  # HC/KS_L16 insur N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'insurance', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_insurance_400_ok():  # HC/KS_L16 insur N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'insurance', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_insurance_500_ok():  # HC/KS_L16 insur N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'insurance', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_insurance_800_ok():  # HC/KS_L16 insur N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'insurance', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_insurance_1k_ok():  # HC/KS_L16 insurance N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'insurance', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_property_10_ok():  # HC/KS_L16 property N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'property', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_property_20_ok():  # HC/KS_L16 property N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'property', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_property_40_ok():  # HC/KS_L16 property N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'property', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_property_50_ok():  # HC/KS_L16 property N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'property', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_property_80_ok():  # HC/KS_L16 property N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'property', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_property_100_ok():  # HC/KS_L16 property N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'property', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_property_200_ok():  # HC/KS_L16 property N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'property', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_property_400_ok():  # HC/KS_L16 property N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'property', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_property_500_ok():  # HC/KS_L16 property N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'property', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_property_800_ok():  # HC/KS_L16 property N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'property', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def systest_run_learn_hc_ks_l16_property_1k_ok():  # HC/KS_L16 property N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'property', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_diarrhoea_ok_10():  # HC/KS_L16 diarrhoea N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'diarrhoea', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_diarrhoea_ok_20():  # HC/KS_L16 diarrhoea N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'diarrhoea', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_diarrhoea_ok_40():  # HC/KS_L16 diarrhoea N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'diarrhoea', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_diarrhoea_ok_50():  # HC/KS_L16 diarrhoea N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'diarrhoea', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_diarrhoea_ok_80():  # HC/KS_L16 diarrhoea N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'diarrhoea', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_diarrhoea_ok_100():  # HC/KS_L16 diarr N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'diarrhoea', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_diarrhoea_ok_200():  # HC/KS_L16 diarr N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'diarrhoea', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_diarrhoea_ok_400():  # HC/KS_L16 diarr N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'diarrhoea', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_diarrhoea_ok_500():  # HC/KS_L16 diarr N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'diarrhoea', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_diarrhoea_ok_800():  # HC/KS_L16 diarr N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'diarrhoea', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_diarrhoea_ok_1k():  # HC/KS_L16 diarrhoea N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'diarrhoea', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_water_ok_10():  # HC/KS_L16 water N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'water', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_water_ok_20():  # HC/KS_L16 water N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'water', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_water_ok_40():  # HC/KS_L16 water N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'water', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_water_ok_50():  # HC/KS_L16 water N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'water', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_water_ok_80():  # HC/KS_L16 water N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'water', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_water_ok_100():  # HC/KS_L16 water N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'water', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_water_ok_200():  # HC/KS_L16 water N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'water', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_water_ok_400():  # HC/KS_L16 water N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'water', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_water_ok_500():  # HC/KS_L16 water N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'water', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_water_ok_800():  # HC/KS_L16 water N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'water', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_water_ok_1k():  # HC/KS_L16 water N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'water', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_mildew_ok_10():  # HC/KS_L16 mildew N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'mildew', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_mildew_ok_20():  # HC/KS_L16 mildew N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'mildew', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_mildew_ok_40():  # HC/KS_L16 mildew N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'mildew', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_mildew_ok_50():  # HC/KS_L16 mildew N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'mildew', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_mildew_ok_80():  # HC/KS_L16 mildew N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'mildew', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_mildew_ok_100():  # HC/KS_L16 mildew N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'mildew', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_mildew_ok_200():  # HC/KS_L16 mildew N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'mildew', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_mildew_ok_400():  # HC/KS_L16 mildew N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'mildew', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_mildew_ok_500():  # HC/KS_L16 mildew N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'mildew', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_mildew_ok_800():  # HC/KS_L16 mildew N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'mildew', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_mildew_ok_1k():  # HC/KS_L16 mildew N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'mildew', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_alarm_ok_10():  # HC/KS_L16 alarm N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'alarm', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_alarm_ok_20():  # HC/KS_L16 alarm N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'alarm', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_alarm_ok_40():  # HC/KS_L16 alarm N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'alarm', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_alarm_ok_50():  # HC/KS_L16 alarm N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'alarm', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_alarm_ok_80():  # HC/KS_L16 alarm N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'alarm', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_alarm_ok_100():  # HC/KS_L16 alarm N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'alarm', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_alarm_ok_200():  # HC/KS_L16 alarm N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'alarm', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_alarm_ok_400():  # HC/KS_L16 alarm N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'alarm', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_alarm_ok_500():  # HC/KS_L16 alarm N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'alarm', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_alarm_ok_800():  # HC/KS_L16 alarm N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'alarm', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_alarm_ok_1k():  # HC/KS_L16 alarm N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'alarm', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_barley_ok_10():  # HC/KS_L16 barley N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'barley', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_barley_ok_20():  # HC/KS_L16 barley N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'barley', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_barley_ok_40():  # HC/KS_L16 barley N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'barley', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_barley_ok_50():  # HC/KS_L16 barley N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'barley', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_barley_ok_80():  # HC/KS_L16 barley N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'barley', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_barley_ok_100():  # HC/KS_L16 barley N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'barley', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_barley_ok_200():  # HC/KS_L16 barley N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'barley', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_barley_ok_400():  # HC/KS_L16 barley N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'barley', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_barley_ok_500():  # HC/KS_L16 barley N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'barley', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_barley_ok_800():  # HC/KS_L16 barley N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'barley', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_barley_ok_1k():  # HC/KS_L16 barley N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'barley', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_hailfinder_ok_10():  # HC/KS_L16 hailf N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hailfinder', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_hailfinder_ok_20():  # HC/KS_L16 hailf N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hailfinder', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hailfinder_ok_40():  # HC/KS_L16 hailf N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hailfinder', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hailfinder_ok_50():  # HC/KS_L16 hailf N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hailfinder', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hailfinder_ok_80():  # HC/KS_L16 hailf N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hailfinder', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hailfinder_ok_100():  # HC/KS_L16 hailf N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hailfinder', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hailfinder_ok_200():  # HC/KS_L16 hailf N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hailfinder', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hailfinder_ok_400():  # HC/KS_L16 hailf N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hailfinder', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hailfinder_ok_500():  # HC/KS_L16 hailf N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hailfinder', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hailfinder_ok_800():  # HC/KS_L16 hailf N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hailfinder', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hailfinder_ok_1k():  # HC/KS_L16 hailf N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hailfinder', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_hepar2_ok_10():  # HC/KS_L16 hepar2 N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hepar2', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_hepar2_ok_20():  # HC/KS_L16 hepar2 N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hepar2', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hepar2_ok_40():  # HC/KS_L16 hepar2 N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hepar2', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hepar2_ok_50():  # HC/KS_L16 hepar2 N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hepar2', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hepar2_ok_80():  # HC/KS_L16 hepar2 N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hepar2', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hepar2_ok_100():  # HC/KS_L16 hepar2 N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hepar2', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hepar2_ok_200():  # HC/KS_L16 hepar2 N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hepar2', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hepar2_ok_400():  # HC/KS_L16 hepar2 N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hepar2', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hepar2_ok_500():  # HC/KS_L16 hepar2 N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hepar2', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hepar2_ok_800():  # HC/KS_L16 hepar2 N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hepar2', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_hepar2_ok_1k():  # HC/KS_L16 hepar2 N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'hepar2', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_win95pts_ok_10():  # HC/KS_L16 win95pts N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'win95pts', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_win95pts_ok_20():  # HC/KS_L16 win95pts N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'win95pts', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_win95pts_ok_40():  # HC/KS_L16 win95pts N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'win95pts', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_win95pts_ok_50():  # HC/KS_L16 win95pts N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'win95pts', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_win95pts_ok_80():  # HC/KS_L16 win95pts N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'win95pts', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_win95pts_ok_100():  # HC/KS_L16 win95pts N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'win95pts', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_win95pts_ok_200():  # HC/KS_L16 win95pts N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'win95pts', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_win95pts_ok_400():  # HC/KS_L16 win95pts N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'win95pts', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_win95pts_ok_500():  # HC/KS_L16 win95pts N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'win95pts', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_win95pts_ok_800():  # HC/KS_L16 win95pts N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'win95pts', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_win95pts_ok_1k():  # HC/KS_L16 win95pts N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'win95pts', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_formed_ok_10():  # HC/KS_L16 formed N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'formed', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_formed_ok_20():  # HC/KS_L16 formed N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'formed', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_formed_ok_40():  # HC/KS_L16 formed N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'formed', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_formed_ok_50():  # HC/KS_L16 formed N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'formed', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_formed_ok_80():  # HC/KS_L16 formed N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'formed', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_formed_ok_100():  # HC/KS_L16 formed N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'formed', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_formed_ok_200():  # HC/KS_L16 formed N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'formed', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_formed_ok_400():  # HC/KS_L16 formed N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'formed', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_formed_ok_500():  # HC/KS_L16 formed N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'formed', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_formed_ok_800():  # HC/KS_L16 formed N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'formed', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


# seems to be inconsistent how mean_N computed - remove test for now

@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_formed_ok_1k():  # HC/KS_L16 formed N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'formed', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_pathfinder_ok_10():  # HC/KS_L16 pathf N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'pathfinder', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def xsystest_run_learn_hc_ks_l16_pathfinder_ok_20():  # HC/KS_L16 pathf N=20
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'pathfinder', 'N': '20', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_pathfinder_ok_40():  # HC/KS_L16 pathf N=40
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'pathfinder', 'N': '40', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_pathfinder_ok_50():  # HC/KS_L16 pathf N=50
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'pathfinder', 'N': '50', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_pathfinder_ok_80():  # HC/KS_L16 pathf N=80
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'pathfinder', 'N': '80', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_pathfinder_ok_100():  # HC/KS_L16 pathf N=100
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'pathfinder', 'N': '100', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_pathfinder_ok_200():  # HC/KS_L16 pathf N=200
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'pathfinder', 'N': '200', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_pathfinder_ok_400():  # HC/KS_L16 pathf N=400
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'pathfinder', 'N': '400', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_pathfinder_ok_500():  # HC/KS_L16 pathf N=500
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'pathfinder', 'N': '500', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_pathfinder_ok_800():  # HC/KS_L16 pathf N=800
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'pathfinder', 'N': '800', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


@pytest.mark.slow
def xsystest_run_learn_hc_ks_l16_pathfinder_ok_1k():  # HC/KS_L16 pathf N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_L16',
                      'networks': 'pathfinder', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')
