
# system testing of run_learn structure learning entry point
# - checking that traces are repeatable for required arc knowledge runs

from run_learn import run_learn
from fileio.common import TESTDATA_DIR


# 16 pieces of required arc info

def systest_run_learn_hc_ks_r16_e100_asia_10_ok():  # HC/KS_R16_E100 asia N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_R16_E100',
                      'networks': 'asia', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_r16_e100_asia_1k_ok():  # HC/KS_R16_E100 asia N=1k
    assert run_learn({'action': 'compare', 'series': 'HC/KS_R16_E100',
                      'networks': 'asia', 'N': '1k', 'nodes': None},
                     TESTDATA_DIR + '/experiments')


def systest_run_learn_hc_ks_r16_sports_10_ok():  # HC/KS_R16_E100 sports N=10
    assert run_learn({'action': 'compare', 'series': 'HC/KS_R16_E100',
                      'networks': 'sports', 'N': '10', 'nodes': None},
                     TESTDATA_DIR + '/experiments')
