
#   Test the basics of Gaussian hc hill-climbing structure learning

# Demonstrates that bnlearn and bnbench produce same results and are
# both snsitive to variable ordering with continuous variables.

import pytest
from pandas import set_option

from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from core.bn import BN
from learn.hc import hc
from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_learn


# Set Pandas options to display whole dataframe
@pytest.fixture
def showall():
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)


# Helper to check bnlearn and bnbench learn same graphs
def check_hc(id, N, expect=None, reverse=False):
    """
        Run hc test, comparing bnbench with bnlearn.

        :param str id: id which specifies datafile
        :param int N: sample size
        :param str expect: expected learnt network specified as string
        :param bool reverse: whether to reverse data set order
    """
    if id == '/xdsl/gauss':
        data = Pandas.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                           dstype='continuous', N=N)
    else:
        bn = BN.read(TESTDATA_DIR + id + '.xdsl')
        data = Pandas(bn.generate_cases(N))
    if reverse is True:
        data.set_order(tuple(list(data.get_order())[::-1]))
    # print('\n\n{}'.format(data.sample.head()))

    context = {'id': ('test/hc/{}/N{}'.format(id.replace('/xdsl/', ''), N)),
               'in': id[1:]}
    params = {'score': 'bic-g'}

    dag, trace = hc(data, context=context, params=params)
    print('\n{} rows for {}:\n{}\n{}'.format(N, id, trace, dag))

    dag_b, trace_b = bnlearn_learn('hc', data, context=context,
                                   params=params)

    print('\nbnlearn:\n{}\n{}'.format(trace_b, dag_b))

    if expect is not None:
        assert dag.to_string() == expect
        if id == '/xdsl/gauss' and dag_b != dag:
            print('\n++ BNlearn:\n{}\n\nBNbench:\n{}\n'.format(dag_b, dag))
        else:
            assert dag_b.to_string() == expect
    else:
        print(dag.to_string())


# --- bivariate networks

# these show variable ordering has an effect - reverse=True, reverses
# order of columns in the DataFrame

# X->Y 10 rows, no trace
@requires_r_and_bnlearn
def test_hc_xy_10_ok(showall):
    check_hc('/xdsl/xy', 10, '[X][Y|X]')


# X->Y 100 rows
@requires_r_and_bnlearn
def test_hc_xy_100_ok(showall):
    check_hc('/xdsl/xy', 100, '[X][Y|X]')
    check_hc('/xdsl/xy', 100, '[X|Y][Y]', reverse=True)


# X->Y 10K rows
@requires_r_and_bnlearn
def test_hc_xy_10k_ok(showall):
    check_hc('/xdsl/xy', 10000, '[X][Y|X]')
    check_hc('/xdsl/xy', 10000, '[X|Y][Y]', reverse=True)


# --- This shows that reversing the function from y = f(x) to x = f(y) has
# --- no effect; variable ordering still the important thing

# X<-Y 10K rows
@requires_r_and_bnlearn
def test_hc_yx_10_ok(showall):
    check_hc('/xdsl/yx', 10, '[X][Y|X]')
    check_hc('/xdsl/yx', 10, '[X|Y][Y]', reverse=True)


# X<-Y 10K rows
@requires_r_and_bnlearn
def test_hc_yx_100_ok(showall):
    check_hc('/xdsl/yx', 100, '[X][Y|X]')
    check_hc('/xdsl/yx', 100, '[X|Y][Y]', reverse=True)


# X<-Y 10K rows
@requires_r_and_bnlearn
def test_hc_yx_10k_ok(showall):
    check_hc('/xdsl/yx', 10000, '[X][Y|X]')
    check_hc('/xdsl/yx', 10000, '[X|Y][Y]', reverse=True)


# --- These all correctly show independence regardless of variable order

# X Y 10 rows
@requires_r_and_bnlearn
def test_hc_x_y_10_ok(showall):
    check_hc('/xdsl/x_y', 10, '[X][Y]')
    check_hc('/xdsl/x_y', 10, '[X][Y]', reverse=True)


# X Y 100 rows
@requires_r_and_bnlearn
def test_hc_x_y_100_ok(showall):
    check_hc('/xdsl/x_y', 100, '[X][Y]')
    check_hc('/xdsl/x_y', 100, '[X][Y]', reverse=True)


# X Y 10k rows
@requires_r_and_bnlearn
def test_hc_x_y_10k_ok(showall):
    check_hc('/xdsl/x_y', 10000, '[X][Y]')
    check_hc('/xdsl/x_y', 10000, '[X][Y]', reverse=True)


# --- Trivariate networks
# --- Chain X->Y->Z again shows influence of variable ordering

# X->Y->Z 10 rows
@requires_r_and_bnlearn
def test_hc_xyz_10_ok(showall):
    check_hc('/xdsl/xyz', 10, '[X][Y|X][Z|Y]')
    check_hc('/xdsl/xyz', 10, '[X|Y][Y|Z][Z]', reverse=True)


# X->Y->Z 100 rows, reversed order
@requires_r_and_bnlearn
def test_hc_xyz_100_ok(showall):
    check_hc('/xdsl/xyz', 100, '[X][Y|X][Z|Y]')
    check_hc('/xdsl/xyz', 100, '[X|Y][Y|Z][Z]', reverse=True)


# X->Y->Z 10k rows
@requires_r_and_bnlearn
def test_hc_xyz_10k_ok(showall):
    check_hc('/xdsl/xyz', 10000, '[X][Y|X][Z|Y]')
    check_hc('/xdsl/xyz', 10000, '[X|Y][Y|Z][Z]', reverse=True)


# --- Reverse chain X<-Y<-Z again shows influence of variable order.
# --- The very weak strength of edge X--Y means it is not discovered at N=10,
# -   At 100 X<-Z<-Y is delvered with XYZ ordering and X Y<-Z with ZYX ordering
# -   At 10K chain learnt with direction driven by variable order

# X<-Y<-Z 10 rows
@requires_r_and_bnlearn
def test_hc_zyx_10_ok(showall):
    check_hc('/xdsl/zyx', 10, '[X][Y][Z|Y]')
    check_hc('/xdsl/zyx', 10, '[X][Y|Z][Z]', reverse=True)


# X<-Y<-Z 100 rows
@requires_r_and_bnlearn
def test_hc_zyx_100_ok(showall):
    check_hc('/xdsl/zyx', 100, '[X|Z][Y][Z|Y]')
    check_hc('/xdsl/zyx', 100, '[X|Z][Y|Z][Z]', reverse=True)


# X<-Y<-Z 10k rows
@requires_r_and_bnlearn
def test_hc_zyx_10k_ok(showall):
    check_hc('/xdsl/zyx', 10000, '[X|Y][Y|Z][Z]', reverse=True)


# --- Collider X->Y<-Z
# --- With XYZ ordering, collider learnt correctly (adds X->Y, then Y<-Z)
# --- With ZYX ordering adds Y->X, then Z->X and then Z->Y to compensate
# --- (at N=10, only Y->X discovered)

# X->Y<-Z 10 rows
@requires_r_and_bnlearn
def test_hc_xy_zy_10_ok(showall):
    check_hc('/xdsl/xy_zy', 10, '[X][Y|X:Z][Z]')
    check_hc('/xdsl/xy_zy', 10, '[X|Y][Y][Z]', reverse=True)


# X->Y<-Z 100 rows
@requires_r_and_bnlearn
def test_hc_xy_zy_100_ok(showall):
    check_hc('/xdsl/xy_zy', 100, '[X][Y|X:Z][Z]')
    check_hc('/xdsl/xy_zy', 100, '[X|Y:Z][Y|Z][Z]', reverse=True)


# X->Y<-Z 10k rows
@requires_r_and_bnlearn
def test_hc_xy_zy_10k_ok(showall):
    check_hc('/xdsl/xy_zy', 10000, '[X][Y|X:Z][Z]')
    check_hc('/xdsl/xy_zy', 10000, '[X|Y:Z][Y|Z][Z]', reverse=True)


# --- Seven node bnlearn example gaussian model.

# Gauss 10 rows
@requires_r_and_bnlearn
def test_hc_gauss_10_ok(showall):
    check_hc('/xdsl/gauss', 10, '[A][B|A][C|A:D][D|A:B:E:G][E][F|C:E:G][G]')
    check_hc('/xdsl/gauss', 10, '[A|D][B|A:D:E:G][C|A:D][D][E][F|C:E:G][G]',
             reverse=True)


# Gauss 100 rows
@requires_r_and_bnlearn
def test_hc_gauss_100_ok(showall):
    check_hc('/xdsl/gauss', 100, '[A][B][C|A:B][D|B][E][F|A:D:E:G][G]')
    check_hc('/xdsl/gauss', 100, '[A][B|D][C|A:B][D][E][F|A:D:E:G][G]',
             reverse=True)


# Gauss 1K rows
@requires_r_and_bnlearn
def test_hc_gauss_1K_ok(showall):
    check_hc('/xdsl/gauss', 1000, '[A][B][C|A:B][D|B][E][F|A:D:E:G][G]')
    check_hc('/xdsl/gauss', 1000, '[A][B|D][C|A:B][D][E][F|A:D:E:G][G]',
             reverse=True)
