
# Testing the Oracle concrete implementation of Data

import pytest

from fileio.common import TESTDATA_DIR
from fileio.data import Data
from fileio.oracle import Oracle
from core.bn import BN


def test_data_type_error_1():  # cannot call constructor directly
    with pytest.raises(TypeError):
        Data()


def test_create_type_error_1():  # bad bn type
    with pytest.raises(TypeError):
        Oracle(bn=None)
    with pytest.raises(TypeError):
        Oracle(bn=2)
    with pytest.raises(TypeError):
        Oracle(bn=False)
    with pytest.raises(TypeError):
        Oracle(bn=12.7)
    with pytest.raises(TypeError):
        Oracle(bn=[2])


def test_create_ab_1_ok():  # A-->B BN
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = Oracle(bn=bn)

    assert isinstance(data, Oracle)
    assert data.bn == bn
    assert data.order == (0, 1)
    assert data.nodes == ('A', 'B')
    assert data.N == 1
    assert data.ext_to_orig == {'A': 'A', 'B': 'B'}
    assert data.orig_to_ext == {'A': 'A', 'B': 'B'}
    assert data.node_types == {'A': 'category',
                               'B': 'category'}
    assert data.dstype == 'categorical'

    assert data.get_order() == ('A', 'B')


def test_create_cancer_2_ok():  # Cancer BN
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Oracle(bn=bn)

    assert isinstance(data, Oracle)
    assert data.bn == bn
    assert data.order == (0, 1, 2, 3, 4)
    assert data.nodes == ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray')
    assert data.N == 1
    assert data.ext_to_orig == \
        {'Cancer': 'Cancer', 'Dyspnoea': 'Dyspnoea', 'Pollution': 'Pollution',
         'Smoker': 'Smoker', 'Xray': 'Xray'}
    assert data.orig_to_ext == \
        {'Cancer': 'Cancer', 'Dyspnoea': 'Dyspnoea', 'Pollution': 'Pollution',
         'Smoker': 'Smoker', 'Xray': 'Xray'}
    assert data.node_types == {'Cancer': 'category',
                               'Dyspnoea': 'category',
                               'Pollution': 'category',
                               'Smoker': 'category',
                               'Xray': 'category'}
    assert data.dstype == 'categorical'

    assert data.get_order() == \
        ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray')


def test_create_asia_2_ok():  # Asia BN
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Oracle(bn=bn)

    assert isinstance(data, Oracle)
    assert data.bn == bn
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.N == 1
    assert data.ext_to_orig == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}

    assert data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.node_types == \
        {'asia': 'category',
         'bronc': 'category',
         'dysp': 'category',
         'either': 'category',
         'lung': 'category',
         'smoke': 'category',
         'tub': 'category',
         'xray': 'category'}
    assert data.dstype == 'categorical'


# Test set_N function

def test_set_N_type_error_1():  # Asia BN - no args
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Oracle(bn=bn)

    with pytest.raises(TypeError):
        data.set_N()


def test_set_N_type_error_2():  # Asia, BN - non-integer arg
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Oracle(bn=bn)

    with pytest.raises(TypeError):
        data.set_N(2.1)
    with pytest.raises(TypeError):
        data.set_N(True)
    with pytest.raises(TypeError):
        data.set_N([2])


def test_set_N_type_error_3():  # Asia, BN - seed is not None
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Oracle(bn=bn)

    with pytest.raises(TypeError):
        data.set_N(10, seed=0)
    with pytest.raises(TypeError):
        data.set_N(10, seed=True)
    with pytest.raises(TypeError):
        data.set_N(10, seed=2.2)


def test_set_N_value_error_4_ok():  # Asia BN - set non-positive N
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Oracle(bn=bn)

    with pytest.raises(ValueError):
        data.set_N(0)
    with pytest.raises(ValueError):
        data.set_N(-3)


def test_set_N_asia_1_ok():  # Asia BN - set N to 50
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Oracle(bn=bn)
    data.set_N(50)

    assert isinstance(data, Oracle)
    assert data.bn == bn
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.N == 50
    assert data.ext_to_orig == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}

    assert data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')

    # Note can increase sample size too

    data.set_N(80)

    assert isinstance(data, Oracle)
    assert data.bn == bn
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert data.N == 80
    assert data.ext_to_orig == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}
    assert data.orig_to_ext == \
        {'asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp', 'either': 'either',
         'lung': 'lung', 'smoke': 'smoke', 'tub': 'tub', 'xray': 'xray'}

    assert data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')


# Test set_order function

def test_set_order_type_error_1_ok():  # Asia BN - no args
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Oracle(bn=bn)

    with pytest.raises(TypeError):
        data.set_order()


def test_set_order_type_error_2_ok():  # Asia BN - bad arg type
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Oracle(bn=bn)

    with pytest.raises(TypeError):
        data.set_order(None)
    with pytest.raises(TypeError):
        data.set_order(12)
    with pytest.raises(TypeError):
        data.set_order(list(data.nodes))
    with pytest.raises(TypeError):
        data.set_order(tuple([1, 2]))


def test_set_order_value_error_1_ok():  # Asia BN - names mismatch
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Oracle(bn=bn)
    nodes = bn.dag.nodes

    with pytest.raises(ValueError):
        data.set_order(tuple())
    with pytest.raises(ValueError):
        data.set_order(tuple(nodes + ['extra']))
    with pytest.raises(ValueError):
        data.set_order(tuple([n for n in nodes if n != 'asia']))


def test_set_order_asia_1_ok():  # Asia BN - optimal/worst/original order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Oracle(bn=bn)
    std_order = tuple(bn.dag.nodes)

    # switch to optimal order

    order = tuple(bn.dag.ordered_nodes())
    assert order == \
        ('asia', 'smoke', 'bronc', 'lung', 'tub', 'either', 'dysp', 'xray')
    data.set_order(order)

    assert isinstance(data, Oracle)
    assert data.bn == bn
    assert data.order == (0, 5, 1, 4, 6, 3, 2, 7)
    assert data.nodes == std_order
    assert data.N == 1
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() does reflect new order

    assert data.get_order() == order

    # switch to worst order

    order = order[::-1]
    assert order == \
        ('xray', 'dysp', 'either', 'tub', 'lung', 'bronc', 'smoke', 'asia')
    data.set_order(order)

    assert isinstance(data, Oracle)
    assert data.bn == bn
    assert data.order == (7, 2, 3, 6, 4, 1, 5, 0)
    assert data.nodes == std_order
    assert data.N == 1
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() does reflect new order

    assert data.get_order() == order

    # revert to standard order

    data.set_order(std_order)

    assert isinstance(data, Oracle)
    assert data.bn == bn
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == std_order
    assert data.N == 1
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() does reflect new order

    assert data.get_order() == std_order
