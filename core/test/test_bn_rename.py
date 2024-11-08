
# Test BN rename function

import pytest
from copy import deepcopy

from core.common import EdgeType
from core.graph import DAG
from core.bn import BN
from fileio.common import TESTDATA_DIR


def test_bn_rename_type_error_1():  # bad argument types
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(TypeError):
        bn.rename()


def test_bn_rename_type_error_2():  # name_map not a dictionary
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(TypeError):
        bn.rename(name_map=None)
    with pytest.raises(TypeError):
        bn.rename(name_map=True)
    with pytest.raises(TypeError):
        bn.rename(name_map=37)
    with pytest.raises(TypeError):
        bn.rename(name_map=[{'A': 'B'}])


def test_bn_rename_type_error_3():  # name_map has non-string keys
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(TypeError):
        bn.rename(name_map={1: 'B'})
    with pytest.raises(TypeError):
        bn.rename(name_map={('A',): 'B'})


def test_bn_rename_type_error_4():  # name_map has non-string values
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(TypeError):
        bn.rename(name_map={'A': 1})
    with pytest.raises(TypeError):
        bn.rename(name_map={'A': ['B']})


def test_bn_rename_value_error_1():  # keys that are not current node name
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    with pytest.raises(ValueError):
        bn.rename(name_map={'C': 'Q'})


# Renames on A-->B

def test_bn_rename_bn_ab_1_ok():  # change first node name, keeping order
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    orig_bn = deepcopy(bn)
    bn.rename(name_map={'A': 'AA', 'B': 'B'})

    assert isinstance(bn.dag, DAG)
    assert bn.dag.is_directed is True
    assert bn.dag.is_partially_directed is True
    assert bn.dag.has_directed_cycles is False
    assert bn.dag.nodes == ['AA', 'B']
    assert bn.dag.edges == {('AA', 'B'): EdgeType.DIRECTED}
    assert bn.dag.parents == {'B': ['AA']}
    assert bn.cnds['AA'].cpt == {'0': 0.75, '1': 0.25}
    assert bn.cnds['B'].cpt == \
        {frozenset({('AA', '0')}): {'0': 0.5, '1': 0.5},
         frozenset({('AA', '1')}): {'0': 0.25, '1': 0.75}}

    bn.rename({'AA': 'A', 'B': 'B'})
    assert bn == orig_bn


def test_bn_rename_bn_ab_2_ok():  # change first node name, changing order
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    orig_bn = deepcopy(bn)
    bn.rename(name_map={'A': 'Z', 'B': 'B'})

    assert isinstance(bn.dag, DAG)
    assert bn.dag.is_directed is True
    assert bn.dag.is_partially_directed is True
    assert bn.dag.has_directed_cycles is False
    assert bn.dag.nodes == ['B', 'Z']
    assert bn.dag.edges == {('Z', 'B'): EdgeType.DIRECTED}
    assert bn.dag.parents == {'B': ['Z']}
    assert bn.cnds['Z'].cpt == {'0': 0.75, '1': 0.25}
    assert bn.cnds['B'].cpt == \
        {frozenset({('Z', '0')}): {'0': 0.5, '1': 0.5},
         frozenset({('Z', '1')}): {'0': 0.25, '1': 0.75}}

    bn.rename({'Z': 'A', 'B': 'B'})
    assert bn == orig_bn


def test_bn_rename_bn_ab_3_ok():  # change both names and order
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    orig_bn = deepcopy(bn)
    bn.rename(name_map={'A': 'X1A', 'B': 'X0B'})

    assert isinstance(bn.dag, DAG)
    assert bn.dag.is_directed is True
    assert bn.dag.is_partially_directed is True
    assert bn.dag.has_directed_cycles is False
    assert bn.dag.nodes == ['X0B', 'X1A']
    assert bn.dag.edges == {('X1A', 'X0B'): EdgeType.DIRECTED}
    assert bn.dag.parents == {'X0B': ['X1A']}
    assert bn.cnds['X1A'].cpt == {'0': 0.75, '1': 0.25}
    assert bn.cnds['X0B'].cpt == \
        {frozenset({('X1A', '0')}): {'0': 0.5, '1': 0.5},
         frozenset({('X1A', '1')}): {'0': 0.25, '1': 0.75}}

    bn.rename({'X1A': 'A', 'X0B': 'B'})
    assert bn == orig_bn


# renames on A-->B<--C

def test_bn_rename_bn_ab_cb_1_ok():  # change all names and order
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_cb.dsc')
    orig_bn = deepcopy(bn)
    bn.rename(name_map={'A': 'X2A', 'B': 'X0B', 'C': 'X1C'})

    assert isinstance(bn.dag, DAG)
    assert bn.dag.is_directed is True
    assert bn.dag.is_partially_directed is True
    assert bn.dag.has_directed_cycles is False
    assert bn.dag.nodes == ['X0B', 'X1C', 'X2A']
    assert bn.dag.edges == {('X2A', 'X0B'): EdgeType.DIRECTED,
                            ('X1C', 'X0B'): EdgeType.DIRECTED}
    assert bn.dag.parents == {'X0B': ['X1C', 'X2A']}
    assert bn.cnds['X2A'].cpt == {'0': 0.6666667, '1': 0.3333333}
    assert bn.cnds['X1C'].cpt == {'0': 0.3333333, '1': 0.6666667}
    assert bn.cnds['X0B'].cpt == \
        {frozenset({('X2A', '0'), ('X1C', '0')}): {'0': 0.1, '1': 0.9},
         frozenset({('X2A', '1'), ('X1C', '0')}): {'0': 0.2, '1': 0.8},
         frozenset({('X1C', '1'), ('X2A', '0')}): {'0': 0.9, '1': 0.1},
         frozenset({('X1C', '1'), ('X2A', '1')}): {'0': 0.7, '1': 0.3}}

    bn.rename({'X2A': 'A', 'X0B': 'B', 'X1C': 'C'})
    assert bn == orig_bn


# Test renames of Gaussian networks

def test_bn_rename_bn_xy_1_ok():  # change first node name, keeping order
    bn = BN.read(TESTDATA_DIR + '/xdsl/xy.xdsl')
    orig_bn = deepcopy(bn)

    name_map = {'X': 'XX', 'Y': 'Y'}
    bn.rename(name_map)
    print('\n\nBN names changed to:\n{}\nCNDs:\n{}\n'
          .format(bn.dag, '\n'.join(['{}: {}'.format(n, c)
                                     for n, c in bn.cnds.items()])))

    assert isinstance(bn.dag, DAG)
    assert bn.dag.is_directed is True
    assert bn.dag.is_partially_directed is True
    assert bn.dag.has_directed_cycles is False
    assert bn.dag.nodes == ['XX', 'Y']
    assert bn.dag.edges == {('XX', 'Y'): EdgeType.DIRECTED}
    assert bn.dag.parents == {'Y': ['XX']}
    assert tuple(bn.cnds) == ('XX', 'Y')

    assert bn.cnds['XX'].mean == 2.0
    assert bn.cnds['XX'].sd == 1.0
    assert bn.cnds['XX'].coeffs == {}

    assert bn.cnds['Y'].mean == 0.5
    assert bn.cnds['Y'].sd == 0.5
    assert bn.cnds['Y'].coeffs == {'XX': 1.5}

    # check can revert back to original BN

    bn.rename({n: o for o, n in name_map.items()})
    assert bn == orig_bn


def test_bn_rename_bn_xy_2_ok():  # change first node name, changing order
    bn = BN.read(TESTDATA_DIR + '/xdsl/xy.xdsl')
    orig_bn = deepcopy(bn)

    name_map = {'X': 'Z', 'Y': 'Y'}
    bn.rename(name_map)
    print('\n\nBN names changed to:\n{}\nCNDs:\n{}\n'
          .format(bn.dag, '\n'.join(['{}: {}'.format(n, c)
                                     for n, c in bn.cnds.items()])))

    assert isinstance(bn.dag, DAG)
    assert bn.dag.is_directed is True
    assert bn.dag.is_partially_directed is True
    assert bn.dag.has_directed_cycles is False
    assert bn.dag.nodes == ['Y', 'Z']
    assert bn.dag.edges == {('Z', 'Y'): EdgeType.DIRECTED}
    assert bn.dag.parents == {'Y': ['Z']}
    assert tuple(bn.cnds) == ('Y', 'Z')

    assert bn.cnds['Z'].mean == 2.0
    assert bn.cnds['Z'].sd == 1.0
    assert bn.cnds['Z'].coeffs == {}

    assert bn.cnds['Y'].mean == 0.5
    assert bn.cnds['Y'].sd == 0.5
    assert bn.cnds['Y'].coeffs == {'Z': 1.5}

    # check can revert back to original BN

    bn.rename({n: o for o, n in name_map.items()})
    assert bn == orig_bn


def test_bn_rename_bn_xy_zy_1_ok():  # change names, keeping order
    bn = BN.read(TESTDATA_DIR + '/xdsl/xy_zy.xdsl')
    orig_bn = deepcopy(bn)

    name_map = {'X': 'XX', 'Y': 'YY', 'Z': 'Z'}
    bn.rename(name_map)
    print('\n\nBN names changed to:\n{}\nCNDs:\n{}\n'
          .format(bn.dag, '\n'.join(['{}: {}'.format(n, c)
                                     for n, c in bn.cnds.items()])))

    assert isinstance(bn.dag, DAG)
    assert bn.dag.is_directed is True
    assert bn.dag.is_partially_directed is True
    assert bn.dag.has_directed_cycles is False
    assert bn.dag.nodes == ['XX', 'YY', 'Z']
    assert bn.dag.edges == {('XX', 'YY'): EdgeType.DIRECTED,
                            ('Z', 'YY'): EdgeType.DIRECTED}
    assert bn.dag.parents == {'YY': ['XX', 'Z']}
    assert tuple(bn.cnds) == ('XX', 'YY', 'Z')

    assert bn.cnds['XX'].mean == 0.0
    assert bn.cnds['XX'].sd == 1.0
    assert bn.cnds['XX'].coeffs == {}

    assert bn.cnds['YY'].mean == 0.5
    assert bn.cnds['YY'].sd == 0.5
    assert bn.cnds['YY'].coeffs == {'XX': 1.5, 'Z': -2.2}

    assert bn.cnds['Z'].mean == -2.0
    assert bn.cnds['Z'].sd == 0.2
    assert bn.cnds['Z'].coeffs == {}

    # check can revert back to original BN

    bn.rename({n: o for o, n in name_map.items()})
    assert bn == orig_bn


def test_bn_rename_bn_xy_zy_2_ok():  # change names, changing order
    bn = BN.read(TESTDATA_DIR + '/xdsl/xy_zy.xdsl')
    orig_bn = deepcopy(bn)

    name_map = {'X': 'B', 'Y': 'Y', 'Z': 'A'}
    bn.rename(name_map)
    print('\n\nBN names changed to:\n{}\nCNDs:\n{}\n'
          .format(bn.dag, '\n'.join(['{}: {}'.format(n, c)
                                     for n, c in bn.cnds.items()])))

    assert isinstance(bn.dag, DAG)
    assert bn.dag.is_directed is True
    assert bn.dag.is_partially_directed is True
    assert bn.dag.has_directed_cycles is False
    assert bn.dag.nodes == ['A', 'B', 'Y']
    assert bn.dag.edges == {('A', 'Y'): EdgeType.DIRECTED,
                            ('B', 'Y'): EdgeType.DIRECTED}
    assert bn.dag.parents == {'Y': ['A', 'B']}
    assert tuple(bn.cnds) == ('A', 'B', 'Y')

    assert bn.cnds['B'].mean == 0.0
    assert bn.cnds['B'].sd == 1.0
    assert bn.cnds['B'].coeffs == {}

    assert bn.cnds['Y'].mean == 0.5
    assert bn.cnds['Y'].sd == 0.5
    assert bn.cnds['Y'].coeffs == {'B': 1.5, 'A': -2.2}

    assert bn.cnds['A'].mean == -2.0
    assert bn.cnds['A'].sd == 0.2
    assert bn.cnds['A'].coeffs == {}

    # check can revert back to original BN

    bn.rename({n: o for o, n in name_map.items()})
    assert bn == orig_bn
