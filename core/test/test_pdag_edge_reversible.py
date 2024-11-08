
#   Check edge_reversible method of PDAG class.
#   Note that analysis/test/test_bn_analysis.py also tests this method.

import pytest

import testdata.example_pdags as ex_pdag


def test_pdag_edge_reversible_type_error_1():  # no args specified
    pdag = ex_pdag.a()
    with pytest.raises(TypeError):
        pdag.edge_reversible()


def test_pdag_edge_reversible_type_error_2():  # argument not a tuple
    pdag = ex_pdag.a()
    with pytest.raises(TypeError):
        pdag.edge_reversible(27)
    with pytest.raises(TypeError):
        pdag.edge_reversible(['A', 'B'])
    with pytest.raises(TypeError):
        pdag.edge_reversible(-1.9)
    with pytest.raises(TypeError):
        pdag.edge_reversible(True)


def test_pdag_edge_reversible_type_error_3():  # tuple doesn't have two elems
    pdag = ex_pdag.a()
    with pytest.raises(TypeError):
        pdag.edge_reversible(tuple(['A']))
    with pytest.raises(TypeError):
        pdag.edge_reversible(('1',))
    with pytest.raises(TypeError):
        pdag.edge_reversible(('1', '2', '3'))


def test_pdag_edge_reversible_type_error_4():  # tuple elements not strings
    pdag = ex_pdag.a()
    with pytest.raises(TypeError):
        pdag.edge_reversible((1, 2))
    with pytest.raises(TypeError):
        pdag.edge_reversible(('1', 3))


def test_pdag_edge_reversible_value_a_ok():  # PDAG A, edge not present
    pdag = ex_pdag.a()
    assert not pdag.edge_reversible(('A', 'A'))


def test_pdag_edge_reversible_ab_ok():  # PDAG A->B
    pdag = ex_pdag.ab()
    assert not pdag.edge_reversible(('A', 'B'))
    assert not pdag.edge_reversible(('B', 'A'))
    assert not pdag.edge_reversible(('A', 'C'))


def test_pdag_edge_reversible_ab3_ok():  # PDAG A-B
    pdag = ex_pdag.ab3()
    assert pdag.edge_reversible(('A', 'B'))
    assert pdag.edge_reversible(('B', 'A'))
    assert not pdag.edge_reversible(('A', 'A'))
    assert not pdag.edge_reversible(('A', 'C'))


def test_pdag_edge_reversible_abc4_ok():  # PDAG A-B-C
    pdag = ex_pdag.abc4()
    assert pdag.edge_reversible(('A', 'B'))
    assert pdag.edge_reversible(('B', 'A'))
    assert pdag.edge_reversible(('B', 'C'))
    assert pdag.edge_reversible(('C', 'B'))
    assert not pdag.edge_reversible(('A', 'C'))
    assert not pdag.edge_reversible(('C', 'A'))


def test_pdag_edge_reversible_ab_cb_ok():  # PDAG A->B<-C
    pdag = ex_pdag.ab_cb()
    assert not pdag.edge_reversible(('A', 'B'))
    assert not pdag.edge_reversible(('B', 'A'))
    assert not pdag.edge_reversible(('B', 'C'))
    assert not pdag.edge_reversible(('C', 'B'))
    assert not pdag.edge_reversible(('A', 'C'))
    assert not pdag.edge_reversible(('C', 'A'))
