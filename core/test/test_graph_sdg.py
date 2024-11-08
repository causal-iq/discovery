
from core.graph import SDG
import testdata.example_sdgs as ex

import pytest


def test_graph_sdg_type_error1():  # bad argument types
    with pytest.raises(TypeError):
        SDG()
    with pytest.raises(TypeError):
        SDG(32)
    with pytest.raises(TypeError):
        SDG('not', 'right')


def test_graph_sdg_type_error2():  # bad type within nodes
    with pytest.raises(TypeError):
        SDG([1], [])


def test_graph_sdg_type_error3():  # bad type within edges
    with pytest.raises(TypeError):
        SDG(['A', 'B'], [3])
    with pytest.raises(TypeError):
        SDG(['A', 'B'], ['S'])
    with pytest.raises(TypeError):
        SDG(['A', 'B'], [('A', '->')])
    with pytest.raises(TypeError):
        SDG(['A', 'B'], [('A', '->', True)])


def test_graph_sdg_value_error1():  # empty node name
    with pytest.raises(ValueError):
        SDG(['A', 'B', ''], [])


def test_graph_sdg_value_error2():  # duplicate node name
    with pytest.raises(ValueError):
        SDG(['A', 'B', 'A'], [])


def test_graph_sdg_value_error3():  # cylic edge
    with pytest.raises(ValueError):
        SDG(['A', 'B'], [('A', '->', 'A')])


def test_graph_sdg_value_error4():  # invalid edge symbol
    with pytest.raises(TypeError):
        SDG(['A', 'B'], [('A', '?', 'B')])


def test_graph_sdg_value_error5():  # edge references unknown node
    with pytest.raises(ValueError):
        SDG(['A', 'B'], [('A', 'o-o', 'C')])


def test_graph_sdg_value_error6():  # duplicate edges
    with pytest.raises(ValueError):
        SDG(['A', 'B'], [('A', 'o-o', 'B'), ('A', '->', 'B')])
    with pytest.raises(ValueError):
        SDG(['A', 'B'], [('A', 'o-o', 'B'), ('B', '->', 'A')])


def test_graph_sdg_ab_undirected_ok():  # A - B graph validates OK
    ex.ab_undirected(ex.ab_undirected())


def test_graph_sdg_abc_mixed_ok():  # three node mixed graph validates OK
    ex.abc_mixed(ex.abc_mixed())


def test_graph_sdg_abc_cycle_ok():  # three node cycle validates OK
    ex.abc_cycle(ex.abc_cycle())


def test_graph_sdg_abc_mixed_eq1():  # graph is equal to itself
    assert ex.abc_mixed() == ex.abc_mixed()
    assert (ex.abc_mixed() != ex.abc_mixed()) is False


def test_graph_sdg_abc_mixed_eq2():  # graph is equal to itself
    assert ex.abc_mixed_2() == ex.abc_mixed_2()
    assert (ex.abc_mixed_2() != ex.abc_mixed_2()) is False


def test_graph_sdg_abc_mixed_eq3():  # graph is equal to identical graph
    assert ex.abc_mixed() == ex.abc_mixed_2()
    assert (ex.abc_mixed() != ex.abc_mixed_2()) is False


def test_graph_sdg_ab_undirected_eq():  # graph is equal to identical graph
    assert ex.ab_undirected() == ex.ab_undirected()
    assert (ex.ab_undirected() != ex.ab_undirected()) is False


def test_graph_sdg_abc_cycle_eq():  # graph is equal to itself
    assert ex.abc_cycle() == ex.abc_cycle()
    assert (ex.abc_cycle() != ex.abc_cycle()) is False


def test_graph_sdg_ne1():  # comparisons with non-graphs work OK
    assert ex.ab_undirected() is not None
    assert (ex.ab_undirected() is None) is False
    assert ex.ab_undirected() != 1
    assert (ex.ab_undirected() == 1) is False


def test_graph_sdg_ne2():  # comparisons with different graphs
    assert ex.ab_undirected() != ex.abc_cycle()
    assert (ex.ab_undirected() == ex.abc_cycle()) is False
    assert ex.abc_mixed() != ex.abc_cycle()
    assert (ex.abc_mixed() == ex.abc_cycle()) is False
    assert ex.ab_undirected() != ex.abc_mixed()
    assert (ex.ab_undirected() == ex.abc_mixed()) is False
