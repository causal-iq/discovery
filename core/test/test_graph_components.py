
#   Test function to list undirected trees

import testdata.example_dags as ex_dag


def test_graph_undirected_trees_empty_ok():
    assert ex_dag.empty().components() == []


def test_graph_undirected_trees_a_ok():
    assert ex_dag.a().components() == [['A']]


def test_graph_undirected_trees_ab_ok():
    assert ex_dag.ab().components() == [['A', 'B']]


def test_graph_undirected_trees_empty_ba_ok():
    assert ex_dag.ba().components() == [['A', 'B']]


def test_graph_undirected_trees_empty_a_b_ok():
    assert ex_dag.a_b().components() == [['A'], ['B']]


def test_graph_undirected_trees_empty_abc_ok():
    assert ex_dag.abc().components() == [['A', 'B', 'C']]


def test_graph_undirected_trees_empty_abc3_ok():
    assert ex_dag.abc3().components() == [['A', 'B', 'C']]


def test_graph_undirected_trees_empty_abc_acyclic_ok():
    assert ex_dag.abc_acyclic().components() == [['A', 'B', 'C']]


def test_graph_undirected_trees_and4_12_ok():
    assert ex_dag.and4_12().components() == [['X1', 'X2', 'X3', 'X4']]
