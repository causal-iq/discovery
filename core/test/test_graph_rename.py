
# Test graph rename function

import pytest

from core.common import EdgeType, adjmat
from core.graph import SDG, PDAG, DAG
import testdata.example_sdgs as ex_sdg
import testdata.example_pdags as ex_pdag
import testdata.example_dags as ex_dag


def test_graph_rename_type_error_1():  # bad argument types
    graph = ex_sdg.ab_undirected()
    with pytest.raises(TypeError):
        graph.rename()


def test_graph_rename_type_error_2():  # name_map not a dictionary
    graph = ex_sdg.ab_undirected()
    with pytest.raises(TypeError):
        graph.rename(name_map=None)
    with pytest.raises(TypeError):
        graph.rename(name_map=True)
    with pytest.raises(TypeError):
        graph.rename(name_map=37)
    with pytest.raises(TypeError):
        graph.rename(name_map=[{'A': 'B'}])


def test_graph_rename_type_error_3():  # name_map has non-string keys
    graph = ex_sdg.ab_undirected()
    with pytest.raises(TypeError):
        graph.rename(name_map={1: 'B'})
    with pytest.raises(TypeError):
        graph.rename(name_map={('A',): 'B'})


def test_graph_rename_type_error_4():  # name_map has non-string values
    graph = ex_sdg.ab_undirected()
    with pytest.raises(TypeError):
        graph.rename(name_map={'A': 1})
    with pytest.raises(TypeError):
        graph.rename(name_map={'A': ['B']})


def test_graph_rename_value_error_1():  # keys that are not current node name
    graph = ex_sdg.ab_undirected()
    with pytest.raises(ValueError):
        graph.rename(name_map={'C': 'Q'})


# Renames on undirected graph

def test_graph_rename_sdg_ab_1_ok():  # change first node name, keeping order
    graph = ex_sdg.ab_undirected()
    graph.rename(name_map={'A': 'AA', 'B': 'B'})

    assert isinstance(graph, SDG)
    assert graph.nodes == ['AA', 'B']
    assert graph.edges == {('AA', 'B'): EdgeType.UNDIRECTED}
    assert graph.parents == {}
    assert graph.is_directed is False
    assert graph.is_partially_directed is True
    assert graph.has_directed_cycles is False


def test_graph_rename_sdg_ab_2_ok():  # change first node name, changing order
    graph = ex_sdg.ab_undirected()
    graph.rename(name_map={'A': 'C', 'B': 'B'})

    assert isinstance(graph, SDG)
    assert graph.nodes == ['B', 'C']
    assert graph.edges == {('B', 'C'): EdgeType.UNDIRECTED}
    assert graph.parents == {}
    assert graph.is_directed is False
    assert graph.is_partially_directed is True
    assert graph.has_directed_cycles is False


def test_graph_rename_sdg_ab_3_ok():  # change second node name, keeping order
    graph = ex_sdg.ab_undirected()
    graph.rename(name_map={'B': 'BB', 'A': 'A'})

    assert isinstance(graph, SDG)
    assert graph.nodes == ['A', 'BB']
    assert graph.edges == {('A', 'BB'): EdgeType.UNDIRECTED}
    assert graph.parents == {}
    assert graph.is_directed is False
    assert graph.is_partially_directed is True
    assert graph.has_directed_cycles is False


def test_graph_rename_sdg_ab_4_ok():  # change both names, changing order
    graph = ex_sdg.ab_undirected()
    graph.rename(name_map={'A': 'X001A', 'B': 'X000B'})

    assert isinstance(graph, SDG)
    assert graph.nodes == ['X000B', 'X001A']
    assert graph.edges == {('X000B', 'X001A'): EdgeType.UNDIRECTED}
    assert graph.parents == {}
    assert graph.is_directed is False
    assert graph.is_partially_directed is True
    assert graph.has_directed_cycles is False


def test_graph_rename_sdg_abc_1_ok():  # change names and order
    graph = ex_sdg.abc_mixed()
    graph.rename(name_map={'A': 'X002A', 'B': 'X001B', 'C': 'X000C'})

    assert isinstance(graph, SDG)
    assert graph.nodes == ['X000C', 'X001B', 'X002A']
    assert graph.edges == {('X001B', 'X002A'): EdgeType.UNDIRECTED,
                           ('X001B', 'X000C'): EdgeType.SEMIDIRECTED}
    assert graph.parents == {}
    assert graph.is_directed is False
    assert graph.is_partially_directed is False
    assert graph.has_directed_cycles is False


# Test rename PDAGs

def test_graph_rename_pdag_a_1_ok():  # single node PDAG
    graph = ex_pdag.a()
    graph.rename(name_map={'A': 'Z'})

    assert isinstance(graph, PDAG)
    assert graph.nodes == ['Z']
    assert graph.edges == {}
    assert graph.is_directed is True
    assert graph.is_partially_directed is True
    assert graph.has_directed_cycles is False
    assert graph.parents == {}
    assert graph.to_adjmat().equals(adjmat({'Z': [0]}))


def test_graph_rename_pdag_ba_1_ok():  # A<--B PDAG
    graph = ex_pdag.ba()
    graph.rename(name_map={'A': 'Z', 'B': 'Y'})

    assert isinstance(graph, PDAG)
    assert graph.nodes == ['Y', 'Z']
    assert graph.edges == {('Y', 'Z'): EdgeType.DIRECTED}
    assert graph.is_directed is True
    assert graph.is_partially_directed is True
    assert graph.has_directed_cycles is False
    assert graph.parents == {'Z': ['Y']}
    assert graph.to_adjmat().equals(adjmat({'Y': [0, 0],
                                            'Z': [1, 0]}))


def test_graph_rename_pdag_ab3_1_ok():  # A--B PDAG
    graph = ex_pdag.ab3()
    graph.rename(name_map={'A': 'Z', 'B': 'B'})

    assert isinstance(graph, PDAG)
    assert graph.nodes == ['B', 'Z']
    assert graph.edges == {('B', 'Z'): EdgeType.UNDIRECTED}
    assert graph.is_directed is False
    assert graph.is_partially_directed is True
    assert graph.has_directed_cycles is False
    assert graph.parents == {}
    print(graph.to_adjmat())
    assert graph.to_adjmat().equals(adjmat({'B': [0, 0],
                                            'Z': [2, 0]}))


def test_graph_rename_pdag_and4_8_1_ok():
    graph = ex_pdag.and4_8()
    graph.rename(name_map={'X1': 'X1', 'X2': 'X2', 'X3': 'Q3', 'X4': 'X4'})

    assert isinstance(graph, PDAG)
    assert graph.nodes == ['Q3', 'X1', 'X2', 'X4']
    assert graph.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                           ('Q3', 'X2'): EdgeType.DIRECTED,
                           ('Q3', 'X4'): EdgeType.UNDIRECTED}
    assert graph.is_directed is False
    assert graph.is_partially_directed is True
    assert graph.has_directed_cycles is False
    assert graph.parents == {'X2': ['Q3', 'X1']}
    assert graph.to_adjmat().equals(adjmat({'Q3': [0, 0, 0, 0],
                                            'X1': [0, 0, 0, 0],
                                            'X2': [1, 1, 0, 0],
                                            'X4': [2, 0, 0, 0]}))


def test_graph_rename_pdag_cancer2_1_ok():  # Cancer with 2 undirected edges
    graph = ex_pdag.cancer2()
    name_map = {n: 'LungCancer' if n == 'Cancer' else n for n in graph.nodes}
    graph.rename(name_map)

    assert isinstance(graph, PDAG)
    assert graph.nodes == ['Dyspnoea', 'LungCancer', 'Pollution', 'Smoker',
                           'Xray']
    assert graph.edges == {('Smoker', 'LungCancer'): EdgeType.DIRECTED,
                           ('Pollution', 'LungCancer'): EdgeType.DIRECTED,
                           ('Dyspnoea', 'LungCancer'): EdgeType.UNDIRECTED,
                           ('LungCancer', 'Xray'): EdgeType.UNDIRECTED}
    assert graph.is_directed is False
    assert graph.is_partially_directed is True
    assert graph.has_directed_cycles is False
    assert graph.is_DAG() is False
    assert graph.is_PDAG() is True
    assert graph.number_components() == 1
    assert graph.parents == {'LungCancer': ['Pollution', 'Smoker']}
    assert graph.to_adjmat().equals(adjmat({'Dyspnoea': [0, 0, 0, 0, 0],
                                            'LungCancer': [2, 0, 1, 1, 0],
                                            'Pollution': [0, 0, 0, 0, 0],
                                            'Smoker': [0, 0, 0, 0, 0],
                                            'Xray': [0, 2, 0, 0, 0]}))


# Test DAG renames

def test_graph_rename_dag_ac_bc_1_ok():  # Cancer with 2 undirected edges

    graph = ex_dag.ac_bc()
    graph.rename({'A': 'X02A', 'B': 'X00B', 'C': 'X01C'})

    assert isinstance(graph, DAG)
    assert graph.nodes == ['X00B', 'X01C', 'X02A']
    assert graph.edges == {('X02A', 'X01C'): EdgeType.DIRECTED,
                           ('X00B', 'X01C'): EdgeType.DIRECTED}
    assert graph.is_directed is True
    assert graph.has_directed_cycles is False
    assert graph.is_DAG() is True
    assert graph.number_components() == 1
    assert graph.parents == {'X01C': ['X00B', 'X02A']}
    assert graph.to_string() == '[X00B][X01C|X00B:X02A][X02A]'
    assert graph.to_adjmat().equals(adjmat({'X00B': [0, 0, 0],
                                            'X01C': [1, 0, 1],
                                            'X02A': [0, 0, 0]}))


def test_graph_rename_dag_asia_1_ok():  # Asia DAG

    graph = ex_dag.asia()
    name_map = {n: n for n in graph.nodes}
    name_map.update({'asia': 'x0asia', 'dysp': 'x1dysp', 'either': 'x2eith'})
    graph.rename(name_map)

    assert isinstance(graph, DAG)
    assert graph.nodes == ['bronc', 'lung', 'smoke', 'tub', 'x0asia', 'x1dysp',
                           'x2eith', 'xray']
    assert graph.edges == {('x0asia', 'tub'): EdgeType.DIRECTED,
                           ('smoke', 'lung'): EdgeType.DIRECTED,
                           ('tub', 'x2eith'): EdgeType.DIRECTED,
                           ('lung', 'x2eith'): EdgeType.DIRECTED,
                           ('x2eith', 'xray'): EdgeType.DIRECTED,
                           ('x2eith', 'x1dysp'): EdgeType.DIRECTED,
                           ('bronc', 'x1dysp'): EdgeType.DIRECTED,
                           ('smoke', 'bronc'): EdgeType.DIRECTED}
    assert graph.is_directed is True
    assert graph.has_directed_cycles is False
    assert graph.is_DAG() is True
    assert graph.number_components() == 1
    assert graph.parents == {'bronc': ['smoke'],
                             'x1dysp': ['bronc', 'x2eith'],
                             'x2eith': ['lung', 'tub'],
                             'lung': ['smoke'],
                             'tub': ['x0asia'],
                             'xray': ['x2eith']}
    assert graph.to_string() == '[bronc|smoke][lung|smoke][smoke]' + \
        '[tub|x0asia][x0asia][x1dysp|bronc:x2eith][x2eith|lung:tub]' + \
        '[xray|x2eith]'
    assert graph.to_adjmat().equals(adjmat({'bronc': [0, 0, 1, 0, 0, 0, 0, 0],
                                            'lung': [0, 0, 1, 0, 0, 0, 0, 0],
                                            'smoke': [0, 0, 0, 0, 0, 0, 0, 0],
                                            'tub': [0, 0, 0, 0, 1, 0, 0, 0],
                                            'x0asia': [0, 0, 0, 0, 0, 0, 0, 0],
                                            'x1dysp': [1, 0, 0, 0, 0, 0, 1, 0],
                                            'x2eith': [0, 1, 0, 1, 0, 0, 0, 0],
                                            'xray': [0, 0, 0, 0, 0, 0, 1, 0]}))

    # revert to original names and check

    name_map = {n: n for n in graph.nodes}
    name_map.update({'x0asia': 'asia', 'x1dysp': 'dysp', 'x2eith': 'either'})
    graph.rename(name_map)
    ex_dag.asia(graph)
