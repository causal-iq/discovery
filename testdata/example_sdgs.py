#
#   Example graphs for testing and demonstration - non DAGs
#
#   Functions follow a common signature of no arguments to generate a graph
#   and a graph argument to validate that graph e.g. ab() generates the A-->B
#   graph, and ab(graph) validates graph as being A-->B
#

from core.common import adjmat
from core.graph import SDG, EdgeType


def ab_undirected(check=None):

    if check is None:
        return SDG(['B', 'A'], [('A', '-', 'B')])

    assert isinstance(check, SDG)
    assert check.nodes == ['A', 'B']
    assert check.edges == {('A', 'B'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0], 'B': [2, 0]}))

    return None


def abc_mixed(check=None):

    if check is None:
        return SDG(['C', 'B', 'A'], [('A', '-', 'B'), ('B', 'o->', 'C')])

    assert isinstance(check, SDG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.UNDIRECTED,
                           ('B', 'C'): EdgeType.SEMIDIRECTED}
    assert check.is_directed is False
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [2, 0, 0],
                                            'C': [0, 4, 0]}))

    return None


def abc_mixed_2(check=None):  # same as abc_mixed but specified differently

    if check is None:
        return SDG(['B', 'A', 'C'], [('B', 'o->', 'C'), ('B', '-', 'A')])

    assert isinstance(check, SDG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.UNDIRECTED,
                           ('B', 'C'): EdgeType.SEMIDIRECTED}
    assert check.is_directed is False
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [2, 0, 0],
                                            'C': [0, 4, 0]}))

    return None


def abc_cycle(check=None):

    if check is None:
        return SDG(['C', 'B', 'A'], [('A', '->', 'B'),
                                     ('B', '->', 'C'),
                                     ('C', '->', 'A')])

    assert isinstance(check, SDG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED,
                           ('C', 'A'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is True
    assert check.is_DAG() is False
    assert check.number_components() == 1
    assert check.parents == {'C': ['B'], 'B': ['A'], 'A': ['C']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 1],
                                            'B': [1, 0, 0],
                                            'C': [0, 1, 0]}))

    return None


def ab(check=None):  # A --> B but just as SDG, not DAG
    if check is None:
        return SDG(['B', 'A'], [('A', '->', 'B')])

    assert isinstance(check, SDG)
    assert check.nodes == ['A', 'B']
    assert check.edges == {('A', 'B'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0],
                                            'B': [1, 0]}))

    return None
