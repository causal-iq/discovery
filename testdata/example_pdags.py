#
#   Example DAGs for testing and demonstration
#
#   Functions follow a common signature of no arguments to generate a graph
#   and a graph argument to validate that graph e.g. ab() generates the A-->B
#   graph, and ab(graph) validates graph as being A-->B
#

from core.common import adjmat
from core.graph import EdgeType, PDAG


def empty(check=None):
    if check is None:
        return PDAG([], [])

    assert isinstance(check, PDAG)
    assert check.nodes == []
    assert check.edges == {}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 0
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({}))


def a(check=None):
    if check is None:
        return PDAG(['A'], [])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A']
    assert check.edges == {}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0]}))

    return None


def ab(check=None):
    if check is None:
        return PDAG(['A', 'B'], [('A', '->', 'B')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'B': ['A']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0], 'B': [1, 0]}))

    return None


def ab_2(check=None):  # same as ab but specified differently
    if check is None:
        return PDAG(['B', 'A'], [('A', '->', 'B')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'B': ['A']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0], 'B': [1, 0]}))

    return None


def ba(check=None):
    if check is None:
        return PDAG(['B', 'A'], [('B', '->', 'A')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B']
    assert check.edges == {('B', 'A'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'A': ['B']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 1], 'B': [0, 0]}))

    return None


def a_b(check=None):
    if check is None:
        return PDAG(['B', 'A'], [])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B']
    assert check.edges == {}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 2
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0], 'B': [0, 0]}))

    return None


def ab3(check=None):
    if check is None:
        return PDAG(['B', 'A'], [('A', '-', 'B')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B']
    assert check.edges == {('A', 'B'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0], 'B': [2, 0]}))

    return None


def a_b_c(check=None):
    if check is None:
        return PDAG(['B', 'A', 'C'], [])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 3
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [0, 0, 0],
                                            'C': [0, 0, 0]}))

    return None


def ac_b(check=None):
    if check is None:
        return PDAG(['B', 'A', 'C'], [('A', '->', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 2
    assert check.parents == {'C': ['A']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [0, 0, 0],
                                            'C': [1, 0, 0]}))


def ac_b2(check=None):
    if check is None:
        return PDAG(['B', 'A', 'C'], [('A', '-', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'C'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 2
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [0, 0, 0],
                                            'C': [2, 0, 0]}))

    return None


def abc(check=None):
    if check is None:
        return PDAG(['B', 'A', 'C'], [('B', '->', 'C'), ('A', '->', 'B')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'C': ['B'], 'B': ['A']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [1, 0, 0],
                                            'C': [0, 1, 0]}))

    return None


def abc2(check=None):
    if check is None:
        return PDAG(['C', 'A', 'B'], [('A', '->', 'B'), ('B', '->', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'C': ['B'], 'B': ['A']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [1, 0, 0],
                                            'C': [0, 1, 0]}))

    return None


def abc3(check=None):
    if check is None:
        return PDAG(['C', 'A', 'B'], [('A', '-', 'B'), ('B', '->', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.UNDIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'C': ['B']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [2, 0, 0],
                                            'C': [0, 1, 0]}))

    return None


def abc4(check=None):
    if check is None:
        return PDAG(['C', 'A', 'B'], [('A', '-', 'B'), ('B', '-', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.UNDIRECTED,
                           ('B', 'C'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [2, 0, 0],
                                            'C': [0, 2, 0]}))

    return None


def abc5(check=None):
    if check is None:
        return PDAG(['C', 'A', 'B'], [('A', '->', 'B'), ('B', '-', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'B': ['A']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [1, 0, 0],
                                            'C': [0, 2, 0]}))

    return None


def abc6(check=None):  # B - A - C
    if check is None:
        return PDAG(['C', 'A', 'B'], [('A', '-', 'B'), ('A', '-', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.UNDIRECTED,
                           ('A', 'C'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [2, 0, 0],
                                            'C': [2, 0, 0]}))

    return None


def ab_ac(check=None):

    if check is None:
        return PDAG(['B', 'A', 'C'], [('A', '->', 'C'), ('A', '->', 'B')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('A', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'C': ['A'], 'B': ['A']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [1, 0, 0],
                                            'C': [1, 0, 0]}))

    return None


def ba_bc(check=None):  # A<-B->C

    if check is None:
        return PDAG(['B', 'A', 'C'], [('B', '->', 'C'), ('B', '->', 'A')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('B', 'A'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'A': ['B'], 'C': ['B']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 1, 0],
                                            'B': [0, 0, 0],
                                            'C': [0, 1, 0]}))

    return None


def ac_bc(check=None):

    if check is None:
        return PDAG(['B', 'A', 'C'], [('A', '->', 'C'), ('B', '->', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'C'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'C': ['A', 'B']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [0, 0, 0],
                                            'C': [1, 1, 0]}))

    return None


def ab_cb(check=None):  # A->B<-C

    if check is None:
        return PDAG(['B', 'A', 'C'], [('A', '->', 'B'), ('C', '->', 'B')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('C', 'B'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'B': ['A', 'C']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [1, 0, 1],
                                            'C': [0, 0, 0]}))

    return None


def abc_acyclic(check=None):  # A->B->C<-A

    if check is None:
        return PDAG(['C', 'B', 'A'], [('A', '->', 'B'), ('B', '->', 'C'),
                                      ('A', '->', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED,
                           ('A', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'C': ['A', 'B'], 'B': ['A']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [1, 0, 0],
                                            'C': [1, 1, 0]}))

    return None


def abc_acyclic2(check=None):  # A --> B --> C, A -- C

    if check is None:
        return PDAG(['C', 'B', 'A'], [('A', '->', 'B'), ('B', '->', 'C'),
                                      ('A', '-', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED,
                           ('A', 'C'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'C': ['B'], 'B': ['A']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [1, 0, 0],
                                            'C': [2, 1, 0]}))

    return None


def abc_acyclic3(check=None):  # A -- B -- C, A --> C

    if check is None:
        return PDAG(['C', 'B', 'A'], [('A', '-', 'B'), ('B', '-', 'C'),
                                      ('A', '->', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.UNDIRECTED,
                           ('B', 'C'): EdgeType.UNDIRECTED,
                           ('A', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'C': ['A']}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [2, 0, 0],
                                            'C': [1, 2, 0]}))

    return None


def abc_acyclic4(check=None):  # A -- B -- C -- A

    if check is None:
        return PDAG(['C', 'B', 'A'], [('A', '-', 'B'), ('B', '-', 'C'),
                                      ('A', '-', 'C')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.UNDIRECTED,
                           ('B', 'C'): EdgeType.UNDIRECTED,
                           ('A', 'C'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [2, 0, 0],
                                            'C': [2, 2, 0]}))

    return None


def cancer1(check=None):  # DAG as a PDAG

    if check is None:
        return PDAG(['Smoker', 'Pollution', 'Cancer', 'Xray', 'Dyspnoea'],
                    [('Smoker', '->', 'Cancer'),
                     ('Pollution', '->', 'Cancer'),
                     ('Cancer', '->', 'Dyspnoea'),
                     ('Cancer', '->', 'Xray')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray']
    assert check.edges == {('Smoker', 'Cancer'): EdgeType.DIRECTED,
                           ('Pollution', 'Cancer'): EdgeType.DIRECTED,
                           ('Cancer', 'Dyspnoea'): EdgeType.DIRECTED,
                           ('Cancer', 'Xray'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'Cancer': ['Pollution', 'Smoker'],
                             'Dyspnoea': ['Cancer'],
                             'Xray': ['Cancer']}
    assert check.to_adjmat().equals(adjmat({'Cancer': [0, 0, 1, 1, 0],
                                            'Dyspnoea': [1, 0, 0, 0, 0],
                                            'Pollution': [0, 0, 0, 0, 0],
                                            'Smoker': [0, 0, 0, 0, 0],
                                            'Xray': [1, 0, 0, 0, 0]}))


def cancer2(check=None):  # 5 node cancer with 2 undirected edges
    if check is None:
        return PDAG(['Smoker', 'Pollution', 'Cancer', 'Xray', 'Dyspnoea'],
                    [('Smoker', '->', 'Cancer'),
                     ('Pollution', '->', 'Cancer'),
                     ('Cancer', '-', 'Dyspnoea'),
                     ('Cancer', '-', 'Xray')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray']
    assert check.edges == {('Smoker', 'Cancer'): EdgeType.DIRECTED,
                           ('Pollution', 'Cancer'): EdgeType.DIRECTED,
                           ('Cancer', 'Dyspnoea'): EdgeType.UNDIRECTED,
                           ('Cancer', 'Xray'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'Cancer': ['Pollution', 'Smoker']}
    assert check.to_adjmat().equals(adjmat({'Cancer': [0, 0, 1, 1, 0],
                                            'Dyspnoea': [2, 0, 0, 0, 0],
                                            'Pollution': [0, 0, 0, 0, 0],
                                            'Smoker': [0, 0, 0, 0, 0],
                                            'Xray': [2, 0, 0, 0, 0]}))


def cancer3(check=None):  # skeleton of 5 node cancer
    if check is None:
        return PDAG(['Smoker', 'Pollution', 'Cancer', 'Xray', 'Dyspnoea'],
                    [('Smoker', '-', 'Cancer'),
                     ('Pollution', '-', 'Cancer'),
                     ('Cancer', '-', 'Dyspnoea'),
                     ('Cancer', '-', 'Xray')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray']
    assert check.edges == {('Cancer', 'Smoker'): EdgeType.UNDIRECTED,
                           ('Cancer', 'Pollution'): EdgeType.UNDIRECTED,
                           ('Cancer', 'Dyspnoea'): EdgeType.UNDIRECTED,
                           ('Cancer', 'Xray'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'Cancer': [0, 0, 0, 0, 0],
                                            'Dyspnoea': [2, 0, 0, 0, 0],
                                            'Pollution': [2, 0, 0, 0, 0],
                                            'Smoker': [2, 0, 0, 0, 0],
                                            'Xray': [2, 0, 0, 0, 0]}))


def asia(check=None):  # Correct PDAG which Asia DAG extends

    if check is None:
        return PDAG(['asia', 'bronc', 'dysp', 'either', 'lung',
                     'smoke', 'tub', 'xray'],
                    [('asia', '-', 'tub'),
                     ('lung', '-', 'smoke'),
                     ('bronc', '-', 'smoke'),
                     ('tub', '->', 'either'),
                     ('lung', '->', 'either'),
                     ('either', '->', 'xray'),
                     ('either', '->', 'dysp'),
                     ('bronc', '->', 'dysp')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['asia', 'bronc', 'dysp', 'either', 'lung',
                           'smoke', 'tub', 'xray']
    assert check.edges == {('asia', 'tub'): EdgeType.UNDIRECTED,
                           ('lung', 'smoke'): EdgeType.UNDIRECTED,
                           ('tub', 'either'): EdgeType.DIRECTED,
                           ('lung', 'either'): EdgeType.DIRECTED,
                           ('either', 'xray'): EdgeType.DIRECTED,
                           ('either', 'dysp'): EdgeType.DIRECTED,
                           ('bronc', 'dysp'): EdgeType.DIRECTED,
                           ('bronc', 'smoke'): EdgeType.UNDIRECTED}
    assert check.has_directed_cycles is False
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'dysp': ['bronc', 'either'],
                             'either': ['lung', 'tub'],
                             'xray': ['either']}
    assert check.to_adjmat().equals(adjmat({'asia':   [0, 0, 0, 0, 0, 0, 0, 0],
                                            'bronc':  [0, 0, 0, 0, 0, 0, 0, 0],
                                            'dysp':   [0, 1, 0, 1, 0, 0, 0, 0],
                                            'either': [0, 0, 0, 0, 1, 0, 1, 0],
                                            'lung':   [0, 0, 0, 0, 0, 0, 0, 0],
                                            'smoke':  [0, 2, 0, 0, 2, 0, 0, 0],
                                            'tub':    [2, 0, 0, 0, 0, 0, 0, 0],
                                            'xray':   [0, 0, 0, 1, 0, 0, 0, 0]
                                            }))

# Exemplar 4 node PDAGs from Andersson et al., 1995


def and4_1(check=None):  # 1  2  3  4

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'], [])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 4
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [0, 0, 0, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_2(check=None):  # 1 - 2  3  4

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X2', '-', 'X1')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 3
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [2, 0, 0, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_3(check=None):  # 1 - 2  3 - 4

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X4', '-', 'X3'),
                     ('X2', '-', 'X1')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.UNDIRECTED,
                           ('X3', 'X4'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 2
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [2, 0, 0, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 2, 0]}))

    return None


def and4_4(check=None):  # 1 - 2 - 3  4

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X2', '-', 'X3'),
                     ('X2', '-', 'X1')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.UNDIRECTED,
                           ('X2', 'X3'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 2
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [2, 0, 0, 0],
                                            'X3': [0, 2, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_5(check=None):  # 1 -> 2 <- 3  4

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X1', '->', 'X2'),
                     ('X3', '->', 'X2')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                           ('X3', 'X2'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 2
    assert check.parents == {'X2': ['X1', 'X3']}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [1, 0, 1, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_6(check=None):  # 1 - 2 - 3 - 1  4

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X1', '-', 'X2'),
                     ('X2', '-', 'X3'),
                     ('X3', '-', 'X1')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.UNDIRECTED,
                           ('X2', 'X3'): EdgeType.UNDIRECTED,
                           ('X1', 'X3'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 2
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [2, 0, 0, 0],
                                            'X3': [2, 2, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_7(check=None):  # 1 - 2 - 3 - 4 (chain)

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X1', '-', 'X2'),
                     ('X2', '-', 'X3'),
                     ('X3', '-', 'X4')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.UNDIRECTED,
                           ('X2', 'X3'): EdgeType.UNDIRECTED,
                           ('X3', 'X4'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [2, 0, 0, 0],
                                            'X3': [0, 2, 0, 0],
                                            'X4': [0, 0, 2, 0]}))

    return None


def and4_8(check=None):  # 1 -> 2 <- 3 - 4

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X1', '->', 'X2'),
                     ('X3', '->', 'X2'),
                     ('X3', '-', 'X4')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                           ('X3', 'X2'): EdgeType.DIRECTED,
                           ('X3', 'X4'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X2': ['X1', 'X3']}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [1, 0, 1, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 2, 0]}))

    return None


def and4_9(check=None):  # 3 - 2 - 1, 2 - 4 (undirected star)

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X1', '-', 'X2'),
                     ('X2', '-', 'X3'),
                     ('X2', '-', 'X4')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.UNDIRECTED,
                           ('X2', 'X3'): EdgeType.UNDIRECTED,
                           ('X2', 'X4'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [2, 0, 0, 0],
                                            'X3': [0, 2, 0, 0],
                                            'X4': [0, 2, 0, 0]}))

    return None


def and4_10(check=None):  # X1 -> X2 -> X4, X3 -> X2

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X1', '->', 'X2'),
                     ('X3', '->', 'X2'),
                     ('X2', '->', 'X4')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                           ('X3', 'X2'): EdgeType.DIRECTED,
                           ('X2', 'X4'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X2': ['X1', 'X3'],
                             'X4': ['X2']}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [1, 0, 1, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 1, 0, 0]}))

    return None


def and4_11(check=None):  # 1 -> 2 <- 4, 3 -> 2 (star collider)

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X1', '->', 'X2'),
                     ('X3', '->', 'X2'),
                     ('X4', '->', 'X2')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                           ('X3', 'X2'): EdgeType.DIRECTED,
                           ('X4', 'X2'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X2': ['X1', 'X3', 'X4']}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [1, 0, 1, 1],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_12(check=None):  # 2 - 3 - 1 - 2 - 4

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X2', '-', 'X3'),
                     ('X3', '-', 'X1'),
                     ('X1', '-', 'X2'),
                     ('X2', '-', 'X4')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.UNDIRECTED,
                           ('X1', 'X3'): EdgeType.UNDIRECTED,
                           ('X2', 'X3'): EdgeType.UNDIRECTED,
                           ('X2', 'X4'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [2, 0, 0, 0],
                                            'X3': [2, 2, 0, 0],
                                            'X4': [0, 2, 0, 0]}))

    return None


def and4_13(check=None):  # 2 <- 1 - 3 -> 2 <- 4

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X1', '->', 'X2'),
                     ('X1', '-', 'X3'),
                     ('X3', '->', 'X2'),
                     ('X4', '->', 'X2')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                           ('X1', 'X3'): EdgeType.UNDIRECTED,
                           ('X3', 'X2'): EdgeType.DIRECTED,
                           ('X4', 'X2'): EdgeType.DIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X2': ['X1', 'X3', 'X4']}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [1, 0, 1, 1],
                                            'X3': [2, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_14(check=None):  # 2 <- 1 -> 3 <- 2 <- 4

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X1', '->', 'X2'),
                     ('X1', '->', 'X3'),
                     ('X2', '->', 'X3'),
                     ('X4', '->', 'X2')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                           ('X1', 'X3'): EdgeType.DIRECTED,
                           ('X2', 'X3'): EdgeType.DIRECTED,
                           ('X4', 'X2'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X2': ['X1', 'X4'],
                             'X3': ['X1', 'X2']}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [1, 0, 0, 1],
                                            'X3': [1, 1, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_15(check=None):  # 2->4<-3, 2->1<-3 (square, 1 collider)

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X2', '->', 'X4'),
                     ('X3', '->', 'X4'),
                     ('X2', '-', 'X1'),
                     ('X3', '-', 'X1')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X2', 'X4'): EdgeType.DIRECTED,
                           ('X3', 'X4'): EdgeType.DIRECTED,
                           ('X1', 'X2'): EdgeType.UNDIRECTED,
                           ('X1', 'X3'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X4': ['X2', 'X3']}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [2, 0, 0, 0],
                                            'X3': [2, 0, 0, 0],
                                            'X4': [0, 1, 1, 0]}))

    return None


def and4_16(check=None):  # 2->4<-3, 2->1<-3 (square colliders)

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X2', '->', 'X4'),
                     ('X3', '->', 'X4'),
                     ('X2', '->', 'X1'),
                     ('X3', '->', 'X1')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X2', 'X4'): EdgeType.DIRECTED,
                           ('X3', 'X4'): EdgeType.DIRECTED,
                           ('X2', 'X1'): EdgeType.DIRECTED,
                           ('X3', 'X1'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X1': ['X2', 'X3'],
                             'X4': ['X2', 'X3']}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 1, 1, 0],
                                            'X2': [0, 0, 0, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 1, 1, 0]}))

    return None


def and4_17(check=None):  # 4 - 3 - 1 - 2 - 4 - 1 (undirected square)

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X4', '-', 'X3'),
                     ('X3', '-', 'X1'),
                     ('X1', '-', 'X2'),
                     ('X2', '-', 'X4'),
                     ('X4', '-', 'X1')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.UNDIRECTED,
                           ('X1', 'X3'): EdgeType.UNDIRECTED,
                           ('X1', 'X4'): EdgeType.UNDIRECTED,
                           ('X2', 'X4'): EdgeType.UNDIRECTED,
                           ('X3', 'X4'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [2, 0, 0, 0],
                                            'X3': [2, 0, 0, 0],
                                            'X4': [2, 2, 2, 0]}))

    return None


def complete4(check=None):  # complete skeleton

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X1', '-', 'X2'),
                     ('X1', '-', 'X3'),
                     ('X1', '-', 'X4'),
                     ('X2', '-', 'X3'),
                     ('X2', '-', 'X4'),
                     ('X3', '-', 'X4')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.UNDIRECTED,
                           ('X1', 'X3'): EdgeType.UNDIRECTED,
                           ('X1', 'X4'): EdgeType.UNDIRECTED,
                           ('X2', 'X3'): EdgeType.UNDIRECTED,
                           ('X2', 'X4'): EdgeType.UNDIRECTED,
                           ('X3', 'X4'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [2, 0, 0, 0],
                                            'X3': [2, 2, 0, 0],
                                            'X4': [2, 2, 2, 0]}))

    return None


def and4_inv1(check=None):  # 1 - 2 - 3 - 4 - 1 (square) - unextendable

    if check is None:
        return PDAG(['X1', 'X2', 'X3', 'X4'],
                    [('X1', '-', 'X2'),
                     ('X2', '-', 'X3'),
                     ('X3', '-', 'X4'),
                     ('X1', '-', 'X4')])

    assert isinstance(check, PDAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.UNDIRECTED,
                           ('X2', 'X3'): EdgeType.UNDIRECTED,
                           ('X3', 'X4'): EdgeType.UNDIRECTED,
                           ('X1', 'X4'): EdgeType.UNDIRECTED}
    assert check.is_directed is False
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is False
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [2, 0, 0, 0],
                                            'X3': [0, 2, 0, 0],
                                            'X4': [2, 0, 2, 0]}))

    return None
