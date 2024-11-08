#
#   Example DAGs for testing and demonstration
#
#   Functions follow a common signature of no arguments to generate a graph
#   and a graph argument to validate that graph e.g. ab() generates the A-->B
#   graph, and ab(graph) validates graph as being A-->B
#

from core.common import adjmat
from core.graph import DAG, EdgeType
from core.bn import BN


def empty(check=None):
    if check is None:
        return DAG([], [])

    assert isinstance(check, DAG)
    assert check.nodes == []
    assert check.edges == {}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 0
    assert check.parents == {}
    assert check.to_string() == ''
    assert check.to_adjmat().equals(adjmat({}))


def a(check=None):
    if check is None:
        return DAG(['A'], [])

    assert isinstance(check, DAG)
    assert check.nodes == ['A']
    assert check.edges == {}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_string() == '[A]'
    assert check.to_adjmat().equals(adjmat({'A': [0]}))

    return None


def x(check=None):
    if check is None:
        return DAG(['X'], [])

    assert isinstance(check, DAG)
    assert check.nodes == ['X']
    assert check.edges == {}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {}
    assert check.to_string() == '[X]'
    assert check.to_adjmat().equals(adjmat({'X': [0]}))

    return None


def ab(check=None, is_bn=False):
    if check is None:
        return DAG(['A', 'B'], [('A', '->', 'B')])

    assert (not is_bn and isinstance(check, DAG)) or (is_bn and
                                                      isinstance(check, BN))
    assert check.nodes == ['A', 'B']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'B': ['A']}
    assert check.to_string() == '[A][B|A]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 0], 'B': [1, 0]}))

    return None


def xy(check=None, is_bn=False):
    if check is None:
        return DAG(['X', 'Y'], [('X', '->', 'Y')])

    assert (not is_bn and isinstance(check, DAG)) or (is_bn and
                                                      isinstance(check, BN))
    assert check.nodes == ['X', 'Y']
    assert check.edges == {('X', 'Y'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'Y': ['X']}
    assert check.to_string() == '[X][Y|X]'
    assert check.to_adjmat().equals(adjmat({'X': [0, 0], 'Y': [1, 0]}))

    return None


def ab_2(check=None):  # same as ab but specified differently
    if check is None:
        return DAG(['B', 'A'], [('A', '->', 'B')])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'B': ['A']}
    assert check.to_string() == '[A][B|A]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 0], 'B': [1, 0]}))

    return None


def ba(check=None):
    if check is None:
        return DAG(['B', 'A'], [('B', '->', 'A')])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B']
    assert check.edges == {('B', 'A'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'A': ['B']}
    assert check.to_string() == '[A|B][B]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 1], 'B': [0, 0]}))

    return None


def a_b(check=None):
    if check is None:
        return DAG(['B', 'A'], [])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B']
    assert check.edges == {}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 2
    assert check.parents == {}
    assert check.to_string() == '[A][B]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 0], 'B': [0, 0]}))

    return None


def x_y(check=None):
    if check is None:
        return DAG(['X', 'Y'], [])

    assert isinstance(check, DAG)
    assert check.nodes == ['X', 'Y']
    assert check.edges == {}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 2
    assert check.parents == {}
    assert check.to_string() == '[X][Y]'
    assert check.to_adjmat().equals(adjmat({'X': [0, 0], 'Y': [0, 0]}))

    return None


def a_b_c(check=None):
    if check is None:
        return DAG(['B', 'A', 'C'], [])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 3
    assert check.parents == {}
    assert check.to_string() == '[A][B][C]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [0, 0, 0],
                                            'C': [0, 0, 0]}))

    return None


def ac_b(check=None):  # A -> C  B
    if check is None:
        return DAG(['B', 'A', 'C'], [('A', '->', 'C')])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 2
    assert check.parents == {'C': ['A']}
    assert check.to_string() == '[A][B][C|A]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [0, 0, 0],
                                            'C': [1, 0, 0]}))

    return None


def ac_b2(check=None):  # C -> A   B
    if check is None:
        return DAG(['B', 'A', 'C'], [('C', '->', 'A')])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('C', 'A'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 2
    assert check.parents == {'A': ['C']}
    assert check.to_string() == '[A|C][B][C]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 1],
                                            'B': [0, 0, 0],
                                            'C': [0, 0, 0]}))

    return None


def abc(check=None, nodes=['A', 'B', 'C'], number_components=1,
        compact='[A][B|A][C|B]'):
    if check is None:
        return DAG(['B', 'A', 'C'], [('B', '->', 'C'), ('A', '->', 'B')])

    assert isinstance(check, DAG)
    assert check.nodes == nodes
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == number_components
    assert check.parents == {'C': ['B'], 'B': ['A']}
    assert check.to_string() == compact
    if compact == '[A][B|A][C|B]':
        assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                                'B': [1, 0, 0],
                                                'C': [0, 1, 0]}))
    elif compact == '[A][B|A][C|B][D]':
        assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0, 0],
                                                'B': [1, 0, 0, 0],
                                                'C': [0, 1, 0, 0],
                                                'D': [0, 0, 0, 0]}))

    return None


def abc_2(check=None, nodes=['A', 'B', 'C'], number_components=1):
    if check is None:
        return DAG(['C', 'A', 'B'], [('A', '->', 'B'), ('B', '->', 'C')])

    assert isinstance(check, DAG)
    assert check.nodes == nodes
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == number_components
    assert check.parents == {'C': ['B'], 'B': ['A']}
    assert check.to_string() == '[A][B|A][C|B]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [1, 0, 0],
                                            'C': [0, 1, 0]}))

    return None


def abc3(check=None):  # A <- B <- C
    if check is None:
        return DAG(['B', 'A', 'C'], [('B', '->', 'A'), ('C', '->', 'B')])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('B', 'A'): EdgeType.DIRECTED,
                           ('C', 'B'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'A': ['B'], 'B': ['C']}
    assert check.to_string() == '[A|B][B|C][C]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 1, 0],
                                            'B': [0, 0, 1],
                                            'C': [0, 0, 0]}))

    return None


def ab_ac(check=None):

    if check is None:
        return DAG(['B', 'A', 'C'], [('A', '->', 'C'), ('A', '->', 'B')])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('A', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'C': ['A'], 'B': ['A']}
    assert check.to_string() == '[A][B|A][C|A]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [1, 0, 0],
                                            'C': [1, 0, 0]}))
    return None


def xyz(check=None):

    if check is None:
        return DAG(['X', 'Y', 'Z'], [('X', '->', 'Y'), ('Y', '->', 'Z')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X', 'Y', 'Z']
    assert check.edges == {('X', 'Y'): EdgeType.DIRECTED,
                           ('Y', 'Z'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'Y': ['X'], 'Z': ['Y']}
    assert check.to_string() == '[X][Y|X][Z|Y]'
    assert check.to_adjmat().equals(adjmat({'X': [0, 0, 0],
                                            'Y': [1, 0, 0],
                                            'Z': [0, 1, 0]}))
    return None


def ba_bc(check=None):  # A<-B->C

    if check is None:
        return DAG(['B', 'A', 'C'], [('B', '->', 'C'), ('B', '->', 'A')])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('B', 'A'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'A': ['B'], 'C': ['B']}
    assert check.to_string() == '[A|B][B][C|B]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 1, 0],
                                            'B': [0, 0, 0],
                                            'C': [0, 1, 0]}))

    return None


def ac_bc(check=None):

    if check is None:
        return DAG(['B', 'A', 'C'], [('A', '->', 'C'), ('B', '->', 'C')])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'C'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'C': ['A', 'B']}
    assert check.to_string() == '[A][B][C|A:B]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [0, 0, 0],
                                            'C': [1, 1, 0]}))
    return None


def xy_zy(check=None):

    if check is None:
        return DAG(['Z', 'X', 'Y'], [('X', '->', 'Y'), ('Z', '->', 'Y')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X', 'Y', 'Z']
    assert check.edges == {('X', 'Y'): EdgeType.DIRECTED,
                           ('Z', 'Y'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'Y': ['X', 'Z']}
    assert check.to_string() == '[X][Y|X:Z][Z]'
    assert check.to_adjmat().equals(adjmat({'X': [0, 0, 0],
                                            'Y': [1, 0, 1],
                                            'Z': [0, 0, 0]}))
    return None


def abc_acyclic(check=None):

    if check is None:
        return DAG(['C', 'B', 'A'], [('A', '->', 'B'), ('B', '->', 'C'),
                                     ('A', '->', 'C')])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('A', 'B'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED,
                           ('A', 'C'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    # assert check.number_components() == 1
    print('\n# components = {}'.format(check.number_components()))
    assert check.parents == {'C': ['A', 'B'], 'B': ['A']}
    assert check.to_string() == '[A][B|A][C|A:B]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0],
                                            'B': [1, 0, 0],
                                            'C': [1, 1, 0]}))

    return None


def abc_acyclic4(check=None):  # another acyclic variant

    if check is None:
        return DAG(['C', 'B', 'A'], [('B', '->', 'A'), ('C', '->', 'B'),
                                     ('C', '->', 'A')])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B', 'C']
    assert check.edges == {('B', 'A'): EdgeType.DIRECTED,
                           ('C', 'B'): EdgeType.DIRECTED,
                           ('C', 'A'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'A': ['B', 'C'], 'B': ['C']}
    assert check.to_string() == '[A|B:C][B|C][C]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 1, 1],
                                            'B': [0, 0, 1],
                                            'C': [0, 0, 0]}))

    return None


def cancer(check=None):

    if check is None:
        return DAG(['Smoker', 'Pollution', 'Cancer', 'Xray', 'Dyspnoea'],
                   [('Smoker', '->', 'Cancer'),
                    ('Pollution', '->', 'Cancer'),
                    ('Cancer', '->', 'Dyspnoea'),
                    ('Cancer', '->', 'Xray')])

    assert isinstance(check, DAG)
    assert check.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray']
    assert check.edges == {('Smoker', 'Cancer'): EdgeType.DIRECTED,
                           ('Pollution', 'Cancer'): EdgeType.DIRECTED,
                           ('Cancer', 'Dyspnoea'): EdgeType.DIRECTED,
                           ('Cancer', 'Xray'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'Cancer': ['Pollution', 'Smoker'],
                             'Dyspnoea': ['Cancer'],
                             'Xray': ['Cancer']}
    assert check.to_string() == '[Cancer|Pollution:Smoker][Dyspnoea|Cancer]' \
        + '[Pollution][Smoker][Xray|Cancer]'
    assert check.to_adjmat().equals(adjmat({'Cancer': [0, 0, 1, 1, 0],
                                            'Dyspnoea': [1, 0, 0, 0, 0],
                                            'Pollution': [0, 0, 0, 0, 0],
                                            'Smoker': [0, 0, 0, 0, 0],
                                            'Xray': [1, 0, 0, 0, 0]}))


def cancer3(check=None):  # variant of Cancer for testing PDAG extension

    if check is None:
        return DAG(['Smoker', 'Pollution', 'Cancer', 'Xray', 'Dyspnoea'],
                   [('Xray', '->', 'Cancer'),
                    ('Cancer', '->', 'Pollution'),
                    ('Cancer', '->', 'Dyspnoea'),
                    ('Cancer', '->', 'Smoker')])

    assert isinstance(check, DAG)
    assert check.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray']
    assert check.edges == {('Xray', 'Cancer'): EdgeType.DIRECTED,
                           ('Cancer', 'Pollution'): EdgeType.DIRECTED,
                           ('Cancer', 'Dyspnoea'): EdgeType.DIRECTED,
                           ('Cancer', 'Smoker'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'Cancer': ['Xray'],
                             'Pollution': ['Cancer'],
                             'Dyspnoea': ['Cancer'],
                             'Smoker': ['Cancer']}
    assert check.to_string() == '[Cancer|Xray][Dyspnoea|Cancer]' \
        + '[Pollution|Cancer][Smoker|Cancer][Xray]'
    assert check.to_adjmat().equals(adjmat({'Cancer': [0, 0, 0, 0, 1],
                                            'Dyspnoea': [1, 0, 0, 0, 0],
                                            'Pollution': [1, 0, 0, 0, 0],
                                            'Smoker': [1, 0, 0, 0, 0],
                                            'Xray': [0, 0, 0, 0, 0]}))


def asia(check=None):  # Standard Asia DAG

    if check is None:
        return DAG(['asia', 'bronc', 'dysp', 'either', 'lung',
                    'smoke', 'tub', 'xray'],
                   [('asia', '->', 'tub'),
                    ('smoke', '->', 'lung'),
                    ('smoke', '->', 'bronc'),
                    ('tub', '->', 'either'),
                    ('lung', '->', 'either'),
                    ('either', '->', 'xray'),
                    ('either', '->', 'dysp'),
                    ('bronc', '->', 'dysp')])

    assert isinstance(check, DAG)
    assert check.nodes == ['asia', 'bronc', 'dysp', 'either', 'lung',
                           'smoke', 'tub', 'xray']
    assert check.edges == {('asia', 'tub'): EdgeType.DIRECTED,
                           ('smoke', 'lung'): EdgeType.DIRECTED,
                           ('tub', 'either'): EdgeType.DIRECTED,
                           ('lung', 'either'): EdgeType.DIRECTED,
                           ('either', 'xray'): EdgeType.DIRECTED,
                           ('either', 'dysp'): EdgeType.DIRECTED,
                           ('bronc', 'dysp'): EdgeType.DIRECTED,
                           ('smoke', 'bronc'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'bronc': ['smoke'],
                             'dysp': ['bronc', 'either'],
                             'either': ['lung', 'tub'],
                             'lung': ['smoke'],
                             'tub': ['asia'],
                             'xray': ['either']}
    assert check.to_string() == '[asia][bronc|smoke][dysp|bronc:either]' + \
        '[either|lung:tub][lung|smoke][smoke][tub|asia][xray|either]'
    assert check.to_adjmat().equals(adjmat({'asia': [0, 0, 0, 0, 0, 0, 0, 0],
                                            'bronc': [0, 0, 0, 0, 0, 1, 0, 0],
                                            'dysp': [0, 1, 0, 1, 0, 0, 0, 0],
                                            'either': [0, 0, 0, 0, 1, 0, 1, 0],
                                            'lung': [0, 0, 0, 0, 0, 1, 0, 0],
                                            'smoke': [0, 0, 0, 0, 0, 0, 0, 0],
                                            'tub': [1, 0, 0, 0, 0, 0, 0, 0],
                                            'xray': [0, 0, 0, 1, 0, 0, 0, 0]}))


def asia2(check=None):  # DAG extended from Asia PDAG

    if check is None:
        return DAG(['asia', 'bronc', 'dysp', 'either', 'lung',
                    'smoke', 'tub', 'xray'],
                   [('tub', '->', 'asia'),
                    ('lung', '->', 'smoke'),
                    ('smoke', '->', 'bronc'),
                    ('tub', '->', 'either'),
                    ('lung', '->', 'either'),
                    ('either', '->', 'xray'),
                    ('either', '->', 'dysp'),
                    ('bronc', '->', 'dysp')])

    assert isinstance(check, DAG)
    assert check.nodes == ['asia', 'bronc', 'dysp', 'either', 'lung',
                           'smoke', 'tub', 'xray']
    assert check.edges == {('tub', 'asia'): EdgeType.DIRECTED,
                           ('lung', 'smoke'): EdgeType.DIRECTED,
                           ('tub', 'either'): EdgeType.DIRECTED,
                           ('lung', 'either'): EdgeType.DIRECTED,
                           ('either', 'xray'): EdgeType.DIRECTED,
                           ('either', 'dysp'): EdgeType.DIRECTED,
                           ('bronc', 'dysp'): EdgeType.DIRECTED,
                           ('smoke', 'bronc'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'asia': ['tub'],
                             'bronc': ['smoke'],
                             'dysp': ['bronc', 'either'],
                             'either': ['lung', 'tub'],
                             'smoke': ['lung'],
                             'xray': ['either']}
    assert check.to_string() == '[asia|tub][bronc|smoke][dysp|bronc:either]' \
        + '[either|lung:tub][lung][smoke|lung][tub][xray|either]'
    assert check.to_adjmat().equals(adjmat({'asia':   [0, 0, 0, 0, 0, 0, 1, 0],
                                            'bronc':  [0, 0, 0, 0, 0, 1, 0, 0],
                                            'dysp':   [0, 1, 0, 1, 0, 0, 0, 0],
                                            'either': [0, 0, 0, 0, 1, 0, 1, 0],
                                            'lung':   [0, 0, 0, 0, 0, 0, 0, 0],
                                            'smoke':  [0, 0, 0, 0, 1, 0, 0, 0],
                                            'tub':    [0, 0, 0, 0, 0, 0, 0, 0],
                                            'xray':   [0, 0, 0, 1, 0, 0, 0, 0]
                                            }))


def gauss(check=None):  # Test BNlearn Gaussian DAG

    if check is None:
        return DAG(['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                   [('A', '->', 'C'),
                    ('B', '->', 'C'),
                    ('B', '->', 'D'),
                    ('A', '->', 'F'),
                    ('D', '->', 'F'),
                    ('E', '->', 'F'),
                    ('G', '->', 'F')])

    assert isinstance(check, DAG)
    assert check.nodes == ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    assert check.edges == {('A', 'C'): EdgeType.DIRECTED,
                           ('B', 'C'): EdgeType.DIRECTED,
                           ('B', 'D'): EdgeType.DIRECTED,
                           ('A', 'F'): EdgeType.DIRECTED,
                           ('D', 'F'): EdgeType.DIRECTED,
                           ('E', 'F'): EdgeType.DIRECTED,
                           ('G', 'F'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.number_components() == 1
    assert check.parents == {'C': ['A', 'B'],
                             'D': ['B'],
                             'F': ['A', 'D', 'E', 'G']}
    assert check.to_string() == \
        '[A][B][C|A:B][D|B][E][F|A:D:E:G][G]'
    assert check.to_adjmat().equals(adjmat({'A': [0, 0, 0, 0, 0, 0, 0],
                                            'B': [0, 0, 0, 0, 0, 0, 0],
                                            'C': [1, 1, 0, 0, 0, 0, 0],
                                            'D': [0, 1, 0, 0, 0, 0, 0],
                                            'E': [0, 0, 0, 0, 0, 0, 0],
                                            'F': [1, 0, 0, 1, 1, 0, 1],
                                            'G': [0, 0, 0, 0, 0, 0, 0]}))


# Exemplar 4 node DAGs from Andersson et al., 1995

def and4_1(check=None):  # 1  2  3  4

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'], [])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 4
    assert check.parents == {}
    assert check.to_string() == '[X1][X2][X3][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [0, 0, 0, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_2(check=None):  # 1 <- 2  3  4

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X2', '->', 'X1')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X2', 'X1'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 3
    assert check.parents == {'X1': ['X2']}
    assert check.to_string() == '[X1|X2][X2][X3][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 1, 0, 0],
                                            'X2': [0, 0, 0, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_3(check=None):  # 1 <- 2  3 <- 4

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X2', '->', 'X1'),
                    ('X4', '->', 'X3')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X2', 'X1'): EdgeType.DIRECTED,
                           ('X4', 'X3'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 2
    assert check.parents == {'X1': ['X2'],
                             'X3': ['X4']}
    assert check.to_string() == '[X1|X2][X2][X3|X4][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 1, 0, 0],
                                            'X2': [0, 0, 0, 0],
                                            'X3': [0, 0, 0, 1],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_4(check=None):  # 1 <- 2 <- 3  4

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X2', '->', 'X1'),
                    ('X3', '->', 'X2')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X2', 'X1'): EdgeType.DIRECTED,
                           ('X3', 'X2'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 2
    assert check.parents == {'X1': ['X2'],
                             'X2': ['X3']}
    assert check.to_string() == '[X1|X2][X2|X3][X3][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 1, 0, 0],
                                            'X2': [0, 0, 1, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_5(check=None):  # 1 -> 2 <- 3  4

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X1', '->', 'X2'),
                    ('X3', '->', 'X2')])

    assert isinstance(check, DAG)
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
    assert check.to_string() == '[X1][X2|X1:X3][X3][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [1, 0, 1, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_6(check=None):  # 1 <- 3 -> 2 -> 1  4

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X2', '->', 'X1'),
                    ('X3', '->', 'X1'),
                    ('X3', '->', 'X2')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X2', 'X1'): EdgeType.DIRECTED,
                           ('X3', 'X1'): EdgeType.DIRECTED,
                           ('X3', 'X2'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 2
    assert check.parents == {'X1': ['X2', 'X3'],
                             'X2': ['X3']}
    assert check.to_string() == '[X1|X2:X3][X2|X3][X3][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 1, 1, 0],
                                            'X2': [0, 0, 1, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_7(check=None):  # 1 <- 2 <- 3 <- 4

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X2', '->', 'X1'),
                    ('X3', '->', 'X2'),
                    ('X4', '->', 'X3')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X2', 'X1'): EdgeType.DIRECTED,
                           ('X3', 'X2'): EdgeType.DIRECTED,
                           ('X4', 'X3'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X1': ['X2'],
                             'X2': ['X3'],
                             'X3': ['X4']}
    assert check.to_string() == '[X1|X2][X2|X3][X3|X4][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 1, 0, 0],
                                            'X2': [0, 0, 1, 0],
                                            'X3': [0, 0, 0, 1],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_8(check=None):  # 1 -> 2 <- 3 <- 4

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X1', '->', 'X2'),
                    ('X3', '->', 'X2'),
                    ('X4', '->', 'X3')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                           ('X3', 'X2'): EdgeType.DIRECTED,
                           ('X4', 'X3'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X2': ['X1', 'X3'],
                             'X3': ['X4']}
    assert check.to_string() == '[X1][X2|X1:X3][X3|X4][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [1, 0, 1, 0],
                                            'X3': [0, 0, 0, 1],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_9(check=None):  # 4 -> 2 -> 1, 2 -> 3

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X2', '->', 'X1'),
                    ('X2', '->', 'X3'),
                    ('X4', '->', 'X2')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X2', 'X1'): EdgeType.DIRECTED,
                           ('X2', 'X3'): EdgeType.DIRECTED,
                           ('X4', 'X2'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X1': ['X2'],
                             'X2': ['X4'],
                             'X3': ['X2']}
    assert check.to_string() == '[X1|X2][X2|X4][X3|X2][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 1, 0, 0],
                                            'X2': [0, 0, 0, 1],
                                            'X3': [0, 1, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_10(check=None):  # 1 -> 2 -> 4, 3 -> 2

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X1', '->', 'X2'),
                    ('X3', '->', 'X2'),
                    ('X2', '->', 'X4')])

    assert isinstance(check, DAG)
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
    assert check.to_string() == '[X1][X2|X1:X3][X3][X4|X2]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [1, 0, 1, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 1, 0, 0]}))

    return None


def and4_11(check=None):  # 1 -> 2 <- 4, 3 -> 2 (star collider)

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X1', '->', 'X2'),
                    ('X3', '->', 'X2'),
                    ('X4', '->', 'X2')])

    assert isinstance(check, DAG)
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
    assert check.to_string() == '[X1][X2|X1:X3:X4][X3][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [1, 0, 1, 1],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_12(check=None):  # 2 -> 1 <- 3 <- 2 <- 4

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X2', '->', 'X1'),
                    ('X3', '->', 'X1'),
                    ('X2', '->', 'X3'),
                    ('X4', '->', 'X2')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X2', 'X1'): EdgeType.DIRECTED,
                           ('X2', 'X3'): EdgeType.DIRECTED,
                           ('X3', 'X1'): EdgeType.DIRECTED,
                           ('X4', 'X2'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X1': ['X2', 'X3'],
                             'X2': ['X4'],
                             'X3': ['X2']}
    assert check.to_string() == '[X1|X2:X3][X2|X4][X3|X2][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 1, 1, 0],
                                            'X2': [0, 0, 0, 1],
                                            'X3': [0, 1, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_13(check=None):  # 2 <- 1 <- 3 -> 2 <- 4

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X1', '->', 'X2'),
                    ('X3', '->', 'X1'),
                    ('X3', '->', 'X2'),
                    ('X4', '->', 'X2')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                           ('X3', 'X1'): EdgeType.DIRECTED,
                           ('X3', 'X2'): EdgeType.DIRECTED,
                           ('X4', 'X2'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X1': ['X3'],
                             'X2': ['X1', 'X3', 'X4']}
    assert check.to_string() == '[X1|X3][X2|X1:X3:X4][X3][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 1, 0],
                                            'X2': [1, 0, 1, 1],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_14(check=None):  # 2 <- 1 -> 3 <- 2 <- 4

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X1', '->', 'X2'),
                    ('X1', '->', 'X3'),
                    ('X2', '->', 'X3'),
                    ('X4', '->', 'X2')])

    assert isinstance(check, DAG)
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
    assert check.to_string() == '[X1][X2|X1:X4][X3|X1:X2][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 0, 0],
                                            'X2': [1, 0, 0, 1],
                                            'X3': [1, 1, 0, 0],
                                            'X4': [0, 0, 0, 0]}))

    return None


def and4_15(check=None):  # 1->2->4<-3->1 (square, 1 collider)

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X1', '->', 'X2'),
                    ('X2', '->', 'X4'),
                    ('X3', '->', 'X4'),
                    ('X3', '->', 'X1')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                           ('X2', 'X4'): EdgeType.DIRECTED,
                           ('X3', 'X4'): EdgeType.DIRECTED,
                           ('X3', 'X1'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X1': ['X3'],
                             'X2': ['X1'],
                             'X4': ['X2', 'X3']}
    assert check.to_string() == '[X1|X3][X2|X1][X3][X4|X2:X3]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 1, 0],
                                            'X2': [1, 0, 0, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 1, 1, 0]}))

    return None


def and4_16(check=None):  # 2->4<-3, 2->1<-3 (square colliders)

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X2', '->', 'X4'),
                    ('X3', '->', 'X4'),
                    ('X2', '->', 'X1'),
                    ('X3', '->', 'X1')])

    assert isinstance(check, DAG)
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
    assert check.to_string() == '[X1|X2:X3][X2][X3][X4|X2:X3]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 1, 1, 0],
                                            'X2': [0, 0, 0, 0],
                                            'X3': [0, 0, 0, 0],
                                            'X4': [0, 1, 1, 0]}))

    return None


def and4_17(check=None):  # 4->3->1->2, 4->1, 4->2

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X4', '->', 'X3'),
                    ('X3', '->', 'X1'),
                    ('X1', '->', 'X2'),
                    ('X4', '->', 'X1'),
                    ('X4', '->', 'X2')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                           ('X3', 'X1'): EdgeType.DIRECTED,
                           ('X4', 'X1'): EdgeType.DIRECTED,
                           ('X4', 'X2'): EdgeType.DIRECTED,
                           ('X4', 'X3'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X1': ['X3', 'X4'],
                             'X2': ['X1', 'X4'],
                             'X3': ['X4']}
    assert check.to_string() == '[X1|X3:X4][X2|X1:X4][X3|X4][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 0, 1, 1],
                                            'X2': [1, 0, 0, 1],
                                            'X3': [0, 0, 0, 1],
                                            'X4': [0, 0, 0, 0]}))

    return None


def complete4(check=None):  # 4 nodes, 6 edges

    if check is None:
        return DAG(['X1', 'X2', 'X3', 'X4'],
                   [('X2', '->', 'X1'),
                    ('X3', '->', 'X1'),
                    ('X3', '->', 'X2'),
                    ('X4', '->', 'X1'),
                    ('X4', '->', 'X2'),
                    ('X4', '->', 'X3')])

    assert isinstance(check, DAG)
    assert check.nodes == ['X1', 'X2', 'X3', 'X4']
    assert check.edges == {('X2', 'X1'): EdgeType.DIRECTED,
                           ('X3', 'X1'): EdgeType.DIRECTED,
                           ('X3', 'X2'): EdgeType.DIRECTED,
                           ('X4', 'X1'): EdgeType.DIRECTED,
                           ('X4', 'X2'): EdgeType.DIRECTED,
                           ('X4', 'X3'): EdgeType.DIRECTED}
    assert check.is_directed is True
    assert check.is_partially_directed is True
    assert check.has_directed_cycles is False
    assert check.is_DAG() is True
    assert check.is_PDAG() is True
    assert check.number_components() == 1
    assert check.parents == {'X1': ['X2', 'X3', 'X4'],
                             'X2': ['X3', 'X4'],
                             'X3': ['X4']}
    assert check.to_string() == '[X1|X2:X3:X4][X2|X3:X4][X3|X4][X4]'
    assert check.to_adjmat().equals(adjmat({'X1': [0, 1, 1, 1],
                                            'X2': [0, 0, 1, 1],
                                            'X3': [0, 0, 0, 1],
                                            'X4': [0, 0, 0, 0]}))

    return None
