#
#   Example CPTs for testing and demonstration
#
#   Functions follow a common signature of no arguments to generate a CPT
#   and a graph argument to validate that graph e.g. ab() generates the A-->B
#   CPT, and ab(CPT) validates CPT as being for A-->B
#

from core.cpt import CPT


def p0_v2_1(check=None):  # 0 parents, 2 node values
    if check is None:
        return CPT({'0': 1.0, '1': 0.0})
    assert isinstance(check, CPT)
    assert check.cdist() == {'0': 1.0, '1': 0.0}
    assert check.has_parents is False
    assert check.free_params == 1
    assert check.values == {'0', '1'}


def p0_v2_1a(check=None):  # as p0_v2_1 but reversed value specification
    if check is None:
        return CPT({'1': 0.0, '0': 1.0})
    assert isinstance(check, CPT)
    assert check.cdist() == {'0': 1.0, '1': 0.0}
    assert check.has_parents is False
    assert check.free_params == 1
    assert check.values == {'0', '1'}


def p0_v2_1b(check=None):  # nearly identical to p0_v2_1a
    if check is None:
        return CPT({'1': 0.0, '0': 0.999999999})
    assert isinstance(check, CPT)
    assert check.cdist() == {'0': 1.0, '1': 0.0}
    assert check.has_parents is False
    assert check.free_params == 1
    assert check.values == {'0', '1'}


def p0_v2_2(check=None):  # 0 parents, 2 node values
    if check is None:
        return CPT({'0': 0.2, '1': 0.8})
    assert isinstance(check, CPT)
    assert check.cdist() == {'0': 0.2, '1': 0.8}
    assert check.has_parents is False
    assert check.free_params == 1
    assert check.values == {'0', '1'}


def p0_v2_3(check=None):  # 0 parents, 2 node values
    if check is None:
        return CPT({'2': 0.2, '3': 0.8})
    assert isinstance(check, CPT)
    assert check.cdist() == {'2': 0.2, '3': 0.8}
    assert check.has_parents is False
    assert check.free_params == 1
    assert check.values == {'2', '3'}


def p0_v3_1(check=None):  # 0 parents, 3 node values
    if check is None:
        return CPT({'A': 0.2, 'B': 0.8, 'C': 0.0})
    assert isinstance(check, CPT)
    assert check.cdist() == {'A': 0.2, 'B': 0.8, 'C': 0.0}
    assert check.has_parents is False
    assert check.free_params == 2
    assert check.values == {'A', 'B', 'C'}


def p1_v2_1(check=None):  # 1 parent, 2 node values
    if check is None:
        return CPT([({'A': '0'}, {'0': 0.2, '1': 0.8}),
                    ({'A': '1'}, {'0': 0.7, '1': 0.3})])
    assert isinstance(check, CPT)
    assert check.cdist({'A': '0'}) == {'0': 0.2, '1': 0.8}
    assert check.cdist({'A': '1'}) == {'0': 0.7, '1': 0.3}
    assert check.has_parents is True
    assert check.free_params == 2
    assert check.values == {'0', '1'}


def p2_v2_1(check=None):  # 2 parents, 2 node values
    if check is None:
        return CPT([({'A': '0', 'B': '0'}, {'0': 0.2, '1': 0.8}),
                    ({'A': '0', 'B': '1'}, {'0': 0.7, '1': 0.3}),
                    ({'B': '0', 'A': '1'}, {'0': 0.5, '1': 0.5}),
                    ({'A': '1', 'B': '1'}, {'1': 0.9, '0': 0.1})])
    assert isinstance(check, CPT)
    assert check.cdist({'A': '0', 'B': '0'}) == {'0': 0.2, '1': 0.8}
    assert check.cdist({'A': '0', 'B': '1'}) == {'0': 0.7, '1': 0.3}
    assert check.cdist({'A': '1', 'B': '0'}) == {'0': 0.5, '1': 0.5}
    assert check.cdist({'A': '1', 'B': '1'}) == {'0': 0.1, '1': 0.9}
    assert check.has_parents is True
    assert check.free_params == 4
    assert check.values == {'0', '1'}


def p2_v2_1a(check=None):  # Identical to p2_v2_1
    if check is None:
        return CPT([({'A': '0', 'B': '0'}, {'1': 0.8, '0': 0.2}),
                    ({'A': '0', 'B': '1'}, {'0': 0.7, '1': 0.3}),
                    ({'A': '1', 'B': '0'}, {'0': 0.5, '1': 0.5}),
                    ({'A': '1', 'B': '1'}, {'1': 0.9, '0': 0.1})])
    assert isinstance(check, CPT)
    assert check.cdist({'A': '0', 'B': '0'}) == {'0': 0.2, '1': 0.8}
    assert check.cdist({'A': '0', 'B': '1'}) == {'0': 0.7, '1': 0.3}
    assert check.cdist({'A': '1', 'B': '0'}) == {'0': 0.5, '1': 0.5}
    assert check.cdist({'A': '1', 'B': '1'}) == {'0': 0.1, '1': 0.9}
    assert check.has_parents is True
    assert check.free_params == 4
    assert check.values == {'0', '1'}


def p2_v2_1b(check=None):  # NEARLY Identical to p2_v2_1
    if check is None:
        return CPT([({'A': '0', 'B': '0'}, {'0': 0.2, '1': 0.8}),
                    ({'A': '0', 'B': '1'}, {'0': 0.7, '1': 0.3}),
                    ({'B': '0', 'A': '1'}, {'0': 0.5, '1': 0.5}),
                    ({'A': '1', 'B': '1'}, {'1': 0.9, '0': 0.1})])
    assert isinstance(check, CPT)
    assert check.cdist({'A': '0', 'B': '0'}) == {'0': 0.2, '1': 0.8}
    assert check.cdist({'A': '0', 'B': '1'}) == {'0': 0.7, '1': 0.3}
    assert check.cdist({'A': '1', 'B': '0'}) == {'0': 0.5, '1': 0.5}
    assert check.cdist({'A': '1', 'B': '1'}) == {'0': 0.1, '1': 0.9}
    assert check.has_parents is True
    assert check.free_params == 4
    assert check.values == {'0', '1'}


def p2_v2_2(check=None):  # 2 parents, 2 node values
    if check is None:
        return CPT([({'A': '0', 'B': '0'}, {'0': 0.2, '1': 0.8}),
                    ({'A': '0', 'B': '1'}, {'0': 0.7001, '1': 0.2999}),
                    ({'B': '0', 'A': '1'}, {'0': 0.5, '1': 0.5}),
                    ({'A': '1', 'B': '1'}, {'1': 0.9, '0': 0.1})])
    assert isinstance(check, CPT)
    assert check.cdist({'A': '0', 'B': '0'}) == {'0': 0.2, '1': 0.8}
    assert check.cdist({'A': '0', 'B': '1'}) == {'0': 0.7001, '1': 0.2999}
    assert check.cdist({'A': '1', 'B': '0'}) == {'0': 0.5, '1': 0.5}
    assert check.cdist({'A': '1', 'B': '1'}) == {'0': 0.1, '1': 0.9}
    assert check.has_parents is True
    assert check.free_params == 4
    assert check.values == {'0', '1'}
