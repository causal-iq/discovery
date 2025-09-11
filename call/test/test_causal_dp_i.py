
# Integration tests of causal-learn exact dp score-based algorithms

import pytest

from call.causal import requires_causal_learn, causal_learn
from fileio.common import TESTDATA_DIR
from fileio.numpy import NumPy
from core.graph import EdgeType, PDAG
from core.metrics import values_same
from learn.trace import Trace


# --- Failure cases

# Astar only supports continuous data
def test_causal_dp_categorical_value_error():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/ab_3.csv',
                      dstype="categorical")
    with pytest.raises(ValueError):
        causal_learn(algorithm="dp", data=data)


# --- Successful cases with continuous data

# XY with 3 rows continuous data
@requires_causal_learn
def test_causal_dp_xy_3_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/xy_3.csv',
                      dstype="continuous")
    graph, trace = causal_learn(algorithm="dp", data=data,
                                context={"in": "xy_3", "id": "xy_3"})
    print(f"\n\nA-Star, XY, N=3:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is True
    assert graph.nodes == ["F1", "F2"]
    assert graph.edges == {}

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -10.400300, sf=8)
    assert values_same(trace.trace["delta/score"][-1], -10.400300, sf=8)


# XYZ with 10 rows continuous data
@requires_causal_learn
def test_causal_dp_xyz_10_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/xyz_10.csv',
                      dstype="continuous")
    graph, trace = causal_learn(algorithm="dp", data=data,
                                context={"in": "xyz_10", "id": "xyz_10"})
    print(f"\n\nDP, XYZ, N=10:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["X", "Y", "Z"]
    assert graph.edges == {
        ("X", "Y"): EdgeType.UNDIRECTED,
        ("Y", "Z"): EdgeType.UNDIRECTED,
    }

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -78.921743, sf=8)
    assert values_same(trace.trace["delta/score"][-1], -80.478955, sf=8)


# Gauss with 100 rows continuous data - very dense graph
@requires_causal_learn
def test_causal_dp_gauss_100_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype="continuous", N=100)
    graph, trace = causal_learn(algorithm="dp", data=data,
                                context={"in": "gauss_100", "id": "gauss_100"})
    print(f"\n\nDP, gauss, N=100:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["A", "B", "C", "D", "E", "F", "G"]
    assert graph.edges == {
        ("A", "B"): EdgeType.DIRECTED,
        ("A", "C"): EdgeType.UNDIRECTED,
        ("A", "F"): EdgeType.DIRECTED,
        ("B", "E"): EdgeType.DIRECTED,
        ("B", "G"): EdgeType.DIRECTED,
        ("C", "B"): EdgeType.DIRECTED,
        ("C", "D"): EdgeType.UNDIRECTED,
        ("D", "B"): EdgeType.DIRECTED,
        ("D", "E"): EdgeType.DIRECTED,
        ("D", "F"): EdgeType.DIRECTED,
        ("D", "G"): EdgeType.DIRECTED,
        ("E", "F"): EdgeType.DIRECTED,
        ("G", "F"): EdgeType.DIRECTED
        }

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -1783.963542, sf=10)
    assert values_same(trace.trace["delta/score"][-1], -1205.873042, sf=10)


# Gauss with 1k rows continuous data
@requires_causal_learn
def test_causal_dp_gauss_1k_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype="continuous", N=1000)
    graph, trace = causal_learn(algorithm="dp", data=data,
                                context={"in": "gauss_1k", "id": "gauss_1k"})
    print(f"\n\nDP, gauss, N=1k:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["A", "B", "C", "D", "E", "F", "G"]
    assert graph.edges == {
        ('A', 'B'): EdgeType.UNDIRECTED,
        ('A', 'C'): EdgeType.UNDIRECTED,
        ('A', 'D'): EdgeType.UNDIRECTED,
        ('A', 'F'): EdgeType.DIRECTED,
        ('B', 'C'): EdgeType.UNDIRECTED,
        ('B', 'D'): EdgeType.UNDIRECTED,
        ('B', 'E'): EdgeType.UNDIRECTED,
        ('B', 'G'): EdgeType.UNDIRECTED,
        ('C', 'D'): EdgeType.UNDIRECTED,
        ('D', 'E'): EdgeType.UNDIRECTED,
        ('D', 'F'): EdgeType.DIRECTED,
        ('D', 'G'): EdgeType.UNDIRECTED,
        ('E', 'F'): EdgeType.DIRECTED,
        ('G', 'F'): EdgeType.DIRECTED
    }

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -17607.686118, sf=10)
    assert values_same(trace.trace["delta/score"][-1], -10659.762071, sf=10)
