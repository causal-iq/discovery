
# Integration tests of causal-learn BOSS algorithm

import pytest

from call.causal import requires_causal_learn, causal_learn
from fileio.common import TESTDATA_DIR
from fileio.numpy import NumPy
from core.graph import EdgeType, PDAG
from core.metrics import values_same
from learn.trace import Trace


# --- Successful cases with categorical data

# AB with 3 rows categorical data
@requires_causal_learn
def test_causal_boss_ab_3_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/ab_3.csv',
                      dstype="categorical")
    graph, trace = causal_learn(algorithm="boss", data=data)
    print(f"\n\nGraph learnt by BOSS from 3 rows of AB:\n{graph}")

    assert trace is None
    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is True
    assert graph.nodes == ["A", "B"]
    assert graph.edges == {}


# AB with 20 rows of strongly dependent data
@requires_causal_learn
def test_causal_boss_ab_strong_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/ab_strong.csv',
                      dstype="categorical")
    graph, trace = causal_learn(algorithm="boss", data=data)
    print(f"\n\nGraph learnt by BOSS from ab_strong:\n{graph}")

    assert trace is None
    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["A", "B"]
    assert graph.edges == {
        ("A", "B"): EdgeType.UNDIRECTED
    }


# AB with 20 rows of random data
@requires_causal_learn
def test_causal_boss_ab_random_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/ab_random.csv',
                      dstype="categorical")
    graph, trace = causal_learn(algorithm="boss", data=data)
    print(f"\n\nGraph learnt by BOSS from ab_random:\n{graph}")

    assert trace is None
    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is True
    assert graph.nodes == ["A", "B"]
    assert graph.edges == {}


# A-->B-->C, 5 rows categorical data
@requires_causal_learn
def test_causal_boss_abc_5_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/abc_5.csv',
                      dstype="categorical")
    graph, trace = causal_learn(algorithm="boss", data=data)
    print(f"\n\nGraph learnt by BOSS from abc_5:\n{graph}")

    assert trace is None
    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is True
    assert graph.nodes == ["A", "B", "C"]
    assert graph.edges == {}


# A-->B-->C, 36 rows categorical data
@requires_causal_learn
def test_causal_boss_abc_36_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/abc_36.csv',
                      dstype="categorical")
    graph, trace = causal_learn(algorithm="boss", data=data)
    print(f"\n\nGraph learnt by BOSS from abc_36:\n{graph}")

    assert trace is None
    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is True
    assert graph.nodes == ["A", "B", "C"]
    assert graph.edges == {}


# Asia, 100 rows
@requires_causal_learn
def test_causal_boss_asia_100_ok():
    data = NumPy.read(filename=TESTDATA_DIR +
                      '/experiments/datasets/asia.data.gz',
                      dstype="categorical", N=100)
    graph, trace = causal_learn(algorithm="boss", data=data,
                                context={"in": "asia.dsc",
                                         "id": "boss/asia/100"})
    print(f"\n\nBOSS, Asia, N=100:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["asia", "bronc", "dysp", "either", "lung", "smoke",
                           "tub", "xray"]
    assert graph.number_components() == 3
    assert graph.edges == {
        ("bronc", "dysp"): EdgeType.UNDIRECTED,
        ("lung", "either"): EdgeType.DIRECTED,
        ("lung", "smoke"): EdgeType.UNDIRECTED,
        ("lung", "xray"): EdgeType.UNDIRECTED,
        ("tub", "either"): EdgeType.DIRECTED
    }

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -330.650761, sf=9)
    assert values_same(trace.trace["delta/score"][-1], -261.469235, sf=9)


# Asia, 10K rows
@requires_causal_learn
def test_causal_boss_asia_1k_ok():
    data = NumPy.read(filename=TESTDATA_DIR +
                      '/experiments/datasets/asia.data.gz',
                      dstype="categorical", N=1000)
    graph, trace = causal_learn(algorithm="boss", data=data,
                                context={"in": "asia.dsc",
                                         "id": "boss/asia/1k"})
    print(f"\n\nBOSS, Asia, N=1k:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["asia", "bronc", "dysp", "either", "lung", "smoke",
                           "tub", "xray"]
    assert graph.number_components() == 2
    assert graph.edges == {
        ("bronc", "dysp"): EdgeType.UNDIRECTED,
        ("bronc", "smoke"): EdgeType.DIRECTED,
        ("dysp", "lung"): EdgeType.UNDIRECTED,
        ("lung", "either"): EdgeType.DIRECTED,
        ("lung", "smoke"): EdgeType.DIRECTED,
        ("lung", "xray"): EdgeType.DIRECTED,
        ("tub", "either"): EdgeType.DIRECTED,
        ("tub", "xray"): EdgeType.DIRECTED
    }

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -3032.945048, sf=10)
    assert values_same(trace.trace["delta/score"][-1], -2270.877997, sf=10)


# --- Test timeout functionality

# Child, 10K rows - causes timeout
@requires_causal_learn
def test_causal_boss_child_1k_timeout_error():
    data = NumPy.read(filename=TESTDATA_DIR +
                      '/experiments/datasets/child.data.gz',
                      dstype="categorical", N=1000)
    with pytest.raises(RuntimeError):
        causal_learn(algorithm="boss", data=data, maxtime=1,
                     context={"in": "asia.dsc", "id": "boss/child/1k"})


# --- Successful cases with continuous data

# XY with 3 rows continuous data
@requires_causal_learn
def test_causal_boss_xy_3_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/xy_3.csv',
                      dstype="continuous")
    graph, trace = causal_learn(algorithm="boss", data=data,
                                context={"in": "xy_3", "id": "xy_3"})
    print(f"\n\nBOSS, XY, N=3:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is True
    assert graph.nodes == ["F1", "F2"]
    assert graph.edges == {}

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -10.400300, sf=8)
    assert values_same(trace.trace["delta/score"][-1], -10.400300, sf=8)


# XYZ with 10 rows continuous data
@requires_causal_learn
def test_causal_boss_xyz_10_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/xyz_10.csv',
                      dstype="continuous")
    graph, trace = causal_learn(algorithm="boss", data=data,
                                context={"in": "xyz_10", "id": "xyz_10"})
    print(f"\n\nBOSS, XY, N=3:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is True
    assert graph.nodes == ["X", "Y", "Z"]
    assert graph.edges == {}

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -78.921743, sf=8)
    assert values_same(trace.trace["delta/score"][-1], -78.921743, sf=8)


# Gauss with 100 rows continuous data - learns true graph
@requires_causal_learn
def test_causal_boss_gauss_100_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype="continuous", N=100)
    graph, trace = causal_learn(algorithm="boss", data=data,
                                context={"in": "gauss_100", "id": "gauss_100"})
    print(f"\n\nBOSS, gauss, N=100:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["A", "B", "C", "D", "E", "F", "G"]
    assert graph.edges == {
        ("A", "C"): EdgeType.DIRECTED,
        ("A", "F"): EdgeType.DIRECTED,
        ("B", "C"): EdgeType.DIRECTED,
        ("B", "D"): EdgeType.UNDIRECTED,
        ("D", "F"): EdgeType.DIRECTED,
        ("E", "F"): EdgeType.DIRECTED,
        ("G", "F"): EdgeType.DIRECTED
        }

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -1783.963542, sf=10)
    assert values_same(trace.trace["delta/score"][-1], -1104.467474, sf=10)


# Gauss with 1k rows continuous data - learns correct graph
@requires_causal_learn
def test_causal_boss_gauss_1k_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype="continuous", N=1000)
    graph, trace = causal_learn(algorithm="boss", data=data,
                                context={"in": "gauss_1k", "id": "gauss_1k"})
    print(f"\n\nBOSS, gauss, N=1k:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["A", "B", "C", "D", "E", "F", "G"]
    assert graph.edges == {
        ("A", "C"): EdgeType.DIRECTED,
        ("A", "F"): EdgeType.DIRECTED,
        ("B", "C"): EdgeType.DIRECTED,
        ("B", "D"): EdgeType.UNDIRECTED,
        ("D", "F"): EdgeType.DIRECTED,
        ("E", "F"): EdgeType.DIRECTED,
        ("G", "F"): EdgeType.DIRECTED
        }

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -17607.686118, sf=10)
    assert values_same(trace.trace["delta/score"][-1], -10638.031475, sf=10)
