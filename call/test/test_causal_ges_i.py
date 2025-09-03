
# Integration tests of causal-learn GES algorithm

import pytest

from call.causal import requires_causal_learn, causal_learn
from fileio.common import TESTDATA_DIR
from fileio.numpy import NumPy
from core.graph import EdgeType, PDAG
from core.metrics import values_same
from learn.trace import Trace


# --- Failure cases

# Unknown algorithm value error
def test_causal_bad_algo_value_error():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/ab_3.csv',
                      dstype="categorical")
    with pytest.raises(ValueError):
        causal_learn(algorithm="unknown", data=data)


# --- Successful cases with categorical data

# AB with 3 rows categorical data
@requires_causal_learn
def test_causal_ges_ab_3_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/ab_3.csv',
                      dstype="categorical")
    graph, trace = causal_learn(algorithm="ges", data=data)
    print(f"\n\nGraph learnt by GES from 3 rows of AB:\n{graph}")

    assert trace is None
    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is True
    assert graph.nodes == ["A", "B"]
    assert graph.edges == {}


# AB with 20 rows of strongly dependent data
@requires_causal_learn
def test_causal_ges_ab_strong_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/ab_strong.csv',
                      dstype="categorical")
    graph, trace = causal_learn(algorithm="ges", data=data)
    print(f"\n\nGraph learnt by GES from ab_strong:\n{graph}")

    assert trace is None
    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["A", "B"]
    assert graph.edges == {
        ("A", "B"): EdgeType.UNDIRECTED
    }


# AB with 20 rows of random data
@requires_causal_learn
def test_causal_ges_ab_random_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/ab_random.csv',
                      dstype="categorical")
    graph, trace = causal_learn(algorithm="ges", data=data)
    print(f"\n\nGraph learnt by GES from ab_random:\n{graph}")

    assert trace is None
    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is True
    assert graph.nodes == ["A", "B"]
    assert graph.edges == {}


# A-->B-->C, 5 rows categorical data
@requires_causal_learn
def test_causal_ges_abc_5_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/abc_5.csv',
                      dstype="categorical")
    graph, trace = causal_learn(algorithm="ges", data=data)
    print(f"\n\nGraph learnt by GES from abc_5:\n{graph}")

    assert trace is None
    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is True
    assert graph.nodes == ["A", "B", "C"]
    assert graph.edges == {}


# A-->B-->C, 36 rows categorical data
@requires_causal_learn
def test_causal_ges_abc_36_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/abc_36.csv',
                      dstype="categorical")
    graph, trace = causal_learn(algorithm="ges", data=data)
    print(f"\n\nGraph learnt by GES from abc_36:\n{graph}")

    assert trace is None
    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is True
    assert graph.nodes == ["A", "B", "C"]
    assert graph.edges == {}


# Asia, 100 rows
@requires_causal_learn
def test_causal_asia_100_ok():
    data = NumPy.read(filename=TESTDATA_DIR +
                      '/experiments/datasets/asia.data.gz',
                      dstype="categorical", N=100)
    graph, trace = causal_learn(algorithm="ges", data=data,
                                context={"in": "asia.dsc",
                                         "id": "ges/asia/100"})
    print(f"\n\nGES, Asia, N=100:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["asia", "bronc", "dysp", "either", "lung", "smoke",
                           "tub", "xray"]
    assert graph.number_components() == 3
    assert graph.edges == {
        ("bronc", "dysp"): EdgeType.UNDIRECTED,
        ("either", "xray"): EdgeType.DIRECTED,
        ("lung", "either"): EdgeType.DIRECTED,
        ("lung", "smoke"): EdgeType.UNDIRECTED,
        ("tub", "either"): EdgeType.DIRECTED
    }

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -330.650761, sf=9)
    assert values_same(trace.trace["delta/score"][-1], -257.786858, sf=9)


# Asia, 10K rows
@requires_causal_learn
def test_causal_asia_1k_ok():
    data = NumPy.read(filename=TESTDATA_DIR +
                      '/experiments/datasets/asia.data.gz',
                      dstype="categorical", N=1000)
    graph, trace = causal_learn(algorithm="ges", data=data,
                                context={"in": "asia.dsc",
                                         "id": "ges/asia/1k"})
    print(f"\n\nGES, Asia, N=1k:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["asia", "bronc", "dysp", "either", "lung", "smoke",
                           "tub", "xray"]
    assert graph.number_components() == 2
    assert graph.edges == {
        ("bronc", "dysp"): EdgeType.DIRECTED,
        ("bronc", "smoke"): EdgeType.UNDIRECTED,
        ("either", "dysp"): EdgeType.DIRECTED,
        ("either", "xray"): EdgeType.DIRECTED,
        ("lung", "either"): EdgeType.DIRECTED,
        ("lung", "smoke"): EdgeType.UNDIRECTED,
        ("tub", "either"): EdgeType.DIRECTED
    }

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -3032.945048, sf=10)
    assert values_same(trace.trace["delta/score"][-1], -2262.291085, sf=10)


# --- Successful cases with continuous data

# XY with 3 rows continuous data
@requires_causal_learn
def test_causal_ges_xy_3_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/xy_3.csv',
                      dstype="continuous")
    graph, trace = causal_learn(algorithm="ges", data=data,
                                context={"in": "xy_3", "id": "xy_3"})
    print(f"\n\nGES, XY, N=3:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["F1", "F2"]
    assert graph.edges == {("F1", "F2"): EdgeType.UNDIRECTED}

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -10.400300, sf=8)
    assert values_same(trace.trace["delta/score"][-1], -10.645278, sf=8)


# XYZ with 10 rows continuous data
@requires_causal_learn
def test_causal_ges_xyz_10_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/xyz_10.csv',
                      dstype="continuous")
    graph, trace = causal_learn(algorithm="ges", data=data,
                                context={"in": "xyz_10", "id": "xyz_10"})
    print(f"\n\nGES, XY, N=3:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["X", "Y", "Z"]
    assert graph.edges == {
        ("X", "Y"): EdgeType.UNDIRECTED,
        ("X", "Z"): EdgeType.UNDIRECTED
        }

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -78.921743, sf=8)
    assert values_same(trace.trace["delta/score"][-1], -80.847896, sf=8)


# Gauss with 100 rows continuous data
@requires_causal_learn
def test_causal_ges_gauss_100_ok():
    data = NumPy.read(filename=TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype="continuous", N=100)
    graph, trace = causal_learn(algorithm="ges", data=data,
                                context={"in": "gauss_100", "id": "gauss_100"})
    print(f"\n\nGES, gauss, N=100:\n{trace}\n{graph}")

    assert isinstance(graph, PDAG)
    assert graph.is_DAG() is False
    assert graph.nodes == ["A", "B", "C", "D", "E", "F", "G"]
    assert graph.edges == {
        ("A", "B"): EdgeType.UNDIRECTED,
        ("A", "C"): EdgeType.UNDIRECTED,
        ("A", "D"): EdgeType.UNDIRECTED,
        ("A", "E"): EdgeType.UNDIRECTED,
        ("A", "F"): EdgeType.UNDIRECTED,
        ("A", "G"): EdgeType.UNDIRECTED
        }

    assert isinstance(trace, Trace)
    assert values_same(trace.trace["delta/score"][0], -1783.963542, sf=10)
    assert values_same(trace.trace["delta/score"][-1], -1793.276536, sf=10)
