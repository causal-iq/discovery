#
#  Module to call causal-learnt Python algorithms
#

from time import time

from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GESUtils import score_g
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.score.LocalScoreFunction import local_score_BDeu
from causallearn.utils.PDAG2DAG import pdag2dag

from learn.trace import CONTEXT_FIELDS, Trace, Activity, Detail
from fileio.pandas import Pandas
from fileio.numpy import NumPy
from core.graph import PDAG

CAUSAL_ALGORITHMS = {
    'ges': 'score'
}


def _validate_learn_params(params: dict, dstype: str):
    """
        Validate parameters supplied for learning by causal-learn.

        :param dict params: parameters as specified in causaliq
        :param str dstype: dataset type: categorical, continuous or mixed

        :raises TypeError: if params have invalid types
        :raises ValueError: if invalid parameters or data values
    """
    # set default parameter values
    params = {} if params is None else params.copy()
    if "score" not in params:
        params.update({"score": "bde"})
    if params["score"] == "bde" and "iss" not in params:
        params.update({"iss": 1})
    if "base" not in params:
        params.update({'base': 'e'})

    # At moment , only discrete data and standard BDeu score supported
    if (dstype != "categorical"
            or params != {"score": "bde", "iss": 1, "base": "e"}):
        raise ValueError("causal_learn: bad parameter values")


def _generate_trace(graph: GeneralGraph, elapsed: float, data: NumPy,
                    params: dict, context: dict,
                    steps: list[GeneralGraph] = None):
    """
        Generate a (minimal) CausalIQ Learning Trace

        :param GeneralGraph graph: the learned graph
        :param float elapsed: elapsed time for learning (seconds)
        :param NumPy data: data graph learned from in CausalIQ format
        :param dict params: the learning parameters
        :param dict context: context information
        :param list[GeneralGraph] steps: sequence of graphs learned

        :returns Trace: minimal CausalIQ trace with initial and final steps
    """
    # set up a scoring function
    score_func = LocalScoreClass(
        data=data.sample,
        local_score_fun=local_score_BDeu,
        parameters=None  # Will use default lambda_value = 1
    )

    # Construct the empty graph and obtain its score
    empty = GeneralGraph(nodes=[GraphNode(n) for n in data.nodes])
    empty_score = score_g(data, empty, score_func, None)
    print(f"Initial score: {empty_score:.5e}\n")

    # Extend learned PDAG to a DAG and obtain its score

    dag = pdag2dag(graph)
    learned_score = score_g(data, dag, score_func, None)
    print(f"Learned BIC score: {learned_score}\n")

    # Instantiate Trace with context details and add init and stop records
    # and learnt graph
    context = context.copy()
    context.update({'algorithm': "GES", 'params': params, 'N': data.N,
                    'external': 'causal-learn', 'dataset': True})
    trace = Trace(context)
    trace.add(Activity.INIT, {Detail.DELTA: empty_score})
    trace.add(Activity.STOP, {Detail.DELTA: learned_score})
    trace.trace['time'][-1] = elapsed
    trace.result = to_causaliq_pdag(graph)

    # print out edge changes in the two phases - this could be the basis of
    # creating a full trace, by adding new direct/undirect actions, or by
    # representing an undirected edge by the two opposing directed arcs.
    # Each iteration generally results in two edge changes.
    if steps is not None:
        prev_edges = set()
        for i, graph in enumerate(steps):
            edges = {str(e) for e in graph.get_graph_edges()}
            print(f"INSERT iteration {i+1} adds : {edges - prev_edges}"
                  f" and drops: {prev_edges - edges}")
            prev_edges = edges

    return trace


def to_causaliq_pdag(graph: GeneralGraph):
    """
        Converts causal-learn GeneralGraph to a CausalIQ PDAG object

        :param GeneralGraph graph: causal-learn general graph

        :returns PDAG: CausalIQ PDAG
    """
    # express edges in format required by PDAG constructor
    edges = [str(e).split(" ") for e in graph.get_graph_edges()]
    edges = [(e[0], "->" if e[1] == "-->" else "-", e[2]) for e in edges]

    return PDAG(nodes=graph.get_node_names(), edges=edges)


def causal_learn(algorithm, data, context=None, params=None):
    """
        Return graph learnt from data using tetrad algorithms

        :param str algorithm: algorithm to use, e.g. 'fges'
        :param Numpy/Pandas/str data: data or data filename to learn from
        :param dict context: context information about the test/experiment
        :param dict params: parameters for algorithm e.g. score to use

        :raises TypeError: if arg types incorrect
        :raises ValueError: if invalid params supplied
        :raises FileNotFoundError: if a specified data file does not exist
        :raises RuntimeError: if unexpected error running Tetrad

        :returns tuple: (DAG/PDAG learnt from data, learning trace)
    """
    if (not isinstance(algorithm, str)
            or not isinstance(data, (NumPy, Pandas))
            or (context is not None and not isinstance(context, dict))
            or (params is not None and not isinstance(params, dict))):
        raise TypeError('causal-learn bad arg types')

    if algorithm not in CAUSAL_ALGORITHMS:
        raise ValueError('causal-learn unsupported algorithm')

    params = _validate_learn_params(params=params, dstype=data.dstype)

    if (context is not None
            and (len(set(context.keys()) - set(CONTEXT_FIELDS))
                 or not {'in', 'id'}.issubset(context.keys()))):
        raise ValueError('causal-learn() bad context values')

    # Validate learning parameters. Return parameters for Trace , for
    # the Tetrad executable itself, and Tetrad version to use.

    print(f"\n\nRunning {algorithm} algorithm in causal-learn ...\n")

    # learn the graph using BDeu
    start = time()
    node_names = [data.orig_to_ext[n] for n in data.nodes]
    results = ges(data.sample, score_func='local_score_BDeu', maxP=None,
                  parameters=None, node_names=node_names)
    elapsed = time() - start
    graph = results["G"]

    trace = (None if context is None
             else _generate_trace(graph, elapsed, data, params, context,
                                  results["G_step1"]))

    return to_causaliq_pdag(graph), trace
