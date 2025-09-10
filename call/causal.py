#
#  Module to call causal-learnt Python algorithms
#

from time import time
from pytest import skip
from numpy import ndarray
from functools import wraps

from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GESUtils import score_g
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.score.LocalScoreFunction import local_score_BDeu, \
    local_score_BIC
from causallearn.utils.PDAG2DAG import pdag2dag

from learn.trace import CONTEXT_FIELDS, Trace, Activity, Detail
from fileio.pandas import Pandas
from fileio.numpy import NumPy
from core.graph import PDAG, DAG
from core.timing import run_with_timeout, TimeoutError

CAUSAL_ALGORITHMS = {
    "astar": "score",
    "boss": "score",
    "ges": "score",
    "grasp": "score"

}


def requires_causal_learn(func):
    """
        Decorator to skip test if causal-learn package is not installed
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            from causallearn.graph.GraphNode import GraphNode
            GraphNode('A')  # just to stop linter error
        except ImportError:
            skip("causal-learn package not installed")
        return func(*args, **kwargs)

    return wrapper


def _validate_learn_params(params: dict, dstype: str):
    """
        Validate parameters supplied for learning by causal-learn.

        :param dict params: parameters as specified in causaliq
        :param str dstype: dataset type: categorical, continuous or mixed

        :raises TypeError: if params have invalid types
        :raises ValueError: if invalid parameters or data values
    """
    # not supporting mixed data types yet
    if dstype not in {'categorical', 'continuous'}:
        raise ValueError('Mixed data types unsupported')

    # set default parameter values
    _params = ({"score": "bde", "iss": 1, "base": "e"}
               if dstype == "categorical" else
               {"score": "bic-g", "k": 1, "base": "e"})

    # override defaults with any specfied parameters
    _params.update(params if params is not None else {})

    # At moment , only discrete data with BDeu and continuous data with BIC
    # score are supported
    if ((dstype == "categorical" and
            _params != {"score": "bde", "iss": 1, "base": "e"}) or
        (dstype == "continuous" and
            _params != {"score": "bic-g", "k": 1, "base": "e"})):
        raise ValueError("causal_learn: bad parameter values")

    return _params


def _generate_trace(graph: GeneralGraph, elapsed: float, data: NumPy,
                    params: dict, context: dict,
                    steps: tuple[list[GeneralGraph], list[GeneralGraph]]
                    = None):
    """
        Generate a (minimal) CausalIQ Learning Trace

        :param GeneralGraph graph: the learned graph
        :param float elapsed: elapsed time for learning (seconds)
        :param NumPy data: data graph learned from in CausalIQ format
        :param dict params: the learning parameters
        :param dict context: context information
        :param tuple steps: 2 sequences of graphs learned

        :returns Trace: minimal CausalIQ trace with initial and final steps
    """
    # set up a scoring function
    score_func = LocalScoreClass(
        data=data.sample,
        local_score_fun=(local_score_BDeu if params['score'] == 'bde'
                         else local_score_BIC),
        parameters=None  # Will use default lambda_value = 1
    )

    # Obtain external (e.g. randomised) node names
    node_names = [data.orig_to_ext[n] for n in data.nodes]

    # Empty graph and obtain its score using causal-learn and CausalIQ
    empty = GeneralGraph(nodes=[GraphNode(n) for n in node_names])
    empty_cl_score = score_g(data, empty, score_func, None)
    empty = DAG(nodes=node_names, edges=[])
    empty_score = (empty.score(data=data,
                               types=[params['score']])[params['score']]).sum()
    print(f"Initial score: {empty_cl_score:.5e}, {empty_score:.5e}\n")

    # Extend learned PDAG to a DAG and obtain its score using CL & CausalIQ

    dag = pdag2dag(graph)
    learned_cl_score = score_g(data, dag, score_func, None)
    if isinstance(learned_cl_score, ndarray):
        learned_cl_score = learned_cl_score.sum()
    pdag = to_causaliq_pdag(graph)
    dag = DAG.extendPDAG(pdag)
    learned_score = dag.score(data=data,
                              types=[params["score"]])[params['score']].sum()
    print(f"Learned score: {learned_cl_score:.5e}, {learned_score:.5e}\n")

    # Instantiate Trace with context details and add init and stop records
    # and learnt graph
    context = context.copy()
    context.update({'algorithm': "GES", 'params': params, 'N': data.N,
                    'external': 'causal-learn', 'dataset': True})
    trace = Trace(context)
    trace.add(Activity.INIT, {Detail.DELTA: empty_score})
    trace.add(Activity.STOP, {Detail.DELTA: learned_score})
    trace.trace['time'][-1] = elapsed
    trace.result = pdag

    # print out edge changes in the two phases - this could be the basis of
    # creating a full trace, by adding new direct/undirect actions, or by
    # representing an undirected edge by the two opposing directed arcs.
    # Each iteration generally results in two edge changes.
    if steps is not None:
        prev_edges = set()
        for i, graph in enumerate(steps[0]):
            edges = {str(e) for e in graph.get_graph_edges()}
            print(f"INSERT iteration {i+1} adds : {edges - prev_edges}"
                  f" and drops: {prev_edges - edges}")
            prev_edges = edges
        for i, graph in enumerate(steps[1]):
            edges = {str(e) for e in graph.get_graph_edges()}
            print(f"DELETE iteration {i+1} adds : {edges - prev_edges}"
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


# Define the GES algorithm execution as a separate function for timeout
def run_algo(algorithm: str, data: NumPy, score: str) -> dict:
    """
        Run the causal-learn algorithm

        :param str algorithm: name of algorithmtorun e.g. "ges" or "astar"
        :param data NumPy: data to learn structure from
        :param str score: score to use (CausalIQ name,e.g. "bde")

        :raises ValueError: if unknown algorithm specified

        :returns dict: of algorithm results
    """
    print(f"\n\nAlgorithm is {algorithm} using {score} score")

    score_func = "local_score_BDeu" if score == 'bde' else "local_score_BIC"
    print(score_func)
    node_names = [data.orig_to_ext[n] for n in data.nodes]
    return ges(data.sample, score_func=score_func, maxP=None,
               parameters=None, node_names=node_names)


def causal_learn(algorithm, data, context=None, params=None, maxtime=None):
    """
        Return graph learnt from data using causal-learn algorithms

        :param str algorithm: algorithm to use, e.g. 'fges'
        :param Numpy/Pandas/str data: data or data filename to learn from
        :param dict context: context information about the test/experiment
        :param dict params: parameters for algorithm e.g. score to use
        :param int maxtime: maximum execution time in seconds,
                          None for no limit

        :raises TypeError: if arg types incorrect
        :raises ValueError: if invalid params supplied
        :raises FileNotFoundError: if a specified data file does not exist
        :raises RuntimeError: if unexpected error running algorithm
        :raises TimeoutError: if execution exceeds maxtime

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

    print(f"\n\nRunning {algorithm} algorithm in causal-learn ...\n")

    # Execute the algorithm with timeout if specified
    start = time()
    try:
        results = run_with_timeout(run_algo,
                                   args=(algorithm, data, params["score"]),
                                   timeout_seconds=maxtime)
        elapsed = time() - start
        graph = results["G"]
    except TimeoutError:
        print(f"*** causal-learn algorithm timed out after {maxtime} seconds")
        raise RuntimeError(f"causal-learn algorithm timed out after "
                           f"{maxtime} seconds")
    except Exception as e:
        elapsed = time() - start
        print(f"*** causal-learn failed: {e}")
        raise RuntimeError(f"causal-learn failed: {e}")

    trace = (None if context is None
             else _generate_trace(graph, elapsed, data, params, context,
                                  (results["G_step1"], results["G_step2"])))

    return to_causaliq_pdag(graph), trace
