#
#  Module to call Tetrad algorithms using dispatch_cmd method
#

from os import remove
from random import random
from re import compile
from datetime import datetime

from core.graph import SDG, PDAG, DAG
from core.score import SCORE_PARAMS
from learn.trace import Trace, Activity, Detail, CONTEXT_FIELDS
from call.cmd import dispatch_cmd
from fileio.pandas import Pandas
from fileio.numpy import NumPy


TETRAD_ALGORITHMS = {
    'fges': 'score'
}

START_SEARCH = 'Start search: '
END_SEARCH = 'End search: '
START_EDGES = 'Graph Edges:'
EDGE_PATTERN = compile(r'^\d+\.\s(\w+)\s([\-o\<])\-([\-o\>])\s(\w+)$')


def _validate_learn_params(algorithm, params, dstype):
    """
        Validate parameters supplied for learning algorithm

        :param str algorithm: algorithm to use, e.g. 'hc', 'pc.stable'
        :param dict params: parameters as specified in bnbench
        :param str dstype: dataset type: categorical, continuous or mixed

        :raises TypeError: if params have invalid types
        :raises ValueError: if invalid parameters or data values

        :returns tuple: (parameters to record in trace,
                         parameters in format required by tetrad)
    """

    # set default parameters if not set

    params = {} if params is None else params.copy()
    if 'score' not in params:
        params.update({'score': 'bic'})
    if 'k' not in params:
        params.update({'k': 1})

    # check params are valid and have valid values

    if (not isinstance(params['score'], str)
            or not isinstance(params['k'], int)):
        raise TypeError('tetrad_learn() bad parameter TypeError')

    # check parameter values

    if (len(set(params) - {'score', 'k'})
        or (dstype == 'categorical' and params['score'] != 'bic')
        or (dstype == 'continuous' and params['score'] != 'bic-g')
            or dstype == 'mixed' or params['k'] != 1):
        raise ValueError('tetrad_learn invalid parameter')

    # setup Tetrad parameters and version

    tetrad_v = '1.3.0'
    tetrad_p = '--algorithm fges --skip-latest'
    tetrad_p += (' --data-type discrete --score disc-bic-score'
                 if dstype == 'categorical' else
                 ' --data-type continuous --score cg-bic-score')

    return (params, tetrad_v, tetrad_p)


def _edge(match):
    """
        Returns edge in native format from pattern match

        :param list match: elements in pattern match

        :returns (nodes, type)
    """
    node1 = match[1]
    ep1 = match[2]
    ep2 = match[3]
    node2 = match[4]

    if ep1 == '-' and ep2 == '>':
        return (node1, '->', node2)
    elif ep1 == '-' and ep2 == '-':
        return ((node1, '-', node2) if node1 < node2
                else (node2, '-', node1))
    else:
        raise ValueError('learn_tetrad() bad edge {}'.format(match))


def _datetime(line):
    """
        Extracts datetime from an output line

        :param str line: formatted output line containing date

        :returns DateTime: corresponding date/time object
    """
    dt = line.replace(START_SEARCH, '').replace(END_SEARCH, '')
    dt = dt.replace('am', 'AM').replace('pm', 'PM')
    return datetime.strptime(dt, '%a, %B %d, %Y %H:%M:%S %p')


def process_output(id, nodes):
    """
        Process Tetrad learning algorithm to obtain learnt graph.

        :param int id: unique id for this run
        :param list nodes: nodes in the graph

        :returns tuple: (graph, elapsed)
    """
    edges_section = False
    edges = []

    file_name = 'call/java/tmp/{}.txt'.format(id)  # for v1.10 will be _out
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            # print('"' + line + '"')
            if line == START_EDGES:
                edges_section = True
            elif not len(line):
                edges_section = False
            elif edges_section is True:
                edge = EDGE_PATTERN.match(line)
                if edge is None:
                    raise ValueError('tetrad_learn() bad line "{}"'
                                     .format(line))
                edge = _edge(edge)
                edges.append(edge)
                # print('*** edge: {}'.format(edge))
            elif line.startswith(START_SEARCH):
                start = _datetime(line)
            elif line.startswith(END_SEARCH):
                elapsed = (_datetime(line) - start).total_seconds()

    graph = SDG(nodes, edges)
    if graph.is_DAG():
        graph = DAG(nodes, edges)
    elif graph.is_PDAG():
        graph = PDAG(nodes, edges)
    else:
        raise ValueError('tetrad_learn() non-PDAG learnt')

    return (graph, elapsed)


def graph_scores(learnt, data, params, context):
    """
        Determine score of initial and learnt graphs.

        :param PDAG learnt: graph learnt by Tetrad
        :param Data data: data graph learnt from
        :param dict params: learning hyperparameters
        :param dict context: parameters for algorithm e.g. score to use

        :returns tuple: (initial, learnt) graph scores
    """
    score_type = params['score']
    score_params = {k: v for k, v in params.items() if k in SCORE_PARAMS}
    score_params.update({'unistate_ok': True})

    # determine score of initial empty graph

    initial = DAG(list(data.get_order()), [])
    score1 = (initial.score(data, score_type, score_params)[score_type]).sum()

    # Try to extend the learnt graph to a DAG to be able to score it

    try:
        learnt = learnt if learnt.is_DAG() else DAG.extendPDAG(learnt)
        score2 = (learnt.score(data, score_type,
                               score_params)[score_type]).sum()
    except ValueError as e:
        print('\n*** Cannot extend PDAG for {} {}:\n{}\n'
              .format(context['id'], e, learnt))
        score2 = 0.0

    return score1, score2


def tetrad_learn(algorithm, data, context=None, params=None):
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
        raise TypeError('tetrad_learn bad arg types')

    if algorithm not in TETRAD_ALGORITHMS:
        raise ValueError('tetrad_learn unsupported algorithm')

    if (context is not None
            and (len(set(context.keys()) - set(CONTEXT_FIELDS))
                 or not {'in', 'id'}.issubset(context.keys()))):
        raise ValueError('tetrad_learn() bad context values')

    # Validate learning parameters. Return parameters for Trace , for
    # the Tetrad executable itself, and Tetrad version to use.

    params, tetrad_v, tetrad_p = _validate_learn_params(algorithm, params,
                                                        data.dstype)

    # Generate or copy data file in Java tmp folder

    id = '{}'.format(int(random() * 10000000))
    tmpfile = 'call/java/tmp/{}.csv'.format(id)

    # Write out data with current column names, order. A bug means that
    # Pandas.write() does not do this, so for now have to use as_df() in
    # this convoluted way at the moment.

    Pandas(df=data.as_df()).write(tmpfile)

    nodes = list(data.get_order())  # gives external node names as required
    if len(nodes) < 2:
        remove(tmpfile)
        raise ValueError('tetrad_learn data < 2 columns')

    try:

        # Call a cmd sub-process to run the causal-cmd Tetrad jar file to
        # perform the learning

        dispatch_cmd(['call\\cmd\\tetrad.bat', id, tetrad_v, tetrad_p])

        # Process the output file to obtain the learnt graph

        graph, elapsed = process_output(id, nodes)

        if context is not None:

            # Generate a (not so useful) trace if a context was specified

            context = context.copy()
            context.update({'algorithm': algorithm.upper(), 'params': params,
                            'N': data.N, 'external': 'TETRAD',
                            'dataset': True})
            trace = Trace(context)

            # Construct simple trace which records elapsed time and scores
            # of and learnt initial graph

            score_i, score_g = graph_scores(graph, data, params, context)
            trace.add(Activity.INIT, {Detail.DELTA: score_i})
            trace.add(Activity.STOP, {Detail.DELTA: score_g})
            trace.trace['time'][-1] = elapsed
            trace.result = graph
        else:
            trace = None

    except Exception as e:
        print('\n*** Exception: tetrad_learn(): {}\n'.format(e))
        raise RuntimeError('tetrad_learn(): ' + str(e))

    finally:
        remove(tmpfile)
        remove(tmpfile.replace('.csv', '.txt'))  # _out for v1.10

    return (graph, trace)
