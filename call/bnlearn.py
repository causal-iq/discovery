#
#  Module to call selected bnlearn R functions using dispatch_r method
#

from pandas import DataFrame, read_csv
from os.path import abspath, exists
from os import remove
from random import random
from re import compile

from call.r import dispatch_r
from core.common import EdgeType
from core.graph import DAG, PDAG
from core.bn import BN
from core.lingauss import LinGauss
from core.score import SCORES, SCORE_PARAMS, check_score_params
from core.indep import TESTS, TEST_PARAMS, check_test_params
from core.indep import check_indep_args, MIN_P_VALUE
from fileio.common import DatasetType, is_valid_path
from fileio.pandas import Pandas
from learn.trace import Trace, Activity, Detail, CONTEXT_FIELDS
from learn.knowledge import Knowledge
from learn.knowledge_rule import Rule

BNLEARN_ALGORITHMS = {
    'gs': 'constraint',
    'h2pc': 'hybrid',
    'hc': 'score',
    'inter.iamb': 'constraint',
    'mmhc': 'hybrid',
    'pc.stable': 'constraint',
    'tabu': 'score'
}


def _arcs_to_edges(arcs):
    """
        Converts arcs in format returned by bnlearn to edges (n1, t, n1)

        :param list arcs: arcs in bnlearn format which is a list of all
                          endpoints, all "from" ones first, and then all "to"
                          ones. Undirected edges represented by two opposing
                          arcs

        :returns list: of tuples in format (n1, '->' or '-', n2)
    """
    halfway = int(len(arcs) / 2)
    arcs = {(arcs[i], arcs[halfway+i]) for i in range(0, halfway)}
    return [(e[0], '-' if (e[1], e[0]) in arcs else '->', e[1])
            for e in arcs if (e[1], e[0]) not in arcs
            or ((e[1], e[0]) in arcs and e[0] < e[1])]


def bnlearn_score(dag, data, types, params, dstype='categorical'):
    """
        Score data against a DAG using bnlearn

        :param DAG dag: DAG to score data against
        :param DataFrame/str data: data or file containing data
        :param str/list types: type/types of score e.g. 'bic', 'bdeu'
        :param dict params: score parameters e.g. 'iss'
        :param str dstype: dataset type: categorical, continuous or mixed

        :raises TypeError: if arg types incorrect
        :raises ValueError: if arg values in valid
        :raises FileNotFoundError: if data file not found
        :raises RuntimeError: if error in R-code

        :returns dict: of scores supported by bnlearn
    """
    if not isinstance(dag, DAG) or \
            (not isinstance(types, str) and not (isinstance(types, list)
             and all([isinstance(t, str) for t in types]))) or \
            (not isinstance(data, DataFrame) and not isinstance(data, str)) \
            or not isinstance(params, dict) or dstype != 'categorical':
        raise TypeError('bnlearn_score called with bad arg types')

    data_file = None if isinstance(data, DataFrame) else data
    if data_file:
        data = Pandas.read(data_file).df

    if len(dag.nodes) < 2 or sorted(data.columns) != dag.nodes:
        raise ValueError('Too few or mismatched variables')

    if any([len(data[c].value_counts()) < 2 for c in data.columns]):
        raise ValueError('Some variables are single-valued')

    types = types if isinstance(types, list) else [types]
    if any([t not in SCORES for t in types]):
        raise ValueError('Unsupported score type specified')

    # setup parameters for R function

    score_params = {p: v for p, v in params.items() if p in SCORE_PARAMS}
    params = {'dag': dag.to_string(), 'types': types, 'dstype': dstype}
    if data_file:
        params['datafile'] = abspath(data_file)
    else:
        params['data'] = {c: list(data[c]) for c in data.columns}

    # Check the score parameters and set default score parameters

    params.update(check_score_params(score_params, scores=types))

    # Make the request to R code

    scores, stdout = dispatch_r('bnlearn', 'dagscore', params)
    for line in stdout:
        print(line)
    return scores


def bnlearn_indep(x, y, z, data, types=['mi', 'x2']):
    """
        Performance an independence test using bnlearn

        :param x str: name of first variable
        :param y str: name of second variable
        :param z str/list/None: name(s) of any conditioning variables
        :param DataFrame/str data: data or file containing data
        :param str/list types: type/types of independence test to perform

        :raises TypeError: if arg types incorrect
        :raises ValueError: if arg values in valid
        :raises FileNotFoundError: if data file not found
        :raises RuntimeError: if error in R-code

        :returns dict: of statistic and p-values
    """
    z, data, data_file, types = check_indep_args(x, y, z, data, types=types)

    # setup parameters for R function
    # (change name of g2 test to mi for bnlearn)

    params = {'x': x, 'y': y, 'z': z, 'types': types, 'dstype': 'categorical'}
    if data_file:
        params['datafile'] = abspath(data_file)
    else:
        params['data'] = {c: list(data[c]) for c in data.columns}

    # Make the request to R Code

    results, _ = dispatch_r('bnlearn', 'citest', params)

    # Rename CI test name from bnlearn's "mi" to "g2" and set p-values
    # below MIN_P_VALUE to zero, and return as DataFrame

    results = {t: {k: 0.0 if k == 'p_value' and v < MIN_P_VALUE else v
                   for k, v in vs.items()} for t, vs in results.items()}
    return DataFrame.from_dict(results)


def bnlearn_import(rda):
    """
        Import a BN bnlearn object from a R .rda file in folder - only supports
        continuous BNs

        :param str rda: name of .rda file

        :raises TypeError: if bad argument type
        :raises FileNotFoundError: if relevant RDA file not found
        :raises ValueError: if problems with RDA file
    """
    RDA = 'call/R/rda/{}.rda'
    DISTKEYS = {'node', 'parents', 'children', 'coefficients', 'sd',
                'residuals', 'fitted.values'}

    if not isinstance(rda, str):
        raise TypeError('bnlearn_import bad arg type')
    is_valid_path(RDA.format(rda), is_file=True)

    # Make the request to R Code

    try:
        bn, _ = dispatch_r('bnlearn', 'import', {'rda': RDA.format(rda)})
    except RuntimeError:
        raise ValueError('bnlearn_import invalid RDA file')

    if (not isinstance(bn, dict)
        or not all([isinstance(n, str) and isinstance(d, dict) and
                    set(d) == DISTKEYS and n == d['node']
                    for n, d in bn.items()])):
        raise ValueError('bnlearn_import invalid RDA file')

    # print('\n\n')
    arcs = []
    cnd_specs = {}
    for node, dist in bn.items():
        parents = (dist['parents'] if isinstance(dist['parents'], list)
                   else [dist['parents']])
        coeffs = {c: float(v) for c, v in dist['coefficients'].items()}
        sd = float(dist['sd'])
        if set(coeffs) != {'(Intercept)'} | set(parents):
            raise ValueError('bnlearn_import invalid RDA file')
        mean = float(coeffs.pop('(Intercept)'))
        # print('Node {} has SD {}, parents {}, mean {}, and coeffs {}'
        #       .format(node, sd, parents, mean, coeffs))
        cnd_specs[node] = (LinGauss,
                           {'coeffs': coeffs, 'mean': mean, 'sd': sd})
        arcs += [(p, '->', node) for p in coeffs]

    return BN(DAG(list(bn), arcs), cnd_specs)


def bnlearn_cpdag(pdag):
    """
        Return CPDAG corresponding to supplied PDAG

        :param PDAG pdag: PDAG to transform to CPDAG

        :raises TypeError: if arg types incorrect
        :raises ValueError: for unsupported PDAGs, e.g. empty
        :raises RuntimeError: if error in R-code

        :returns PDAG: CPDAG corresponding to pdag argument
    """
    if not isinstance(pdag, PDAG):
        raise TypeError('bnlearn_cpdag bad arg types')

    if not pdag.nodes:
        raise ValueError('bnlearn_cpdag unsupported PDAG')

    # Build request and send to R code

    params = {'nodes': '[' + ']['.join(pdag.nodes) + ']',
              'edges': [(e[0], '->' if t == EdgeType.DIRECTED else '-', e[1])
                        for e, t in pdag.edges.items()]}
    arcs, _ = dispatch_r('bnlearn', 'pdag2cpdag', params)

    return PDAG(pdag.nodes, _arcs_to_edges(arcs))


def bnlearn_compare(pdag, ref):
    """
        Compares pdag structure to a reference pdag and returns confusion
        matrix elements TP, FP and FN and SHD (Note bnlearn changes DAGs to
        CPDAGs before doing SHD computation only)

        :param PDAG pdag: PDAG to compare against ...
        :param PDAG ref: ... the reference PDAG

        :raises TypeError: if arg types incorrect
        :raises RuntimeError: if error in R-code

        :returns dict: {'tp': int1, 'fp': int2, 'fn': int3, 'shd': int4}
    """
    if not isinstance(pdag, PDAG) or not isinstance(ref, PDAG):
        raise TypeError('bnlearn_compare bad arg types')

    if not pdag.nodes or pdag.nodes != ref.nodes:
        raise ValueError('bnlearn_compare bad pdag values')

    # Build request and send to R code

    params = {'nodes': '[' + ']['.join(ref.nodes) + ']',
              'ref': [(e[0], '->' if t == EdgeType.DIRECTED else '-', e[1])
                      for e, t in ref.edges.items()],
              'edges': [(e[0], '->' if t == EdgeType.DIRECTED else '-', e[1])
                        for e, t in pdag.edges.items()]}

    return dispatch_r('bnlearn', 'compare', params)[0]


def _get_hc_trace(stdout, algorithm, context, params, N, dag, elapsed):
    """
        Extract bnlearn HC trace from debug in stdout

        :param list stdout: output lines produced by R subprocess
        :param str algorithm: algorithm to use, e.g. 'hc', 'pc.stable'
        :param dict context: context information for experiment, e.g. id
        :param dict params: parameters for learning process e.g. score
        :param int N: number of rows in dataset
        :param DAG dag: learnt dag to add into trace
        :param float elapsed: elapsed time for learning graph

        :returns Trace: trace file for learning process
    """

    # print('\n{}'.format('\n'.join(stdout)))

    score_pattern = compile(r'^\* current score: ([+-]?\d*\.?\d*)\s*$')
    best_pattern = compile(r'^\* best operation was: ' +
                           r'(.*)\s+(\S+) \-\> (\S+)\s*\.$')

    delta_pattern = compile(r'^    \> delta between scores for nodes ' +
                            r'(\S+) (\S+) is ([+-]?\d*\.?\d*)\.$')
    poss_pattern = compile(r'^    \@ (adding|removing|reversing) ' +
                           r'(\S+) \-\> (\S+)\s*\.$')
    algo_pattern = compile(r'^\* Running bnlearn algorithm (\S+) \.\.\.$')
    block_pattern = compile(r'^    \> not (add|remov|revers)ing' +
                            r'\, network matches element (\d+)' +
                            r' in the tabu list\.$')

    context = context.copy()
    context.update({'algorithm': algorithm.upper(), 'params': params, 'N': N,
                    'external': 'BNLEARN', 'dataset': True})
    trace = Trace(context)
    activity = Activity.INIT

    algo = None  # algorithm being run
    arc = None
    delta = None  # last delta read
    deltas = {}  # possible changes when scanning deltas
    best_score = None  # best score of any DAG so far
    blocked = None  # holds the blocked changes
    iter = 0

    for line in stdout:
        score_match = score_pattern.match(line)  # current score of DAG
        if score_match is not None:
            score = float(score_match.group(1))
            if activity == Activity.INIT:
                best_score = score
                trace.add(activity, {Detail.DELTA: score,
                                     Detail.BLOCKED: blocked})
            elif score >= best_score:
                best_score = score

        algo_match = algo_pattern.match(line)  # detect the algorithm being run
        if algo_match is not None:
            algo = algo_match.group(1)
            blocked = [] if algo == 'tabu' else None

        block_match = block_pattern.match(line)  # change blocked by tabu list
        if block_match is not None:

            # This change is blocked because it would create a DAG currently
            # in Tabu list.

            activity = (Activity.ADD if block_match.group(1) == 'add' else
                        (Activity.DEL if block_match.group(1) == 'remov' else
                         Activity.REV))
            change = (activity.value, (delta[0], delta[1]), delta[2],
                      {'elem': int(block_match.group(2))})
            if change not in blocked:
                blocked.append(change)

        delta_match = delta_pattern.match(line)  # delta for a change
        if delta_match is not None:
            delta = (delta_match.group(1), delta_match.group(2),
                     float(delta_match.group(3)))

        poss_match = poss_pattern.match(line)  # a change with highest delta
        if poss_match is not None:
            arc = (poss_match.group(2), poss_match.group(3))
            if arc != (delta[0], delta[1]):
                raise RuntimeError('bnlearn(hc): unexpected arc {}'
                                   .format(arc))
            deltas.update({(poss_match.group(1), arc[0],
                            arc[1]): delta[2]})

        best_match = best_pattern.match(line)  # this is a change being made
        if best_match is not None:
            arc = (best_match.group(2), best_match.group(3))
            best = (best_match.group(1), best_match.group(2),
                    best_match.group(3))
            if best not in deltas:
                raise RuntimeError('bnlearn(hc): no delta for {}'
                                   .format(best))
            if best_match.group(1) == 'adding':
                activity = Activity.ADD
            elif best_match.group(1) == 'removing':
                activity = Activity.DEL
            else:
                activity = Activity.REV

            iter += 1
            # print('Change at iter {} is {} of {}, score {} and blocked {}\n'
            #       .format(iter, activity,
            #               (best_match.group(2), best_match.group(3)),
            #               deltas[best], blocked))
            trace.add(activity, {Detail.DELTA: deltas[best],
                                 Detail.ARC: (best_match.group(2),
                                              best_match.group(3)),
                                 Detail.BLOCKED: blocked})
            blocked = None if blocked is None else []

    trace.add(Activity.STOP, {Detail.DELTA: best_score,
                              Detail.BLOCKED: blocked})

    trace.trace['time'] = [None] * len(trace.trace['time'])
    trace.trace['time'][-1] = elapsed
    trace.result = dag

    return trace


def _get_pc_trace(stdout, algorithm, context, params, N, pdag, elapsed):
    """
        Extract bnlearn HC trace from debug in stdout

        :param list stdout: output lines produced by R subprocess
        :param str algorithm: algorithm to use, e.g. 'hc', 'pc.stable'
        :param dict context: context information for experiment, e.g. id
        :param dict params: parameters for learning process e.g. score
        :param int N: number of rows in dataset
        :param PDAG pdag: learnt pdag to add into trace
        :param float elapsed: elapsed time for learning graph

        :returns Trace: trace file for learning process
    """
    # print('\n{}'.format('\n'.join(stdout)))

    test_pattern = compile(r'^    \> node (\S+) is (ind|d)ependent ' +
                           r'(?:on|from) (\S+)(.+)\(p-value\: (\S+)\)\.$')
    vstruct_pattern = compile(r'^    \@ detected v-structure (\S+) \-\> ' +
                              r'(\S+) \<\- (\S+) from d-separating set\.$')
    context = context.copy()
    context.update({'algorithm': algorithm.upper(),
                    'params': params if params is not None else {}, 'N': N,
                    'external': 'BNLEARN', 'dataset': True})
    # print(context)
    trace = Trace(context).add(Activity.INIT, {Detail.DELTA: -1.0})

    for line in stdout:
        break
        # print(line)
        test = test_pattern.match(line)
        if test is not None:
            print('*** {} is {} {}{}, p-value: {}'
                  .format(test.group(1),
                          ('dependent on' if test.group(2) == 'd'
                           else 'independent of'), test.group(3),
                          test.group(4), test.group(5)))
        vstruct = vstruct_pattern.match(line)
        if vstruct is not None:
            print('*** VSTRUCT: {} -> {} <- {}'
                  .format(vstruct.group(1), vstruct.group(2),
                          vstruct.group(3)))

    trace.add(Activity.STOP, {Detail.DELTA: -1.0})
    trace.trace['time'] = [None] * len(trace.trace['time'])
    trace.trace['time'][-1] = elapsed
    trace.result = pdag

    return trace


def _validate_learn_params(algorithm, params, dstype, knowledge):
    """
        Validate parameters supplied for learning algorithm

        :param str algorithm: algorithm to use, e.g. 'hc', 'pc.stable'
        :param dict params: parameters as specified in bnbench
        :param str dstype: dataset type: categorical/mixed/continuous
        :param Knowledge/None knowledge: knowledge constraints

        :raises TypeError: if params have invalid types
        :raises ValueError: if invalid parameters or data values

        :returns tuple: (parameters to record in trace,
                         parameters in format required by bnlearn)
    """

    # set defaults for CI test and score if needed and not set

    params = {} if params is None else params.copy()
    algo_class = BNLEARN_ALGORITHMS[algorithm]
    if 'score' not in params and algo_class != 'constraint':
        params.update({'score': 'bic'})
    if 'test' not in params and algo_class != 'score':
        params.update({'test': 'mi'})

    # check score and test have valid values (mi-g additionally supported)

    if (('score' in params and params['score'] not in SCORES) or
            ('test' in params and params['test'] not in TESTS + ['mi-g'])):
        raise ValueError('invalid test or score')

    # now validate and add in the score and/or test parameters

    if BNLEARN_ALGORITHMS[algorithm] != 'constraint':
        score_params = {p: v for p, v in params.items() if p in SCORE_PARAMS}
        params.update(check_score_params(score_params, [params['score']]))
    if BNLEARN_ALGORITHMS[algorithm] != 'score':
        test_params = {p: v for p, v in params.items() if p in TEST_PARAMS}
        params.update(check_test_params(test_params))
    if 'base' in params and params['base'] != 'e':
        raise ValueError('bnlearn_learn called with invalid base')

    # validate parameters against dstype - mixed not supported, and continuous
    # data must use 'mi-g' CI test and 'bic-g' scores.

    print(dstype, algo_class, params)

    if (dstype == 'mixed' or
        (dstype == 'categorical' and
         ((algo_class != 'constraint' and params['score'] == 'bic-g') or
          (algo_class != 'score') and params['test'] == 'mi-g')) or
        (dstype == 'continuous' and
         ((algo_class != 'constraint' and params['score'] != 'bic-g') or
          (algo_class != 'score') and params['test'] != 'mi-g'))):
        raise ValueError('bnlearn_learn has dstype/test/score mismatch')

    if (knowledge is not None and knowledge.rules.rules != [Rule.REQD_ARC]
            and knowledge.rules.rules != [Rule.STOP_ARC]):
        raise ValueError('bnlearn_learn() bad knowledge value')

    # Set up whitelist and blacklist

    bnlearn_params = params.copy()
    bnlearn_params.pop('base', None)
    bnlearn_params.update({'algorithm': algorithm})
    whitelist = {'from': [], 'to': []}
    blacklist = {'from': [], 'to': []}
    if knowledge is not None:
        whitelist['from'] = [r[0] for r in knowledge.reqd]
        whitelist['to'] = [r[1] for r in knowledge.reqd]
        blacklist['from'] = [r[0] for r in knowledge.stop]
        blacklist['to'] = [r[1] for r in knowledge.stop]
    bnlearn_params.update({'whitelist': whitelist})
    bnlearn_params.update({'blacklist': blacklist})

    return (params, bnlearn_params)


def bnlearn_learn(algorithm, data, context=None, params=None,
                  dstype='categorical', knowledge=None):
    """
        Return graph learnt from data using bnlearn algorithms

        :param str algorithm: algorithm to use, e.g. 'hc', 'pc.stable'
        :param DataFrame/str data: data or data filename to learn from
        :param dict context: context information about the test/experiment
        :param dict params: parameters for algorithm e.g. score to use
        :param str dstype: dataset type: categorical, continuous or mixed
        :param Knowledge/None knowledge: knowledge constraints

        :raises TypeError: if arg types incorrect
        :raises ValueError: if invalid params supplied
        :raises FileNotFoundError: if a specified data file does not exist
        :raises RuntimeError: if error in R-code

        :returns tuple: (DAG/PDAG learnt from data, learning trace)
    """
    if not isinstance(algorithm, str) or\
            (not isinstance(data, DataFrame) and not isinstance(data, str)) \
            or (context is not None and not isinstance(context, dict)) \
            or (params is not None and not isinstance(params, dict)) \
            or dstype not in {v for v in DatasetType} \
            or (knowledge is not None
                and not isinstance(knowledge, Knowledge)):
        raise TypeError('bnlearn_learn bad arg types')

    if algorithm not in BNLEARN_ALGORITHMS:
        raise ValueError('bnlearn_learn unsupported algorithm')

    if isinstance(data, str):
        tmpfile = None
        if not exists(data):
            raise FileNotFoundError('bnlearn_learn data file does not exist')
        if algorithm not in ['hc', 'tabu', 'h2pc', 'mmhc'] \
                and len(read_csv(data).columns) <= 2:
            raise ValueError('bnlearn_learn data < 3 columns')
    else:
        tmpfile = 'call/R/tmp/{}.csv'.format(int(random() * 10000000))
        Pandas(df=data).write(tmpfile)
        if algorithm not in ['hc', 'tabu', 'h2pc', 'mmhc'] \
                and len(data.columns) <= 2:
            raise ValueError('bnlearn_learn data < 3 columns')

    if (context is not None
            and (len(set(context.keys()) - set(CONTEXT_FIELDS))
                 or not {'in', 'id'}.issubset(context.keys()))):
        raise ValueError('bnlearn_learn() bad context values')

    # Validate learning parameters and return in format required by Trace and
    # by the bnlearn algorithms

    params, bnlearn_params = _validate_learn_params(algorithm, params, dstype,
                                                    knowledge)
    bnlearn_params.update({'datafile': data if tmpfile is None else tmpfile,
                           'dstype': dstype})

    # Call a R sub-process to perform the learning

    # print(bnlearn_params)
    graph, stdout = dispatch_r('bnlearn', 'learn', bnlearn_params)
    elapsed = graph['elapsed']

    if algorithm in ['hc', 'tabu', 'h2pc', 'mmhc']:
        graph = DAG(graph['nodes'], _arcs_to_edges(graph['arcs']))
        trace = _get_hc_trace(stdout, algorithm, context, params, len(data),
                              graph, elapsed) if context else None
    else:
        graph = PDAG(graph['nodes'], _arcs_to_edges(graph['arcs']))
        trace = _get_pc_trace(stdout, algorithm, context, params, len(data),
                              graph, elapsed) if context else None

    if tmpfile is not None:
        remove(tmpfile)

    return (graph, trace)
