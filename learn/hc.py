

#   Hill-climbing family of score-based structure learning algorithms

from pandas import DataFrame
from statistics import mean
from enum import Enum
from copy import deepcopy
from time import time

from learn.common import TreeStats
from learn.hc_worker import HCWorker, Prefer
from fileio.data import Data
from fileio.pandas import Pandas
from fileio.numpy import NumPy
from core.graph import DAG, SDG
from core.score import check_score_params, SCORE_PARAMS
from core.metrics import values_same
from learn.tabulist import TabuList
from learn.knowledge import Knowledge
from learn.knowledge_rule import RuleSet, Rule

HC_PARAMS = {'score',  # score to use e.g. 'bic', 'bde'
             'maxiter',  # limit on number of iterations
             'tabu',  # length of tabulist, if not specified HC algorithm runs
             'noinc',  # num of iterations where delta <= 0 allowed in Tabu
             'prefer',  # whether adding un/connected arcs preferred
             'tree',  # enhanced tree type search
             'output',  # type of output required, eg 'dag'
             'stable',  # stabilisation approach e.g. 'entropy+'
             'bnlearn'  # processing identical to bnlearn
             } | set(SCORE_PARAMS)


class Stability(Enum):  # types of stability enforcement supported
    INC_SCORE = 'inc_score'  # nodes in increasing score order
    DEC_SCORE = 'dec_score'  # nodes in decreasing score order
    DEC_1 = 'dec_1'  # nodes in decreasing score order - 1 key only
    DEC_2 = 'dec_2'  # nodes in decreasing score order - 2 keys only   
    SCORE = 'score'  # try both inc/dec_score use best one
    SCORE_PLUS = 'score+'  # try both inc/dec_score, use result order
    SC4_PLUS = 'sc4+'  # 4 orders tried


def _validate_tabu_params(params):
    """
        Validate any Tabu list related parameters.

        :param dict params: parameters for HC algorithm

        :raises TypeError: if arguments have bad types
        :raises ValueError: if arguments have bad values

        returns dict: (possibly modified) learning parameters
    """
    if 'tabu' in params and params['tabu'] is not None:
        TabuList(params['tabu'])
        if 'noinc' in params and params['noinc'] is not None:
            if (not isinstance(params['noinc'], int)
                    or isinstance(params['noinc'], bool)):
                raise TypeError('hc() bad noinc type')
            if params['noinc'] < 1 or params['noinc'] > 100:
                raise ValueError('hc() bad noinc value')
    elif 'noinc' in params and params['noinc'] is not None:
        raise ValueError('hc() noinc specified without tabu')

    if 'bnlearn' in params:
        if not isinstance(params['bnlearn'], bool):
            raise TypeError('hc() bad bnlearn type')
    else:
        params['bnlearn'] = True

    return params


def _validate_tree_params(params, knowledge, context):
    """
        Validate any tree search or stability related parameters.

        :param dict/None params: parameters for HC algorithm
        :param Knowledge/None knowledge: knowledge to help algorithm
        :param dict context: context information for trace, or None if
                             tracing not required

        :raises TypeError: if arguments have bad types
        :raises ValueError: if arguments have bad values

        :returns dict: (possibly modified) learning parameters
    """
    if params is None:
        return

    if 'stable' in params:
        stable = params['stable']
        if not isinstance(stable, (bool, str)):
            raise TypeError('hc(): bad stable parameter type')
        allowed = {s.value for s in Stability}
        if isinstance(stable, str) and stable not in allowed:
            raise ValueError('hc(): bad stable parameter value')
        if knowledge is not False:
            raise ValueError('hc(): stable and knowledge incompatible')
        if stable is False:
            params.pop('stable')
        elif stable is True:
            params['stable'] = Stability.DEC_SCORE  # for compatability
        else:
            params['stable'] = Stability(params['stable'])

    if 'tree' in params:
        tree = params['tree']
        if (not isinstance(tree, tuple) or len(tree) != 3
            or any([not isinstance(i, int) or isinstance(i, bool)
                    for i in tree])):
            raise TypeError('hc(): bad tree parameter type')

        if (tree[0] < 1 or tree[0] > 10 or tree[1] < -10 or tree[1] > 100
                or -1 * tree[1] > tree[0] or tree[2] < -1):
            raise ValueError('hc(): bad tree values')

        if knowledge is not False:
            raise ValueError('hc() tree and Knowledge specified')

        if context is None:
            raise ValueError('hc() tree but no context specified')

    return params


def _validate_hc_arguments(data, params, knowledge, context):
    """
        Validate input arguments for hc

        :param Data data: data to learn graph from
        :param dict params: parameters for algorithm e.g. score to use
        :param Knowledge knowledge: knowledge to help algorithm
        :param dict context: context information for trace, or None if
                             tracing not required

        :raises TypeError: if arguments have bad types
        :raises ValueError: if arguments have bad values

        :returns dict: checked and updated HC parameters
    """

    if not isinstance(data, Data) \
            or (params is not None and not isinstance(params, dict)) \
            or (knowledge is not False
                and not isinstance(knowledge, Knowledge)) \
            or (context is not None and not isinstance(context, dict)):
        raise TypeError('hc bad arg type')

    # Set some defaults if params not specified

    params = {} if params is None else params
    if 'score' not in params:
        params['score'] = 'bic' if data.dstype == 'categorical' else 'bic-g'
    if 'prefer' not in params:
        params['prefer'] = Prefer.NONE

    # Check only supported parameters are present

    if not set(params).issubset(HC_PARAMS):
        raise ValueError('hc() bad param name')
    if (('score' in params and params['score']
         not in {'bde', 'bds', 'bic', 'loglik', 'bic-g', 'bge'})
        or ('maxiter' in params and
            (not isinstance(params['maxiter'], int) or params['maxiter'] < 1))
        or ('prefer' in params and (not isinstance(params['prefer'], Prefer)))
        or (data.dstype == 'categorical' and
            params['score'] in {'bic-g', 'bge'})
            or (data.dstype == 'continuous' and
                params['score'] not in {'bic-g', 'bge'})):
        raise ValueError('hc() bad param value')

    # Check any tabulist and score related parameters

    params = _validate_tabu_params(params)
    score_params = {p: v for p, v in params.items() if p in SCORE_PARAMS}
    check_score_params(score_params)

    params = _validate_tree_params(params, knowledge, context)

    # If an initial DAG specified in Knowledge, check it doesn't contain any
    # nodes not present in data

    if (knowledge is not False and knowledge.initial is not None and
            len(set(knowledge.initial.nodes) - set(data.get_order()))):
        raise ValueError('hc() initial DAG contains unknown nodes')

    # Check context keys all valid

    if (context is not None and
            len(set(context.keys()) -
                {'in', 'id', 'score', 'randomise', 'var_order', 'pretime'})):
        raise ValueError('hc() bad context keys')

    # Set up default parameter values if not specified, and check score params

    score_params = {p: v for p, v in params.items() if p in SCORE_PARAMS}
    params.update(check_score_params(score_params, [params['score']]))

    return params


def reorder_list(strings, num_parts):
    """
        Divide a list into `num_parts` equal parts and perform two reorderings:
        1. Reverse the strings in each part and concatenate the parts in their
           original order.
        2. Keep the strings in their original order within each part but
           reverse the order of the parts.

        :param list strings: The list to reorder.
        :param int num_parts: The number of parts to divide the list into.
        :return: A tuple containing two reordered lists.
    """
    if num_parts <= 0 or num_parts > len(strings):
        raise ValueError("num_parts must be between 1 and length of the list.")

    # Calculate the size of each part
    part_size = len(strings) // num_parts
    remainder = len(strings) % num_parts

    # Divide the list into parts
    parts = []
    start = 0
    for i in range(num_parts):
        # Add extra element to some parts if list length not divisible evenly
        end = start + part_size + (1 if i < remainder else 0)
        parts.append(strings[start:end])
        start = end

    # 1st reordering: Reverse strings in each part, parts in original order
    reversed_within_parts = [item for part in parts for item in reversed(part)]

    # 2nd reordering: just reverse the order of the parts
    reversed_parts_order = [item for part in reversed(parts) for item in part]

    return [tuple(reversed_within_parts), tuple(reversed_parts_order)]


def set_stable_order(data, params):
    """
        Sets a data dependent node order for stable searches depending upon
        the values of params['stable'] of type Stability.

        :param Data data: data used to learn structure
        :param dict params: learning parameters

        :returns tuple: (Data: data with determined stable order set,
                         float: elapsed time to determine stable order)
    """
    def _node_order(parents, current):  # return topological-based order
        return tuple([n for g in SDG.partial_order(parents)
                      for n in current if n in g])

    # Obtain node order with decreasing score - DEC_1 and DEC_2 use fewer
    # elements in the sort key to test the effect of the key elements
    start = time()
    _params = deepcopy(params)
    num_keys = (1 if _params['stable'] == Stability.DEC_1 else
                (2 if _params['stable'] == Stability.DEC_2 else 3))
    _order = _score_order(data, _params, num_keys)
    print('Score order completed after {:.3f}s'.format(time() - start))

    # if using a decreasing order no need to do anything
    if _params['stable'] not in {Stability.DEC_SCORE, Stability.DEC_1,
                                 Stability.DEC_2}:
        _rev_order = tuple([n for n in _order][::-1])

        # just use reverse order to get increasing score order
        if _params['stable'] == Stability.INC_SCORE:
            print('reversing {}'.format(_order))
            _order = _rev_order

        # Going to run HC or Tabu with the different orders and use the order
        # that results in the graph with the best score
        else:
            orders = [_order, _rev_order]
            if _params['stable'] == Stability.SC4_PLUS:
                orders += reorder_list(orders[0], 2)
                orders += reorder_list(orders[1], 2)

            # run the algorithm with the different (stable) orders
            best = None
            for order in orders:
                data.set_order(order)
                hcw = HCWorker(data, _params, False, None, False).run()
                if best is None or (hcw.score > best[1] and not
                                    values_same(hcw.score, best[1], sf=10)):
                    best = (order, hcw.score, hcw.parents)
                print('Order {}, {} ... {}, {} gives score {:.3f} after {:.3f}'
                      .format(order[0], order[1], order[-2], order[-1],
                              hcw.score, time() - start))

            # choose order which gave highest score (decreasing if same)
            _order = best[0]

            if _params['stable'] != Stability.SCORE:
                _order = _node_order(best[2], _order)

    data.set_order(_order)
    if _params['score'] == 'loglik':
        HCWorker.init_score_cache()
    elapsed = time() - start
    print('Stable {} order: {} in {:.3f}s'
          .format(params['stable'].value, _order, elapsed))
    return (data, elapsed)


def _score_order(data, params, num_keys=3):
    """
        Returns stable node order based on decreasing score or values using:
            (1) unconditional score
            (2) mean conditional score averaged across all other nodes
            (3) string of ordered values and their counts

        These all depend solely upon the data and not column or name
        ordering. If the two variables are the same over all three measures
        then it is highly likely they are identical.

        :param Data data: data used to learn structure
        :param dict params: learning parameters

        :returns tuple: of nodes in decreasing score order
    """
    order = None  # entries are tuples: (score, cscore, [nodes])
    N = data.N
    nvs = data.node_values if isinstance(data, (NumPy, Pandas)) else None

    for node in data.get_order():

        # Compute nodes' parentless score and mean of all its single-parent
        # scores

        score = HCWorker.nscore(node, set(), data, params)[0] / N
        c_score = mean([HCWorker.nscore(node, set([p]), data, params)[0] / N
                        for p in data.get_order() if node != p])

        if order is None:
            order = [(score, c_score, [node])]
        else:
            for i, entry in enumerate(order):
                same_score = values_same(score, entry[0], sf=10)
                same_c_score = (num_keys < 2
                                or values_same(c_score, entry[1], sf=10))

                # If scores are both the same then make a
                # comparsion on the variables' sequences of values

                vals_same = vals_after = None
                if (same_score and same_c_score
                        and isinstance(data, (Pandas, NumPy))):
                    print('*** Scores same for {} and {}'
                          .format(node, entry[2][0]))
                    vals_same = (num_keys < 3 or
                                 (data.as_df()[node].to_numpy() ==
                                  data.as_df()[entry[2][0]].to_numpy()).all())
                    if not vals_same:
                        v_n = '{}'.format({v: nvs[node][v]
                                           for v in sorted(nvs[node])})
                        v_e = '{}'.format({v: nvs[entry[2][0]][v]
                                           for v in sorted(nvs[entry[2][0]])})
                        print('*** Values counts {} and {}'.format(v_n, v_e))
                        vals_after = v_n > v_e

                # Insert node in correct place in order

                if same_score and same_c_score and vals_same:
                    print('*** Variable {} and {} same position'
                          .format(node, order[i][2][0]))
                    order[i][2].append(node)
                    break

                elif ((not same_score and score > entry[0]) or
                      (same_score and c_score > entry[1]) or
                      (same_score and same_c_score and vals_after)):
                    order.insert(i, (score, c_score, [node]))
                    break

                elif i == len(order) - 1:
                    order.append((score, c_score, [node]))
                    break

    order = tuple([n for e in order for n in e[2]])
    print('\n\nDecreasing score order is {}'.format(order))

    return order


def hc_prune(hcws, params, stats):
    """
        Prune a set of HCWorkers by score returning the highest "limit" scoring
        ones of them. HCWorkers with same score are retained in the same order
        that they are listed in the input argument.

        :param list hcws: list of HCWorker's to prune
        :param dict params: search parameters including tree
        :param TreeStats stats: statistics about the tree search

        :returns list: of pruned HCWs in descending score order
    """
    _, width, lookahead = params['tree']

    # First prune any sequences which produced the same DAG

    ps = [hcw.parents for hcw in hcws]
    keep = []
    for i, hcw in enumerate(hcws):
        if hcw.parents in ps[:i]:
            stats.update_state(hcw, 'XD')
        else:
            keep.append(hcw)
    if len(keep) < len(hcws):
        print('+++ {} duplicate DAGs pruned'.format(len(hcws) - len(keep)))
    hcws = keep

    # if lookahead is non-zero we will lookahead what the score WILL be if
    # we do a further lookahead number of steps where we choose 'F' as the
    # equivalent add decision - this means dataset variable ordering is used

    if lookahead > 0:
        for hcw in hcws:
            if not hcw.paused:
                continue
            _seq = tuple(list(hcw.knowledge.sequence) + ([False] * lookahead))
            _hcw = hcw.clone(_seq, True)
            _hcw.run()
            hcw.score2 = _hcw.score
            print('+++ lookahead {} score for {} is {:.4f} cf {:.4f}'
                  .format(lookahead, TreeStats.key(hcw),
                          hcw.score2 / hcw.data.N, hcw.score / hcw.data.N))

    # Sort HCWorks in descending score2 (which is score with any lookahead)

    hcws = sorted(hcws, key=lambda hcw: hcw.score2, reverse=True)

    # if width is positive prune off lowest scoring HCWorkers

    if width > 0 and len(hcws) > width:
        print('+++ {} lowest scoring pruned'.format(len(hcws) - width))
        for hcw in hcws[width:]:
            stats.update_state(hcw, 'XP')
        hcws = hcws[:width]
    return hcws


def hc_subtree(root, data, params, context, stats, init_cache):
    """
        Explores a subtree within the HC tree search.

        :param HCWorker/None root: a HCWorker in a paused state which is the
                                   root of this subtree search.
        :param Data data: data to learn graph from
        :param dict params: parameters for algorithm e.g. score to use
        :param dict context: context information for trace, or None if
                             tracing not required
        :param bool init_cache: whether score cache should be initialised
    """
    depth, width, lookahead = params['tree']

    if root is None:

        # root is None so right at start of tree search. Need to create an
        # initial HCWorker from which others will be cloned

        _know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                          params={'sequence': (False, )})
        root = HCWorker(data, params, _know, context, init_cache)
        root_seq = []
    else:
        root_seq = list(root.knowledge.sequence)

    iseq = 0
    hcws = []
    while iseq < 2**depth:

        # convert bits of iseq into tuple of booleans, and append this to
        # sequence of booleans to get complete list of decision booleans
        # for this subtree

        sequence = tuple(root_seq + [True if (iseq >> s) & 1 == 1 else False
                                     for s in range(depth)][::-1])

        # Clone an HCWorker with this extended sequence and run it

        hcw = root.clone(sequence, False if lookahead == -1 else True)
        hcw.run()
        stats.add(hcw, depth)

        # If learning has completed for this HCWorker, then see how much of the
        # decision sequence was used - this is then used to increment iseq by
        # more than 1 so that redundant sub-tree brances are not explored.
        # For example, if the decision sequence is TFTF but it completed after
        # TFT, there is no point exploring TFTT because that too will terminate
        # at TFT with the same result.

        depth_used = depth
        if hcw.paused is False:
            decisions = [k[1] for k in hcw.trace.trace['knowledge']
                         if k is not None and k[0] == Rule.EQUIV_SEQ.value]
            decisions = sum(d is True for d in decisions)
            depth_used += decisions - len(sequence)
        iseq += 2**(depth - depth_used)

        # seq_str = ''.join(['T' if b is True else 'F' for b in sequence])
        # print('Sequence {} {}ed, score {:.3f}, depth used {}'
        #       .format(seq_str, 'paus' if hcw.paused else 'stopp',
        #               hcw.score / data.N, depth_used))

        hcws.append(hcw)

    # Delete statistics information associated with the root because this has
    # now been subsumed into the statistics for the leaves of the subtree.

    if len(root_seq):
        stats.delete(root)

    # Negative width means subtree leaf pruning. Do the pruning and update
    # state of pruned sequences in the statistics.

    if width < 0:
        limit = 2 ** (depth + width)
        hcws = sorted(hcws, key=lambda hcw: hcw.score2, reverse=True)
        for hcw in hcws[limit:]:
            stats.update_state(hcw, 'XS')
        hcws = hcws[:limit]

    return hcws


def hc_tree(data, params, context, init_cache):
    """
        Explores a tree of sequences of equivalent add decisions.

        :param Data data: data to learn graph from
        :param dict params: parameters for algorithm e.g. score to use
        :param dict context: context information for trace, or None if
                             tracing not required
        :param bool init_cache: whether score cache should be initialised

        :returns tuple: dags (single or list depending upon output param),
                        Trace for highest scoring DAG found
    """
    print('\n\nhc_tree search ...')

    hcws = [None]  # list of running HCWorkers
    finished = False  # whether tree search has completed
    stats = TreeStats()  # statistics from tree search

    while finished is False:

        # Loop through all currently active HCWorkers using each one as a root
        # of a subtree, aggregating a list of all subtree leave nodes.

        new_hcws = []
        for root in hcws:
            if root is None or root.paused is True:
                new_hcws += hc_subtree(root, data, params, context, stats,
                                       init_cache)
            else:
                new_hcws += [root]
        hcws = new_hcws

        # Prune leaves of all subtrees

        hcws = hc_prune(hcws, params, stats)

        stats.update_ranks(hcws)

        print('\nTree stats:\n{}'.format(stats.to_string()))

        finished = all([hcw.paused is False or
                        len(hcw.knowledge.sequence) >= 60 for hcw in hcws])

    hcws[0].trace.set_treestats(stats)
    return hcws[0]


def hc(data, params=None, knowledge=False, context=None,
       init_cache=True):
    """
        Performs Hill-Climbing Structure Learning.

        :param Data/DataFrame data: data to learn graph from
        :param dict params: parameters for algorithm e.g. score to use
        :param Knowledge/bool knowledge: knowledge to aid structure learning
        :param dict context: context information for trace, or None if
                             tracing not required
        :param bool init_cache: whether score cache should be initialised

        :raises TypeError: if arguments have bad types
        :raises ValueError: if arguments have bad values

        :returns tuple: (DAG learnt, learning trace)
    """

    # support data being specified as a Pandas DataFrame

    if isinstance(data, DataFrame):
        data = Pandas(df=data)

    # validate algorithm parameters

    params = deepcopy(params)
    params = _validate_hc_arguments(data, params, knowledge, context)

    # Initialise score cache if required

    if init_cache is True:
        HCWorker.init_score_cache()

    # Set stable node order if required
    if 'stable' in params and isinstance(data, (Pandas, NumPy)):
        data, pretime = set_stable_order(data, params)
        if context is not None:
            context.update({'pretime': pretime})

    # Perform multiple hc runs if tree parameter specified
    if 'tree' in params:
        hcw = hc_tree(data, params, context, init_cache=False)
    else:
        hcw = HCWorker(data, params, knowledge, context, init_cache=False)
        hcw.run()

    # Construct DAG object from parents information and return with trace

    arcs = [(c, '->', p) for p, children
            in hcw.parents.items() for c in children]
    dag = DAG(list(hcw.parents.keys()), arcs)
    return ((dag, None) if hcw.trace is None
            else (dag, hcw.trace.set_result(dag)))
