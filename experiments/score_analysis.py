
# Analyses node scores in experiemntal results

from pandas import DataFrame, Series, concat
from itertools import combinations

from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from experiments.common import NETWORKS_GRID_DESIGN, reference_bn, series_props
from experiments.plot import relplot
from learn.trace import Trace
from analysis.trace import TraceAnalysis
from core.score import node_score
from core.graph import DAG
from core.bn import BN
from core.cpt import CPT
from core.metrics import kl
from core.indep import indep


def _get_scores_for_figure(nodes, Ns, ref_parents, params, data):
    """
        Get score data which will be plotted on the figure.

        :param dict nodes: {node: candidate parent sets}
        :param list Ns: of sample sizes to process
        :param dict ref_parents: {node: parents} in reference network
        :param dict params: learning parameters {name: value}
        :param Data data: data from which scores generated

        :returns tuple: (DataFrame of scores, total score range,
                         score range for largest sample size)
    """
    parents = ref_parents.copy()  # working copy of parents of nodes
    score_types = (params['score'] if isinstance(params['score'], list) else
                   [params['score']])

    scores = {'subplot': [], 'x_val': [], 'y_var': [], 'y_val': []}
    total_range = None
    for N in Ns:
        data.set_N(N)
        last_range = None  # score range at each sample size
        for node in nodes:

            # loop over differnt parent sets to get their score

            for node_parents in nodes[node]:
                if len(node_parents):
                    parents.update({node: list(node_parents)})
                else:
                    parents.pop(node, None)

                # obtain node score with specified parents

                cps_scores = node_score(node, parents, types=score_types,
                                        params=params, data=data)

                # accummuate variable values in long form array

                for type, score in cps_scores.items():
                    norm_score = score / N
                    scores['subplot'].append(node)
                    scores['x_val'].append(N)
                    y_var = (', '.join(node_parents) if len(node_parents)
                             else 'None') + ('-' + type if len(score_types) > 1
                                             else '')
                    scores['y_var'].append(y_var)
                    scores['y_val'].append(norm_score)

                # record min and max parent set score at this iteration

                if last_range is None:
                    last_range = (norm_score, norm_score)
                elif norm_score < last_range[0]:
                    last_range = (norm_score, last_range[1])
                elif norm_score > last_range[1]:
                    last_range = (last_range[0], norm_score)

            # reset parents of node back to those of reference network

            if node not in ref_parents:
                parents.pop(node, None)
            else:
                parents[node] = ref_parents

        if total_range is None:
            total_range = last_range
        else:
            if last_range[0] < total_range[0]:
                total_range = (last_range[0], total_range[1])
            if last_range[1] > total_range[1]:
                total_range = (total_range[0], last_range[1])

    print('\nTotal range is {}, and last range {}\n'
          .format(total_range, last_range))

    return (DataFrame(scores), total_range, last_range)


def _get_parents_to_score(series, network, ref, nodes, Ns):
    """
        Get sets of parent sets whose scores will be plotted. These will
        be any parent set that has been encountered during any of the
        learning processes at each sample size, as well as the true
        parents.

        :param str series: series being considered
        :param str network: network being considered
        :param BN ref: reference BN for the network
        :param list nodes: list of child nodes of interest
        :param list Ns: of sample sizes to process

        :returns dict: {node: [parent sets to score]}
    """
    def _change_parent(child, parent, add, parents, nodes):
        """
            Records when an iteration creates a new parent set for one of
            the nodes of interest.

            :param str child: node of interest
            :param str parent: node being added/removed as parent of child
            :param bool add: whether parent is being added or removed
            :param dict parents: current parents of each node during learning
                                 process
            :param dict nodes: nodes of interest and parent sets used during
                               the learning process {node: [parent sets]}
        """
        if child in parents:
            # need to work with copy of current parents ... directly
            # modifying parents[child] here inexplicably modifies nodes as well

            c_parents = parents[child].copy()
            if add is True:
                c_parents.add(parent)
            else:
                c_parents.remove(parent)

            if c_parents not in nodes[child]:  # a new parent set discovered
                nodes[child].append(c_parents)

            parents[child] = c_parents

    # Create dict which will hold parent sets used during the learning
    # process for each node of interest

    nodes = {n: [set()] for n in ref.dag.nodes if n in nodes}

    # Obtain learning traces for network at each sample size

    traces = Trace.read(series + '/' + network)
    if traces is None:
        return None

    # Loop by increasing sample size

    sorted_N = sorted([int(k.replace('N', '')) for k in traces])
    for N in sorted_N:
        if N not in Ns:
            continue
        parents = {n: set() for n in nodes}.copy()  # current node parents
        trace = traces['N{}'.format(N)]
        trace = trace.trace

        # Loop through iterations in each trace to see if that change creates
        # a novel parent set for any node of interest.

        for iter, activity in enumerate(trace['activity']):
            arc = trace['arc'][iter]
            if activity == 'add':
                _change_parent(arc[1], arc[0], True, parents, nodes)
            elif activity == 'delete':
                _change_parent(arc[1], arc[0], False, parents, nodes)
            elif activity == 'reverse':
                _change_parent(arc[1], arc[0], False, parents, nodes)
                _change_parent(arc[0], arc[1], True, parents, nodes)

    # Ensure the true parent set is also included if it wasn't encountered
    # during the learning process

    for node in nodes:
        ref_parents = (set(ref.dag.parents[node]) if node in ref.dag.parents
                       else set())
        if ref_parents not in nodes[node]:
            print('Adding in true parents {} of node {}'
                  .format(ref_parents, node))
            nodes[node].append(ref_parents)

    return nodes


def score_analysis(series, networks, nodes, Ns):
    """
        Perform an analysis of critical nodes scores for specified series
        and networks.

        :param list series: series to include
        :param list networks: networks to include
        :param list nodes: nodes (patterns) to include
        :param list Ns: of sample sizes to process
    """
    def _parent_combos(parents):  # all possible combinations of parents
        ps = []
        for n in range(len(parents) + 1):
            ps += combinations(parents, n)
        return ps

    print('\nAnalysing scores in networks {} in series {} ...'
          .format(networks, series))

    for _series in series:
        props = series_props(_series)
        for network in networks:
            ref, _ = reference_bn(network)

            # identify nodes to focus on and their candidate parent sets

            nodes = _get_parents_to_score(series=_series, network=network,
                                          ref=ref, nodes=nodes, Ns=Ns)
            print('\nScoring nodes and parent sets: {}'.format(nodes))

            data = Pandas.read(EXPTS_DIR + '/datasets/' + network + '.data.gz',
                               dstype='categorical', N=Ns[-1])

            # generate table of scores to be plotted

            scores, total_range, last_range = \
                _get_scores_for_figure(nodes=nodes, Ns=Ns,
                                       ref_parents=ref.dag.parents,
                                       params=props['params'], data=data)

            # ymax = total_range[1] + 0.1 * (total_range[1] - total_range[0])
            # ymin = total_range[0] - 0.1 * (total_range[1] - total_range[0])

            props = NETWORKS_GRID_DESIGN
            props.update({'figure.title': 'Node scores in ' + network,
                          # 'legend.labels': metrics,
                          'legend.title': 'Parent nodes',
                          'yaxis.range': (-1.0, 0),
                          'figure.per_row': 1,
                          'figure.subplots_left': 0.15,
                          'figure.subplots_top': 0.83,
                          'figure.subplots_right': 0.65,
                          'xaxis.label': 'Sample size',
                          'yaxis.label': 'BIC score of node'})

            plot_file = ('{}/analysis/scores/{}_{}.png'
                         .format(EXPTS_DIR, _series.replace('/', '_'),
                                 network))

            relplot(scores, props, plot_file)
            print('Plot file {} written'.format(plot_file))


def score2_analysis(networks, Ns):
    """
        Obtain the lowest sample size for each network at which the reference
        score becomes higher than the learnt graph score. This is evidence
        that score is not a reliable indicator of graph accuracy.

        :param list networks: networks to include
        :param list Ns: list of sample sizes to process
    """
    print('\n\nCompare reference and learnt scores ...\n')
    for network in networks:
        ref, _ = reference_bn(network)

        # Get traces for optimum order series, BIC traces has maxiter of 1
        # and was just used to generate reference scores, and variable
        # order traces

        opt_traces = Trace.read('HC/OPT/' + network, EXPTS_DIR)
        bic_traces = Trace.read('HC/SCORE/BIC/' + network, EXPTS_DIR)
        ord_traces = Trace.read('HC/ORDER/BASE/' + network, EXPTS_DIR)
        min_sample = None
        for N in Ns:
            key = 'N{}'.format(N)

            # try and get trace which contains reference graph score

            ref_trace = (bic_traces[key] if key in bic_traces
                         else (ord_traces[key + '_0']
                         if key + '_0' in ord_traces else None))
            if ref_trace is None:
                continue
            ref_score = ref_trace.context['score']

            # Get learnt graph score with optimal ordering

            opt_trace = TraceAnalysis(opt_traces[key], ref.dag)
            opt_score = opt_trace.summary['score']

            # Note the lowest sample size at which the reference score
            # is higher than the learnt score

            # print(network, N, ref_score, opt_score,
            #       ('REF' if ref_score >= opt_score else 'OPT'))
            if min_sample is None and ref_score >= opt_score:
                min_sample = N

        print('Reference score bigger than learnt score for {} at {}'
              .format(network, min_sample))


def score3_analysis():
    """
        Plot BIC and loglik for candidate parent sets over different
        parameter values.
    """
    print('\n\nAnalyse impact of parameter values on scores ...\n')

    subplot_titles = {}
    scores = None
    extrap = False

    for n in [1, 2, 4, 8, 16, 32, 64, 128, 0]:

        child = 'C{}'.format(n)
        p1 = n if n != 0 else 1
        p2 = n + 1 if n != 0 else 1
        pmf1 = {'0': p1 / (p1 + p2), '1': p2 / (p1 + p2)}
        pmf2 = {'0': p2 / (p1 + p2), '1': p1 / (p1 + p2)}
        dag = DAG(['P1', child], [('P1', '->', child)])
        bn = BN(dag, {'P1': (CPT, {'0': 0.5, '1': 0.5}),
                      child: (CPT, [({'P1': '0'}, pmf1),
                                    ({'P1': '1'}, pmf2)])})
        for node, cpt in bn.cnds.items():
            print('{}: {}'.format(node, cpt))

        data = Pandas(df=bn.generate_cases(1000000))

        _kl = kl(Series(bn.cnds[child].cdist({'P1': '1'})),
                 Series(bn.cnds[child].cdist({'P1': '0'})))
        pval_bn = (indep(child, 'P1', z=None, data=None,
                         bn=bn, N=10000))['mi']['p_value']
        pval_data = (indep(child, 'P1', z=None,
                           data=data.sample[:10000]))['mi']['p_value']
        print(pval_bn, pval_data)

        subplot_title = (('Param ratio {}:{}, KL={:.4f}\n10K p-value(bn)' +
                          '={:.4f}, (data)={:.4f}')
                         .format(p1, p2, _kl, pval_bn, pval_data))
        subplot_titles.update({child: subplot_title})

        types = ['loglik'] if extrap else ['loglik', 'bic']
        range = (800, 2000) if extrap else (10, 1000000)
        _scores, _, _ = _get_scores_for_figure({child: [[], ['P1']]}, range,
                                               {child: ['P1'], 'P1': []},
                                               {'score': types, 'k': 1}, data)
        scores = (_scores if scores is None
                  else concat([scores, _scores], ignore_index=True))

    props = NETWORKS_GRID_DESIGN
    props.update({'figure.title': ('BIC & loglik scores for zero and one' +
                                   ' parent for different param ratios'),
                  'legend.title': 'Parents & score',
                  'subplot.title': subplot_titles,
                  'subplot.title_fontsize': 12,
                  'subplot.kind': 'regression' if extrap else 'line',
                  'yaxis.range': (-0.3025, -0.2990),  # (-0.3012, -0.3009),
                  'figure.per_row': 2 if extrap else 3,
                  'figure.subplots_left': 0.10,
                  'figure.subplots_top': 0.89,
                  'figure.subplots_right': 0.87,
                  'xaxis.label': 'Sample size',
                  'yaxis.label': 'Score of parent sets',
                  # 'yaxis.scale': 'log' if extrap else 'linear',
                  'xaxis.scale': 'linear' if extrap else 'log',
                  'figure.dpi': 200})

    plot_file = ('{}/analysis/scores/params.png'.format(EXPTS_DIR))

    relplot(scores, props, plot_file)
    print('Plot file {} written'.format(plot_file))
