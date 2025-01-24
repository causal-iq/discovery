
# Analyses learning traces in experimental results

from pandas import DataFrame, set_option
from itertools import zip_longest, chain

from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from experiments.common import reference_bn, NETWORKS_GRID_DESIGN, \
    SERIES_GROUPS, FIGURE_PARAMS
from learn.trace import Trace
from analysis.trace import TraceAnalysis
from analysis.graphviz import traceviz
from experiments.plot import relplot


class ArbitraryAnalysis():
    """
        Analyses traces to identify cummulative proportion of changes
        which were arbitrary.

        :ivar data dict: arbitrary change data keyed by for each series and
                         network {(series, network):
                                  {'total': num changes at each iter,
                                   'count': arbitrary changes at each iter}}
    """
    def __init__(self):
        self.data = {}

    def collect(self, series, network, analysis):
        """
            Collect arbitrary change data for a specific experiment.

            :param str series: series which experiment belongs to
            :param str network: network which experiment done on
            :param TraceAnalysis analysis: analysis of learning trace
        """
        if (series, network) not in self.data:
            self.data.update({(series, network): {'count': [], 'total': []}})

        arbitrary = [1 if m == 0 else 0
                     for m in analysis.trace['margin'][1:-1]]
        self.data[(series, network)]['count'] = \
            [sum(x) for x in
             zip_longest(self.data[(series, network)]['count'],
                         [1] * len(arbitrary), fillvalue=0)]
        self.data[(series, network)]['total'] = \
            [sum(x) for x in
             zip_longest(self.data[(series, network)]['total'],
                         arbitrary, fillvalue=0)]

    def plot(self, max_iteration, fig_p, file):
        """
            Plot the arbitrary change data.

            :param int max_iteration: only plot changes upto this iteration
            :param dict fig_p: figure parameters
            :param str file: output file name
        """
        data = {'subplot': [], 'x_val': [], 'y_var': [], 'y_val': []}
        for key, values in self.data.items():
            cummulative = 0.0
            for i in range(len(values['count'])):
                if max_iteration is not None and i > max_iteration - 1:
                    continue
                cummulative += values['total'][i] / values['count'][i]
                data['subplot'].append(key[0])  # series
                data['y_var'].append(key[1])  # network
                data['x_val'].append(i + 1)  # iteration
                data['y_val'].append(cummulative / (i + 1))  # at this iter

        data = DataFrame(data)
        props = NETWORKS_GRID_DESIGN
        props.update({'figure.title': ('Arbitrary changes in hill-climbing' +
                                       '\n- selected networks, 10K data rows'),
                      'figure.title_fontsize': 20,
                      'figure.per_row': 1,
                      'figure.dpi': 200,
                      'figure.subplots_left': 0.18,
                      'figure.subplots_right': 0.78,
                      'figure.subplots_top': 0.80,
                      'subplot.aspect': 1.0,
                      'subplot.title': '',
                      'line.sizes': [2] * len(data['y_var'].unique()),
                      'xaxis.scale': 'linear',
                      'xaxis.label': 'Hill-climbing iteration',
                      'yaxis.label': 'Proportion of arbitrary changes',
                      'legend.title': 'Network',
                      'legend.title_fontsize': 12})

        if fig_p is not None:
            props.update(FIGURE_PARAMS[fig_p])

        plot_file = EXPTS_DIR + (file if file is not None else
                                 '/analysis/traces/arbitrary.png')

        print('... generating plot file "{}"'.format(plot_file))
        relplot(data, props, plot_file)


def _edges(edges, updates=None, prev_N=None, N=None):
    """
        Update or print analysis of the matched, missing etc edges

        :param dict edges: analysis of matched, missing etc edges
        :param dict/None updates: updates to edge analysis, or None
                                  to return existing analysis in readable
                                  format.
        :param int/None prev_N: sample size previous update applied to
        :param int N: sample size that this update relates to

        :returns dict/str: updated edge analysis or printable analysis text.
    """
    if updates is None:  # return edge analysis in readable form
        text = '\nAnalysis of edges:\n'
        for metric in edges:
            text += 'Metric {}:\n'.format(metric)
            for edge in edges[metric]:
                text += '  {}: {}\n'.format(edge, edges[metric][edge])
        return text

    for metric, edge_list in updates.items():
        if edge_list == set():
            continue
        if metric not in edges:
            edges.update({metric: {}})
        for edge in edge_list:
            if edge not in edges[metric]:
                edges[metric][edge] = [(N, N)]
            elif prev_N == edges[metric][edge][-1][1]:
                edges[metric][edge][-1] = (edges[metric][edge][-1][0], N)
            else:
                edges[metric][edge].append((N, N))

    return edges


def _check_trace_args(series, networks, Ns, params, root_dir):
    """
        Check args specified for trace analysis are valid

        :param list series: series to include
        :param list networks: networks to include
        :param list Ns: of sample sizes (Ns) to process
        :param dict/None params: trace-specific parameters
        :param str root_dir: root directory holding trace files

        :raises TypeError: if any bad argument types
        :raises ValueError: if any bad argument values

        :returns tuple: of arb_p, metrics_p, graph_p
    """
    if (not isinstance(series, list) or not isinstance(networks, list)
            or not isinstance(Ns, list)
            or not isinstance(root_dir, str)
            or any([not isinstance(s, str) for s in series])
            or any([not isinstance(n, str) for n in networks])
            or (params is not None and not isinstance(params, dict))):
        raise TypeError('trace_analysis() bad arg types')

    # check only valid parameter keys

    params = params.copy() if params is not None else {}
    if len(set(params.keys()) - {'arb', 'metrics', 'graph', 'fig'}) > 0:
        raise ValueError('trace_analysis() unknown param keys')

    # check graph parameter supplied with no value

    graph_p = ((True if params['graph'] == '' else params['graph'])
               if 'graph' in params else None)
    if graph_p is not None and graph_p is not False:
        raise ValueError('trace_analysis() bad value supplied for graph')

    # check arb parameter is an integer

    arb_p = params['arb'] if 'arb' in params else 0
    if (not isinstance(arb_p, int) or isinstance(arb_p, bool) or
            arb_p < 0 or arb_p > 200):
        raise ValueError('trace_analysis() bad value supplied for arb')

    # check fig parameter is valid

    fig_p = params['fig'] if 'fig' in params else None
    if fig_p is not None and fig_p not in FIGURE_PARAMS:
        raise ValueError('trace_analysis() bad value supplied for fig')

    # check only valid metrics values specified

    metrics_p = (set(params['metrics'].split(',')) if 'metrics' in params
                 and params['metrics'] is not None else set())
    if len(metrics_p - {'mi', 'omi'}) != 0:
        raise ValueError('trace_analysis() bad metrics values')

    return (graph_p, metrics_p, arb_p, fig_p)


def trace_analysis(series, networks, Ns, Ss=None, file=None, params=None,
                   root_dir=EXPTS_DIR):
    """
        Prints specified traces

        :param list series: series to include
        :param list networks: networks to include
        :param list Ns: of sample sizes (Ns) to process
        :param tuple Ss: sub-samples to process
        :param str file: output file name
        :param dict/None params: trace-specific parameters
        :param str root_dir: root directory holding trace files

        :returns tuple: (summaries, last trace, last diffs) for system tetsing
    """
    graph_p, metrics_p, arb_p, fig_p = \
        _check_trace_args(series, networks, Ns, params, root_dir)

    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)

    exp_series = list(chain(*[SERIES_GROUPS[s] if s in SERIES_GROUPS else [s]
                              for s in series]))

    print('\n\n**Series {}'.format(exp_series))

    summaries = []  # summary information from each trace
    edges = {}  # edges matched, missing, extra etc.
    diffs = None  # trace differences
    trace = None

    # Loop over specified networks and series

    previous_trace = None
    arbitrary = ArbitraryAnalysis() if arb_p > 0 else None
    for network in networks:
        for _series in exp_series:

            # get series properties, reference BN and learning traces

            ref, _ = reference_bn(network)
            N_scale = 1 if isinstance(Ns[0], int) else ref.free_params
            _Ns = [round(N_scale * N) for N in Ns]
            traces = Trace.read(_series + '/' + network, root_dir)
            if traces is None:
                print('\nNo traces available for network {} in series {}'
                      .format(network, _series))
                continue

            # if MI is required need to get dataset

            if 'mi' in metrics_p and Ns[-1] <= 10**7:
                data = Pandas.read(root_dir + '/datasets/' + network +
                                   '.data.gz', dstype='categorical', N=Ns[-1])
            else:
                data = None

            # loop over traces by increasing sample size order + sample number

            sorted_keys = sorted([tuple(k.split('_')) for k in traces])
            sorted_keys = {t[0]: [t2[1] for t2 in sorted_keys
                                  if t2[0] == t[0] and len(t2) == 2]
                           for t in sorted_keys}
            for N, samples in sorted_keys.items():
                N = int(N.replace('N', ''))
                if N not in _Ns:
                    continue  # ignore sample sizes not required

                samples = [None] if samples == [] else samples
                for sample in samples:
                    if (Ss is not None and sample is not None and
                            (int(sample) < Ss[0] or int(sample) > Ss[1])):
                        continue
                    key = 'N{}'.format(N) + ('' if sample is None
                                             else '_{}'.format(sample))
                    trace = traces[key]

                    analysis = TraceAnalysis(trace, ref if 'omi' in metrics_p
                                             else ref.dag, None if data is None
                                             else data[:N])

                    if analysis.summary['type'] == 'DAG':
                        edges = _edges(edges, analysis.edges['result'],
                                       (None if previous_trace is None else
                                        previous_trace[2].context['N']), N)

                    if arbitrary is not None:
                        arbitrary.collect(_series, network, analysis)

                    # Compare previous trace with this one

                    if previous_trace is not None:
                        print('\n\nDifferences of {}/{}/{} from {}/{}/{}:'
                              .format(_series, network, trace.context['N'],
                                      previous_trace[0], previous_trace[1],
                                      previous_trace[2].context['N']))
                        diffs = trace.diffs_from(previous_trace[2])
                        print('{}\n'.format(diffs[2] if diffs is not None
                                            else 'None'))

                    previous_trace = (_series, network, trace)
                    print('\n{}'.format(analysis))
                    summary = {'series': _series, 'network': network,
                               'sample': sample}
                    summary.update(analysis.summary)
                    summaries.append(summary)

                    if graph_p is True:
                        traceviz(analysis, root_dir + '/analysis/graphviz')

    if len(summaries):
        print('\n\n{}'.format(DataFrame(summaries)
                              .sort_values(by=['network', 'N', 'series'])))

        if len(edges) and len(summaries) == 1:
            print('\n{}'.format(_edges(edges)))

        # generate arbitrary change plot if just one trace and not sys testing

        if arbitrary is not None:
            arbitrary.plot(max_iteration=arb_p, fig_p=fig_p, file=file)

    else:
        print('\n\nNo traces found.')

    return (summaries, trace, diffs)


def check_traces(series, networks, Ns):
    """
        Check whether traces in two series are near-enough identical.

        :param list series: series to compare (first and last is compared)
        :param list networks: compare traces from these networks
        :param list Ns: of sample sizes to process
    """
    print('\n\nCheck traces from series {} against {}, networks {}, Ns {}-{}'
          .format(series[0], series[-1], networks, Ns[0], Ns[-1]))

    for network in networks:
        stats = {'same': 0, 'diff': 0, 'only_1': 0, 'only_2': 0, 'neither': 0}
        traces_1 = Trace.read(series[0] + '/' + network)
        traces_2 = Trace.read(series[-1] + '/' + network)
        for N in Ns:
            key = 'N{}'.format(N)
            if key not in traces_1 and key not in traces_2:
                stats['neither'] += 1
                msg = 'in neither series'
            elif key not in traces_1:
                stats['only_2'] += 1
                msg = 'only in {}'.format(series[-1])
            elif key not in traces_2:
                stats['only_1'] += 1
                msg = 'only in {}'.format(series[0])
            else:
                diffs = traces_2[key].diffs_from(traces_1[key], strict=False)
                if diffs is None or diffs[0] == {}:
                    stats['same'] += 1
                    msg = 'same in the two series'
                else:
                    print(diffs)
                    stats['diff'] += 1
                    msg = ('DIFFERENT in the two series:\n{}\n\n{}\n\n{}'
                           .format(diffs[2], traces_1[key], traces_2[key]))
            print('\nNetwork {}, sample size {} is {}'
                  .format(network, N, msg))

        print(('\nNetwork {} has {} traces same, {} different, {} only in {}'
               + ', {} only in {} and {} in neither series')
              .format(network, stats['same'], stats['diff'], stats['only_1'],
                      series[0], stats['only_2'], series[-1],
                      stats['neither']))
