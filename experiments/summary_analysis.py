
# Performs analysis for the tree runs

from itertools import chain
from statistics import mean, stdev
from pandas import DataFrame, set_option
from numpy import NaN
from math import isnan

from fileio.common import EXPTS_DIR
from experiments.common import SERIES_GROUPS, reference_bn
from core.graph import PDAG
from learn.trace import Trace
from analysis.trace import TraceAnalysis


def _generate_table(reqd_metric, raw_metrics):
    """
        Generate the tree analysis tables

        :param str reqd_metric: metric which is shown in table
        :param dict raw_metrics: raw data metrics

        :returns dict: {series: mean value} of metric for each series
    """
    def _round(value, dp=3):
        return NaN if all([isnan(v) for v in value]) else round(value, dp)

    # Generate the basic rows in the table

    table = {}
    for network, _series in raw_metrics.items():
        table[network] = {}
        for series, Ns in _series.items():

            values = []  # array of metric values across sample sizes

            for N, samples in Ns.items():

                # samples is array of metrics dict for each sub-sample

                if reqd_metric == 'f1-e-std':  # st. dev of f1-e values
                    values += ([stdev([s['f1-e'] for s in samples])]
                               if len(samples) > 2 else [NaN])

                elif reqd_metric == 'expts':  # number of sub-samples
                    values += [len(samples)]

                elif reqd_metric == 'dens':  # # edges / # nodes i.e. density
                    values += [s['|E|'] / s['n'] for s in samples]

                else:  # just take metric from TraceAnalysis
                    values += [s[reqd_metric] for s in samples]

            # if reqd_metric == 'time':  # use first time to avoid caching bias
            #     values = [samples[0]['time'] if samples[0]['time'] > 0.1
            #              else 0.1]

            if reqd_metric == 'expts':  # we want total sum for this metric
                values = [sum(values)]

            # We now take mean value over all sample sizes except for expts
            # which we sum

            print(series, reqd_metric, values)
            table[network][series] = (round(sum(values), 0) if reqd_metric ==
                                      'expts' else round(mean(values), 4)
                                      if len(values) else None)
    table = DataFrame(table)
    if 'series' in table.columns:
        table.set_index('series')
    table.index.name = 'series'
    table['MEAN'] = table.mean(axis='columns')
    print('\n{} metric\n{}\n'.format(reqd_metric, table))
    means = table['MEAN'].to_dict()
    table.drop(columns=['MEAN'], inplace=True)

    return {s: round(v, 1 if reqd_metric in {'time', 'pretime'} else 4)
            for s, v in means.items()}


def summary_analysis(series, networks, Ns, Ss=None, metrics=None, maxtime=None,
                     file=None, params=None, root_dir=EXPTS_DIR):
    """
        Perform a summary analysis

        :param list series: series to include
        :param list networks: networks to include
        :param list Ns: of sample sizes (Ns) to process
        :param tuple Ss: sub-samples to process
        :param list metrics: metrics to obtain data for
        :param int maxtime: maximum execution time in minutes
        :param str file: output file name
        :param dict/None params: trace-specific parameters
        :param str root_dir: root directory holding trace files

        :returns DataFrame: means of each metric for each series
    """
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)

    exp_series = list(chain(*[SERIES_GROUPS[s] if s in SERIES_GROUPS else [s]
                              for s in series]))

    print('\n\n**Series {}'.format(exp_series))

    summaries = []  # summary information from each trace
    trace = None
    raw_metrics = {}  # raw metrics across networks and series

    # identify network and sample sizes to ignore

    ignore = ({(e.split('@')[0], int(e.split('@')[1]))
               for e in params['ignore']}
              if 'ignore' in params else set())
    print('\n\n** {} \n'.format(ignore))

    # Loop over specified networks and series

    for network in networks:
        ref, _ = reference_bn(network)
        if network not in raw_metrics:
            raw_metrics[network] = {}

        for _series in exp_series:
            print('Scanning traces of {} in {}'.format(network, _series))

            # get series properties, reference BN and learning traces

            traces = Trace.read(_series + '/' + network, root_dir)
            if traces is None:
                print('\nNo traces available for network {} in series {}'
                      .format(network, _series))
                continue

            if _series not in raw_metrics[network]:
                raw_metrics[network][_series] = {}

            # loop over traces by increasing sample size order + sample number

            sorted_keys = sorted([tuple(k.split('_')) for k in traces])
            sorted_keys = {t[0]: [t2[1] for t2 in sorted_keys
                                  if t2[0] == t[0] and len(t2) == 2]
                           for t in sorted_keys}
            for N, samples in sorted_keys.items():
                N = int(N.replace('N', ''))
                if N not in Ns or (network, N) in ignore:
                    continue  # ignore sample sizes not required

                if N not in raw_metrics[network][_series]:
                    raw_metrics[network][_series][N] = []

                samples = [None] if samples == [] else samples
                for sample in samples:
                    if (Ss is not None and sample is not None and
                            (int(sample) < Ss[0] or int(sample) > Ss[1])):
                        continue
                    key = 'N{}'.format(N) + ('' if sample is None
                                             else '_{}'.format(sample))
                    trace = traces[key]

                    # Get standard metrics for trace

                    analysis = TraceAnalysis(trace, ref.dag, None)
                    pretime = (round(trace.context['pretime'], 1)
                               if 'pretime' in trace.context else 0.0)
                    analysis.summary.update({'pretime': pretime})

                    # Compute non-standard Prec, Recall

                    if 'p-e' in metrics or 'r-e' in metrics:
                        try:
                            graph = PDAG.toCPDAG(trace.result)
                        except ValueError:
                            graph = trace.result
                        ref_cpdag = PDAG.toCPDAG(ref.dag)
                        metrics_e = graph.compared_to(ref_cpdag)
                        analysis.summary.update({'p-e': (metrics_e['p'] if
                                                         metrics_e['p'] is not
                                                         None else 0.0),
                                                 'r-e': metrics_e['r']})

                    summary = {'series': _series, 'network': network,
                               'sample': sample}
                    summary.update(analysis.summary)

                    summaries.append(summary)

                    if (maxtime is None
                            or analysis.summary['time'] <= maxtime * 60):
                        raw_metrics[network][_series][N] \
                            .append(analysis.summary)

    if len(summaries):
        if 'summ' in params:
            print('\n\n{}'.format(DataFrame(summaries)
                                  .sort_values(by=['network', 'N', 'series'])))

        means = {}
        for metric in metrics:
            stat_means = _generate_table(metric, raw_metrics)
            means.update({metric: stat_means})
        means = DataFrame(means)
        print('\n\nMeans across all sample sizes and networks:\n\n{}'
              .format(means))

    else:
        means = None
        print('\n\nNo traces found.')

    return means
