
# Performs analysis of a group of learning experiment series

from itertools import chain
from pandas import DataFrame, set_option, concat
from math import isnan
from statistics import mean, stdev

from fileio.common import EXPTS_DIR
from experiments.common import SERIES_GROUPS, reference_bn
from core.graph import PDAG
from learn.trace import Trace
from analysis.trace import TraceAnalysis

NaN = float("nan")


def impute_missing(metric, reqd_metric):
    """
        Impute missing metric values as average value of that metric over the
        other series that do have a value for the same network and sample
        size.
    """
    def impute(network, N):  # compute a single imputed metric value
        values = []
        for series in metric[network]:
            if not isnan(metric[network][series][N]):
                values.append(metric[network][series][N])
        return mean(values) if len(values) else NaN

    # Loop over all combinations of network, series and sample size imputing
    # any missing values with mean over other series for that sample size and
    # network.

    imputed = {}
    for network, _series in metric.items():
        for series, Ns in _series.items():
            for N in Ns:
                if isnan(metric[network][series][N]):
                    if network not in imputed:
                        imputed[network] = {}
                    if series not in imputed[network]:
                        imputed[network][series] = {}
                    imputed[network][series][N] = impute(network, N)

    # Copy over any imputed values into metric structure:

    for network, _series in imputed.items():
        for series, Ns in _series.items():
            for N in Ns:
                metric[network][series][N] = imputed[network][series][N]
                print("{} imputed for {}/{}/N{} as {:.4f}"
                      .format(reqd_metric, network, series, N,
                              imputed[network][series][N]))


def _generate_table2(reqd_metric, raw_metrics, ignore, impute):
    """
        Generate the tree analysis tables

        :param str reqd_metric: metric which is shown in table
        :param dict raw_metrics: raw data metrics
        :param set ignore: {(network, N), ...} of cases to ignore
        :param set impute: set of metrics to impute values for

        :returns tuple: ({series: mean value} of metric for each series,
                         DataFrame: metric value for each network)
    """

    # Construct empty metrics table - ensure it has all combinations of
    # network, series and N even if there is no corresponding entry in
    # raw_metrics, but don't include networks and ssample sizes in "ignore".

    metric = {}
    all_series = set()
    all_Ns = set()
    for network, _series in raw_metrics.items():
        for series, Ns in _series.items():
            all_series.add(series)
            for N in Ns:
                all_Ns.add(N)
    for network in raw_metrics:
        metric[network] = {}
        for series in all_series:
            metric[network][series] = {}
            for N in all_Ns:
                if (network, N) not in ignore:
                    metric[network][series][N] = (0 if reqd_metric == 'expts'
                                                  else NaN)

    # Place values in metric linked-lists structure

    for network, _series in raw_metrics.items():
        for series, Ns in _series.items():
            for N, samples in Ns.items():
                if (network, N) in ignore:
                    continue

                if reqd_metric == 'f1-e-std':  # st. dev of f1-e values
                    values = [s['f1-e'] for s in samples
                              if not isnan(s['f1-e'])]
                    value = stdev(values) if len(values) > 2 else NaN

                elif reqd_metric == 'score-std':  # st. dev of score values
                    values = [s['score'] for s in samples
                              if 'score' in s and not isnan(s['score'])]
                    value = stdev(values) if len(values) > 2 else NaN

                elif reqd_metric == 'loglik-std':  # st. dev of loglik values
                    values = [s['loglik'] for s in samples
                              if 'loglik' in s and s['loglik'] is not None
                              and not isnan(s['loglik'])]
                    value = stdev(values) if len(values) > 2 else NaN

                elif reqd_metric == 'expts':  # number of sub-samples
                    value = len(samples)

                elif reqd_metric == 'dens':  # edges / # nodes i.e. density
                    values = [2.0 * s['|E|'] / s['n'] for s in samples
                              if not isnan(s['|E|'])]
                    value = mean(values) if len(values) else NaN

                elif reqd_metric == 'dens-std':  # density SD
                    values = [2.0 * s['|E|'] / s['n'] for s in samples
                              if not isnan(s['|E|'])]
                    value = stdev(values) if len(values) > 2 else NaN

                elif reqd_metric == 'nonex':  # non-extendable PDAGs
                    values = [1 for s in samples if s['type'] == 'NONEX']
                    value = len(values)

                else:  # just take metric as-is from TraceAnalysis
                    values = [s[reqd_metric] for s in samples
                              if reqd_metric in s and
                              s[reqd_metric] is not None
                              and not isnan(s[reqd_metric])]
                    value = ((values[0] if reqd_metric == 'time'
                              else mean(values)) if len(values) else NaN)

                metric[network][series][N] = value

    # impute missing metrics here if possible - NEED TO HANDLE IGNORE

    if reqd_metric in impute:
        impute_missing(metric, reqd_metric)

    # Now average metric over sample sizes

    for network, _series in metric.items():
        for series, Ns in _series.items():
            values = [v for v in Ns.values() if not isnan(v)]
            value = (NaN if not len(values) else
                     (round(sum(values)) if reqd_metric in ['expts', 'nonex']
                      else round(mean(values), 4)))
            metric[network][series] = value

    # Convert linked list structure to a Dataframe, and add column of mean
    # values over networks which is returned from this function.

    table = DataFrame(metric)
    if 'series' in table.columns:
        table.set_index('series')
    table.index.name = 'series'
    table['MEAN'] = table.mean(axis='columns')
    print('\n{} metric\n{}\n'.format(reqd_metric, table))
    means = table['MEAN'].to_dict()
    table = table.drop(columns=['MEAN']).reset_index()
    table = table.melt(id_vars=table.columns[0], var_name='network',
                       value_name='value')
    table['metric'] = reqd_metric

    return ({s: round(v, 1 if reqd_metric in {'time', 'pretime'} else 4)
             for s, v in means.items()}, table)


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
        :param dict/None params: analysis-specific parameters:
                                 'impute': ('metric1', ....)
                                 'ignore': ('network1@N', ...)
        :param str root_dir: root directory holding trace files

        :returns tuple: (DataFrame: means of each metric for each series,
                         DataFrame: details of metrics per series/network)
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
    if len(ignore):
        print('\n\nIgnoring these experiments: {} \n'.format(ignore))
    impute = set(params['impute']) if 'impute' in params else set()

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
                            or 'time' not in analysis.summary
                            or (analysis.summary['time'] <= maxtime * 60
                                and analysis.summary['time'] >= 0)):
                        raw_metrics[network][_series][N] \
                            .append(analysis.summary)

    if len(summaries):
        if 'summ' in params:
            print('\n\n{}'.format(DataFrame(summaries)
                                  .sort_values(by=['network', 'N', 'series'])))

        means = {}
        details = []
        for metric in metrics:
            stat_means, detail = _generate_table2(metric, raw_metrics, ignore,
                                                  impute)
            details.append(detail)
            means.update({metric: stat_means})
        means = DataFrame(means).reindex(series)
        details = concat(details, ignore_index=True)
        print('\n\nMeans across all sample sizes and networks:\n\n{}'
              .format(means))

    else:
        means = None
        print('\n\nNo traces found.')

    return (means, details)
