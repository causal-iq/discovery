
# Analyses metrics from experiments

from pandas import DataFrame
from math import ceil, sqrt

from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from experiments.common import NETWORKS_GRID_DESIGN, reference_bn, METRICS, \
    Ordering, compare_series_properties, comma_to_and, FIGURE_PARAMS
from experiments.plot import relplot
from learn.trace import Trace
from analysis.trace import TraceAnalysis


def _get_metrics_for_figure(compare_metrics, series, networks, metrics, Ns):
    """
        Assembles metrics for figure in 'long-form' required by plot libraries

        :param bool compare_metrics: are we comparing metrics in each figure,
                                     otherwise series
        :param list series: series to obtain data for
        :param list networks: networks to obtain data for
        :param list metrics: metrics to obtain data for
        :param list Ns: of sample sizes to process
    """
    data = {'subplot': [], 'x_val': [], 'y_var': [], 'y_val': []}

    # Loop over series required, obtaining series properties

    for _series in series:

        # Loop over networks required, obtaining reference BN and traces
        # for that network

        for network in networks:
            ref, _ = reference_bn(network)
            traces = Trace.read(_series + '/' + network)
            if traces is None:
                continue

            df = (Pandas.read(EXPTS_DIR + '/datasets/' + network + '.data.gz',
                              dstype='categorical').sample
                  if 'loglik' in metrics else None)

            # Loop over traces for each sample size obtaining trace analysis

            for trace in traces.values():
                if trace.context['N'] not in Ns:
                    continue

                sample = None if df is None else df[:trace.context['N']]
                summary = TraceAnalysis(trace, ref.dag, sample).summary

                # Loop over metrics required, storing data point for the
                # specific network, sample size, metric and metric value.

                for metric in metrics:
                    data['subplot'].append(network)
                    data['x_val'].append(summary['N'])

                    # y_var is either the metric or the series

                    data['y_var'].append(metric if compare_metrics is True
                                         else _series)

                    # arc metric value is scaled by the number of edges

                    data['y_val'].append(summary[metric] / len(ref.dag.edges)
                                         if metric.startswith('a-') or
                                         metric.startswith('e-') else
                                         summary[metric])
    return DataFrame(data)


def _plot_file(compare_metrics, fig_var, series, metrics, params):
    """
        Generate name of the plot file

        :param bool compare_metrics: whether figures compare metrics or series
        :param str fig_var: distinguishing variable for this figure - either
                            a metric or a series
        :param list series: series to include
        :param list metrics: metrics being processed
        :param dict params: user specified parameters

        :returns str: full path name of plot file
    """
    path = '{}/analysis/{}/'.format(EXPTS_DIR, 'metrics' if compare_metrics
                                    else 'series')
    file = (params['fig'] if 'fig' in params else
            ((metrics if compare_metrics else series) + '_' + fig_var))
    return path + file.replace('/', '_').replace(',', '_') + '.png'


def series_label(different):
    """
        Generates legend label for set of properties which differentiate
        this series from the other series being analysed.

        :param dict different: those properties which differentiate this series
                               from the other series and which are therefore
                               used to construct the legend label {prop: value}

        :return str: human readable legend label
    """
    if 'knowledge' not in different:
        label = '{}'.format(','.join(['{}'.format(v)
                                      for v in different.values()]))

    else:  # legend labels specifically tailored for knowledge comparisons

        if different['knowledge'] is False:
            label = 'No knowledge'
        else:
            label = []
            if 'limit' in different:
                label.append('{} requests'
                             .format(different['limit'] if different['limit']
                                     is not False else 'Unlimited'))
            if 'expertise' in different:
                label.append('expertise: {}'
                             .format(different['expertise']))
            if 'ignore' in different:
                label.append('{:.0f} ignored'.format(different['ignore'])
                             if different['ignore'] != 0
                             else 'None ignored')
            label = 'Knowledge:\n' + '\n'.join(label)

    return label


def _labels(compare_metrics, fig_var, common, different, metrics):
    """
        Generate labels for the metrics figure.

        :param bool compare_metrics: whether figures compare metrics or series
        :param str fig_var: distinguishing variable for this figure - either
                            a metric or a series
        :param dict common: properties that are common to the series
        :param dict different: properties that differ between series
        :param list metrics: metrics being processed

        :returns dict: of label properties for figure e.g. figure.title
    """
    text = {'algorithm': '{} algorithm', 'package': '{} package',
            'score': '{} score', 'ordering': '{} variable ordering'}
    xlabel = 'Sample size'
    if compare_metrics is True:

        # The title of a "by metrics" figure is based on the properties of the
        # series that that figure is showing - which is the composite of the
        # common properties and the specific properties for that series.

        either = {**common, **(different[fig_var])}
        title = ', '.join([f.format(either[p]) for p, f in text.items()])
        title = comma_to_and(title)

        # y-label and legend labels are created from the metric labels. If the
        # metrics include a-ok and F1 then just shorten ylabel to
        # "F1 and arc metrics"

        ylabel = ('F1 and arc metrics' if 'a-ok' in metrics and 'f1' in metrics
                  else ', '.join([METRICS[m]['label'] for m in metrics]))
        leg_title = 'Metrics'
        leg_labels = {m: METRICS[m]['label'] for m in metrics}

    else:

        # The title of a "by series" figure shows which properties we are
        # varying on the charts, and which series properties (the common
        # properties across all series) are common across all the lines in
        # the subcharts

        varying = ', '.join([f.format('#') for p, f in text.items()
                             if p in list(different.values())[0]])
        constant = ', '.join([f.format(common[p]) for p, f in text.items()
                              if p in common])
        title = ('Varying ' + comma_to_and(varying.replace('# ', ''))
                 + ' for ' + comma_to_and(constant))

        # The y-label is the specific metric for this figure

        ylabel = METRICS[fig_var]['label']
        leg_title = comma_to_and(varying.replace('# ', ''))
        leg_labels = {s: series_label(d) for s, d in different.items()}

    # Add a second line to the title which describe what is plotted against
    # what in each subchart.

    title += '\n' + ylabel + ' vs. ' + xlabel.lower()

    return {'figure.title': title, 'xaxis.label': xlabel,
            'yaxis.label': ylabel, 'legend.title': leg_title,
            'legend.labels': leg_labels}


def _styles(compare_metrics, common, different, metrics):
    """
        Generate styles (colours, line widths and styles etc.) for the metrics
        figure.

        :param bool compare_metrics: whether figures compare metrics or series
        :param str fig_var: distinguishing variable for this figure - either
                            a metric or a series
        :param dict common: properties that are common to the series
        :param dict different: properties that differ between series
        :param list metrics: metrics being processed

        :returns dict: of label properties for figure e.g. figure.title
    """
    styles = {}
    if compare_metrics:

        # Different lines are different metrics - just take colour, width
        # and dashes as defined in METRICS constant

        styles['palette'] = [METRICS[m]['colour'] for m in metrics]
        styles['line.sizes'] = {m: METRICS[m]['size'] for m in metrics}
        styles['line.dashes'] = {m: METRICS[m]['dashes'] for m in metrics}
        styles['figure.subplots_right'] = 0.78

    elif (set(different[list(different)[0]]) == {'ordering'}
          and len(different) == 3):

        # Different lines are different series, and in this case there are
        # three series and only the variable ordering is varying - so use
        # green/black/red colouring for optimal, standard and worst naming

        colours = {n.value['name']: n.value['colour'] for n in Ordering}
        styles['palette'] = [colours[different[s]['ordering']]
                             for s in different]
        styles['figure.subplots_right'] = 0.86

    return styles


def metric_analysis(compare_metrics, series, networks, metrics, Ns, params,
                    args):
    """
        Analyses learnt graph metrics and generates one or more figures

        :param bool compare_metrics: are we comparing metrics in each figure,
                                     otherwise series
        :param list series: series argument specified
        :param list networks: networks to include
        :param list metrics: metrics to include
        :param list Ns: of sample sizes to process
        :param dict params: command line parameters
        :param dict args: raw command line args - to name plot file
    """
    common, different = compare_series_properties(series)
    comp_know = 'knowledge' in set((list(different.values())[0]).keys())
    print('Common series properties are {} and different ones are {}'
          .format(common, different))

    # Loop over figures - each figure has a defined figure_var - either a
    # metric or a series

    for fig_var in series if compare_metrics else metrics:

        # Obtain labels and styles for this figure

        labels = _labels(compare_metrics, fig_var, common, different, metrics)
        styles = _styles(compare_metrics, common, different, metrics)

        print('\ngetting data for figure "{}" ...'
              .format(labels['figure.title']))

        if compare_metrics:
            data = _get_metrics_for_figure(compare_metrics, [fig_var],
                                           networks, metrics, Ns)
        else:
            data = _get_metrics_for_figure(compare_metrics, series, networks,
                                           [fig_var], Ns)

        if 'sd' in params:
            print('\n\n*** compute SD ***\n{}\n'.format(data))
            std = (data.groupby(['subplot', 'x_val', 'y_var'])['y_val']
                   .agg('std').reset_index())
            print(std)
            std = (std.groupby(['y_var']).mean()
                   .drop(columns=['x_val'])
                   .rename(columns={'y_val': 'std'}))
            print('\n\nMean SD over variable orderings per series:\n{}\n\n'
                  .format(std))

        props = NETWORKS_GRID_DESIGN
        props.update(labels)
        props.update(styles)
        props.update({'figure.per_row': ceil(sqrt(len(networks))),
                      'yaxis.range': ((0, 2.05) if fig_var == 'shd-e'
                                      else (0, 1.05)),
                      'xaxis.range': tuple((Ns[0], Ns[-1])),
                      'yaxis.shared': fig_var not in ['score', 'time'],
                      'figure.subplots_left': 0.05,
                      'figure.subplots_right': 0.85,
                      'figure.subplots_wspace': 0.20,
                      'legend.fontsize': 22,
                      'legend.title_fontsize': 22,
                      'subplot.axes_fontsize': 22,
                      'subplot.title_fontsize': 22,
                      'xaxis.ticks_fontsize': 22,
                      'yaxis.ticks_fontsize': 22})
        if comp_know:
            props.update({'figure.subplots_right': 0.84})
        if Ns[0] == 10 and Ns[-1] == 10**7:
            props.update({'xaxis.ticks': [10, 100, 1000, 10000, 100000,
                                          1000000, 10000000]})
        if 'fig' in params and params['fig'] in FIGURE_PARAMS:
            props.update(FIGURE_PARAMS[params['fig']])

        props.update(params)
        print(params)

        props = {k: v for k, v in props.items() if v is not None}

        if props['subplot.kind'] in ['boxplot', 'violin']:
            print('\n\n***re-organise data\n')
            data.drop(columns=['x_val'], inplace=True)
            data.rename(columns={'y_var': 'x_val'}, inplace=True)
            del props['xaxis.range']  # needed for categorical x-axis
            del props['xaxis.scale']  # needed for categorical x-axis

        plot_file = _plot_file(compare_metrics, fig_var, args['series'],
                               args['metrics'], params)

        print('... generating plot file "{}"'.format(plot_file))
        relplot(data, props, plot_file)
