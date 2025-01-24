
# Analysis of the overall impact of a factor e.g. sample size, var. ordering
# on the learnt graphs

from pandas import DataFrame, set_option, read_csv
from statistics import mean
from enum import Enum
from numpy import std

from fileio.common import EXPTS_DIR
from learn.trace import Trace
from analysis.trace import TraceAnalysis
from experiments.plot import relplot
from analysis.bn import BN_PROPERTIES
from experiments.config import NETWORKS
from experiments.common import series_props, reference_bn, sample_sizes, \
    NETWORKS_GRID_DESIGN, Algorithm, compare_series_properties, METRICS, \
    SERIES_COMPARATORS, series_comparator, FIGURE_PARAMS


def _algo_label(algo_name):
    """
    """
    for a in Algorithm:
        if a.value['name'] == algo_name:
            return a.value['label']
    return None


def _props_key(series=None, property=None):
    """
        Return elements of key based on series properties, or index for a
        specific property

        :param str series: if specified, series to obtain key elements for
        :param str property: if specified, property to return tuple index of

        :returns tuple/int: series comparison properties/tuple index
    """

    # return index of property

    if property is not None:
        return SERIES_COMPARATORS.index(property)

    # return tuple of comparison properties for series

    props = series_props(series)
    props.update(props['params'] if 'params' in props else {})
    props.update(props['kparams'] if 'kparams' in props
                 and props['kparams'] is not None else {})
    return tuple([((props[p].value['name'] if isinstance(props[p].value, dict)
                    else props[p].value)
                   if isinstance(props[p], Enum)
                   else props[p]) if p in props else None
                  for p in SERIES_COMPARATORS])


def _get_rawdata(series, networks, metrics, Ns, Ss, expts_dir):
    """
        Return rawdata required for impact analysis.

        :param list series: series being analysed
        :param list networks: networks being analysed
        :param str metric: metric being analysed
        :param list Ns: list of sample sizes to process
        :param tuple Ss: sub-samples to process
        :param str expts_dir: location of experiments directory

        :returns tuple: dict of metric values:
                        {(N, series properties, metric, network): value}
                        Sample sizes which have data:
                        {N1, N2, ...} or {network1: {N1, N2, ...}, ...}
    """
    rawdata = {}  # contains all metric values for all series, networks and N
    Ns_data = set() if isinstance(Ns[0], int) else {}

    for _series in series:
        print('Processing series {} with properties {} ...'
              .format(_series, series_props(_series)))
        for network in networks:

            # Get reference BN for network and all traces for this series

            ref, _ = reference_bn(network)
            traces = Trace.read(_series + '/' + network, root_dir=expts_dir)
            if traces is None:
                continue
            N_scale = 1 if isinstance(Ns[0], int) else ref.free_params
            _Ns = [round(N_scale * N) for N in Ns]

            if isinstance(Ns[0], int):
                Ns_set = Ns_data
            else:
                Ns_set = Ns_data[network] = set()

            print('... {} traces read for {}'.format(len(traces), network))
            for trace in traces.values():
                summary = TraceAnalysis(trace, ref.dag).summary
                N = summary['N']
                sample = summary['sample']
                if (N not in _Ns or (Ss is not None and
                                     (sample < Ss[0] or sample > Ss[1]))):
                    continue
                Ns_set.add(N)

                for metric in metrics:
                    if metric is None:
                        continue
                    key = (network, N, summary['sample'],
                           *_props_key(series=_series), metric)
                    rawdata[key] = summary[metric]
    return rawdata, Ns_data


def _sample_impact(series, networks, metric, rawdata, data,
                   multiples=[10, 100]):
    """
        Assembles the plot data measuring the impact of increasing the sample
        size by specified multiples

        :param list series: series being analysed
        :param list networks: networks being analysed
        :param str metric: metric being analysed
        :param dict rawdata: data keyed by series properties, network & N
        :param dict data: data in 'long format' required by plot functions
        :param list multiples: multiples of sampe sizes to analyse

        :returns dict: updated plot data
    """
    for _series in series:
        props = series_props(_series)
        samples = ([s for s in range(props['randomise'][1])]
                   if props['randomise'] is not False else [None])
        for network in networks:
            for N in sample_sizes('10-10m')[0]:
                for sample in samples:
                    key1 = (network, N, sample, *_props_key(series=_series),
                            metric)
                    if key1 not in rawdata:
                        continue
                    value1 = rawdata[key1]
                    for mult in multiples:
                        key2 = (network, mult * N, sample,
                                *_props_key(series=_series), metric)
                        if key2 not in rawdata:
                            continue
                        data['x_val'].append(('Increase sample\nsize by {}' +
                                              ' times').format(mult))
                        data['y_val'].append(rawdata[key2] - value1)
                        data['subplot'].append('unused')

    return data


def series_comparable(required, props1, props2):
    """
        Decide whether two series are comparable based upon their properties.
        To be comparable the series must have the values specified in the
        "required" argument for each series, and all other values must be the
        same in the two series.

        :param dict required: {idx: (val1, val2)} index position and value
                              required in each series
        :param tuple props1: properties of the first series
        :param tuple props2: properties of the second series

        :raises TypeError: if arguments have bad types
        :raises ValueError: if arguments have wrong sizes

        :returns bool: True if series are comparable, otherwise False
    """
    if (not isinstance(required, dict) or not isinstance(props1, tuple)
            or not isinstance(props2, tuple)
            or not all([isinstance(i, int) for i in required.keys()])
            or not all([isinstance(v, tuple) for v in required.values()])):
        raise TypeError('series_comparable: bad arg type')

    if (not len(required) or not len(props1) or len(props1) != len(props2)
            or any([len(v) != 2 for v in required.values()])
            or any([i < 0 or i > len(props1) - 1 for i in required.keys()])):
        raise ValueError('series_comparable: bad arg sizes/values')

    for i in range(len(props1)):
        if i in required:
            if props1[i] != required[i][0] or props2[i] != required[i][1]:
                return False
        elif props1[i] != props2[i]:
            return False
    return True


def series_impact(required, series, networks, metric, rawdata, data,
                  x_val=None, Ns=None):
    """
        Assembles the plot data measuring the impact of changing one
        of the series properties

        :param dict required: {prop: (val1, val2)} property values which must
                              be present in series being compared
        :param list series: series being analysed
        :param list networks: networks being analysed
        :param str metric: metric being analysed
        :param dict rawdata: data keyed by series properties, network & N
        :param dict data: data in 'long format' required by plot functions
        :param str x_val: optionally specify the x_val to use
        :param set/dict/None Ns: optionally, sample sizes which have rawdata

        :returns dict: updated plot data
    """
    Ns = set(sample_sizes('10-10m')[0]) if Ns is None else Ns

    print('\n\nIn series_impact, Ns are: {}\n\n'.format(Ns))
    # Collect the series properties of each series and formulate _required in
    # the structure needed by series_comparable. Also construct samples as a
    # list of the number of randomisation num_samples in each series in the
    # order defined by series arguments

    series_keys = [_props_key(series=s) for s in series]
    _required = {_props_key(property=p): v for p, v in required.items()}
    num_samples = [p['randomise'][1] if p['randomise'] is not False else None
                   for p in [series_props(s) for s in series]]
    print('\nComparing values {} ({}) over {} samples'
          .format(required, _required, num_samples))

    # comparisons is a list of 3-tuples of the series properties key
    # of each pair of series to be compared, and the number of sub-samples
    # each comparison should be made over for each sample size

    comparisons = []
    for i, props1 in enumerate(series_keys):
        for j, props2 in enumerate(series_keys):
            if series_comparable(_required, props1, props2):
                comparisons.append((tuple(props1), tuple(props2),
                                    num_samples[i]))
    for comparison in comparisons:
        print('  {} with {} over {} samples'
              .format(comparison[0], comparison[1], comparison[2]))
    base_done = len(data['x_val']) > 0

    # Now loop over each comparison and retrieve the raw data for each
    # network and sample size where that comparison is possible

    if x_val is None:
        x_val = {p: v for p, v in required.items()
                 if v[0] is not None and v[1] is not None}
        x_val = '\n'.join(['{} from\n{} to {}'.format(p, v[0], v[1])
                           for p, v in x_val.items()])
        x_val = x_val.replace('bde', 'bdeu')
    for comparison in comparisons:
        samples = ([None] if comparison[2] is None
                   else [s for s in range(comparison[2])])
        for network in networks:
            _Ns = Ns[network] if isinstance(Ns, dict) else Ns
            for N in _Ns:
                y_val0 = []
                y_val1 = []
                for sample in samples:
                    key0 = (network, N, sample, *(comparison[0]), metric)
                    key1 = (network, N, sample, *(comparison[1]), metric)
                    # print('Comparing {} and {}'.format(key0, key1))
                    if (key0 in rawdata and key1 in rawdata and rawdata[key0]
                            is not None and rawdata[key1] is not None):
                        data['x_val'].append(x_val)
                        data['y_val'].append(rawdata[key1] - rawdata[key0])
                        data['subplot'].append('unused')
                        # print('** value is {}'.format(rawdata[key0]))
                        y_val0.append(rawdata[key0])
                        y_val1.append(rawdata[key1])
                if len(y_val1) > 100:
                    data['x_val'].append('STD ' + x_val)
                    data['y_val'].append(std(y_val1))
                    data['subplot'].append('unused')
                    if not base_done:
                        data['x_val'].append('STD Base')
                        data['y_val'].append(std(y_val0))
                        data['subplot'].append('unused')

                    # print(' ** Values are {} and STD is {:.4f}'
                    #       .format(y_val1, std(y_val1)))

    return data


def _get_property_impact(base, required, series, networks, metric, rawdata,
                         data, y_var=None):
    """
        Get impact of varying a specific property for each individual
        algorithm.

        :param str base: base series for this algorithm
        :param dict required: {prop: (val1, val2)} property values which must
                              be present in series being compared
        :param list series: list of series names to possibly get data from
        :param list networks: networks to get data for
        :param str metric: metric to get data for e.g. "f1" or "f1-e"
        :param dict rawdata: data keyed by series properties, network & N
        :param list data: data for plotting functions, which this function
                          adds to
        :param str y_var: explicitly specify y_var value
    """
    # Get properties of base series for this algorithm, determine algorithm
    # name, and set up lookup dict from properties to series name.

    base_props = _props_key(series=base)
    algo = base_props[_props_key(property='algorithm')]
    lookup = {_props_key(series=s): s for s in series}

    # Construct the properties required in the two series to be compared by
    # merging the base properties with the mandatory and comparison properties.

    _required = {_props_key(property=p): v for p, v in required.items()}
    s1_props = tuple([_required[i][0] if i in _required else v
                      for i, v in enumerate(base_props)])
    s2_props = tuple([_required[i][1] if i in _required else v
                      for i, v in enumerate(base_props)])

    # Check that the two series with the required properties are available ...

    if s1_props in lookup and s2_props in lookup:

        # ... if so, compare the two relevant series to see impact of
        # property (2nd argument specifies the two series being compared)

        values = series_impact(required, [lookup[s1_props], lookup[s2_props]],
                               networks, metric, rawdata,
                               {'subplot': [], 'x_val': [], 'y_val': []})

        # this analysis only requires the mean impact not the distribution

        if len(values):
            data.append({'y_val': mean(values['y_val']),
                         'y_var': (values['x_val'][0] if y_var is None
                                   else y_var),
                         'subplot': 'unused', 'x_val': _algo_label(algo)})


def impact_vs_factor_plot_data(series, networks, metric, rawdata, common):
    """
        Generate plot format data showing impact on accuracy of various
        factors for a specific algorithm.

        :param list series: series being analysed
        :param list networks: networks being analysed
        :param str metric: metric being analysed
        :param dict rawdata: data keyed by series properties, network & N
        :param dict common: properties common to all series

        :returns tuple: (plot data, plot properties)
    """
    std_algo = 'BNLEARN/{}_STD'.format(common['algorithm'])
    data = {'subplot': [], 'x_val': [], 'y_val': []}

    data = _sample_impact([std_algo], networks, metric, rawdata, data)
    data = series_impact({'ordering': ('alphabetic', 'optimal')},
                         series, networks, metric, rawdata, data)
    data = series_impact({'ordering': ('worst', 'optimal')},
                         series, networks, metric, rawdata, data)
    data = series_impact({'score': ('bic', 'bde'), 'k': (1, None),
                          'iss': (None, 1)}, series, networks, metric,
                         rawdata, data)
    data = series_impact({'score': ('bic', 'bds'), 'k': (1, None),
                          'iss': (None, 1)}, series, networks, metric,
                         rawdata, data)
    data = series_impact({'test': ('mi', 'x2')}, series, networks, metric,
                         rawdata, data)
    data = series_impact({'k': (1, 5)},  series, networks, metric, rawdata,
                         data, x_val='complexity scaling\nfrom 1 to 5')
    data = series_impact({'iss': (1, 5)}, series, networks, metric,
                         rawdata, data, x_val="ESS from\n1 to 5")
    data = series_impact({'alpha': (0.05, 0.01)}, series, networks, metric,
                         rawdata, data)
    props = {'figure.title': (common['algorithm'] + ' algorithm - impact '
                              + 'of sample size, variable ordering,\nscore'
                              + ' and hyper-parameters on '
                              + METRICS[metric]['label']),
             'xaxis.label': 'Changes made to experiment',
             'yaxis.range': (-1.02, 1.02),
             'subplot.axes_fontsize': 12,
             'xaxis.label_fontsize': 12,
             'xaxis.ticks_fontsize': 10,
             'xaxis.ticks_rotation': -30,
             'xaxis.ticks_halign': 'left',
             'yaxis.ticks_fontsize': 10,
             'yaxis.label.fontsize': 12,
             'figure.subplots_left': 0.09,
             'figure.subplots_right': 0.96,
             'figure.subplots_bottom': 0.3,
             'subplot.aspect': 1.7}

    return (data, props)


def algo_vs_hc_plot_data(series, networks, metric, rawdata, common, different):
    """
        Generate plot format data comparing accuracy of algorithm against
        HC algorithm.

        :param list series: series being analysed
        :param list networks: networks being analysed
        :param str metric: metric being analysed
        :param dict rawdata: data keyed by series properties, network & N
        :param dict common: properties common to all series
        :param dict different: properties which differ between series

        :returns tuple: (plot data, plot properties)
    """
    data = {'subplot': [], 'x_val': [], 'y_val': []}
    algo_colours = {a.value['name']: a.value['colour'] for a in Algorithm}
    palette = []
    for diffs in different.values():
        if diffs['algorithm'] == 'HC':
            continue
        required = {'algorithm': ('HC', diffs['algorithm'])}
        if diffs['algorithm'] in ['PC', 'GS', 'IIAMB']:
            required.update({'score': ('bic', None), 'test': (None, 'mi'),
                             'k': (1, None), 'alpha': (None, 0.05)})
        elif diffs['algorithm'] == 'FGES':
            required.update({'package': ('BNLEARN', 'TETRAD')})
        data = series_impact(required, series, networks, metric, rawdata,
                             data, x_val=_algo_label(diffs['algorithm']))
        palette.append(algo_colours[diffs['algorithm']])

    props = {'figure.title': ('Impact of changing from HC algorithm to a '
                              + 'different algorithm\n on '
                              + METRICS[metric]['label'] + ', using '
                              + common['ordering'] + ' variable ordering'),
             'xaxis.label': 'Algorithm',
             'subplot.axes_fontsize': 12,
             'xaxis.label_fontsize': 12,
             'xaxis.ticks_fontsize': 10,
             'yaxis.ticks_fontsize': 10,
             'yaxis.label.fontsize': 12,
             'yaxis.range': (-0.80, 1.02),
             'subplot.aspect': 1.4,
             'palette': palette}

    return (data, props)


def impact_vs_algo_plot_data(networks, metric, rawdata, different):
    """
        Generate plot format data for summary bar chart of impact of factors
        such as sample size, ordering, score etc. for each algorithm.

        :param list networks: networks being analysed
        :param str metric: metric being analysed
        :param dict rawdata: data keyed by series properties, network & N
        :param dict different: properties which differ between series

        :returns tuple: (plot data, plot properties)
    """
    # Assemble series available for each algorithm, and define the
    # baseline properties for score/hybrid and constraint algorithms.

    compare = {p1['algorithm']: [s2 for s2, p2 in different.items()
               if p2['algorithm'] == p1['algorithm']]
               for s1, p1 in different.items()}
    algo_idx = _props_key(property='algorithm')
    score_baseline = list(_props_key(series='BNLEARN/HC_STD'))
    score_baseline.pop(algo_idx)
    constraint_baseline = list(_props_key(series='BNLEARN/PC_STD'))
    constraint_baseline.pop(algo_idx)
    fges_baseline = list(_props_key(series='TETRAD/FGES_STD'))
    fges_baseline.pop(algo_idx)
    BASELINES = [tuple(score_baseline), tuple(constraint_baseline),
                 tuple(fges_baseline)]
    print(BASELINES)

    # Loop over each algorithm getting impact data for it

    data = []
    for _algo, algo_series in compare.items():

        # first check that there is a baseline series for this algorithm

        base = None
        for _series in algo_series:
            _props = list(_props_key(series=_series))
            _props.pop(algo_idx)
            if tuple(_props) in BASELINES:
                base = _series
                break
        if base is None:
            continue

        # get impact of changing sample size by 100x on base series

        values = _sample_impact([base], networks, metric, rawdata,
                                {'subplot': [], 'x_val': [], 'y_val': []},
                                multiples=[100])
        if len(values):
            data.append({'y_val': mean(values['y_val']),
                         'y_var': values['x_val'][0],
                         'subplot': 'unused',
                         'x_val': _algo_label(_algo)})

        # compare series to get impact of ordering, score/test or hypers

        _get_property_impact(base, {'ordering': ('worst', 'optimal')},
                             algo_series, networks, metric, rawdata, data)
        _get_property_impact(base, {'score': ('bic', 'bde'),
                                    'k': (1, None), 'iss': (None, 1)},
                             algo_series, networks, metric, rawdata, data,
                             y_var="objective score\nor CI test")
        _get_property_impact(base, {'test': ('mi', 'x2')},
                             algo_series, networks, metric, rawdata, data,
                             y_var="objective score\nor CI test")
        _get_property_impact(base, {'k': (1, 5)},
                             algo_series, networks, metric, rawdata, data,
                             y_var="hyper-parameter")
        _get_property_impact(base, {'alpha': (0.05, 0.01)},
                             algo_series, networks, metric, rawdata, data,
                             y_var="hyper-parameter")

    props = {'figure.title': ('Sensitivity of algorithms to ordering, '
                              + 'sample size,\nscore or CI test and '
                              + 'hyper-parameters ('
                              + METRICS[metric]['label'] + ')'),
             'xaxis.label': '\nAlgorithm',
             'xaxis.label_fontsize': 10,
             'xaxis.ticks_fontsize': 10,
             'yaxis.range': (-0.12, 0.42),
             'yaxis.ticks_fontsize': 10,
             'yaxis.label.fontsize': 10,
             'legend.title': 'Change to\nexperiment',
             'legend.title_fontsize': 10,
             'legend.fontsize': 10,
             'figure.subplots_left': 0.09,
             'figure.subplots_right': 0.78,
             'figure.subplots_bottom': 0.13,
             'subplot.aspect': 1.4}

    return (data, props)


def sensitivity_vs_algo_plot_data(networks, metric, rawdata, different):
    """
        Generate plot format  data for sensitivity to ordering vs algorithm.

        :param list networks: networks being analysed
        :param str metric: metric being analysed
        :param dict rawdata: data keyed by series properties, network & N
        :param dict different: properties which differ between series

        :returns tuple: (plot data, plot properties)
    """
    data = {'subplot': [], 'x_val': [], 'y_val': []}
    algo_colours = {a.value['name']: a.value['colour'] for a in Algorithm}
    palette = []
    compare = {p1['algorithm']: [s2 for s2, p2 in different.items()
               if p2['algorithm'] == p1['algorithm']]
               for s1, p1 in different.items()}

    for _algo, _series in compare.items():
        data = series_impact({'ordering': ('worst', 'optimal')},
                             _series, networks, metric, rawdata, data,
                             x_val=_algo_label(_algo))
        palette.append(algo_colours[_algo])

    props = {'figure.title': ('Sensitivity of algorithm to variable '
                              + 'ordering\n- impact of changing from worst'
                              + ' to optimal ordering on '
                              + METRICS[metric]['label']),
             'xaxis.label': 'Algorithm',
             'subplot.axes_fontsize': 12,
             'xaxis.label_fontsize': 12,
             'xaxis.ticks_fontsize': 10,
             'yaxis.ticks_fontsize': 10,
             'yaxis.label.fontsize': 12,
             'yaxis.range': (-0.6, 1.02),
             'subplot.aspect': 1.4,
             'palette': palette}

    return (data, props)


def impact_vs_knowledge_plot_data(series, networks, Ns, metric, rawdata,
                                  common, different, params, xtick_labels):
    """
        Generate plot format data showing impact of knowledge on the bnbench
        HC algorithm.

        :param list series: series being analysed
        :param list networks: networks being analysed
        :param set/dict Ns: sample sizes which have rawdata - a set if
                            absolute sizes or {network: {N1, N2, ...}, ...}
        :param str metric: metric being analysed
        :param dict rawdata: data keyed by series properties, network & N
        :param dict common: properties common to all series
        :param dict different: properties which differ between series
        :param dict params: command line parameters
        :param tuple xtick_labels: custom xtick labels

        :returns tuple: (plot data, plot properties)
    """
    KPARAMS = ['knowledge', 'limit', 'ignore', 'expertise', 'reqd',
               'threshold', 'earlyok', 'partial', 'stop', 'nodes']
    data = {'subplot': [], 'x_val': [], 'y_val': []}
    print('\nDifferent is {}, common {}'.format(different, common))
    # print(rawdata)
    # for key, val in rawdata.items():
    #     print(key, val)

    # Identify 'base' series which doesn't use knowledge

    base = [s for s, p in different.items() if p['knowledge'] is False]
    if len(base) != 1:
        raise ValueError('Knowledge impact series must have 1 base')
    base = base[0]

    # Identify parameters which vary across the knowledge series and which
    # will be used to construct labels on the impact plot.

    kdiff = {p: {d[p] for d in different.values() if d['knowledge']
                 is not False} for p in list(different.values())[0]}
    kdiff = {p for p, v in kdiff.items() if len(v) > 1}
    xaxis_label = ', '.join(kdiff) + ' or sample sizes'

    # Loop over knowledge series collecting impacts

    bparams = series_comparator(base, as_dict=True)
    for i, s in enumerate(series):
        if s == base:
            continue
        sparams = series_comparator(s, as_dict=True)
        compare = {p: (bparams[p], sparams[p]) for p in KPARAMS}
        x_val = (',\n'.join(['{} = {}'.format(p, sparams[p]) for p in KPARAMS
                            if p in kdiff and sparams[p] is not None])
                 if xtick_labels is None or i > len(xtick_labels)
                 else xtick_labels[i-1])

        data = series_impact(compare, series, networks, metric, rawdata, data,
                             x_val=x_val, Ns=Ns)

    if 'N.impact' in params:
        data = _sample_impact([base], networks, metric, rawdata, data,
                              multiples=[10, 100])

    props = {'figure.title': 'Impact of ' + xaxis_label,
             'xaxis.label': 'Change made: ' + xaxis_label,
             'yaxis.range': (-0.30, 1.02),
             'subplot.axes_fontsize': 12,
             'xaxis.label_fontsize': 12,
             'xaxis.ticks_fontsize': 10,
             'xaxis.ticks_rotation': -30,
             'xaxis.ticks_halign': 'left',
             'yaxis.ticks_fontsize': 10,
             'yaxis.label.fontsize': 12,
             'figure.subplots_left': 0.09,
             'figure.subplots_right': 0.88,
             'figure.subplots_bottom': 0.3,
             'subplot.aspect': 1.7}

    return (data, props)


def _get_plot_data(series, networks, Ns, metric, rawdata, common, different,
                   params):
    """
        Extract the data in long-form required by the plotting function for
        a specific plot, and return this and any specific chart properties.

        :param list series: series being analysed
        :param list networks: networks being analysed
        :param set/dict Ns: sample sizes which have rawdata - a set if
                            absolute sizes or {network: {N1, N2, ...}, ...}
        :param str metric: metric being analysed
        :param dict rawdata: data keyed by series properties, network & N
        :param dict common: properties common to all series
        :param dict different: properties which differ between series
        :param dict params: command line parameters

        :returns tuple: (DataFrame: data required by plot function,
                         dict: chart properies specific to this chart)
    """
    xtick_labels = (FIGURE_PARAMS[params['fig']]['xaxis.tick_labels']
                    if 'fig' in params and params['fig'] in FIGURE_PARAMS
                    and 'xaxis.tick_labels' in FIGURE_PARAMS[params['fig']]
                    else None)

    diff_props = set(different[series[0]])
    print('\nDifferent properties are {}\n'.format(diff_props))

    if 'algorithm' not in diff_props and 'knowledge' not in diff_props:

        # looking at impact of sample size, ordering, score and
        # hyperparameters for a specific algorithm

        data, props = impact_vs_factor_plot_data(series, networks, metric,
                                                 rawdata, common)

    elif 'algorithm' not in diff_props and 'knowledge' in diff_props:

        # analyse effect of knowledge

        data, props = impact_vs_knowledge_plot_data(series, networks, Ns,
                                                    metric, rawdata, common,
                                                    different, params,
                                                    xtick_labels)

    elif diff_props == {'algorithm'} or diff_props == {'algorithm', 'package'}:

        # different algorithms but one ordering so compare each
        # algorithm with HC

        data, props = algo_vs_hc_plot_data(series, networks, metric, rawdata,
                                           common, different)

    elif diff_props == {'algorithm', 'ordering'}:

        # different algorithms and ordering so compare worst --> optimal
        # ordering impact for each algorithm

        data, props = sensitivity_vs_algo_plot_data(networks, metric, rawdata,
                                                    different)

    else:

        # only package is same, so analysing mean impact of sample size,
        # ordering and hyper-parameters across all algorithms

        data, props = impact_vs_algo_plot_data(networks, metric, rawdata,
                                               different)

    return (DataFrame(data), props)


def impact_analysis(series, networks, metrics, Ns, Ss, params, args,
                    expts_dir=EXPTS_DIR):
    """
        Analyse and plot impact of var. ordering, sample size, score and
        algorithm across a range of experiments

        :param list series: series required
        :param list networks: networks to include
        :param list metrics: metrics required
        :param list Ns: list of sample sizes to process
        :param tuple Ss: sub-samples to process
        :param dict params: command line parameters
        :param dicts args: raw command line args used to name plot file
        :param str expts_dir: location of experiments dir - use non-default
                              for testing

        :returns tuple: (data, props, stats) for module/system tests
    """

    # determine common and different properties in series

    common, different = compare_series_properties(series)
    print('\nCommon properties are {}, and different ones are {}\n'
          .format(list(common.keys()),
                  list(different[list(different.keys())[0]].keys())))

    # get all the raw data metrcs for all series, networks and sample sizes

    rawdata, Ns = _get_rawdata(series, networks, metrics, Ns, Ss, expts_dir)

    # plot chart for each metric requested

    for metric in metrics:

        # Get long-form data and chart properties required for this specific
        # chart

        data, _props = _get_plot_data(series, networks, Ns, metric, rawdata,
                                      common, different, params)
        if not len(data):
            print('\n*** No data to plot\n')
            break
        subplot_kind = 'bar' if 'y_var' in data.columns else 'box'

        # Compute some more statistics for each x_val in boxplot e.g. mean

        if subplot_kind in {'box', 'violin'}:

            stats = data.copy().groupby(['subplot', 'x_val']
                                        ).agg(['min', 'max', 'mean', 'std',
                                               'count'])
            stats = {level: {k: {x[1]: round(y, 3) for x, y in d.items()}
                             for k, d in stats.xs(level)
                             .to_dict('index').items()}
                     for level in stats.index.levels[0]}

        else:
            stats = None

        # Merge specific and general chart properties

        props = NETWORKS_GRID_DESIGN
        props.pop('xaxis.scale', None)  # unspecify so as to get categories
        props.update({'figure.per_row': 1,
                      'figure.title_fontsize': 14,
                      'figure.subplots_left': 0.12,
                      'figure.subplots_right': 0.95,
                      'figure.subplots_top': 0.8,
                      'figure.dpi': 150,
                      'subplot.kind': subplot_kind,
                      'subplot.title': '',
                      'yaxis.label': ('Resulting change in '
                                      + METRICS[metric]['label']),
                      'xaxis.ticks_fontsize': 6})
        props.update(_props)
        if 'fig' in params and params['fig'] in FIGURE_PARAMS:
            props.update(FIGURE_PARAMS[params['fig']])
        props.update(params)
        if subplot_kind == 'box':
            props.pop('legend.title', None)

        plot_file = (expts_dir + '/analysis/impact/'
                     + ((args['series']).replace('/', '_').replace(',', '_') +
                        '_' + metric if 'fig' not in params
                        else params['fig']) + '.png')

        if expts_dir == EXPTS_DIR:
            print('... generating plot file "{}"'.format(plot_file))
            relplot(data, props, plot_file, info=stats)

        return (data, props, stats)


def network_impact(networks=NETWORKS):  # analyse effect network properties
    """
        Summarise the effects of network properties on f1 and optimal sample
        size for the hill-climbing experiments

        :param list networks: networks to examine effect over
    """
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)

    bn_data = read_csv(EXPTS_DIR + '/analysis/bn/bn_analysis.csv',
                       index_col='network',
                       dtype={'n': int, '|A|': int, 'in-max': int})

    summary = []                        # tabular summary information
    data = []                           # long-form data for plots
    subplots = ['aligned', 'n', '|A|', 'deg-avg', 'in-avg',
                'in-max', 'mb-avg', 'card.avg', 'free.avg', 'opt.score',
                'k-l.avg', 'reversible']
    y_vars = ['opt.f1', 'std.f1', 'opt.N', 'std.N']

    for network in networks:
        print('Processing results for {} ...'.format(network))
        row = {'network': network}
        row.update(bn_data.loc[network].to_dict())
        for order in ['opt', 'std', 'bad']:
            if order == 'opt':
                series = 'HC_N_3'
            elif order == 'std':
                series = 'HC_N_1'
            else:
                series = 'HC_N_4'
            analysis = TraceAnalysis.select(series, network, bn_dir='/bn')
            for metric in ['N', 'f1', 'f1-e', 'shd-s', 'shd-e', 'score']:
                order_metric = order + '.' + metric
                row.update({order_metric: analysis.summary[metric]})
        row.update({'ratio.f1': (round(row['opt.f1'] / row['bad.f1'], 3)
                                 if row['bad.f1'] else None),
                    'ratio.f1-e': (round(row['opt.f1-e'] / row['bad.f1-e'], 3)
                                   if row['bad.f1-e'] else None)})
        summary.append(row)
        long = [{'network': network, 'subplot': subplot, 'x_val': row[subplot],
                'y_var': y_var, 'y_val': row[y_var]}
                for subplot in subplots for y_var in y_vars
                if subplot != 'aligned' or y_var not in ['opt.f1', 'opt.N']]
        data += long

    ints = {c: int for c in ['n', 'in-max', 'deg-max', 'mb-max', '|A|',
                             'opt.N', 'std.N', 'bad.N']}
    summary = DataFrame(summary).set_index('network').astype(ints)
    summary.to_csv(EXPTS_DIR + '/analysis/impact/networks.csv')
    print('\n\n{}'.format(summary))

    data = DataFrame(data)
    plot_file = EXPTS_DIR + '/analysis/impact/f1_networks.png'
    xaxis_labels = BN_PROPERTIES
    xaxis_labels.update({'opt.score': 'BIC score per row'})
    properties = {'xaxis.label': xaxis_labels,
                  'xaxis.shared': False,
                  'xaxis.scale': {'card.avg': 'log', 'free.avg': 'log',
                                  'free.max': 'log'},
                  'yaxis.label': 'F1',
                  'yaxis.range': (0, 1.05),
                  'figure.per_row': 4,
                  'figure.title': 'F1 against various network properties',
                  'figure.subplots_top': 0.9,
                  'figure.subplots_right': 0.88,
                  'figure.subplots_hspace': 0.25,
                  'legend.title': 'Variable ordering',
                  'legend.title_fontsize': 18,
                  'legend.fontsize': 12,
                  'legend.outside': True,
                  'legend.labels': {'std.f1': 'Standard ordering',
                                    'opt.f1': 'Optimal ordering'},
                  'subplot.kind': 'regression'}
    properties.update({'subplot.title': {p: 'F1 vs. ' + t.lower() for p, t in
                                         properties['xaxis.label'].items()}})

    relplot(data.loc[(data['y_var'] == 'opt.f1') |
                     (data['y_var'] == 'std.f1')], properties, plot_file)

    plot_file = EXPTS_DIR + '/analysis/impact/N_networks.png'
    del properties['yaxis.range']
    properties.update({'yaxis.label': 'Optimal sample size',
                       'yaxis.scale': 'log',
                       'figure.title': 'Optimal sample size (best F1) against '
                                       + 'various network properties',
                       'legend.labels': {'std.N': 'Standard ordering',
                                         'opt.N': 'Optimal ordering'},
                       'subplot.title': {p: 'Sample size vs. ' + t.lower()
                                         for p, t in
                                         properties['xaxis.label'].items()}})
    relplot(data.loc[(data['y_var'] == 'opt.N') |
                     (data['y_var'] == 'std.N')], properties, plot_file)
