
#   Plot experiment results and analyses

import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pandas import DataFrame

#   Matplotlib/Seaborn properties which can be modified in format
#   {bnbench property name: Matplotlib/Seaborn param name}

#   Individual properties set explicitly
#
#       figure.title
#       figure.title_fontsize
#       line.sizes
#       line.dashes
#       palette
#       subplot.kind
#       subplot.aspect


#   Properties set on an individual axis set

AXES_PROPS = {'xaxis.label': 'xlabel',
              'xaxis.range': 'xlim',
              'xaxis.scale': 'xscale',
              'xaxis.ticks': 'xticks',
              'yaxis.label': 'ylabel',
              'yaxis.range': 'ylim',
              'yaxis.scale': 'yscale',
              'yaxis.ticks': 'yticks',
              'subplot.title': 'title'}

#   Properties e.g. font sizes, set through rcParams
#   see https://github.com/mwaskom/seaborn/blob/master/seaborn/rcmod.py

CONTEXT_PROPS = {'legend.fontsize': 'legend.fontsize',
                 'legend.title_fontsize': 'legend.title_fontsize',
                 'subplot.title_fontsize': 'axes.titlesize',
                 'subplot.axes_fontsize': 'axes.labelsize',
                 'xaxis.ticks_fontsize': 'xtick.labelsize',
                 'yaxis.ticks_fontsize': 'ytick.labelsize',
                 'subplot.background': 'axes.facecolor',
                 'subplot.grid': 'axes.grid',
                 'subplot.grid_colour': 'grid.color',
                 'figure.background': 'figure.facecolor'}

SUBPLOT_ADJUST = {'figure.subplots_top': 'top',
                  'figure.subplots_left': 'left',
                  'figure.subplots_right': 'right',
                  'figure.subplots_bottom': 'bottom',
                  'figure.subplots_hspace': 'hspace',
                  'figure.subplots_wspace': 'wspace'}

FACET_PROPS = {'xaxis.shared': 'sharex',
               'yaxis.shared': 'sharey',
               'legend.outside': 'legend_out'}

VIOLIN_PROPS = {'violin.scale': 'scale',  # scale: area, count or width
                'violin.width': 'width'}  # absolute width of violin


def _set_axes_props(axes, properties, subplot=None):
    """
        Set up axes properties.

        :param Axes axes: axes to modify
        :param dict properties: axis properties required
        :param str subplot: identifies particular subplot axes belongs to
    """
    params = {}
    current = axes.properties()
    for key, param in AXES_PROPS.items():
        if key in properties:
            if not isinstance(properties[key], dict):
                params.update({param: properties[key]})
            elif subplot and subplot in properties[key]:
                params.update({param: properties[key][subplot]})
            else:
                params.update({param: current[param]})
    axes.set(**params)

    # Set some properties not supported by axes.set function

    # set custom x-axis tick labels
    if 'xaxis.tick_labels' in properties:
        axes.set_xticklabels(properties['xaxis.tick_labels'])

    # rotation of x-axis tick labels
    if 'xaxis.ticks_rotation' in properties:
        axes.set_xticklabels(axes.get_xticklabels(),
                             rotation=properties['xaxis.ticks_rotation'])

    # horizontal alignment of xaxis tick labels
    if 'xaxis.ticks_halign' in properties:
        axes.set_xticklabels(axes.get_xticklabels(),
                             horizontalalignment=properties['xaxis.ticks_'
                                                            + 'halign'])

    # invert y-axis - used that negative bars grow upwards
    if ('yaxis.invert' in properties
            and subplot in properties['yaxis.invert']
            and 'yaxis.range' in properties
            and subplot in properties['yaxis.range']):
        print("Inverting")
        for bar in axes.patches:
            bar.set_y(properties['yaxis.range'][subplot][0])
            bar.set_height(bar.get_height() -
                           properties['yaxis.range'][subplot][0])


def _report_boxplot_values(axes, info=None):
    """
        Report and plot the boxplot values - percentiles, whiskers and
        extremes.

        :param Axes axes: matplotlib axes object for a subplot
        :param dict info: optional info to add to box plots
    """
    PLOT_VALUES = ['p25', 'p75', 'lo_whisker', 'hi_whisker', 'p50']
    lines = axes.get_lines()
    x_labels = axes.get_xticklabels()

    data = []
    for x_idx in axes.get_xticks():
        x_label = x_labels[x_idx].get_text()
        print('Data for {} is {}'.format(x_label, info[x_label]))
        values = {key: round(list(lines[6 * x_idx + i].get_ydata())[0], 3)
                  for i, key in enumerate(PLOT_VALUES)}
        outliers = list(lines[6 * x_idx + 5].get_ydata())
        values.update({'min': (round(min(outliers), 3)
                               if len(outliers)
                               and min(outliers) < values['lo_whisker']
                               else None),
                       'max': (round(max(outliers), 3)
                               if len(outliers)
                               and max(outliers) > values['hi_whisker']
                               else None),
                       'comparing': x_label})
        data.append(values)

        axes.text(x_idx, info[x_label]['mean'],
                  '{}'.format(info[x_label]['mean']),
                  ha='center', va='center', fontweight='bold', size=9,
                  color='white',
                  bbox={'facecolor': '#445A64', 'pad': 1,
                        'alpha': 0.5})

    data = DataFrame(data)
    data = data[['comparing', 'min', 'lo_whisker', 'p25', 'p50', 'p75',
                 'hi_whisker', 'max']]
    print('\nBox plot values are:\n{}\n'.format(data))


def _plot_violin_means(axes, info, props):
    """
        Display the means on violin plots.

        :param Axes axes: matplotlib axes object for a subplot
        :param dict info: optional info to add to box plots
        :param dict props: customisable properties of chart
    """
    # PLOT_VALUES = ['p25', 'p75', 'lo_whisker', 'hi_whisker', 'p50']
    # lines = axes.get_lines()
    x_labels = axes.get_xticklabels()

    for x_idx in axes.get_xticks():
        x_label = x_labels[x_idx].get_text()
        print('Data for {} is {}'.format(x_label, info[x_label]))
        # values = {key: round(list(lines[6 * x_idx + i].get_ydata())[0], 3)
        #           for i, key in enumerate(PLOT_VALUES)}
        # outliers = list(lines[6 * x_idx + 5].get_ydata())

        size = props['violin.fontsize'] if 'violin.fontsize' in props else 9
        axes.text(x_idx, info[x_label]['mean'],
                  '{}'.format(info[x_label]['mean']),
                  ha='center', va='center', fontweight='bold', size=size,
                  color='white',
                  bbox={'facecolor': '#445A64', 'pad': 1,
                        'alpha': 0.5})


def relplot(data, props, plot_file, info=None):
    """
        Plot a set of relational charts

        :param DataFrame data: data in 'long form' with following columns
                               subplot - identifies subplot
                               x_val - x values
                               y_var - y-variable on each subplot
                               y_val - y values
        :param dict props: customisable properties of chart
        :param str plot_file: path of output plot file
        :param info dict: [optional] extra information to print on chart

        :raises ValueError: if unsupported subplot.kind requested
    """
    for p in sorted(props):
        print('{} = {}'.format(p, props[p]))

    # Get the unique y_var values

    if 'y_var' in data.columns:
        y_vars = list(data['y_var'].unique())
        print('Unique y_vars are: {}'.format(y_vars))

        sizes = (props['line.sizes'] if 'line.sizes' in props
                 else {y_var: 3 for y_var in y_vars})  # default width 3

        dashes = (props['line.dashes'] if 'line.dashes' in props
                  else {y_var: (1, 0) for y_var in y_vars})  # default solid

    palette = [tuple(int(h[i:i+2], 16) / 255 for i in (1, 3, 5))
               for h in props['palette']] if 'palette' in props else None

    facet_kws = {p: props[k] for k, p in FACET_PROPS.items() if k in props}

    col_wrap = props['figure.per_row'] if 'figure.per_row' in props else None

    plt.rcParams.update({p: props[k] for k, p in CONTEXT_PROPS.items()
                         if k in props})

    kind = props['subplot.kind'] if 'subplot.kind' in props else None
    aspect = props['subplot.aspect'] if 'subplot.aspect' in props else 1

    if kind == 'line':
        g = sns.relplot(data=data, x="x_val", y="y_val", hue='y_var',
                        kind=kind, sizes=sizes, col='subplot', size='y_var',
                        facet_kws=facet_kws, col_wrap=col_wrap, style='y_var',
                        palette=palette, dashes=dashes, aspect=aspect,
                        ci='sd')

    elif kind == 'regression':
        g = sns.lmplot(data=data, x='x_val', y='y_val', hue='y_var',
                       col='subplot', facet_kws=facet_kws, col_wrap=col_wrap,
                       palette=palette)

    elif kind == 'histogram':
        g = sns.displot(data=data, x="x_val", col='subplot', hue='y_var',
                        col_wrap=col_wrap, kind='kde', weights='weight')

    elif kind == 'box':
        g = sns.catplot(x="x_val", y="y_val", data=data, kind=kind,
                        col_wrap=col_wrap, col='subplot', aspect=aspect,
                        whis=[0, 100], palette=palette)

    elif kind == 'violin':
        kwargs = {VIOLIN_PROPS[arg]: value for arg, value in props.items()
                  if arg in VIOLIN_PROPS}
        print(kwargs)
        g = sns.catplot(x="x_val", y="y_val", data=data, kind=kind,
                        col_wrap=col_wrap, col='subplot', aspect=aspect,
                        cut=0, palette=palette, **kwargs)

    elif kind == 'bar':
        sharex = props['xaxis.shared'] if 'xaxis.shared' in props else True
        sharey = props['yaxis.shared'] if 'yaxis.shared' in props else True
        g = sns.catplot(x="x_val", y="y_val", data=data, col_wrap=col_wrap,
                        col='subplot', hue='y_var', kind=kind, aspect=aspect,
                        sharex=sharex, sharey=sharey)

    else:
        raise ValueError('relplot() bad arg values')

    # modify figure level properties

    if 'figure.title' in props:
        size = (props['figure.title_fontsize']
                if 'figure.title_fontsize' in props else 30)
        g.fig.suptitle(props['figure.title'], size=size)
    adjust = {p: props[k] for k, p in SUBPLOT_ADJUST.items() if k in props}
    if len(adjust):
        g.fig.subplots_adjust(**adjust)  # adjust the subplot area

    #   Modify the properties of the axes of each subplot

    title_pattern = re.compile(r'^subplot\s\=\s(.+)$')
    for axes in list(g.axes):
        subplot = title_pattern.match(axes.properties()['title']).group(1)
        print("Sub plot is {}".format(subplot))
        _set_axes_props(axes, props, subplot)
        if info is not None:
            print("\n\nInfo is {}\n\n".format(info[subplot]))
        if kind == 'box':
            _report_boxplot_values(axes, info=(None if info is None
                                               else info[subplot]))
        elif kind == 'violin' and info is not None:
            _plot_violin_means(axes, info[subplot], props)

    legend = g._legend

    # legend.key property used to manually define a legend

    if legend is None and 'legend.key' in props:
        loc = props['legend.loc'] if 'legend.loc' in props else None
        ncol = props['legend.ncol'] if 'legend.ncol' in props else 1
        artists = [Patch(fc=colour, ec='gray', label=key) for key, colour
                   in props['legend.key'].items()]
        legend = plt.legend(handles=artists, handlelength=1, ncol=ncol,
                            loc=loc)

    print('Legend is {}'.format(legend))
    if legend is not None and 'legend.title' in props:
        g._legend.set_title(props['legend.title'])
    if legend is not None and 'legend.labels' in props:
        for label in legend.texts:
            metric = label.get_text()
            if metric in props['legend.labels']:
                label.set_text(props['legend.labels'][metric])

    #   Save plot to a file

    dpi = props['figure.dpi'] if 'figure.dpi' in props else 80
    plt.savefig(plot_file, dpi=dpi)


def plot_scatter(data, props, plot_file):
    """
        Plot one or more scatter plots.

        :param DataFrame data: data in 'long form' with following columns
                               subplot - identifies subplot
                               x_val - x values
                               y_var - y-variable on each subplot
                               y_val - y values
        :param dict props: customisable properties of chart
        :param str plot_file: path of output plot file

        :raises ValueError: if data doesn't have required columns
    """
    # for p in sorted(props):
    #     print('{} = {}'.format(p, props[p]))

    # sizes = props['line.sizes'] if 'line.sizes' in props else None

    palette = [tuple(int(h[i:i+2], 16) / 255 for i in (1, 3, 5))
               for h in props['palette']] if 'palette' in props else None

    facet_kws = {p: props[k] for k, p in FACET_PROPS.items() if k in props}

    col_wrap = props['figure.per_row'] if 'figure.per_row' in props else None

    plt.rcParams.update({p: props[k] for k, p in CONTEXT_PROPS.items()
                         if k in props})

    g = sns.lmplot(data=data, x='x_val', y='y_val', hue='y_var', col='subplot',
                   facet_kws=facet_kws, col_wrap=col_wrap, palette=palette)

    # modify figure level properties

    if 'figure.title' in props:
        g.fig.suptitle(props['figure.title'], size=30)
    adjust = {p: props[k] for k, p in SUBPLOT_ADJUST.items() if k in props}
    if len(adjust):
        g.fig.subplots_adjust(**adjust)  # adjust the Figure

    #   Modify axes of each subplot

    title_pattern = re.compile(r'^subplot\s\=\s(.+)$')
    for axes in list(g.axes):
        subplot = title_pattern.match(axes.properties()['title']).group(1)
        _set_axes_props(axes, props, subplot)

    #   modify the legend

    legend = g._legend
    if 'legend.title' in props:
        g._legend.set_title(props['legend.title'], {'size': 18})
    if 'legend.labels' in props:
        for label in legend.texts:
            metric = label.get_text()
            if metric in props['legend.labels']:
                label.set_text(props['legend.labels'][metric])

    plt.savefig(plot_file, dpi=600)


def plot_degree_distribution(dist, plot_file):
    g = sns.FacetGrid(dist, col="network", hue="metric", col_wrap=5)
    g.map(sns.histplot, "value", discrete=True)
    g.add_legend()
    g.fig.subplots_adjust(top=0.9)  # adjust the Figure
    g.fig.suptitle('Node degree distributions for networks', size=30)
    g.set_axis_labels(x_var='Number of nodes')
    g.set_axis_labels(y_var='Node in-degree or total degree')
    legend = g.fig.get_children()[-1]
    for label in legend.texts:
        text = label.get_text()
        label.set_text('In-degree' if text == 'in' else 'Total degree')
    plt.savefig(plot_file, dpi=600)
