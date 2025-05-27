
# Generate the tables and charts for R2 of Tabu-Stable paper for IJAR 2025

from pandas import DataFrame, Categorical
from numpy import unique, isclose, arange
from numpy.random import choice, seed
from scipy.stats import entropy
import matplotlib.pyplot as plt

from experiments.run_analysis import run_analysis
from experiments.summary_analysis import summary_analysis
from experiments.plot import relplot
from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from learn.hc import hc
from learn.trace import Trace
from experiments.latex import to_table

# Categorical variable networks studied
CATEGORICAL = ('asia', 'sports', 'sachs', 'covid', 'child',
               'insurance', 'property', 'diarrhoea', 'water',
               'mildew', 'alarm', 'barley', 'hailfinder',
               'hepar2', 'win95pts', 'formed', 'pathfinder')

# Continuous variable networks
CONTINUOUS = ('sachs_c', 'covid_c', 'building_c', 'magic-niab_c',
              'magic-irri_c', 'ecoli70_c', 'arth150_c')

# Set of series investigating impact of different stable orders
ORDERS_SERIES = ('TABU/BASE3',
                 'TABU/STABLE3/DEC_SCORE',
                 'TABU/STABLE3/INC_SCORE',
                 'TABU/STABLE3/SCORE_PLUS',
                 'TABU/STABLE3/SC4_PLUS',
                 'TABU/STABLE3/SP_GREEDY',
                 'HC/BASE3',
                 'HC/STABLE3/SCORE_PLUS',
                 'HC/SCORE/REF')

ORDERS_SERIES_2 = ('TABU/BAD',
                   'TABU/BASE3',
                   'TABU/STABLE3/SCORE_PLUS',
                   'TABU/STABLE3/SC4_PLUS',
                   'TABU/STABLE3/SP_GREEDY',
                   'TABU/OPT')

SAMPLING_SERIES = ('TABU/STD',
                   'TABU/BASE3',
                   'TABU/STABLE3/SCORE_PLUS',
                   'TABU/STABLE3/SP_GREEDY',
                   'TABU/SAMPLE/STD',
                   'TABU/SAMPLE/BASE',
                   'TABU/SAMPLE/STABLE',
                   'TABU/SAMPLE/GREEDY'
                   )

# Comparing algorithms
ALGO_SERIES = {'TABU/STABLE3/SCORE_PLUS': 'Tabu-Stable',
               'TABU/STABLE3/SP_GREEDY': 'Tabu-StableAdapt',
               'HC/STABLE3/SCORE_PLUS': 'HC-Stable',
               'TABU/BASE3': 'Tabu',
               'HC/BASE3': 'HC',
               'TETRAD/FGES_BASE3': 'FGES',
               'BNLEARN/MMHC_BASE3': 'MMHC',
               'BNLEARN/H2PC_BASE3': 'H2PC',
               'BNLEARN/PC_BASE3': 'PC-Stable',
               'BNLEARN/GS_BASE3': 'GS',
               'BNLEARN/IIAMB_BASE3': 'Inter-IAMB'
               }

# Comparing algorithms with row sampling
ALGO_SAMPLE_SERIES = {'TABU/SAMPLE/STABLE': 'Tabu-Stable',
                      'TABU/SAMPLE/GREEDY': 'Tabu-StableAdapt',
                      'HC/SAMPLE/STABLE': 'HC-Stable',
                      'TABU/SAMPLE/BASE': 'Tabu',
                      'HC/SAMPLE/BASE': 'HC',
                      'TETRAD/FGES_BASE4': 'FGES',
                      'BNLEARN/MMHC_BASE4': 'MMHC',
                      'BNLEARN/H2PC_BASE4': 'H2PC',
                      'BNLEARN/PC_BASE4': 'PC-Stable',
                      'BNLEARN/GS_BASE4': 'GS',
                      'BNLEARN/IIAMB_BASE4': 'Inter-IAMB'
                      }

SAMPLE_SIZES = [100, 1000, 10000, 100000]

RANDOM = (0, 24)
# RANDOM = (0, 4)

SING_VAL = ['insurance@100', 'water@100', 'barley@100', 'hailfinder@100',
            'hailfinder2@100', 'win95pts@100', 'win95pts2@100',
            'formed@100', 'pathfinder@100']

IJAR_STAB_METRICS = ('expts,f1-e,f1-e-std,f1,bsf-e,score,score-std,' +
                     'time,p-e,r-e,loglik,loglik-std')

ORDERS_METRICS = ('expts', 'f1-e', 'f1-e-std', 'f1', 'bsf-e', 'score',
                  'score-std', 'time', 'p-e', 'r-e', 'loglik', 'loglik-std')

ALGO_BAR_PROPS = {  # Properties of the algorithm comparison bar plots
    'subplot.kind': 'bar',
    'figure.per_row': 2,
    'figure.dpi': 300,
    'xaxis.ticks_rotation': -60,
    'xaxis.label': 'Algorithm',
    'yaxis.label': {'f1-e': 'F1', 'f1-e-std': 'F1 S.D.',
                    'p-e': 'Precision', 'r-e': 'Recall',
                    'bsf-e': 'BSF', 'time': 'Time',
                    'score': 'Normalised BIC',
                    'score-std': 'Normalised BIC SD',
                    'loglik': 'Normalised log-lik.',
                    'loglik-std': 'Normalised log-lik. SD'},
    'subplot.title': {'f1-e': '(a) F1', 'f1-e-std': '(b) F1 SD',
                      'p-e': '(c) Precision', 'r-e': '(d) Recall',
                      'bsf-e': '(e) BSF', 'time': '(f) Time (seconds)',
                      'score': '(g) Normalised BIC',
                      'score-std': '(h) Normalised BIC SD',
                      'loglik': '(i) Normalised log-likelihood',
                      'loglik-std': '(j) Normalised log-likelihood SD'},
    'subplot.aspect': 2.0,
    'legend.title': ('Exclude\nidentical &\nsingle-valued\nvariables'),
    'figure.subplots_wspace': 0.3,
    'figure.subplots_hspace': 0.55,
    'figure.subplots_bottom': 0.06,
    'figure.subplots_left': 0.04,
    'figure.subplots_right': 0.92,
    'subplot.grid': True,
    'xaxis.shared': False,
    'yaxis.shared': False}


# helper functions for analysis

def _pivot(series, means, y_var, correct=True):
    """
        Pivot summary means data into long form required by relplot,
        and adjust F1 to account for number of experiments.

        :param dict series: {series id: algo name} for algos in comparison
        :param DataFrame means: metric mean values, columns are metrics
        :param str y_var: value for y_var column in long form
        :param bool correct: whether to correct metrics on basis of how
                             many experiments contributed to the value

    """
    means = means.rename(index=series).to_dict()
    means = {m: {a: vs[a] for a in series.values()}
             for m, vs in means.items()}
    if correct is True:
        scaling = {a: v / means['expts']['Tabu-Stable']
                   for a, v in means['expts'].items()}
        print('\nScaling for metrics: {}\n'.format(scaling))
    else:
        scaling = {a: 1.0 for a in means['expts']}
    means['f1-e'] = {a: v * scaling[a]
                     for a, v in means['f1-e'].items()}
    if 'p-e' in means:
        means['p-e'] = {a: v * scaling[a] for a, v in means['p-e'].items()}
    if 'r-e' in means:
        means['r-e'] = {a: v * scaling[a] for a, v in means['r-e'].items()}
    data = [{'subplot': m, 'x_val': a, 'y_val': v, 'y_var': y_var}
            for m, vs in means.items() for a, v in vs.items()
            if m in ['f1-e', 'f1-e-std', 'p-e', 'r-e', 'score', 'score-std',
                     'time', 'bsf-e', 'loglik', 'loglik-std']]
    return data


# Add BIC or loglik scores to existing FGES and bnlearn constraint-based
# algorithms

def values_ijar2_stab_score_graphs():
    """
        Adds score to FGES graph traces
    """
    print('\n')
    # SERIES = '/BNLEARN/PC_BASE3'
    # SERIES = '/BNLEARN/GS_BASE3'
    # SERIES = '/BNLEARN/IIAMB_BASE3'
    # networks = CONTINUOUS

    series = ['/HC/BASE3', 'HC/SCORE/EMPTY', 'HC/SCORE/REF',
              'HC/STABLE3/SCORE_PLUS', 'TABU/BASE3', 'TABU/STABLE3/DEC_SCORE',
              'TABU/STABLE3/INC_SCORE', 'TABU/STABLE3/SCORE_PLUS']

    series = ['TETRAD/FGES_BASE3',
              'BNLEARN/MMHC_BASE3',
              'BNLEARN/H2PC_BASE3',
              'BNLEARN/PC_BASE3',
              'BNLEARN/GS_BASE3',
              'BNLEARN/IIAMB_BASE3']

    series = ['HC/SAMPLE/STABLE']
    # compute loglik all the categorical networks for Tabu & HC

    # networks = list(CATEGORICAL) + ['hailfinder2', 'win95pts2']
    # for s in series:
    #     Trace.update_scores(s, networks, 'loglik')

    # compute loglik all the continuous networks for Tabu & HC

    networks = list(CONTINUOUS)
    for s in series:
        Trace.update_scores(s, networks, 'loglik')


# Draws sequence of arcs added when learning Asia from 10K rows.

def graph_stab_arbitrary_arcs():
    """
        Draws learnt DAG showing sequence of changes
        NOTE: this judges whether an arc is an "equivalent add" based on the
              true graph ... this is not the SAME as the arc being added in
              an arbitrary direction .. the relevant figure in the paper is
              constructued manually from the trace.
    """
    args = {'action': 'trace',
            'series': 'HC/STD',
            'networks': 'asia',
            'N': '10k;1;0',
            'file': None,
            'params': 'graph'}
    run_analysis(args)


# Leans Asia from 10K rows with a modified order that prevents some
# errors when default alphabetic order is used. Hence illustrates how errors
# propagate in the learning process.

def values_stab_hc_asia_10K():
    """
        Learn Asia 10K with HC with slightly modified order to show propogation
        of misorientations.
    """
    data = Pandas.read(EXPTS_DIR + '/datasets/asia.data.gz',
                       dstype='categorical', N=10000)

    # Switch order of 'lung' and 'either' so second arc added has correct
    # orientation.

    data.set_order(('asia', 'bronc', 'dysp', 'lung', 'either', 'smoke', 'tub',
                    'xray'))

    context = {'id': 'tabu_stable/alt_order', 'in': 'asia'}
    _, trace = hc(data=data, context=context)
    print(trace)


# Chart comparing f1-e with he different stability approaches with categorical
# data and BIC score

def chart_ijar2_stab_cat_f1():
    """
        Chart showing F1 against sample size for each stability approach and
        for each categorical variable network.
    """
    args = {'action': 'series',
            'series': (
                       'TABU/BASE3,' +
                       'TABU/STABLE3/DEC_SCORE,' +
                       'TABU/STABLE3/INC_SCORE,' +
                       'TABU/STABLE3/SCORE_PLUS'
                       ),
            'file': EXPTS_DIR + '/papers/ijar_stability/ijar_stab_cat_f1.png',
            'metrics': 'f1-e',
            'networks': ','.join(CATEGORICAL),
            'N': '100-100k',
            'params': ('fig:ijar_stab_cat_f1;' +
                       'figure.title;' +
                       'legend.fontsize:24;' +
                       'xaxis.ticks_fontsize:24;' +
                       'yaxis.ticks_fontsize:24;' +
                       'subplot.axes_fontsize:24;' +
                       'subplot.title_fontsize:26;' +
                       'subplot.title:{' +
                       ','.join([n + ',' + n
                                 for n in CATEGORICAL]) + '};' +
                       'legend.labels:{' +
                       'TABU/BASE3,Standard unstable Tabu \n' +
                       'using variable order\n,' +
                       'TABU/STABLE3/DEC_SCORE,Tabu using' +
                       '\ndecreasing score order\n,' +
                       'TABU/STABLE3/INC_SCORE,Tabu using' +
                       '\nincreasing score order\n,' +
                       'TABU/STABLE3/SCORE_PLUS,Tabu-Stable' +
                       '};' +
                       'subplot.aspect:1.70;' +
                       'legend.box:True;' +
                       'palette:(#55a868,#CC0000,#0000CC,#000000);' +
                       'figure.per_row:3;' +
                       'figure.subplots_left:0.05;' +
                       'figure.subplots_right:0.81;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.24;' +
                       'figure.subplots_wspace:0.20')}
    run_analysis(args)


# Chart comparing f1-e with the different stability approaches with continuous
# data and BIC score

def chart_ijar2_stab_con_f1():
    """
        Chart showing F1 against sample size for each stability approach and
        for each continuous variable network.
    """
    args = {'action': 'series',
            'series': (
                       'TABU/BASE3,' +
                       'TABU/STABLE3/DEC_SCORE,' +
                       'TABU/STABLE3/INC_SCORE,' +
                       'TABU/STABLE3/SCORE_PLUS'
                       ),
            'file': EXPTS_DIR + '/papers/ijar_stability/ijar_stab_con_f1.png',
            'metrics': 'f1-e',
            'networks': ','.join(CONTINUOUS),
            'N': '100-100k',
            'params': ('fig:tabu_stab_con_f1;' +
                       'figure.title;' +
                       'legend.fontsize:24;' +
                       'xaxis.ticks_fontsize:24;' +
                       'yaxis.ticks_fontsize:24;' +
                       'subplot.axes_fontsize:24;' +
                       'subplot.title_fontsize:26;' +
                       'subplot.title:{' +
                       ','.join([n + ',' + n
                                 for n in CONTINUOUS]) + '};' +
                       'legend.labels:{' +
                       'TABU/BASE3,Tabu using\nvariable order\n,' +
                       'TABU/STABLE3/DEC_SCORE,Tabu using' +
                       '\ndecreasing score order\n,' +
                       'TABU/STABLE3/INC_SCORE,Tabu using' +
                       '\nincreasing score order\n,' +
                       'TABU/STABLE3/SCORE_PLUS,Tabu-Stable' +
                       '};' +
                       'subplot.aspect:1.70;' +
                       'legend.box:True;' +
                       'palette:(#55a868,#CC0000,#0000CC,#000000);' +
                       'figure.per_row:3;' +
                       'figure.subplots_left:0.05;' +
                       'figure.subplots_right:0.81;' +
                       'figure.subplots_top:0.96;' +
                       'figure.subplots_hspace:0.24;' +
                       'figure.subplots_wspace:0.20')}
    run_analysis(args)


def value_by_series_and_network(details: DataFrame, reqd_value: str,
                                networks: tuple):
    """
        Generate DataFrame of specified value broken down by series and
        network.

        :param DataFrane details: metrics and scores for all networks and
                                  series
        :param str reqd_value: value required
        :param tuple networks: networks in order required

        :returns DataFrame: row, columns are networks, series
    """

    # remove series and metrics not required
    reqd_series = [s for s in ORDERS_SERIES if not s.endswith('_SCORE')]
    table = details[(details['metric'] == reqd_value) &
                    (details['series'].isin(reqd_series))]

    # remove metric column and then pivot so columns are series
    table = table.drop(columns=['metric'])
    table = table.pivot(index='network', columns='series', values='value')
    table = table[reqd_series].reset_index()

    # ensure rows in required network order
    table['network'] = Categorical(table['network'], categories=networks,
                                   ordered=True)
    table = table.sort_values('network')

    return table


# Table showing results from different stability approaches for categorical
# and continuous data using BIC and BDeu scores
def table_ijar2_stab_ord_cat_bic():
    """
        Table summarising HC/Tabu stability approaches - categorical
    """
    means, details = summary_analysis(series=ORDERS_SERIES,
                                      networks=CATEGORICAL,
                                      Ns=SAMPLE_SIZES,
                                      Ss=RANDOM,
                                      metrics=ORDERS_METRICS,
                                      params={},
                                      maxtime=180)

    # Generate tables with summary results for each metric and series
    means = means.T.loc[['p-e', 'r-e', 'f1-e', 'f1-e-std', 'bsf-e', 'score',
                         'score-std', 'loglik', 'loglik-std']]
    means = means.reset_index().rename(columns={'index': 'Metric'})
    print(to_table(df=means.drop(columns=['TABU/STABLE3/SC4_PLUS',
                                          'TABU/STABLE3/SP_GREEDY',
                                          'HC/SCORE/REF']),
                   options={'decimals': 4, 'label': 'tab:?',
                            'caption': 'Categ. order comparison'}))
    print(to_table(df=means.drop(columns=['TABU/STABLE3/INC_SCORE',
                                          'TABU/STABLE3/DEC_SCORE',
                                          'HC/BASE3',
                                          'HC/STABLE3/SCORE_PLUS',
                                          'HC/SCORE/REF']),
                   options={'decimals': 4, 'label': 'tab:?',
                            'caption': 'Categ. adv order comparison'}))

    # Generate tables of BIC & loglik scores by series and network
    networks = CATEGORICAL
    print(to_table(value_by_series_and_network(details, 'f1-e', networks),
                   {'label': '?', 'decimals': 4,
                    'caption': 'F1 by categorical network and series'}))
    print(to_table(value_by_series_and_network(details, 'score', networks),
                   {'label': '?', 'decimals': 4,
                    'caption': 'BIC by categorical network and series'}))
    print(to_table(value_by_series_and_network(details, 'loglik', networks),
                   {'label': '?', 'decimals': 4,
                    'caption': 'Loglik by categorical network and series'}))


def table_ijar2_stab_ord2_cat_bic():
    """
        Table summarising HC/Tabu stability approaches - categorical
    """
    means, details = summary_analysis(series=ORDERS_SERIES_2,
                                      networks=CATEGORICAL,
                                      Ns=SAMPLE_SIZES,
                                      Ss=RANDOM,
                                      metrics=ORDERS_METRICS,
                                      params={},
                                      maxtime=180)

    # Generate Table with mean results for each metric and series
    means = means.T[list(ORDERS_SERIES_2)]
    means = means.loc[['p-e', 'r-e', 'f1-e', 'f1-e-std', 'bsf-e', 'score',
                       'score-std', 'loglik', 'loglik-std']]
    means = means.reset_index().rename(columns={'index': 'Metric'})
    print(to_table(df=means,
                   options={'decimals': 4, 'label': 'tab:?',
                            'caption': 'Categ. scores/metrics by series'}))


def table_ijar2_stab_ord_con_bic():
    """
        Table summarising HC/Tabu stability approaches - continuous
    """
    means, details = summary_analysis(series=ORDERS_SERIES,
                                      networks=CONTINUOUS,
                                      Ns=SAMPLE_SIZES,
                                      Ss=RANDOM,
                                      metrics=ORDERS_METRICS,
                                      params={},
                                      maxtime=180)

    # Generate tables with summary results for each metric and series
    means = means.T.loc[['p-e', 'r-e', 'f1-e', 'f1-e-std', 'bsf-e', 'score',
                         'score-std', 'loglik', 'loglik-std']]
    means = means.reset_index().rename(columns={'index': 'Metric'})
    print(to_table(df=means.drop(columns=['TABU/STABLE3/SC4_PLUS',
                                          'TABU/STABLE3/SP_GREEDY',
                                          'HC/SCORE/REF']),
                   options={'decimals': 4, 'label': 'tab:?',
                            'caption': 'Categ. order comparison'}))
    print(to_table(df=means.drop(columns=['TABU/STABLE3/INC_SCORE',
                                          'TABU/STABLE3/DEC_SCORE',
                                          'HC/BASE3',
                                          'HC/STABLE3/SCORE_PLUS',
                                          'HC/SCORE/REF']),
                   options={'decimals': 4, 'label': 'tab:?',
                            'caption': 'Categ. adv order comparison'}))

    # Generate tables of BIC & loglik scores by series and network
    networks = CONTINUOUS
    print(to_table(value_by_series_and_network(details, 'f1-e', networks),
                   {'label': '?', 'decimals': 4,
                    'caption': 'F1 by continuous network and series'}))
    print(to_table(value_by_series_and_network(details, 'score', networks),
                   {'label': '?', 'decimals': 4,
                    'caption': 'BIC by continuous network and series'}))
    print(to_table(value_by_series_and_network(details, 'loglik', networks),
                   {'label': '?', 'decimals': 4,
                    'caption': 'Loglik by continuous network and series'}))


def table_ijar2_stab_ord2_con_bic():
    """
        Table summarising HC/Tabu stability approaches - continuous
    """
    means, details = summary_analysis(series=ORDERS_SERIES_2,
                                      networks=CONTINUOUS,
                                      Ns=SAMPLE_SIZES,
                                      Ss=RANDOM,
                                      metrics=ORDERS_METRICS,
                                      params={},
                                      maxtime=180)

    # Generate Table with mean results for each metric and series
    means = means.T[list(ORDERS_SERIES_2)]
    means = means.loc[['p-e', 'r-e', 'f1-e', 'f1-e-std', 'bsf-e', 'score',
                       'score-std', 'loglik', 'loglik-std']]
    means = means.reset_index().rename(columns={'index': 'Metric'})
    print(to_table(df=means,
                   options={'decimals': 4, 'label': 'tab:?',
                            'caption': 'Cont. scores/metrics by series'}))


def table_ijar2_stab_residuals():
    """
        F1 S.D. for each network for each sample size
    """
    Ns = [100, 1000, 10000, 100000]
    Ss = (0, 24)
    metrics = ['f1-e', 'f1-e-std', 'expts']
    networks = CATEGORICAL + ['hailfinder2', 'win95pts2']

    for N in Ns:
        print('\n\n*** RESULTS FOR N={} ***\n'.format(N))
        summary_analysis(series=['TABU/STABLE3/SCORE_PLUS'], networks=networks,
                         Ns=[N], Ss=Ss, metrics=metrics, params={})[0]


# Generate charts which compare algorithms

def chart_ijar2_stab_algos_cat_bic():
    """
        Algorithm comparson for categorical data learnt using BIC score using
        modified Hailfinder and Pathfinder networks.
        Experiments with single-valued datasets ignored, and missing
        metric values NOT imputed.
    """
    metrics = ['f1-e', 'f1-e-std', 'p-e', 'r-e', 'bsf-e', 'time',
               'score', 'score-std', 'loglik', 'loglik-std', 'nonex', 'expts',
               'dens', 'dens-std', 'n', '|E|', '|A|']

    # categorical, & replace hailfinder & win95pts with modified versions
    networks = list(CATEGORICAL)
    networks[networks.index('hailfinder')] = 'hailfinder2'
    networks[networks.index('win95pts')] = 'win95pts2'

    means = summary_analysis(series=list(ALGO_SERIES),
                             networks=networks, Ns=SAMPLE_SIZES,
                             Ss=RANDOM, metrics=metrics, maxtime=180,
                             params={'ignore': SING_VAL})[0]
    data = DataFrame(_pivot(ALGO_SERIES, means, 'no', False))

    props = ALGO_BAR_PROPS.copy()
    props.update({'xaxis.ticks_fontsize': 14,
                  'yaxis.ticks_fontsize': 14,
                  'subplot.title_fontsize': 16,
                  'subplot.axes_fontsize': 14,
                  'figure.subplots_hspace': 0.8,
                  'figure.subplots_left': 0.06,
                  'figure.subplots_bottom': 0.08,
                  'yaxis.invert': {'score', 'loglik'},
                  'yaxis.range': {'f1-e': (0.2, 0.6),
                                  'p-e': (0.3, 0.6),
                                  'r-e': (0.1, 0.6),
                                  'bsf-e': (0.2, 0.7),
                                  'score':  (-31.0, -24.0),
                                  'loglik':  (-31.0, -22.0)
                                  }})
    print(data)
    relplot(data=data, props=props,
            plot_file=(EXPTS_DIR +
                       '/papers/ijar_stability/algos-cat-bic.png'))


# Algorithm comparison for continuous networks

def chart_ijar2_stab_algos_con_bic():
    """
        Comparsion of different algorithms with continuous data and using
        BIC score.
    """
    metrics = ['f1-e', 'f1-e-std', 'p-e', 'r-e', 'bsf-e', 'time',
               'score', 'score-std', 'loglik', 'loglik-std', 'nonex', 'expts',
               'dens', 'dens-std', 'n', '|E|', '|A|']
    means = summary_analysis(series=list(ALGO_SERIES),
                             networks=CONTINUOUS,
                             Ns=SAMPLE_SIZES,
                             Ss=RANDOM, metrics=metrics, maxtime=180,
                             params={'ignore': ['arth150_c@100000']})[0]
    data = DataFrame(_pivot(ALGO_SERIES, means, 'no', False))
    props = ALGO_BAR_PROPS.copy()
    props.update({'xaxis.ticks_fontsize': 14,
                  'yaxis.ticks_fontsize': 14,
                  'subplot.title_fontsize': 16,
                  'subplot.axes_fontsize': 14,
                  'figure.subplots_hspace': 0.8,
                  'figure.subplots_left': 0.06,
                  'figure.subplots_bottom': 0.08,
                  'yaxis.invert': {'score', 'loglik'},
                  'yaxis.range': {'f1-e': (0.4, 0.7),
                                  'p-e': (0.4, 0.8),
                                  'r-e': (0.3, 0.7),
                                  'bsf-e': (0.5, 0.8),
                                  'score': (-59.0, -52.0),
                                  'loglik': (-56.0, -51.0)}})

    print(data)
    relplot(data=data, props=props, plot_file=EXPTS_DIR +
            '/papers/ijar_stability/algos-con-bic.png')


def values_ijar2_stab_entropy_ties():

    def _entropy(column):
        _, counts = unique(column, return_counts=True)
        probs = counts / len(column)
        return entropy(probs, base=2)

    def _entropy_tie(N):
        col1 = choice(['0', '1'], size=N, p=[1/2, 1/2])
        col2 = choice(['0', '1'], size=N, p=[26/50, 24/50])
        return isclose(_entropy(col1), _entropy(col2), atol=10e-10)

    def _prob_tie(N, num_trials):
        ties = 0
        for _ in range(num_trials):
            if _entropy_tie(N):
                ties += 1
        return ties / num_trials

    seed(42)

    Ns = arange(10, 1010, 10)
    probs = [_prob_tie(N, 5000) for N in Ns]
    print(probs)

    plt.figure(figsize=(8, 5))
    plt.plot(Ns, probs, marker='o', linestyle='-',
             label='Empirical Tie Probability')
    plt.xlabel('Number of Rows (n)')
    plt.ylabel('Probability of Entropy Tie')
    plt.title('Decay of Tie Probability with Increasing n')
    plt.legend()
    plt.grid()
    plt.show()


# Table showing results from different row sampling for categorical networks
def table_ijar2_sampling_cat():
    """
        Table summarising Tabu categorical sampling results
    """
    # categorical, & replace hailfinder & win95pts with modified versions
    networks = list(CATEGORICAL)
    networks[networks.index('hailfinder')] = 'hailfinder2'
    networks[networks.index('win95pts')] = 'win95pts2'

    means, _ = summary_analysis(series=SAMPLING_SERIES,
                                networks=networks,
                                Ns=SAMPLE_SIZES,
                                Ss=RANDOM,
                                metrics=ORDERS_METRICS,
                                params={'ignore': SING_VAL},
                                maxtime=180)

    # Generate tables with summary results for each metric and series
    means = means[['f1-e', 'f1-e-std', 'score', 'score-std',
                   'loglik', 'loglik-std']]
    means = means.reset_index().rename(columns={'index': 'Algo/rando'})
    print(to_table(df=means, options={'decimals': 4, 'label': 'tab:?',
                                      'caption': 'Cat. sampling'}))


# Table showing results from different row sampling for continuous networks
def table_ijar2_sampling_con():
    """
        Table summarising Tabu categorical sampling results
    """
    means, _ = summary_analysis(series=SAMPLING_SERIES,
                                networks=CONTINUOUS,
                                Ns=SAMPLE_SIZES,
                                Ss=RANDOM,
                                metrics=ORDERS_METRICS,
                                params={'ignore': SING_VAL},
                                maxtime=180)

    # Generate tables with summary results for each metric and series
    means = means[['f1-e', 'f1-e-std', 'score', 'score-std',
                   'loglik', 'loglik-std']]
    means = means.reset_index().rename(columns={'index': 'Algo/rando'})
    print(to_table(df=means, options={'decimals': 4, 'label': 'tab:?',
                                      'caption': 'Con. sampling'}))


def chart_ijar2_stab_algos_cat_sampling():
    """
        Algorithm comparson for categorical data learnt using BIC score using
        modified Hailfinder and Pathfinder networks and row sampling.
        Experiments with single-valued datasets ignored, and missing
        metric values NOT imputed.
    """
    metrics = ['f1-e', 'f1-e-std', 'p-e', 'r-e', 'bsf-e', 'time',
               'score', 'score-std', 'loglik', 'loglik-std', 'nonex', 'expts',
               'dens', 'dens-std', 'n', '|E|', '|A|']

    # categorical, & replace hailfinder & win95pts with modified versions
    networks = list(CATEGORICAL)
    networks[networks.index('hailfinder')] = 'hailfinder2'
    networks[networks.index('win95pts')] = 'win95pts2'

    means = summary_analysis(series=list(ALGO_SAMPLE_SERIES),
                             networks=networks, Ns=SAMPLE_SIZES,
                             Ss=RANDOM, metrics=metrics, maxtime=180,
                             params={'ignore': SING_VAL})[0]
    data = DataFrame(_pivot(ALGO_SAMPLE_SERIES, means, 'no', False))

    # remove metrics not interested in
    data = data[~data['subplot'].isin(['p-e', 'r-e', 'bsf-e', 'time'])]

    props = ALGO_BAR_PROPS.copy()
    props.update({'xaxis.ticks_fontsize': 14,
                  'yaxis.ticks_fontsize': 14,
                  'subplot.title_fontsize': 16,
                  'subplot.axes_fontsize': 14,
                  'figure.subplots_hspace': 0.8,
                  'figure.subplots_left': 0.06,
                  'figure.subplots_bottom': 0.08,
                  'subplot.title': {'f1-e': '(a) F1',
                                    'f1-e-std': '(b) F1 SD',
                                    'score': '(c) Normalised BIC',
                                    'score-std': '(d) Normalised BIC SD',
                                    'loglik': '(e) Normalised log-likelihood',
                                    'loglik-std': ('(f) Normalised ' +
                                                   'log-likelihood SD')},
                  'yaxis.invert': {'score', 'loglik'},
                  'yaxis.range': {'f1-e': (0.2, 0.6),
                                  'score':  (-31.0, -24.0),
                                  'loglik':  (-31.0, -22.0)
                                  }})
    print(data)
    relplot(data=data, props=props,
            plot_file=(EXPTS_DIR +
                       '/papers/ijar_stability/algos-cat-sampling.png'))


def chart_ijar2_stab_algos_con_sampling():
    """
        Algorithm comparson for categorical data learnt using BIC score
        and row sampling.
        Experiments with single-valued datasets ignored, and missing
        metric values NOT imputed.
    """
    metrics = ['f1-e', 'f1-e-std', 'p-e', 'r-e', 'bsf-e', 'time',
               'score', 'score-std', 'loglik', 'loglik-std', 'nonex', 'expts',
               'dens', 'dens-std', 'n', '|E|', '|A|']

    means = summary_analysis(series=list(ALGO_SAMPLE_SERIES),
                             networks=list(CONTINUOUS), Ns=SAMPLE_SIZES,
                             Ss=RANDOM, metrics=metrics, maxtime=180,
                             params={'ignore': SING_VAL})[0]
    data = DataFrame(_pivot(ALGO_SAMPLE_SERIES, means, 'no', False))

    # remove metrics not interested in
    data = data[~data['subplot'].isin(['p-e', 'r-e', 'bsf-e', 'time'])]

    props = ALGO_BAR_PROPS.copy()
    props.update({'xaxis.ticks_fontsize': 14,
                  'yaxis.ticks_fontsize': 14,
                  'subplot.title_fontsize': 16,
                  'subplot.axes_fontsize': 14,
                  'figure.subplots_hspace': 0.8,
                  'figure.subplots_left': 0.06,
                  'figure.subplots_bottom': 0.08,
                  'subplot.title': {'f1-e': '(a) F1',
                                    'f1-e-std': '(b) F1 SD',
                                    'score': '(c) Normalised BIC',
                                    'score-std': '(d) Normalised BIC SD',
                                    'loglik': '(e) Normalised log-likelihood',
                                    'loglik-std': ('(f) Normalised ' +
                                                   'log-likelihood SD')},
                  'yaxis.invert': {'score', 'loglik'},
                  'yaxis.range': {'f1-e': (0.2, 0.6),
                                  'score':  (-60.0, -52.0),
                                  'loglik':  (-58.0, -51.0)
                                  }})
    print(data)
    relplot(data=data, props=props,
            plot_file=(EXPTS_DIR +
                       '/papers/ijar_stability/algos-con-sampling.png'))
