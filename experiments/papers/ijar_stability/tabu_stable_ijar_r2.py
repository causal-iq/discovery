
# Generate the tables and charts for R2 of Tabu-Stable paper for IJAR 2025

from pandas import DataFrame
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


CATEGORICAL = ('asia,sports,sachs,covid,child,' +
               'insurance,property,diarrhoea,water,' +
               'mildew,alarm,barley,hailfinder,' +
               'hepar2,win95pts,formed,pathfinder')

CONTINUOUS = ('sachs_c,covid_c,building_c,magic-niab_c,magic-irri_c,' +
              'ecoli70_c,arth150_c')

SERIES2ALGO = {'TABU/STABLE3/SCORE_PLUS': 'Tabu-Stable',
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


BDEU2ALGO = {'TABU/STABLE3/BDEU_PLUS': 'Tabu-Stable',
             'HC/STABLE3/SCORE_PLUS': 'HC-Stable',
             'TABU/STABLE3/BDEU_BASE': 'Tabu',
             'HC/STABLE3/BDEU_BASE': 'HC',
             'TETRAD/FGES_BDEU': 'FGES'
             }

SAMPLE_SIZES = [100, 1000, 10000, 100000]

RANDOM = (0, 24)
# RANDOM = (0, 2)

SING_VAL = ['insurance@100', 'water@100', 'barley@100', 'hailfinder@100',
            'hailfinder2@100', 'win95pts@100', 'win95pts2@100',
            'formed@100', 'pathfinder@100']

IJAR_STAB_METRICS = ('expts,f1-e,f1-e-std,f1,bsf-e,score,score-std,' +
                     'time,p-e,r-e,loglik,loglik-std')

ALGO_BAR_PROPS = {  # Properties of the algorithm comparison bar plots
    'subplot.kind': 'bar',
    'figure.per_row': 2,
    'figure.dpi': 300,
    'xaxis.ticks_rotation': -60,
    'xaxis.label': 'Algorithm',
    'yaxis.label': {'f1-e': 'F1', 'f1-e-std': 'F1 S.D.',
                    'p-e': 'Precision', 'r-e': 'Recall',
                    'loglik': 'Normalised Log Likelihood',
                    'loglik-std': 'Normalised Log Likelihood SD',
                    'bsf-e': 'BSF', 'time': 'Time'},
    'subplot.title': {'f1-e': '(a) F1', 'f1-e-std': '(b) F1 SD',
                      'p-e': '(c) Precision', 'r-e': '(d) Recall',
                      'loglik': '(e) Normalised Log Likelihood',
                      'loglik-std': '(f) Normalised Log Likelihood SD',
                      'bsf-e': '(g) BSF', 'time': '(h) Time (seconds)'},
    'subplot.aspect': 1.6,
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

def _fges_correct(metric, data, others, failure_rate):
    """
        Replace zero for FGES failed experiments with the average over all
        other algorithms to prevent bias against FGES.

        :param str metric: metric to correct e.g. f1-e
        :param DataFrame data: metrics over all algorithms
        :param DataFrame others: metrics over all algorithms except FGES
        :param float failure_rate: fraction of FGES experiments that failed
    """
    o_mean = others[others['subplot'] == metric]['y_val'].mean()

    # get position of FGES value and hences its value

    fges_idx = data.index[(data['subplot'] == metric)
                          & (data['x_val'] == 'FGES')].tolist()[0]
    fges_val = data.at[fges_idx, 'y_val']

    data.at[fges_idx, 'y_val'] = (o_mean * failure_rate +
                                  fges_val * (1 - failure_rate))
    print('FGES {} corrected from {:.4f} --> {:.4f}'
          .format(metric, fges_val, data.at[fges_idx, 'y_val']))

    return data


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
            if m in ['f1-e', 'f1-e-std', 'p-e', 'r-e', 'loglik', 'loglik-std',
                     'time', 'bsf-e']]
    return data


def _pgm_fges_correct(series, metrics):
    """
        Corrects FGES accuracy metric values using the approach employed in
        the PGM paper. In this approach, an overall mean metric value is
        computed WITHOUT weighting each network's average according to the
        number of sample sizes utilised. This overall average is then
        corrected using the mean metric for Hailfinder2 and Pathfinder for
        10K and 100K (i.e. FGES failure cases) over all algorithms except FGES.
    """
    Ns = [100, 1000, 10000, 100000]
    Ss = (0, 24)
    SING_VAL = ['insurance@100', 'water@100', 'barley@100', 'hailfinder@100',
                'hailfinder2@100', 'win95pts@100', 'win95pts2@100',
                'formed@100', 'pathfinder@100']

    # Get summary EXcluding networks with identical variables for all
    # algorithms

    networks = CATEGORICAL.replace('lfinder',
                                   'lfinder2').replace('pts', 'pts2')

    means = summary_analysis(series=list(series),
                             networks=networks.split(','), Ns=Ns, Ss=Ss,
                             metrics=metrics, params={'ignore': SING_VAL})
    print(means)
    data = DataFrame(_pivot(series, means, 'no', False))
    print(data)

    # Replace FGES failure cases for hailfinder and pathfinder with
    # averages over all other algorithms for those cases. 10K and 100K
    # for hailfinder2 and pathfinder failed with BIC score. i.e. 4/68
    # cases in all. For BDeu it was 1K, 10K and 100K, for hailfinder2
    # and pathfinder, i.e. 6/68 cases.

    others = series.copy()
    others.pop('TETRAD/FGES_BASE3', None)
    others.pop('TETRAD/FGES_BDEU', None)
    failure_rate = 6 / 68 if 'TETRAD/FGES_BDEU' in series else 4 / 68
    failure_Ns = ([1000, 10000, 100000] if 'TETRAD/FGES_BDEU' in series
                  else [10000, 100000])
    means = summary_analysis(series=list(others),
                             networks=['hailfinder2', 'pathfinder'],
                             Ns=failure_Ns, Ss=Ss,
                             metrics=metrics, params={'ignore': SING_VAL})
    others = DataFrame(_pivot(others, means, 'no'))
    for metric in set(metrics) & {'f1-e', 'p-e', 'r-e', 'bsf-e', 'score'}:
        data = _fges_correct(metric, data, others, failure_rate)

    return data


# Add BIC or loglik scores to existing FGES and bnlearn constraint-based
# algorithms

def values_ijar_stab_score_graphs():
    """
        Adds score to FGES graph traces
    """
    print('\n')
    # SERIES = '/BNLEARN/PC_BASE3'
    # SERIES = '/BNLEARN/GS_BASE3'
    # SERIES = '/BNLEARN/IIAMB_BASE3'
    # networks = CONTINUOUS.split(',')

    series = ['/HC/BASE3', 'HC/SCORE/EMPTY', 'HC/SCORE/REF',
              'HC/STABLE3/SCORE_PLUS', 'TABU/BASE3', 'TABU/STABLE3/DEC_SCORE',
              'TABU/STABLE3/INC_SCORE', 'TABU/STABLE3/SCORE_PLUS']

    series = ['TETRAD/FGES_BASE3',
              'BNLEARN/MMHC_BASE3',
              'BNLEARN/H2PC_BASE3',
              'BNLEARN/PC_BASE3',
              'BNLEARN/GS_BASE3',
              'BNLEARN/IIAMB_BASE3']

    # compute loglik all the categorical networks for Tabu & HC

    networks = (CATEGORICAL + ',hailfinder2,win95pts2').split(',')
    for s in series:
        Trace.update_scores(s, networks, 'loglik')

    # compute loglik all the continuous networks for Tabu & HC

    networks = CONTINUOUS.split(',')
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

def chart_ijar_stab_cat_f1():
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
            'networks': CATEGORICAL,
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
                                 for n in CATEGORICAL.split(',')]) + '};' +
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

def chart_ijar_stab_con_f1():
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
            'networks': CONTINUOUS,
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
                                 for n in CONTINUOUS.split(',')]) + '};' +
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
    run_analysis(args)


# Table showing results from different stability approaches for categorical
# and continuous data using BIC and BDeu scores

def table_ijar_stab_ord_cat_bic():
    """
        Table summarising HC/Tabu stability approaches - categorical
    """
    # CATEGORICAL = ('asia,sports,sachs,covid,child,insurance,property,' +
    #                'diarrhoea,water')  # smaller networks
    # CATEGORICAL = ('mildew,alarm,barley,hailfinder,hepar2,win95pts,' +
    #                'formed,pathfinder,gaming')  # larger networks
    SERIES = ('TABU/BASE3,' +
              'TABU/STABLE3/DEC_SCORE,' +
              'TABU/STABLE3/INC_SCORE,' +
              'TABU/STABLE3/SCORE_PLUS,' +
              'HC/BASE3,' +
              'HC/STABLE3/SCORE_PLUS,' +
              'HC/SCORE/REF,' +
              'HC/SCORE/EMPTY'
              )
    args = {'action': 'summary',
            'series': SERIES,
            'networks': CATEGORICAL,
            'N': '100-100k;1;0-24',
            'metrics': IJAR_STAB_METRICS,
            'maxtime': '180',
            'file': None}
    run_analysis(args)


def table_ijar_stab_ord_con_bic():
    """
        Table summarising HC/Tabu stability approaches - continuous
    """
    SERIES = ('TABU/BASE3,' +
              'TABU/STABLE3/DEC_SCORE,' +
              'TABU/STABLE3/INC_SCORE,' +
              'TABU/STABLE3/SCORE_PLUS,' +
              'HC/BASE3,' +
              'HC/STABLE3/SCORE_PLUS,' +
              'HC/SCORE/REF,' +
              'HC/SCORE/EMPTY'
              )
    args = {'action': 'summary',
            'series': SERIES,
            'networks': CONTINUOUS,
            'N': '100-100k;1;0-24',
            'metrics': IJAR_STAB_METRICS,
            'maxtime': '180',
            'file': None}
    run_analysis(args)


def table_ijar_stab_residuals():
    """
        F1 S.D. for each network for each sample size
    """
    Ns = [100, 1000, 10000, 100000]
    Ss = (0, 24)
    metrics = ['f1-e', 'f1-e-std', 'expts']
    networks = CATEGORICAL.split(',') + ['hailfinder2', 'win95pts2']

    for N in Ns:
        print('\n\n*** RESULTS FOR N={} ***\n'.format(N))
        summary_analysis(series=['TABU/STABLE3/SCORE_PLUS'], networks=networks,
                         Ns=[N], Ss=Ss, metrics=metrics, params={})


# Generate charts which compare algorithms

def chart_ijar_stab_algos_cat_bic():
    """
        Algorithm comparson for categorical data learnt using BIC score using
        modified Hailfinder and Pathfinder networks.
        Experiments with single-valued datasets ignored, and missing
        metric values NOT imputed.
    """
    metrics = ['f1-e', 'f1-e-std', 'p-e', 'r-e', 'loglik', 'loglik-std',
               'bsf-e', 'time', 'score', 'score-std', 'nonex', 'expts',
               'dens', 'dens-std', 'n', '|E|', '|A|']
    networks = CATEGORICAL.replace('lfinder',
                                   'lfinder2').replace('pts', 'pts2')

    means = summary_analysis(series=list(SERIES2ALGO),
                             networks=networks.split(','), Ns=SAMPLE_SIZES,
                             Ss=RANDOM, metrics=metrics, maxtime=180,
                             params={'ignore': SING_VAL})
    data = DataFrame(_pivot(SERIES2ALGO, means, 'no', False))

    props = ALGO_BAR_PROPS.copy()
    props.update({'xaxis.ticks_fontsize': 14,
                  'yaxis.ticks_fontsize': 14,
                  'subplot.title_fontsize': 16,
                  'subplot.axes_fontsize': 14,
                  'figure.subplots_hspace': 0.8,
                  'figure.subplots_left': 0.06,
                  'figure.subplots_bottom': 0.08,
                  'yaxis.range': {'f1-e': (0.2, 0.6),
                                  'p-e': (0.3, 0.6),
                                  'r-e': (0.1, 0.6),
                                  'bsf-e': (0.2, 0.7),
                                  'loglik': (-30.0, -22.0)}})
    print(data)
    relplot(data=data, props=props,
            plot_file=(EXPTS_DIR +
                       '/papers/ijar_stability/algos-cat-bic.png'))


# Algorithm comparison for continuous networks

def chart_ijar_stab_algos_con_bic():
    """
        Comparsion of different algorithms with continuous data and using
        BIC score.
    """
    metrics = ['f1-e', 'f1-e-std', 'p-e', 'r-e', 'loglik', 'loglik-std',
               'bsf-e', 'time', 'score', 'score-std', 'nonex', 'expts',
               'dens', 'dens-std', 'n', '|E|', '|A|']
    means = summary_analysis(series=list(SERIES2ALGO),
                             networks=CONTINUOUS.split(','), Ns=SAMPLE_SIZES,
                             Ss=RANDOM, metrics=metrics, maxtime=180,
                             params={'ignore': ['arth150_c@100000']})
    data = DataFrame(_pivot(SERIES2ALGO, means, 'no', False))
    props = ALGO_BAR_PROPS.copy()
    props.update({'xaxis.ticks_fontsize': 14,
                  'yaxis.ticks_fontsize': 14,
                  'subplot.title_fontsize': 16,
                  'subplot.axes_fontsize': 14,
                  'figure.subplots_hspace': 0.8,
                  'figure.subplots_left': 0.06,
                  'figure.subplots_bottom': 0.08,
                  'yaxis.range': {'f1-e': (0.4, 0.7),
                                  'p-e': (0.4, 0.8),
                                  'r-e': (0.3, 0.7),
                                  'bsf-e': (0.5, 0.8),
                                  'loglik': (-56.0, -50.0)}})
    print(data)
    relplot(data=data, props=props, plot_file=EXPTS_DIR +
            '/papers/ijar_stability/algos-con-bic.png')


def values_ijar_stab_entropy_ties():

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
