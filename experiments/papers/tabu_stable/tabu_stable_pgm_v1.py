
# Generate the tables and charts for V1 of Tabu-Stable paper for PGHM 2024

from pandas import DataFrame

from experiments.run_analysis import run_analysis
from experiments.summary_analysis import summary_analysis
from experiments.plot import relplot
from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from learn.hc import hc


CATEGORICAL = ('asia,sports,sachs,covid,child,' +
               'insurance,property,diarrhoea,water,' +
               'mildew,alarm,barley,hailfinder,' +
               'hepar2,win95pts,formed,pathfinder')

CONTINUOUS = ('sachs_c,covid_c,building_c,magic-niab_c,magic-irri_c,' +
              'ecoli70_c,arth150_c')


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


def chart_tabu_stab_order_f1():
    args = {'action': 'series',
            'series': (
                       'TABU/BASE3,' +
                       'TABU/STABLE3/DEC_SCORE,' +
                       'TABU/STABLE3/INC_SCORE,' +
                       'TABU/STABLE3/SCORE_PLUS'
                       ),
            'metrics': 'f1-e',
            'networks': CATEGORICAL,
            'N': '100-100k',
            'params': ('fig:tabu_stab_order_f1;' +
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
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.24;' +
                       'figure.subplots_wspace:0.20')}
    run_analysis(args)


def chart_tabu_stab_order_score():
    """
        Score vs. sample size for different stability approaches
    """
    args = {'action': 'series',
            'series': ('TABU/STABLE/DEC_SCORE,' +
                       'TABU/STABLE/INC_SCORE,' +
                       'TABU/STABLE/SCORE,' +
                       'TABU/STABLE/SCORE_PLUS'),
            'metrics': 'score',
            'networks': CATEGORICAL,
            'N': '100-100k',
            'params': ('fig:tabu_stab_order_score;' +
                       'figure.title;' +
                       'legend.labels:{' +
                       'TABU/STABLE/DEC_SCORE,Decreasing\nscore,' +
                       'TABU/STABLE/INC_SCORE,Increasing\nscore,' +
                       'TABU/STABLE/SCORE,Best\nscore,' +
                       'TABU/STABLE/SCORE_PLUS,Learnt\norder};' +
                       'subplot.aspect:1.05;'
                       'figure.per_row:3;' +
                       'figure.subplots_left:0.06;' +
                       'figure.subplots_right:0.87;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.22;' +
                       'figure.subplots_wspace:0.40;' +
                       'yaxis.label:Normalised BIC Score;' +
                       'yaxis.range:¬')}
    run_analysis(args)


def table_tabu_stab_baseline_cat():
    """
        Comparison of stable algorithms with baselines for categorical data
    """
    args = {'action': 'summary',
            'series': ('HC/BASE3,' +
                       'HC/STABLE3/SCORE_PLUS,' +
                       'TABU/BASE3,' +
                       'TABU/STABLE3/DEC_SCORE,' +
                       'TABU/STABLE3/INC_SCORE,' +
                       'TABU/STABLE3/SCORE_PLUS'),
            'networks': CATEGORICAL,
            'N': '100-100k;1;0-24',
            'metrics': 'expts,f1-e,f1-e-std,score,time,p-e,r-e',
            # 'maxtime': '180',
            'file': None}
    run_analysis(args)


def table_tabu_stab_baseline_con():
    """
        Comparison of stable algorithms with baselines for continuous data
    """
    args = {'action': 'summary',
            'series': ('HC/BASE3,' +
                       'HC/STABLE3/SCORE_PLUS,' +
                       'TABU/BASE3,' +
                       'TABU/STABLE3/DEC_SCORE,' +
                       'TABU/STABLE3/INC_SCORE,' +
                       'TABU/STABLE3/SCORE_PLUS,' +
                       'TABU/OPT,' +
                       'TETRAD/FGES_BASE3'),
            'networks': CONTINUOUS,
            'N': '100-100k;1;0-24',
            'metrics': 'expts,f1-e,f1-e-std,score,time,p-e,r-e',
            # 'maxtime': '180',
            'file': None}
    run_analysis(args)


def table_tabu_stab_residuals():
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
                         Ns=[N], Ss=Ss, metrics=metrics, params={})[0]


def chart_tabu_stab_algos():
    """
        Comparsion of different algorithms - ignoring cases with
        identical or single-valued variables, and correcting FGES so
        cases with no results are assigned an F1 of 0.0
    """
    def _pivot(means, y_var, correct=True):
        """
            Pivot summary means data into long form required by relplot,
            and adjust F1 to account for number of experiments.

            :param DataFrame means: metric mean values, columns are metrics
            :param str y_var: value for y_var column in long form
            :param bool correct: whether to correct metrics on basis of how
                                 many experiments contributed to the value

        """
        means = means.rename(index=SERIES2ALGO).to_dict()
        means = {m: {a: vs[a] for a in SERIES2ALGO.values()}
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
                if m in ['f1-e', 'f1-e-std', 'p-e', 'r-e']]
        return data

    """
        Replace zero for FGES failed experiments with the average over all
        other algorithms to prevent bias against FGES.

    """
    def _fges_correct(metric, data, others):
        """
            :param str metric: metric to correct e.g. f1-e
            :param DataFrame data: metrics over all algorithms
            :param DataFrame others: metrics over all algorithms except FGES
        """
        o_mean = others[others['subplot'] == metric]['y_val'].mean()

        # get position of FGES value and hences its value

        fges_idx = data.index[(data['subplot'] == metric)
                              & (data['x_val'] == 'FGES')].tolist()[0]
        fges_val = data.at[fges_idx, 'y_val']

        data.at[fges_idx, 'y_val'] = (o_mean + 16 * fges_val) / 17
        print('FGES {} corrected from {:.4f} --> {:.4f}'
              .format(metric, fges_val, data.at[fges_idx, 'y_val']))

        return data

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
    # CATEGORICAL = 'sports'
    Ns = [100, 1000, 10000, 100000]
    Ss = (0, 24)
    metrics = ['f1-e', 'f1-e-std', 'p-e', 'r-e', 'expts']
    # metrics = ['f1-e', 'f1-e-std', 'expts']
    SING_VAL = ['insurance@100', 'water@100', 'barley@100', 'hailfinder@100',
                'hailfinder2@100', 'win95pts@100', 'win95pts2@100',
                'formed@100', 'pathfinder@100']

    # Get summary EXcluding networks with identical variables for all
    # algorithms

    networks = CATEGORICAL.replace('lfinder',
                                   'lfinder2').replace('pts', 'pts2')
    means = summary_analysis(series=list(SERIES2ALGO),
                             networks=networks.split(','), Ns=Ns, Ss=Ss,
                             metrics=metrics, params={'ignore': SING_VAL})[0]
    data = DataFrame(_pivot(means, 'no', False))  # don't correct FGES F1 here

    # Replace FGES failure cases for hailfinder and pathfinder at 10K and
    # 100K with averages over all other algorithms for those cases

    others = SERIES2ALGO
    others.pop('TETRAD/FGES_BASE3')
    others = summary_analysis(series=list(others),
                              networks=['hailfinder2', 'pathfinder'],
                              Ns=[10000, 100000], Ss=Ss,
                              metrics=metrics, params={'ignore': SING_VAL})[0]
    others = DataFrame(_pivot(others, 'no'))
    for metric in set(metrics) - {'f1-e-std', 'expts'}:
        data = _fges_correct(metric, data, others)

    print(data)

    if len(metrics) > 3:
        hspace = 0.55
        yaxis_label = {'f1-e': 'F1', 'f1-e-std': 'F1 S.D.',
                       'p-e': 'Precision', 'r-e': 'Recall'}
        subplot_title = {'f1-e': '(a) F1', 'f1-e-std': '(b) F1 S.D.',
                         'p-e': '(c) Precision', 'r-e': '(d) Recall'}
        subplot_aspect = 1.5
        subplots_bottom = 0.12
    else:
        hspace = 0.0
        yaxis_label = {'f1-e': 'F1', 'f1-e-std': 'F1 S.D.'}
        subplot_title = {'f1-e': '(a) F1', 'f1-e-std': '(b) F1 S.D.'}
        subplot_aspect = 1.3
        subplots_bottom = 0.26

    relplot(data=data,
            props={'subplot.kind': 'bar',
                   'figure.per_row': 2,
                   'figure.dpi': 300,
                   'xaxis.ticks_rotation': -60,
                   'xaxis.label': 'Algorithm',
                   'yaxis.label': yaxis_label,
                   'subplot.title': subplot_title,
                   'subplot.aspect': subplot_aspect,
                   'legend.title': ('Exclude\nidentical &\n' +
                                    'single-valued\nvariables'),
                   # 'yaxis.range': {'F1': (0.0, 0.6), 'SD': (0.0, 0.08)},
                   'figure.subplots_wspace': 0.3,
                   'figure.subplots_hspace': hspace,
                   'figure.subplots_bottom': subplots_bottom,
                   'figure.subplots_left': 0.04,
                   'figure.subplots_right': 0.90,
                   'subplot.grid': True,
                   'xaxis.shared': False,
                   'yaxis.shared': False},
            plot_file=EXPTS_DIR + '/papers/tabu_stable/tabu_stab_algos.png')


def values_tabu_stab_revised_f1():
    """
        Compute CPDAG F1 S.D. over all networks using revised definitions of
        Hailfinder and Win95pts.
    """
    args = {'action': 'summary',
            'series': ('TABU/STABLE3/SCORE_PLUS,' +
                       'TETRAD/FGES_BASE3'),
            'networks': CATEGORICAL.replace('lfinder',
                                            'lfinder2').replace('pts', 'pts2'),
            'N': '100-100K;1;0-24',
            'metrics': 'expts,f1-e,f1-e-std,score,time',
            # 'maxtime': '180',
            'file': None}
    run_analysis(args)
