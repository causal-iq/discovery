
# Generate the tables and charts for V1 of Tabu-Stable paper for IJAR 2025

from pandas import DataFrame

from experiments.common import reference_bn, sample_sizes
from experiments.run_analysis import run_analysis
from experiments.summary_analysis import summary_analysis
from experiments.plot import relplot
from core.graph import DAG
from core.score import SCORE_PARAMS
from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from fileio.numpy import NumPy
from learn.hc import hc
from learn.trace import Trace, Activity, Detail
from learn.knowledge import Knowledge, RuleSet


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
               'BNLEARN/GS_BASE3': 'GS'
               # 'BNLEARN/IIAMB_BASE3': 'Inter-IAMB'
               }


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


def chart_ijar_stab_cat_f1():
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
            'params': ('fig:tabu_stab_cat_f1;' +
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


def chart_ijar_stab_con_f1():
    args = {'action': 'series',
            'series': (
                       'TABU/BASE3,' +
                       'TABU/STABLE3/DEC_SCORE,' +
                       'TABU/STABLE3/INC_SCORE,' +
                       'TABU/STABLE3/SCORE_PLUS'
                       ),
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


def chart_ijar_stab_order_score():
    """
        Chart of core vs. sample size for different stability approaches - cat
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


def table_ijar_stab_cat():
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
              'TABU/STABLE3/SCORE,' +
              'HC/BASE3,' +
              'HC/STABLE3/SCORE_PLUS'
              )
    args = {'action': 'summary',
            'series': SERIES,
            'networks': CATEGORICAL,
            'N': '100-100k;1;0-24',
            'metrics': 'f1-e,f1-e-std,score',  # 'expts,f1-e,f1-e-std,score,time,p-e,r-e',
            # 'maxtime': '180',
            'file': None}
    run_analysis(args)


def table_ijar_stab_con():
    """
        Table summarising HC/Tabu stability approaches - continuous
    """
    SERIES = ('HC/BASE3,' +
              'HC/STABLE3/SCORE_PLUS,' +
              'HC/OPT,' +
              'HC/SCORE/EMPTY,' +
              'HC/SCORE/REF,' +
              'HC/SCORE/HCREF,' +
              'TABU/BASE3,' +
              'TABU/STABLE3/DEC_SCORE,' +
              'TABU/STABLE3/INC_SCORE,' +
              'TABU/STABLE3/SCORE,' +
              'TABU/STABLE3/SCORE_PLUS,' +
              'TABU/OPT,' +
              'TETRAD/FGES_BASE3')
    SERIES = ('TABU/BASE3,' +
              'TABU/STABLE3/DEC_SCORE,' +
              'TABU/STABLE3/INC_SCORE,' +
              'TABU/STABLE3/SCORE,' +
              'HC/BASE3,' +
              'HC/STABLE3/SCORE_PLUS'
              )
    args = {'action': 'summary',
            'series': SERIES,
            'networks': CONTINUOUS,
            'N': '100-100k;1;0-24',
            'metrics': 'f1-e,f1-e-std,score',  # 'expts,f1-e,f1-e-std,f1,score,time,p-e,r-e',
            # 'maxtime': '180',
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


def _fges_correct(metric, data, others):
    """
        Replace zero for FGES failed experiments with the average over all
        other algorithms to prevent bias against FGES.

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


def chart_ijar_stab_algos_cat():
    """
        Comparsion of different algorithms for categorical data - ignoring
        cases with identical or single-valued variables, and correcting FGES so
        cases with no results are assigned an F1 of 0.0.
    """

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
                             metrics=metrics, params={'ignore': SING_VAL})
    data = DataFrame(_pivot(means, 'no', False))  # don't correct FGES F1 here

    # Replace FGES failure cases for hailfinder and pathfinder at 10K and
    # 100K with averages over all other algorithms for those cases

    others = SERIES2ALGO
    others.pop('TETRAD/FGES_BASE3')
    others = summary_analysis(series=list(others),
                              networks=['hailfinder2', 'pathfinder'],
                              Ns=[10000, 100000], Ss=Ss,
                              metrics=metrics, params={'ignore': SING_VAL})
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
            plot_file=(EXPTS_DIR +
                       '/papers/tabu_stable/ijar_stab_algos_cat.png'))


def chart_ijar_stab_algos_con():
    """
        Comparsion of different algorithms - ignoring cases with
        identical or single-valued variables, and correcting FGES so
        cases with no results are assigned an F1 of 0.0
    """

    Ns = [100, 1000, 10000, 100000]
    Ss = (0, 24)
    metrics = ['f1-e', 'f1-e-std', 'p-e', 'r-e', 'expts']

    # Get summary EXcluding networks with identical variables for all
    # algorithms

    means = summary_analysis(series=list(SERIES2ALGO),
                             networks=CONTINUOUS.split(','), Ns=Ns, Ss=Ss,
                             metrics=metrics, params={})
    data = DataFrame(_pivot(means, 'no', False))  # don't correct FGES F1 here

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
            plot_file=(EXPTS_DIR +
                       '/papers/tabu_stable/ijar_stab_algos_con.png'))


def values_ijar_stab_revised_f1():
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


def values_ijar_stab_score_fges_graphs():
    """
        Adds score to FGES graph traces
    """
    print('\n')
    FGES_SERIES = '/TETRAD/FGES_BASE3'

    params = {'base': 'e', 'unistate_ok': True}
    for network in (CATEGORICAL + ',' + CONTINUOUS).split(','):
        if network != 'asia':
            break

        # read traces for this network

        print('\nReading {} traces for {} ...'.format(FGES_SERIES, network))
        traces = Trace.read(FGES_SERIES + '/' + network)
        if traces is None:
            print(' ... no traces found for {}'.format(network))
            continue

        # Determine sample sizes used for this network, read in enough data
        # for largest sample size, and determine score of the initial (empty)
        # graph at each sample size.

        Ns = {int(id.split('_')[0][1:]) for id in traces}
        dstype = 'continuous' if network.endswith('_c') else 'categorical'
        score = 'bic' if dstype == 'categorical' else 'bic-g'
        data = NumPy.read(EXPTS_DIR + '/datasets/' + network + '.data.gz',
                          dstype=dstype, N=max(Ns))
        initial = DAG(list(data.get_order()), [])
        initial_score = {}
        for N in Ns:
            data.set_N(N)
            initial_score[N] = (initial.score(data, score,
                                              params)[score]).sum()

        # Determine score of each learnt graph and update trace with initial
        # and learnt graph scores

        for id, trace in traces.items():
            if trace.trace['delta/score'][-1] != 0.0:
                continue
            N = int(id.split('_')[0][1:])
            if N != data.N:
                data.set_N(N)
            learnt = trace.result
            try:
                learnt = learnt if learnt.is_DAG() else DAG.extendPDAG(learnt)
                learnt_score = (learnt.score(data, score,
                                             params)[score]).sum()
            except ValueError:
                print('\n*** Cannot extend PDAG for {}\n'.format(id))
                learnt_score = 0.0

            print('{}: {} score {:.3e} --> {:.3e}'
                  .format(id, score, initial_score[N], learnt_score))
            trace.trace['delta/score'][0] = initial_score[N]
            trace.trace['delta/score'][-1] = learnt_score
            trace.save()


def values_ijar_stab_hc_ref():
    """
        Learn and score true graphs using hc
    """
    Ns, _ = sample_sizes('10-100K;1')
    params = {'k': 1, 'unistate_ok': True}

    for network in ('cancer,' + CATEGORICAL + ',' + CONTINUOUS).split(','):
        # if network != 'sachs_c':
        #     continue

        # Obtain reference & empty DAG and data for network

        ref, _in = reference_bn(network)
        ref = ref.dag
        empty = DAG(ref.nodes, [])
        dstype = 'continuous' if network.endswith('_c') else 'categorical'
        score = 'bic' if dstype == 'categorical' else 'bic-g'
        params.update({'score': score})
        data = NumPy.read(EXPTS_DIR + '/datasets/' + network + '.data.gz',
                          dstype=dstype, N=max(Ns))
        print('\n\nLearning {} using {} score ...'.format(network, score))

        # Construct Knowledge object that stops all arcs not in the
        # reference graph

        ref_know = {}
        for n1 in ref.nodes:
            for n2 in ref.nodes:
                if n1 != n2 and (n1, n2) not in ref.edges:
                    ref_know[(n1, n2)] = True
        ref_know = Knowledge(rules=RuleSet.STOP_ARC, params={'stop': ref_know})
        data.set_order(tuple([n for n in ref.ordered_nodes()]))

        for N in Ns:
            data.set_N(N)
            _params = {k: v for k, v in params.items() if k in SCORE_PARAMS}

            # create a minimal trace for empty graph which includes its score

            score_e = (empty.score(data, score, _params)[score]).sum()
            id = 'HC/SCORE/EMPTY/{}/N{}_0'.format(network, N)
            context = {'id': id, 'in': _in, 'algorithm': 'DAG_SCORE',
                       'N': data.N, 'params': params, 'dataset': True}
            trace = Trace(context.copy())
            trace.add(Activity.INIT, {Detail.DELTA: score_e})
            trace.add(Activity.STOP, {Detail.DELTA: score_e})
            trace.result = empty
            trace.save()
            print('{:>8} rows, emp: {:.3e}'.format(N, score_e))

            # create a minimal trace for ref graph which includes its score

            score_r = (ref.score(data, score, _params)[score]).sum()
            context.update({'id':
                            'HC/SCORE/REF/{}/N{}_0'.format(network, N)})
            trace = Trace(context.copy())
            trace.add(Activity.INIT, {Detail.DELTA: score_e})
            trace.add(Activity.STOP, {Detail.DELTA: score_r})
            trace.result = ref
            trace.save()
            print('{:>8} rows, ref: {:.3e}'.format(N, score_r))

            # Learn graph with Knowledge prohibiting all arcs not in the
            # reference graph - one might expect this to learn the ref graph.

            context = {'id': 'HC/SCORE/HCREF/{}/N{}_0'.format(network, N),
                       'in': _in}
            _, trace = hc(data, params=params, knowledge=ref_know,
                          context=context, init_cache=True)
            trace.save()
            print('{:>8} rows,  hc: {:.3e}        [{} iters]\n'
                  .format(N, trace.trace['delta/score'][-1],
                          len(trace.trace['delta/score']) - 2))
