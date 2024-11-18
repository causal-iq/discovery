
# Generate the results for the Gaming ppaer from real data

from pandas import DataFrame

from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from fileio.numpy import NumPy
from fileio.bayesys import read
from learn.hc import hc
from call.bnlearn import bnlearn_learn
from call.tetrad import tetrad_learn
from fileio.tetrad import write as write_tetrad, read as read_tetrad
from fileio.bayesys import write as write_bayesys
from core.bn import BN
from core.graph import DAG
from core.timing import Timing

PATH = EXPTS_DIR + '/papers/gaming/'
TETRAD_FILE = EXPTS_DIR + '/papers/gaming/tetrad/{}_disc.tetrad'


def run_bnlearn(algo):
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical')
    print(data.df.tail())

    context = {'id': 'bnlearn_' + algo + '_disc/N1000000', 'in': 'ht_disc'}
    dag, trace = bnlearn_learn(algo, data, context=context)

    print(trace)
    print(dag)
    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/bnlearn_' + algo + '_disc.tetrad')
    write_bayesys(dag, PATH + 'bayesys/bnlearn_' + algo + '_disc.csv')


def values_gaming_tabu_stable_1k_disc():
    """
        Learn Gaming 1K discrete with Tabu-Stable
    """
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical', N=1000)
    print(data.df.tail())

    params = {'tabu': 10, 'stable': 'score+'}
    context = {'id': 'tabu-stable_disc/N1000', 'in': 'ht_disc'}

    dag, trace = hc(data, params=params, context=context)

    print(trace)
    print(dag)
    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/tabu-stable_1k_disc.tetrad')
    write_bayesys(dag, PATH + 'bayesys/tabu-stable_1k_disc.csv')


def values_gaming_tabu_stable_disc():
    """
        Learn Gaming discrete with Tabu-Stable
    """
    N = 1000000
    data = NumPy.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                      dstype='categorical', N=N)
    print(data.as_df().tail())

    params = {'tabu': 10, 'stable': 'score+'}
    context = {'id': 'tabu-stable_disc/N1000000', 'in': 'ht_disc'}

    Timing.on(True)
    start = Timing.now()
    dag, trace = hc(data, params=params, context=context)
    Timing.record('tabu', N, start)

    print(trace)
    print(dag)
    print(Timing)

    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/tabu-stable_disc.tetrad')
    write_bayesys(dag, PATH + 'bayesys/tabu-stable_disc.csv')


def values_gaming_hc_stable_disc():
    """
        Learn Gaming discrete with HC-Stable
    """
    N = 1000000
    data = NumPy.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                      dstype='categorical', N=N)
    print(data.as_df().tail())

    params = {'stable': 'score+'}
    context = {'id': 'hc-stable_disc/N1000000', 'in': 'ht_disc'}

    Timing.on(True)
    start = Timing.now()
    dag, trace = hc(data, params=params, context=context)
    Timing.record('hc', N, start)

    print(trace)
    print(dag)
    print(Timing)

    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/hc-stable_disc.tetrad')
    write_bayesys(dag, PATH + 'bayesys/hc-stable_disc.csv')


def values_gaming_bnlearn_tabu_disc():
    """
        Learn Gaming discrete with bnlearn Tabu
    """
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical')
    print(data.df.tail())

    context = {'id': 'bnlearn_tabu_disc/N1000000', 'in': 'ht_disc'}
    dag, trace = bnlearn_learn('tabu', data, context=context)

    print(trace)
    print(dag)
    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/bnlearn_tabu_disc.tetrad')
    write_bayesys(dag, PATH + 'bayesys/bnlearn_tabu_disc.csv')


def values_gaming_bnlearn_hc_disc():
    """
        Learn Gaming discrete with bnlearn HC
    """
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical')
    print(data.df.tail())

    context = {'id': 'bnlearn_hc_disc/N1000000', 'in': 'ht_disc'}
    dag, trace = bnlearn_learn('hc', data, context=context)

    print(trace)
    print(dag)
    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/bnlearn_hc_disc.tetrad')
    write_bayesys(dag, PATH + 'bayesys/bnlearn_hc_disc.csv')


def values_gaming_bnlearn_pc_stable_disc():
    """
        Learn Gaming discrete with bnlearn PC Stable
    """
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical')
    print(data.df.tail())

    context = {'id': 'bnlearn_pc-stable_disc/N1000000', 'in': 'ht_disc'}
    dag, trace = bnlearn_learn('pc.stable', data, context=context)

    print(trace)
    print(dag)
    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/bnlearn_pc-stable_disc.tetrad')
    write_bayesys(dag, PATH + 'bayesys/bnlearn_pc-stable_disc.csv')


def values_gaming_bnlearn_mmhc_disc():
    """
        Learn Gaming discrete with bnlearn MMHC
    """
    run_bnlearn('mmhc')


def values_gaming_bnlearn_h2pc_disc():
    """
        Learn Gaming discrete with bnlearn H2PC
    """
    run_bnlearn('h2pc')


def values_gaming_bnlearn_gs_disc():
    """
        Learn Gaming discrete with bnlearn GS
    """
    run_bnlearn('gs')


def values_gaming_tetrad_fges_disc():
    """
        Learn Gaming discrete with Tetrad FGES
    """
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical')
    print(data.df.tail())

    params = {'score': 'bic', 'k': 1}
    context = {'id': 'tetrad_fges_disc/N1000000', 'in': 'ht_disc'}
    dag, trace = tetrad_learn('fges', data, context=context, params=params)

    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/tetrad_fges_disc.tetrad')
    write_bayesys(dag, PATH + 'bayesys/tetrad_fges_disc.csv')


def graph_gaming_knowledge():
    """
        Process Bayesys format knowledge file and create dsc and xdsl files
    """
    dag = read(EXPTS_DIR + '/bn/bayesys/gaming.csv')

    print('\n\nReading data file ...\n')
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical', N=100000)
    print(data.df.tail())

    print('\nFitting data to DAG ...')
    bn = BN.fit(dag, data)

    print('\nWriting .dsc and .xdsl files ...')
    bn.write(EXPTS_DIR + '/bn/gaming.dsc')

    bn.write(EXPTS_DIR + '/bn/xdsl/gaming.xdsl')


def graph_gaming_synthetic():
    """
        Generate 10M rows of synthetic data file for Gaming
    """
    bn = BN.read(EXPTS_DIR + '/bn/xdsl/gaming.xdsl')

    bn.generate_cases(1000000, EXPTS_DIR + '/datasets/gaming.data.gz')


def graph_gaming_analysis():
    """
        Analyse graphs produced by different algorithms
    """
    ALGOS = ['bnlearn_gs', 'bnlearn_hc', 'bnlearn_mmhc', 'bnlearn_pc-stable',
             'bnlearn_tabu', 'hc-stable', 'tabu-stable', 'tetrad_fges']
    ref = BN.read(EXPTS_DIR + '/bn/gaming.dsc').dag

    comparisons = []
    for algo in ALGOS:
        graph = read_tetrad(TETRAD_FILE.format(algo))
        if isinstance(graph, DAG):
            kind = 'DAG'
        else:
            try:
                kind = 'CPDAG' if graph.is_CPDAG() else 'PDAG'
            except ValueError:
                kind = 'NONEX'
        comparison = {'algo': algo, 'type': kind}
        comparison.update(graph.compared_to(ref))
        comparisons.append(comparison)

    print('\n\n{}'.format(DataFrame(comparisons)))


def graph_gaming_extend():
    """
        Extend the CPDAGs produced by FGES and GES
    """
    cpdag = read_tetrad(TETRAD_FILE.format('bnlearn_gs'))
    print('\n\nExtending bnlearn_gs ...\n')
    dag = DAG.extendPDAG(cpdag)
    write_tetrad(dag, PATH + 'tetrad/bnlearn_gs_disc_dag.tetrad')
    write_bayesys(dag, PATH + 'bayesys/bnlearn_gs_disc_dag.csv')

    cpdag = read_tetrad(TETRAD_FILE.format('tetrad_fges'))
    print('\n\nExtending tetrad_fges ...\n')
    dag = DAG.extendPDAG(cpdag)
    write_tetrad(dag, PATH + 'tetrad/tetrad_fges_disc_dag.tetrad')
    write_bayesys(dag, PATH + 'bayesys/tetrad_fges_disc_dag.csv')
