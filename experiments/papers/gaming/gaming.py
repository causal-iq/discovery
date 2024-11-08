
# Generate the results for the Gaming ppaer from real data

from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from learn.hc import hc
from call.bnlearn import bnlearn_learn
from call.tetrad import tetrad_learn
from fileio.tetrad import write as write_tetrad
from fileio.bayesys import write as write_bayesys

PATH = EXPTS_DIR + '/papers/gaming/'


def run_bnlearn(algo):
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical')
    print(data.df.tail())

    context = {'id': 'bnlearn_' + algo + '_disc/N1000000', 'in': 'ht_disc'}
    dag, trace = bnlearn_learn(algo, data.df, context=context)

    print(trace)
    print(dag)
    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/bnlearn_' + algo + '_disc.txt')
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
    write_tetrad(dag, PATH + 'tetrad/tabu-stable_1k_disc.txt')
    write_bayesys(dag, PATH + 'bayesys/tabu-stable_1k_disc.csv')


def values_gaming_tabu_stable_disc():
    """
        Learn Gaming discrete with Tabu-Stable
    """
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical')
    print(data.df.tail())

    params = {'tabu': 10, 'stable': 'score+'}
    context = {'id': 'tabu-stable_disc/N1000000', 'in': 'ht_disc'}

    dag, trace = hc(data, params=params, context=context)

    print(trace)
    print(dag)
    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/tabu-stable_disc.txt')
    write_bayesys(dag, PATH + 'bayesys/tabu-stable_disc.csv')


def values_gaming_hc_stable_disc():
    """
        Learn Gaming discrete with HC-Stable
    """
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical')
    print(data.df.tail())

    params = {'stable': 'score+'}
    context = {'id': 'hc-stable_disc/N1000000', 'in': 'ht_disc'}

    dag, trace = hc(data, params=params, context=context)

    print(trace)
    print(dag)
    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/hc-stable_disc.txt')
    write_bayesys(dag, PATH + 'bayesys/hc-stable_disc.csv')


def values_gaming_bnlearn_tabu_disc():
    """
        Learn Gaming discrete with bnlearn Tabu
    """
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical')
    print(data.df.tail())

    context = {'id': 'bnlearn_tabu_disc/N1000000', 'in': 'ht_disc'}
    dag, trace = bnlearn_learn('tabu', data.df, context=context)

    print(trace)
    print(dag)
    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/bnlearn_tabu_disc.txt')
    write_bayesys(dag, PATH + 'bayesys/bnlearn_tabu_disc.csv')


def values_gaming_bnlearn_hc_disc():
    """
        Learn Gaming discrete with bnlearn HC
    """
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical')
    print(data.df.tail())

    context = {'id': 'bnlearn_hc_disc/N1000000', 'in': 'ht_disc'}
    dag, trace = bnlearn_learn('hc', data.df, context=context)

    print(trace)
    print(dag)
    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/bnlearn_hc_disc.txt')
    write_bayesys(dag, PATH + 'bayesys/bnlearn_hc_disc.csv')


def values_gaming_bnlearn_pc_stable_disc():
    """
        Learn Gaming discrete with bnlearn PC Stable
    """
    data = Pandas.read(EXPTS_DIR + '/realdata/ht_disc.data.gz',
                       dstype='categorical')
    print(data.df.tail())

    context = {'id': 'bnlearn_pc-stable_disc/N1000000', 'in': 'ht_disc'}
    dag, trace = bnlearn_learn('pc.stable', data.df, context=context)

    print(trace)
    print(dag)
    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/bnlearn_pc-stable_disc.txt')
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
    dag, trace = tetrad_learn('fges', data.df, context=context, params=params)

    trace.save(PATH + 'trace')
    write_tetrad(dag, PATH + 'tetrad/tetrad_fges_disc.txt')
    write_bayesys(dag, PATH + 'bayesys/tetrad_fges_disc.csv')
