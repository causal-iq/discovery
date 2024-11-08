
# Generate the graphs for paper comparing LLM with BNSL for causal discovery

# This version has some deficiencies
#   - diarrhoea in bnbench has incorrect spelling of DIA_HadDiarrhoea, so
#     need to edit this variable name in produced csv files
#   - formed real data file uses original names containing spaces which FGES
#     doesn't support. They are replaced by underscores which must be edited
#     in output file. Also, bnbench names are camel cased so can't do
#     reference comparison.
#   - Diarrhoea + pc.stable doesn't run so using Bayesys file from Diarrhoea
#     as the source here

import time
from os import listdir
from glob import glob

from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from fileio.bayesys import write as write_bayesys, read as read_bayesys, \
    read_constraints
from fileio.xdsl import write as write_xdsl
from fileio.tetrad import read as read_tetrad, write as write_tetrad
from call.bnlearn import bnlearn_learn
from call.tetrad import tetrad_learn
from core.graph import PDAG, NotPDAGError
from core.bn import BN
from experiments.common import reference_bn

OUTPUT = EXPTS_DIR + '/papers/llm_cf_bnsl/{}_{}.{}'


def learn_graph(netw, algo):
    """
        Learning graph from real dataset and writing result to Bayesys and
        Genie compatible files

        :param str netw: network to learn
        :param str algo: algorithm to use e.g. "pc.stable"
    """
    print('\n\n\nLearning {} from real data using {} ...\n'.format(netw, algo))

    # Read in real data

    data = Pandas.read(EXPTS_DIR + '/realdata/' + netw + '.data.gz',
                       dstype='categorical').df
    print('{} rows of {} data read OK:'.format(len(data), netw))
    data.columns = [c.replace(' ', '_') for c in data.columns]  # for FGES
    print(data.head())

    # Run the appropriate bnlearn or Tetrad structure learning algorithm

    start = time.time()
    if algo in {'pc.stable', 'mmhc'}:
        graph, _ = bnlearn_learn(algo, data)
    elif algo == 'fges':
        graph, _ = tetrad_learn('fges', data)
    else:
        raise ValueError('Invalid algorithm: {}'.format(algo))
    print('\nLearnt graph in {:.2f} seconds and is:\n{}\n'
          .format((time.time() - start), graph))

    # Writing graph to Bayesys output file

    print('Writing graph in Bayesys format ...')
    write_bayesys(graph, OUTPUT.format(netw, algo, 'csv'))

    if netw == 'formed':  # has ref / data name mismatches so can't check
        return

    if graph.is_DAG():
        bn = BN.fit(graph, Pandas(df=data))
        print('Writing graph (DAG) in Genie format ...')
        write_xdsl(bn, OUTPUT.format(netw, algo, 'xdsl'), True)
        cpdag = PDAG.fromDAG(graph)
    else:
        cpdag = graph

    # Structural comparison with reference DAG

    ref, _ = reference_bn(netw)
    print('\nComparison with reference DAG:')
    for metric, value in graph.compared_to(ref.dag).items():
        print('{}: {}'.format(metric, round(value, 3)))

    # Structural comparison between CPDAGs

    ref_cpdag = PDAG.fromDAG(ref.dag)
    print('\nCPDAG comparison with reference:')
    for metric, value in cpdag.compared_to(ref_cpdag).items():
        print('{}: {}'.format(metric, round(value, 3)))


def constrained_graph(netw, algo, type, level='0'):
    """
        Learning graph from real dataset using specified constraints file

        :param str netw: network to learn
        :param str algo: algorithm to use e.g. "pc.stable"
        :param str type: constraint type e.g. 'Directed'
        :param start level: constraints level - 4, 6 or 7

        :returns PDAG: learnt graph
    """

    # Obtain reference file ... note some name/content differences between
    # bnbench synthetic networks and networks used in LLM study

    ref = (BN.read(EXPTS_DIR + '/bn/' +
                   netw.replace('-19', '').replace('hoea', 'hoea_c')
                   + '.dsc').dag if netw != 'formed' else
           read_bayesys(EXPTS_DIR +
                        '/papers/llm_cf_bnsl/DAGtrue_FORMED_real.csv'))

    # Read constraints implied from LLM responses

    if type == '':
        c_file = ('experiments/papers/llm_cf_bnsl/constraints/constraints' +
                  type + netw.upper() + '.csv')
        knowledge = None
    else:
        c_file = ('experiments/papers/llm_cf_bnsl/constraints/constraints' +
                  type + '_' + netw.upper() + '_' + level + '.csv')
        print('\nFile is {}'.format(c_file))
        knowledge = read_constraints(c_file, set(ref.nodes))

    # Read the real data file

    d_file = EXPTS_DIR + '/realdata/' + netw + '.data.gz'
    data = Pandas.read(d_file.replace('-19', '_disc_kmeans'),
                       dstype='categorical')
    print('\n{} rows read from data file'.format(data.N))
    print(data.sample.head())

    # Use bnlearn to learn graph from real world data with constraints

    id = '{}_{}_{}'.format(algo, netw, level)
    pdag, _ = bnlearn_learn(algo, data.sample, knowledge=knowledge,
                            context={'in': d_file, 'id': id})
    print('\nGraph learnt by {} for {} with level {} {}:\n{}'
          .format(algo, netw, level, type, pdag))

    # Structural comparison between reference and learnt graph

    diffs = pdag.compared_to(ref)
    print(diffs)

    write_bayesys(pdag, c_file.replace('/constraints/constraints',
                                       '/learnt_graphs/' + algo + '_'))

    return pdag


# Learning Diahorrea - pc run is from M.Sc. work


def xgraph_diarrhoea_pc():

    # pc.stable doesn't complete in reasonable time ... use result
    # from M.Sc. work by reading in Bayesys format file

    graph = read_tetrad(EXPTS_DIR + '/papers/llm_cf_bnsl/d7a-pc.tetrad')
    print('\n\nGraph read from Tetrad file:\n{}'.format(graph))
    print('Writing graph in Bayesys format ...')
    write_bayesys(graph, OUTPUT.format('diarrhoea', 'pc.stable', 'csv'))

    # Structural comparison with reference DAG

    ref, _ = reference_bn('diarrhoea')
    print('\nComparison with reference DAG:')
    for metric, value in graph.compared_to(ref.dag).items():
        print('{}: {}'.format(metric, round(value, 3)))

    # Structural comparison between CPDAGs

    cpdag = PDAG.fromDAG(graph)
    ref_cpdag = PDAG.fromDAG(ref.dag)
    print('\nCPDAG comparison with reference:')
    for metric, value in cpdag.compared_to(ref_cpdag).items():
        print('{}: {}'.format(metric, round(value, 3)))


def xgraph_diarrhoea_fges():
    learn_graph('diarrhoea', 'fges')


# Generate Formed reference graph without synthetic variables fitted to
# real data

def graph_reference_formed():
    ref = read_bayesys(EXPTS_DIR +
                       '/papers/llm_cf_bnsl/DAGtrue_FORMED_real.csv')
    print(ref)

    data = Pandas.read(EXPTS_DIR + '/realdata/formed.data.gz',
                       dstype='categorical')

    bn = BN.fit(ref, data)  # VERY slow - fitting Violence with many parents

    bn.write(EXPTS_DIR + '/papers/llm_cf_bnsl/formed_real.dsc')


# Generate the Tetrad Reference graphs

def graph_llm_tetrad_refs():

    # Sports is one used in KK work

    bn = BN.read(EXPTS_DIR + '/bn/sports.dsc')
    write_tetrad(bn.dag, EXPTS_DIR +
                 '/papers/llm_cf_bnsl/tetrad-7.1.2-2/sports_ref.txt')

    # Covid is one used in KK work

    bn = BN.read(EXPTS_DIR + '/bn/covid.dsc')
    write_tetrad(bn.dag, EXPTS_DIR +
                 '/papers/llm_cf_bnsl/tetrad-7.1.2-2/covid_ref.txt')

    # Formed is version without synthetic nodes - obtain from Bayesys format
    # file

    dag = read_bayesys(EXPTS_DIR
                       + '/papers/llm_cf_bnsl/DAGtrue_FORMED_real.csv')
    write_tetrad(dag, EXPTS_DIR +
                 '/papers/llm_cf_bnsl/tetrad-7.1.2-2/formed_ref.txt')

    # Diarrhoea using version with corrected DIA_HadDiarrhoea name

    bn = BN.read(EXPTS_DIR + '/bn/diarrhoea_c.dsc')
    write_tetrad(bn.dag, EXPTS_DIR +
                 '/papers/llm_cf_bnsl/tetrad-7.1.2-2/diarrhoea_ref.txt')


# Do the bnlearn LLM Directed constraint runs now
# NB Diarrhoea not included for pc.stable as does not run

def graph_sports_pc():
    constrained_graph('sports', 'pc.stable', '')


def graph_sports_mmhc():
    pdag = constrained_graph('sports', 'mmhc', '')
    prev = read_bayesys(EXPTS_DIR +
                        '/papers/llm_cf_bnsl/sports_mmhc.csv')
    diffs = pdag.compared_to(prev)
    print(diffs)


def graph_diarrhoea_mmhc():
    constrained_graph('diarrhoea', 'mmhc', '')


def graph_diarrhoea_fges():
    pdag = read_tetrad(EXPTS_DIR + '/papers/llm_cf_bnsl/tetrad-7.1.2-2' +
                       '/learnt_graphs/fges_DIARRHOEA.tetrad')
    prev = read_bayesys(EXPTS_DIR +
                        '/papers/llm_cf_bnsl/diarrhoea_fges.csv')
    diffs = pdag.compared_to(prev)
    print(diffs)


def graph_formed_pc():
    constrained_graph('formed', 'pc.stable', '')


def graph_formed_mmhc():
    constrained_graph('formed', 'mmhc', '')


def graph_covid_mmhc():
    constrained_graph('covid-19', 'mmhc', '')


def graph_covid_pc():
    constrained_graph('covid-19', 'pc.stable', '')


def graph_directed_sports_pc_4():
    constrained_graph('sports', 'pc.stable', 'Directed', '4')


def graph_directed_sports_pc_5():
    constrained_graph('sports', 'pc.stable', 'Directed', '5')


def graph_directed_sports_pc_7():
    constrained_graph('sports', 'pc.stable', 'Directed', '7')


def graph_directed_sports_mmhc_4():
    constrained_graph('sports', 'mmhc', 'Directed', '4')


def graph_directed_sports_mmhc_5():
    constrained_graph('sports', 'mmhc', 'Directed', '5')


def graph_directed_sports_mmhc_7():
    constrained_graph('sports', 'mmhc', 'Directed', '7')


def graph_directed_covid_pc_4():
    constrained_graph('covid-19', 'pc.stable', 'Directed', '4')


def graph_directed_covid_pc_5():
    constrained_graph('covid-19', 'pc.stable', 'Directed', '5')


def graph_directed_covid_pc_7():
    constrained_graph('covid-19', 'pc.stable', 'Directed', '7')


def graph_directed_covid_mmhc_4():
    constrained_graph('covid-19', 'mmhc', 'Directed', '4')


def graph_directed_covid_mmhc_5():
    constrained_graph('covid-19', 'mmhc', 'Directed', '5')


def graph_directed_covid_mmhc_7():
    constrained_graph('covid-19', 'mmhc', 'Directed', '7')


def graph_directed_diarrhoea_pc_4():
    constrained_graph('diarrhoea', 'pc.stable', 'Directed', '4')


def graph_directed_diarrhoea_pc_5():
    constrained_graph('diarrhoea', 'pc.stable', 'Directed', '5')


def graph_directed_diarrhoea_pc_7():
    constrained_graph('diarrhoea', 'pc.stable', 'Directed', '7')


def graph_directed_diarrhoea_mmhc_4():
    constrained_graph('diarrhoea', 'mmhc', 'Directed', '4')


def graph_directed_diarrhoea_mmhc_5():
    constrained_graph('diarrhoea', 'mmhc', 'Directed', '5')


def graph_directed_diarrhoea_mmhc_7():
    constrained_graph('diarrhoea', 'mmhc', 'Directed', '7')


def graph_directed_formed_pc_4():
    constrained_graph('formed', 'pc.stable', 'Directed', '4')


def graph_directed_formed_pc_5():
    constrained_graph('formed', 'pc.stable', 'Directed', '5')


def graph_directed_formed_mmhc_4():
    constrained_graph('formed', 'mmhc', 'Directed', '4')


def graph_directed_formed_mmhc_5():
    constrained_graph('formed', 'mmhc', 'Directed', '5')


# Do the bnlearn LLM Temporal constraint runs now

def graph_temporal_sports_pc_4():
    constrained_graph('sports', 'pc.stable', 'Temporal', '4')


def graph_temporal_sports_pc_5():
    constrained_graph('sports', 'pc.stable', 'Temporal', '5')


def graph_temporal_sports_pc_7():
    constrained_graph('sports', 'pc.stable', 'Temporal', '7')


def graph_temporal_sports_mmhc_4():
    constrained_graph('sports', 'mmhc', 'Temporal', '4')


def graph_temporal_sports_mmhc_5():
    constrained_graph('sports', 'mmhc', 'Temporal', '5')


def graph_temporal_sports_mmhc_7():
    constrained_graph('sports', 'mmhc', 'Temporal', '7')


def graph_temporal_covid_pc_4():
    constrained_graph('covid-19', 'pc.stable', 'Temporal', '4')


def graph_temporal_covid_pc_5():
    constrained_graph('covid-19', 'pc.stable', 'Temporal', '5')


def graph_temporal_covid_pc_7():
    constrained_graph('covid-19', 'pc.stable', 'Temporal', '7')


def graph_temporal_covid_mmhc_4():
    constrained_graph('covid-19', 'mmhc', 'Temporal', '4')


def graph_temporal_covid_mmhc_5():
    constrained_graph('covid-19', 'mmhc', 'Temporal', '5')


def graph_temporal_covid_mmhc_7():
    constrained_graph('covid-19', 'mmhc', 'Temporal', '7')


def graph_temporal_diarrhoea_pc_4():
    constrained_graph('diarrhoea', 'pc.stable', 'Temporal', '4')


def graph_temporal_diarrhoea_pc_5():
    constrained_graph('diarrhoea', 'pc.stable', 'Temporal', '5')


def graph_temporal_diarrhoea_pc_7():
    constrained_graph('diarrhoea', 'pc.stable', 'Temporal', '7')


def graph_temporal_diarrhoea_mmhc_4():
    constrained_graph('diarrhoea', 'mmhc', 'Temporal', '4')


def graph_temporal_diarrhoea_mmhc_5():
    constrained_graph('diarrhoea', 'mmhc', 'Temporal', '5')


def graph_temporal_diarrhoea_mmhc_7():
    constrained_graph('diarrhoea', 'mmhc', 'Temporal', '7')


def graph_temporal_formed_pc_4():
    constrained_graph('formed', 'pc.stable', 'Temporal', '4')


def graph_temporal_formed_pc_5():
    constrained_graph('formed', 'pc.stable', 'Temporal', '5')


def graph_temporal_formed_mmhc_4():
    constrained_graph('formed', 'mmhc', 'Temporal', '4')


def graph_temporal_formed_mmhc_5():
    constrained_graph('formed', 'mmhc', 'Temporal', '5')


def graph_tetrad_to_bayesys():
    """
        Convert manually generated Tetrad graphs to Bayesys format files
    """
    print('\n')
    tetrad = EXPTS_DIR + '/papers/llm_cf_bnsl/tetrad-7.1.2-2/learnt_graphs/'
    bayesys = EXPTS_DIR + '/papers/llm_cf_bnsl/learnt_graphs/'
    for name in listdir(tetrad):
        try:
            pdag = read_tetrad(tetrad + name)
            write_bayesys(pdag, bayesys + name.replace('.tetrad', '.csv'))
            print('{} converted OK'.format(name))
        except NotPDAGError:
            print('*** {} not a PDAG'.format(name))


def graph_llm_summarise():
    """
        Summarise accuracy of learnt graphs
    """
    PATTERN = EXPTS_DIR + '/papers/llm_cf_bnsl/learnt_graphs/{}*{}*.csv'
    print('\n')
    for netw in ['SPORTS', 'COVID-19', 'DIARRHOEA', 'FORMED']:
        ref = read_tetrad(EXPTS_DIR + '/papers/llm_cf_bnsl/tetrad-7.1.2-2/'
                          + netw.lower().replace('-19', '') + '_ref.tetrad')
        print('\n\n{} {}'.format(netw, len(ref.nodes)))
        for algo in ['pc.stable', 'pc-stable', 'mmhc', 'fges']:
            for file in glob(PATTERN.format(algo, netw)):
                pdag = read_bayesys(file)
                edges = [(e[0], t.value[3], e[1])
                         for e, t in pdag.edges.items()]
                pdag = PDAG(ref.nodes, edges)
                file = (file.split('\\')[-1]).replace('.csv', '')
                diffs = pdag.compared_to(ref)
                print('{}: {:.3f}'.format(file, diffs['f1']))
            print()
