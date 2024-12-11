
#   Test calling the tetrad PC-stable structure learning algorithm

import pytest

from core.common import EdgeType
from call.tetrad import tetrad_learn
from fileio.common import TESTDATA_DIR
from fileio.numpy import NumPy
from core.bn import BN
from core.graph import PDAG, DAG


@pytest.fixture(scope="module")  # simple ab DataFrame
def ab3():
    return NumPy.read(TESTDATA_DIR + '/simple/ab_3.csv',
                      dstype='categorical')


@pytest.fixture(scope="module")  # simple ab DataFrame
def xy3():
    return NumPy.read(TESTDATA_DIR + '/simple/xy_3.csv',
                      dstype='continuous')


def test_tetrad_fges_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        tetrad_learn()


def test_tetrad_fges_type_error_2(ab3):  # only one argument
    with pytest.raises(TypeError):
        tetrad_learn(ab3)
    with pytest.raises(TypeError):
        tetrad_learn('fges')


def test_tetrad_fges_type_error_3(ab3):  # bad algorithm type
    with pytest.raises(TypeError):
        tetrad_learn(1, ab3)
    with pytest.raises(TypeError):
        tetrad_learn(None, ab3)
    with pytest.raises(TypeError):
        tetrad_learn(['fges'], ab3)


def test_tetrad_fges_type_error_4(ab3):  # bad data type
    with pytest.raises(TypeError):
        tetrad_learn('fges', ab3.as_df())
    with pytest.raises(TypeError):
        tetrad_learn('fges', None)
    with pytest.raises(TypeError):
        tetrad_learn('fges', False)


def test_tetrad_fges_type_error_5(ab3):  # bad context type
    with pytest.raises(TypeError):
        tetrad_learn('fges', ab3, context='invalid')
    with pytest.raises(TypeError):
        tetrad_learn('fges', ab3, context=True)


def test_tetrad_fges_type_error_6(ab3):  # Bad params type
    with pytest.raises(TypeError):
        tetrad_learn('fges', ab3, params=True)
    with pytest.raises(TypeError):
        tetrad_learn('fges', ab3, params=['score', 'bic'])
    with pytest.raises(TypeError):
        tetrad_learn('fges', ab3, params='bic')


def test_tetrad_fges_type_error_7(ab3):  # Wrong type for score param
    with pytest.raises(TypeError):
        tetrad_learn('fges', ab3, params={'score': 3})


def test_tetrad_fges_type_error_8(ab3):  # Wrong type for k param
    with pytest.raises(TypeError):
        tetrad_learn('fges', ab3, params={'k': None})


def test_tetrad_fges_value_error_1(ab3):  # Invalid algorithm name
    with pytest.raises(ValueError):
        tetrad_learn('unknown', ab3)


def test_tetrad_fges_value_error_2(ab3):  # Context has wrong elements
    with pytest.raises(ValueError):
        tetrad_learn('fges', ab3, {'invalid': 3})


def test_tetrad_fges_value_error_3(ab3):  # Context missing mandatory elements
    with pytest.raises(ValueError):
        tetrad_learn('fges', ab3, {'dataset': True})


def test_tetrad_fges_value_error_4(ab3):  # Params has invalid key
    with pytest.raises(ValueError):
        tetrad_learn('fges', ab3, params={'invalid': 3})


def test_tetrad_fges_value_error_5(ab3):  # Params has bad score
    with pytest.raises(ValueError):
        tetrad_learn('fges', ab3, params={'score': 'unknown'})


def test_tetrad_fges_value_error_6(ab3):  # categorical & bic-g incompatible
    with pytest.raises(ValueError):
        tetrad_learn('fges', ab3, params={'score': 'bic-g'})


def test_tetrad_fges_value_error_7(xy3):  # continuous & bic incompatible
    with pytest.raises(ValueError):
        tetrad_learn('fges', xy3, params={'score': 'bic'})


def test_tetrad_fges_value_error_8(ab3):  # Params has bad k penalty
    with pytest.raises(ValueError):
        tetrad_learn('fges', ab3, params={'k': 4})


def test_tetrad_fges_value_error_9(ab3):  # Params has bad iss value
    with pytest.raises(ValueError):
        tetrad_learn('fges', ab3, params={'iss': 2})


def test_tetrad_fges_ab_ok_1(ab3):  # Learning from an ab datafile, no context
    graph, _ = tetrad_learn('fges', ab3)
    print('\nGraph learnt from ab_3.csv by Tetrad FGES:\n{}'.format(graph))
    assert isinstance(graph, DAG)
    assert graph.nodes == ['A', 'B']
    assert graph.edges == {}


def test_tetrad_fges_ab_ok_2(ab3):  # Learning from an ab datafile, context
    _, trace = tetrad_learn('fges', ab3, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from ab_3.csv by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, DAG)
    assert trace.result.nodes == ['A', 'B']
    assert trace.result.edges == {}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_ab_ok_3():  # Learning generated data, no context
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    graph, _ = tetrad_learn('fges', data)
    print('\nGraph learnt from ab, N=1K by Tetrad FGES:\n{}'.format(graph))
    assert isinstance(graph, DAG)
    assert graph.nodes == ['A', 'B']
    assert graph.edges == {}


def test_tetrad_fges_ab_ok_4():  # Learning generated data, check trace
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data,
                            context={'in': 'in', 'id': 'TEST/AB_OK_4'})
    print('\nGraph learnt from ab, N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, DAG)
    assert trace.result.nodes == ['A', 'B']
    assert trace.result.edges == {}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_ab_cb_ok_1():  # A -> B <- C, N=100
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_cb.dsc')
    data = NumPy.from_df(bn.generate_cases(100), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from a->b<-c, N=100 by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {('B', 'C'): EdgeType.UNDIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_ab_cb_ok_2():  # A -> B <- C, N=1K
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_cb.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from a->b<-c, N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, DAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {('A', 'B'): EdgeType.DIRECTED,
                                  ('C', 'B'): EdgeType.DIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_abc_ok_1():  # A -> B -> C, N=1K
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from a->b->c, N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {('A', 'B'): EdgeType.UNDIRECTED,
                                  ('B', 'C'): EdgeType.UNDIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_abc_dual_ok_1():  # A -> B -> C, A -> C N=10
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc_dual.dsc')
    data = NumPy.from_df(bn.generate_cases(10), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from a->b->c, a->c N=100 by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {('A', 'C'): EdgeType.UNDIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_abc_dual_ok_2():  # A -> B -> C, A -> C N=100
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc_dual.dsc')
    data = NumPy.from_df(bn.generate_cases(100), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from a->b->c, a->c N=100 by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {('A', 'C'): EdgeType.UNDIRECTED,
                                  ('B', 'C'): EdgeType.UNDIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_abc_dual_ok_3():  # A -> B -> C, A -> C N=1K
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc_dual.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from a->b->c, a->c N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {('A', 'B'): EdgeType.UNDIRECTED,
                                  ('A', 'C'): EdgeType.UNDIRECTED,
                                  ('B', 'C'): EdgeType.UNDIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_and4_10_ok_1():  # and4_10, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from and4_10 N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['X1', 'X2', 'X3', 'X4']
    assert trace.result.edges == {('X1', 'X2'): EdgeType.UNDIRECTED,
                                  ('X2', 'X3'): EdgeType.UNDIRECTED,
                                  ('X2', 'X4'): EdgeType.UNDIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_and4_10_ok_2():  # and4_10, 10K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    data = NumPy.from_df(bn.generate_cases(10000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from and4_10 N=10K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, DAG)
    assert trace.result.nodes == ['X1', 'X2', 'X3', 'X4']
    assert trace.result.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                                  ('X2', 'X4'): EdgeType.DIRECTED,
                                  ('X3', 'X2'): EdgeType.DIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_cancer_ok_1():  # Cancer, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from Cancer N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                                  'Xray']
    assert {('Cancer', 'Dyspnoea'): EdgeType.UNDIRECTED,
            ('Cancer', 'Smoker'): EdgeType.UNDIRECTED,
            ('Cancer', 'Xray'): EdgeType.UNDIRECTED} == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_cancer_ok_2():  # Cancer, 10K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = NumPy.from_df(bn.generate_cases(10000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from Cancer N=10K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, DAG)
    assert trace.result.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                                  'Xray']
    assert {('Cancer', 'Dyspnoea'): EdgeType.DIRECTED,
            ('Pollution', 'Cancer'): EdgeType.DIRECTED,
            ('Smoker', 'Cancer'): EdgeType.DIRECTED,
            ('Cancer', 'Xray'): EdgeType.DIRECTED} == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_asia_ok_1():  # Asia, 100 rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = NumPy.from_df(bn.generate_cases(100), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from Cancer N=10K by Tetrad FGES:\n{}'
          .format(trace.result))
    print('\nGraph learnt from Asia N=100 by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['asia', 'bronc', 'dysp', 'either', 'lung',
                                  'smoke', 'tub', 'xray']
    assert {('bronc', 'smoke'): EdgeType.UNDIRECTED,
            ('bronc', 'dysp'): EdgeType.UNDIRECTED,
            ('either', 'xray'): EdgeType.UNDIRECTED,
            ('either', 'lung'): EdgeType.UNDIRECTED,
            ('lung', 'smoke'): EdgeType.UNDIRECTED} == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_asia_ok_2():  # Asia, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from Asia N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['asia', 'bronc', 'dysp', 'either', 'lung',
                                  'smoke', 'tub', 'xray']
    assert {('bronc', 'smoke'): EdgeType.UNDIRECTED,
            ('bronc', 'dysp'): EdgeType.UNDIRECTED,
            ('either', 'xray'): EdgeType.DIRECTED,
            ('lung', 'either'): EdgeType.DIRECTED,
            ('lung', 'smoke'): EdgeType.UNDIRECTED,
            ('tub', 'either'): EdgeType.DIRECTED} == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_asia_ok_3():  # Asia, 10K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = NumPy.from_df(bn.generate_cases(10000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from Asia N=10K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['asia', 'bronc', 'dysp', 'either', 'lung',
                                  'smoke', 'tub', 'xray']
    assert {('bronc', 'smoke'): EdgeType.UNDIRECTED,
            ('bronc', 'dysp'): EdgeType.DIRECTED,
            ('either', 'xray'): EdgeType.DIRECTED,
            ('lung', 'either'): EdgeType.DIRECTED,
            ('either', 'dysp'): EdgeType.DIRECTED,
            ('asia', 'tub'): EdgeType.UNDIRECTED,
            ('lung', 'smoke'): EdgeType.UNDIRECTED,
            ('tub', 'either'): EdgeType.DIRECTED} == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_child_1k_ok_1():  # Child, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/child.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'})
    print('\nGraph learnt from Child N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert ['Age', 'BirthAsphyxia', 'CO2', 'CO2Report', 'CardiacMixing',
            'ChestXray', 'Disease', 'DuctFlow', 'Grunting', 'GruntingReport',
            'HypDistrib', 'HypoxiaInO2', 'LVH', 'LVHreport', 'LowerBodyO2',
            'LungFlow', 'LungParench', 'RUQO2',
            'Sick', 'XrayReport'] == trace.result.nodes
    assert {('Age', 'Disease'): EdgeType.UNDIRECTED,
            ('Age', 'Sick'): EdgeType.UNDIRECTED,
            ('CO2', 'CO2Report'): EdgeType.UNDIRECTED,
            ('CardiacMixing', 'Disease'): EdgeType.UNDIRECTED,
            ('CardiacMixing', 'HypoxiaInO2'): EdgeType.UNDIRECTED,
            ('ChestXray', 'XrayReport'): EdgeType.DIRECTED,
            ('Disease', 'LVH'): EdgeType.UNDIRECTED,
            ('Disease', 'DuctFlow'): EdgeType.UNDIRECTED,
            ('DuctFlow', 'HypDistrib'): EdgeType.UNDIRECTED,
            ('Grunting', 'GruntingReport'): EdgeType.UNDIRECTED,
            ('Grunting', 'LungParench'): EdgeType.UNDIRECTED,
            ('HypoxiaInO2', 'LowerBodyO2'): EdgeType.UNDIRECTED,
            ('LVH', 'LVHreport'): EdgeType.UNDIRECTED,
            ('LungFlow', 'ChestXray'): EdgeType.DIRECTED,
            ('Disease', 'LungFlow'): EdgeType.UNDIRECTED,
            ('CO2', 'LungParench'): EdgeType.UNDIRECTED,
            ('LungParench', 'ChestXray'): EdgeType.DIRECTED,
            ('Disease', 'LungParench'): EdgeType.UNDIRECTED,
            ('HypoxiaInO2', 'RUQO2'): EdgeType.UNDIRECTED
            } == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_gauss_1_ok():  # Gaussian example, 100 rows
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous', N=100)
    pdag, trace = tetrad_learn('fges', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'gauss'})

    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))

    assert pdag.nodes == ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('A', '->', 'F'),
         ('B', '-', 'D'),
         ('A', '->', 'C'),
         ('E', '->', 'F'),
         ('D', '->', 'F'),
         ('B', '->', 'C'),
         ('G', '->', 'F')}


def test_tetrad_fges_gauss_2_ok():  # Gaussian example, 100 rows, rev ord
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous', N=100)
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = tetrad_learn('fges', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'gauss'})

    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))

    assert pdag.nodes == ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('A', '->', 'F'),
         ('B', '-', 'D'),
         ('A', '->', 'C'),
         ('E', '->', 'F'),
         ('D', '->', 'F'),
         ('B', '->', 'C'),
         ('G', '->', 'F')}


def test_tetrad_fges_gauss_3_ok():  # Gaussian example, 5K rows
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous', N=5000)
    pdag, trace = tetrad_learn('fges', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'gauss'})

    print('\nPDAG learnt from 5K rows of gauss: {}\n\n{}'.format(pdag, trace))

    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert pdag.nodes == ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    assert edges == \
        {('A', '->', 'F'),
         ('B', '-', 'D'),
         ('A', '->', 'C'),
         ('E', '->', 'F'),
         ('D', '->', 'F'),
         ('B', '->', 'C'),
         ('G', '->', 'F')}


def test_tetrad_fges_gauss_4_ok():  # Gaussian example, 5K rows, rev ord
    data = NumPy.read(TESTDATA_DIR + '/simple/gauss.data.gz',
                      dstype='continuous', N=5000)
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = tetrad_learn('fges', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'gauss'})

    print('\nPDAG learnt from 5K rows of gauss: {}\n\n{}'.format(pdag, trace))

    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert pdag.nodes == ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    assert edges == \
        {('A', '->', 'F'),
         ('B', '-', 'D'),
         ('A', '->', 'C'),
         ('E', '->', 'F'),
         ('D', '->', 'F'),
         ('B', '->', 'C'),
         ('G', '->', 'F')}


def test_tetrad_fges_sachs_c_1_ok():  # Sachs gauss example, 1K rows
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz',
                      dstype='continuous', N=1000)
    pdag, trace = tetrad_learn('fges', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'gauss'})

    assert pdag.nodes == ['Akt', 'Erk', 'Jnk', 'Mek', 'P38', 'PIP2', 'PIP3',
                          'PKA', 'PKC', 'Plcg', 'Raf']
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('Mek', '-', 'Raf'),
         ('PKA', '->', 'Erk'),
         ('P38', '-', 'PKA'),
         ('Mek', '-', 'PKC'),
         ('Jnk', '-', 'PKC'),
         ('Plcg', '->', 'PIP2'),
         ('Akt', '->', 'Erk'),
         ('PKC', '-', 'Raf'),
         ('PIP3', '->', 'PIP2'),
         ('P38', '-', 'PKC')}


def test_tetrad_fges_sachs_c_2_ok():  # Sachs gauss example, rev, 1K rows
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz',
                      dstype='continuous', N=1000)
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = tetrad_learn('fges', data, params={'score': 'bic-g'},
                               context={'in': 'in', 'id': 'gauss'})

    assert pdag.nodes == ['Akt', 'Erk', 'Jnk', 'Mek', 'P38', 'PIP2', 'PIP3',
                          'PKA', 'PKC', 'Plcg', 'Raf']
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('Mek', '-', 'Raf'),
         ('PKA', '->', 'Erk'),
         ('P38', '-', 'PKA'),
         ('Mek', '-', 'PKC'),
         ('Jnk', '-', 'PKC'),
         ('Plcg', '->', 'PIP2'),
         ('Akt', '->', 'Erk'),
         ('PKC', '-', 'Raf'),
         ('PIP3', '->', 'PIP2'),
         ('P38', '-', 'PKC')}


# BDeu Score

def test_tetrad_fges_bdeu_ab_ok_2(ab3):  # AB 3 rows
    _, trace = tetrad_learn('fges', ab3, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde', 'iss': 1})
    print('\nGraph learnt from ab_3.csv by Tetrad FGES (BDeu):\n{}'
          .format(trace.result))
    assert isinstance(trace.result, DAG)
    assert trace.result.nodes == ['A', 'B']
    assert trace.result.edges == {}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_ab_cb_ok_1():  # A -> B <- C, N=100
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_cb.dsc')
    data = NumPy.from_df(bn.generate_cases(100), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from a->b<-c, N=100 by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {('B', 'C'): EdgeType.UNDIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_ab_cb_ok_2():  # A -> B <- C, N=1K
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_cb.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from a->b<-c, N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, DAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {('A', 'B'): EdgeType.DIRECTED,
                                  ('C', 'B'): EdgeType.DIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_abc_ok_1():  # A -> B -> C, N=1K
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from a->b->c, N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {('A', 'B'): EdgeType.UNDIRECTED,
                                  ('B', 'C'): EdgeType.UNDIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_abc_dual_ok_1():  # A -> B -> C, A -> C N=10
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc_dual.dsc')
    data = NumPy.from_df(bn.generate_cases(10), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from a->b->c, a->c N=100 by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_abc_dual_ok_2():  # A -> B -> C, A -> C N=100
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc_dual.dsc')
    data = NumPy.from_df(bn.generate_cases(100), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from a->b->c, a->c N=100 by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_abc_dual_ok_3():  # A -> B -> C, A -> C N=1K
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc_dual.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from a->b->c, a->c N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['A', 'B', 'C']
    assert trace.result.edges == {('A', 'C'): EdgeType.DIRECTED,
                                  ('B', 'C'): EdgeType.DIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_and4_10_ok_1():  # and4_10, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from and4_10 N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['X1', 'X2', 'X3', 'X4']
    assert trace.result.edges == {('X2', 'X3'): EdgeType.UNDIRECTED,
                                  ('X2', 'X4'): EdgeType.UNDIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_and4_10_ok_2():  # and4_10, 10K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    data = NumPy.from_df(bn.generate_cases(10000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from and4_10 N=10K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, DAG)
    assert trace.result.nodes == ['X1', 'X2', 'X3', 'X4']
    assert trace.result.edges == {('X1', 'X2'): EdgeType.DIRECTED,
                                  ('X2', 'X4'): EdgeType.DIRECTED,
                                  ('X3', 'X2'): EdgeType.DIRECTED}
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_cancer_ok_1():  # Cancer, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from Cancer N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                                  'Xray']
    assert {('Cancer', 'Smoker'): EdgeType.UNDIRECTED,
            ('Cancer', 'Xray'): EdgeType.UNDIRECTED} == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_cancer_ok_2():  # Cancer, 10K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = NumPy.from_df(bn.generate_cases(10000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from Cancer N=10K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, DAG)
    assert trace.result.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                                  'Xray']
    assert {('Cancer', 'Dyspnoea'): EdgeType.DIRECTED,
            ('Pollution', 'Cancer'): EdgeType.DIRECTED,
            ('Smoker', 'Cancer'): EdgeType.DIRECTED,
            ('Cancer', 'Xray'): EdgeType.DIRECTED} == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_asia_ok_1():  # Asia, 100 rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = NumPy.from_df(bn.generate_cases(100), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from Asia N=100 by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['asia', 'bronc', 'dysp', 'either', 'lung',
                                  'smoke', 'tub', 'xray']
    assert {('bronc', 'dysp'): EdgeType.UNDIRECTED,
            ('either', 'xray'): EdgeType.DIRECTED,
            ('lung', 'either'): EdgeType.DIRECTED,
            ('lung', 'smoke'): EdgeType.UNDIRECTED,
            ('tub', 'either'): EdgeType.DIRECTED} == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_asia_ok_2():  # Asia, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from Asia N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['asia', 'bronc', 'dysp', 'either', 'lung',
                                  'smoke', 'tub', 'xray']
    assert {('bronc', 'dysp'): EdgeType.DIRECTED,
            ('bronc', 'smoke'): EdgeType.UNDIRECTED,
            ('either', 'dysp'): EdgeType.DIRECTED,
            ('either', 'xray'): EdgeType.DIRECTED,
            ('lung', 'either'): EdgeType.DIRECTED,
            ('lung', 'smoke'): EdgeType.UNDIRECTED,
            ('tub', 'either'): EdgeType.DIRECTED} == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_asia_ok_3():  # Asia, 10K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = NumPy.from_df(bn.generate_cases(10000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from Asia N=10K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert trace.result.nodes == ['asia', 'bronc', 'dysp', 'either', 'lung',
                                  'smoke', 'tub', 'xray']
    assert {('bronc', 'dysp'): EdgeType.DIRECTED,
            ('bronc', 'smoke'): EdgeType.UNDIRECTED,
            ('either', 'dysp'): EdgeType.DIRECTED,
            ('either', 'xray'): EdgeType.DIRECTED,
            ('lung', 'either'): EdgeType.DIRECTED,
            ('lung', 'smoke'): EdgeType.UNDIRECTED,
            ('tub', 'either'): EdgeType.DIRECTED} == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']


def test_tetrad_fges_bdeu_child_1k_ok_1():  # Child, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/child.dsc')
    data = NumPy.from_df(bn.generate_cases(1000), dstype='categorical',
                         keep_df=False)
    _, trace = tetrad_learn('fges', data, context={'in': 'in', 'id': 'id'},
                            params={'score': 'bde'})
    print('\nGraph learnt from Child N=1K by Tetrad FGES:\n{}'
          .format(trace.result))
    assert isinstance(trace.result, PDAG)
    assert ['Age', 'BirthAsphyxia', 'CO2', 'CO2Report', 'CardiacMixing',
            'ChestXray', 'Disease', 'DuctFlow', 'Grunting', 'GruntingReport',
            'HypDistrib', 'HypoxiaInO2', 'LVH', 'LVHreport', 'LowerBodyO2',
            'LungFlow', 'LungParench', 'RUQO2',
            'Sick', 'XrayReport'] == trace.result.nodes
    assert {('Age', 'Disease'): EdgeType.UNDIRECTED,
            ('Age', 'Sick'): EdgeType.UNDIRECTED,
            ('CO2', 'CO2Report'): EdgeType.UNDIRECTED,
            ('CardiacMixing', 'Disease'): EdgeType.UNDIRECTED,
            ('CardiacMixing', 'HypoxiaInO2'): EdgeType.UNDIRECTED,
            ('ChestXray', 'XrayReport'): EdgeType.DIRECTED,
            ('Disease', 'LVH'): EdgeType.UNDIRECTED,
            ('Disease', 'DuctFlow'): EdgeType.UNDIRECTED,
            ('DuctFlow', 'HypDistrib'): EdgeType.UNDIRECTED,
            ('Grunting', 'GruntingReport'): EdgeType.UNDIRECTED,
            ('Grunting', 'LungParench'): EdgeType.UNDIRECTED,
            ('HypoxiaInO2', 'LowerBodyO2'): EdgeType.UNDIRECTED,
            ('LVH', 'LVHreport'): EdgeType.UNDIRECTED,
            ('LungFlow', 'ChestXray'): EdgeType.DIRECTED,
            ('Disease', 'LungFlow'): EdgeType.UNDIRECTED,
            ('CO2', 'LungParench'): EdgeType.UNDIRECTED,
            ('LungParench', 'ChestXray'): EdgeType.DIRECTED,
            ('Disease', 'LungParench'): EdgeType.UNDIRECTED,
            ('HypoxiaInO2', 'RUQO2'): EdgeType.UNDIRECTED
            } == trace.result.edges
    assert trace.trace['activity'] == ['init', 'stop']
