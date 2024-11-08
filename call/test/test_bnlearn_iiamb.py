
#   Test calling the bnlearn PC-stable structure learning algorithm

import pytest

from core.common import EdgeType
from call.bnlearn import bnlearn_learn
from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from core.bn import BN
import testdata.example_pdags as ex_pdag


def test_bnlearn_iiamb_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        bnlearn_learn()


def test_bnlearn_iiamb_type_error_2():  # only one argument
    with pytest.raises(TypeError):
        bnlearn_learn(32.23)
    with pytest.raises(TypeError):
        bnlearn_learn('inter.iamb')


def test_bnlearn_iiamb_type_error_3():  # invalid types
    with pytest.raises(TypeError):
        bnlearn_learn(32.23, '/discrete/tiny/ab_cb.dsc')
    with pytest.raises(TypeError):
        bnlearn_learn('inter.iamb', [1, 2])


def test_bnlearn_iiamb_filenotfound_error_1():  # non-existent data file
    with pytest.raises(FileNotFoundError):
        bnlearn_learn('inter.iamb', 'nonexistent.txt')


def test_bnlearn_iiamb_value_error_1():  # DataFrame has too few columns
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        bnlearn_learn('inter.iamb', data)


def test_bnlearn_iiamb_value_error_2():  # Data file has too few columns
    with pytest.raises(ValueError):
        bnlearn_learn('inter.iamb', TESTDATA_DIR + '/discrete/tiny/ab.dsc')


def test_bnlearn_iiamb_ab_cb_1k_ok_1():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_cb.dsc')
    data = bn.generate_cases(1000)
    pdag, _ = bnlearn_learn('inter.iamb', data)
    print('\nPDAG learnt by iiamb from 1K rows of ab_cb:\n{}'.format(pdag))
    assert pdag == ex_pdag.ab_cb()


def test_bnlearn_iiamb_ab_cb_1k_ok_2():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_cb.dsc')
    data = bn.generate_cases(1000)
    pdag, _ = bnlearn_learn('inter.iamb', data, params={'iss': 10})
    print('\nPDAG learnt by iiamb from 1K rows of ab_cb:\n{}'.format(pdag))
    assert pdag == ex_pdag.ab_cb()


def test_bnlearn_iiamb_abc_1k_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = bn.generate_cases(1000)
    pdag, _ = bnlearn_learn('inter.iamb', data)
    print('\nPDAG learnt by iiamb from 1K rows of abc:\n{}'.format(pdag))
    assert pdag == ex_pdag.abc4()


def test_bnlearn_iiamb_abc_dual_1k_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc_dual.dsc')
    data = bn.generate_cases(1000)
    pdag, _ = bnlearn_learn('inter.iamb', data)
    print('\nPDAG learnt by iiamb from 1K rows of abc_dual:\n{}'
          .format(pdag))
    assert pdag == ex_pdag.abc_acyclic4()


def test_bnlearn_iiamb_and4_10_1k_ok():  # 1->2->4, 3->2, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    data = bn.generate_cases(1000)
    pdag, _ = bnlearn_learn('inter.iamb', data)
    print('\nPDAG learnt by iiamb from 1K rows of and4_10:\n{}'
          .format(pdag))
    print(ex_pdag.and4_11())
    assert pdag.edges == {('X1', 'X2'): EdgeType.UNDIRECTED,
                          ('X2', 'X3'): EdgeType.UNDIRECTED,
                          ('X2', 'X4'): EdgeType.UNDIRECTED}


def test_bnlearn_iiamb_cancer_1k_ok_1():  # Cancer, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = bn.generate_cases(1000)
    pdag, _ = bnlearn_learn('inter.iamb', data)
    print('\nPDAG learnt by iiamb from 1K rows of Cancer:\n{}'
          .format(pdag))
    assert pdag.edges == {('Cancer', 'Dyspnoea'): EdgeType.UNDIRECTED,
                          ('Cancer', 'Smoker'): EdgeType.UNDIRECTED,
                          ('Cancer', 'Xray'): EdgeType.UNDIRECTED}


def test_bnlearn_iiamb_cancer_1k_ok_2():  # Cancer, 1K rows, alpha = 0.001
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = bn.generate_cases(1000)
    pdag, _ = bnlearn_learn('inter.iamb', data, params={'alpha': 0.001})
    print('\nPDAG learnt by iiamb, 1K rows of Cancer (alpha=0.001):\n{}'
          .format(pdag))
    assert pdag.edges == {('Cancer', 'Smoker'): EdgeType.UNDIRECTED,
                          ('Cancer', 'Xray'): EdgeType.UNDIRECTED}


def test_bnlearn_iiamb_asia_1k_ok_1():  # Asia, 1K rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = bn.generate_cases(1000)
    pdag, _ = bnlearn_learn('inter.iamb', data)
    print('\nPDAG learnt by iiamb from 1K rows of Asia:\n{}'
          .format(pdag))
    assert pdag.edges == {('bronc', 'smoke'): EdgeType.UNDIRECTED,
                          ('bronc', 'dysp'): EdgeType.UNDIRECTED,
                          ('lung', 'either'): EdgeType.DIRECTED,
                          ('tub', 'either'): EdgeType.DIRECTED}


def test_bnlearn_iiamb_asia_1k_ok_2():  # Asia, 1K rows, alpha=0.01
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = bn.generate_cases(1000)
    pdag, _ = bnlearn_learn('inter.iamb', data, params={'alpha': 1E-4})
    print('\nPDAG learnt by iiamb from 1K rows Asia (alpha=1E-4):\n{}'
          .format(pdag))
    assert pdag.edges == {('bronc', 'smoke'): EdgeType.UNDIRECTED,
                          ('bronc', 'dysp'): EdgeType.UNDIRECTED,
                          ('lung', 'either'): EdgeType.DIRECTED,
                          ('tub', 'either'): EdgeType.DIRECTED}


# Gaussian datasets

def test_bnlearn_iiamb_gauss_1_ok():  # Gaussian example, 100 rows
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = Pandas.read(_in, dstype='continuous', N=100)
    pdag, trace = bnlearn_learn('inter.iamb', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
                                params={'test': 'mi-g'})
    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('F', '-', 'G'),
         ('B', '-', 'D')}


def test_bnlearn_iiamb_gauss_2_ok():  # Gaussian example, 100 rows, rev ord
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = Pandas.read(_in, dstype='continuous', N=100)
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = bnlearn_learn('inter.iamb', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
                                params={'test': 'mi-g'})
    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('F', '-', 'G'),
         ('B', '-', 'D')}


def test_bnlearn_iiamb_gauss_3_ok():  # Gaussian example, 5K rows
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = Pandas.read(_in, dstype='continuous')
    pdag, trace = bnlearn_learn('inter.iamb', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
                                params={'test': 'mi-g'})
    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('B', '-', 'D'),
         ('E', '->', 'F'),
         ('A', '->', 'F'),
         ('B', '->', 'C'),
         ('D', '->', 'F'),
         ('G', '->', 'F'),
         ('A', '->', 'C')}


def test_bnlearn_iiamb_gauss_4_ok():  # Gaussian example, 5K rows, rev ord
    _in = TESTDATA_DIR + '/simple/gauss.data.gz'
    data = Pandas.read(_in, dstype='continuous')
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = bnlearn_learn('inter.iamb', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
                                params={'test': 'mi-g'})
    print('\nPDAG learnt from 100 rows of gauss: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('B', '-', 'D'),
         ('E', '->', 'F'),
         ('A', '->', 'F'),
         ('B', '->', 'C'),
         ('D', '->', 'F'),
         ('G', '->', 'F'),
         ('A', '->', 'C')}


def test_bnlearn_iiamb_sachs_c_1_ok():  # Sachs gauss example, 1K rows
    _in = TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz'
    data = Pandas.read(_in, dstype='continuous')
    pdag, trace = bnlearn_learn('inter.iamb', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
                                params={'test': 'mi-g'})
    print('\nPDAG rom 1K rows of sachs_c: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('Mek', '-', 'Raf'),
         ('Jnk', '-', 'PKC'),
         ('Mek', '-', 'PKA'),
         ('PKC', '->', 'P38'),
         ('PIP3', '-', 'Plcg'),
         ('Mek', '-', 'PKC'),
         ('PKA', '->', 'Erk'),
         ('PIP3', '-', 'PKA'),
         ('PKA', '->', 'P38'),
         ('Akt', '->', 'Erk'),
         ('PKC', '-', 'Raf'),
         ('PKA', '-', 'Raf'),
         ('PIP2', '-', 'Plcg'),
         ('PIP2', '-', 'PIP3')}


def test_bnlearn_iiamb_sachs_c_2_ok():  # Sachs gauss example, rev, 1K rows
    _in = TESTDATA_DIR + '/experiments/datasets/sachs_c.data.gz'
    data = Pandas.read(_in, dstype='continuous')
    data.set_order(tuple(list(data.get_order())[::-1]))
    pdag, trace = bnlearn_learn('inter.iamb', data.sample, dstype='continuous',
                                context={'in': _in, 'id': 'gauss'},
                                params={'test': 'mi-g'})
    print('\nPDAG rom 1K rows of sachs_c: {}\n\n{}'.format(pdag, trace))
    edges = {(e[0], t.value[3], e[1]) for e, t in pdag.edges.items()}
    assert edges == \
        {('Mek', '-', 'Raf'),
         ('Jnk', '-', 'PKC'),
         ('Mek', '-', 'PKA'),
         ('PKC', '->', 'P38'),
         ('PIP3', '-', 'Plcg'),
         ('Mek', '-', 'PKC'),
         ('PKA', '->', 'Erk'),
         ('PIP3', '-', 'PKA'),
         ('PKA', '->', 'P38'),
         ('Akt', '->', 'Erk'),
         ('PKC', '-', 'Raf'),
         ('PKA', '-', 'Raf'),
         ('PIP2', '-', 'Plcg'),
         ('PIP2', '-', 'PIP3')}
