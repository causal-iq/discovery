
# Check bnbench and bnlearn metrics give same answers

import pytest
from pandas import DataFrame
from random import random
from os import remove

from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_score
import testdata.example_dags as dag
from fileio.common import TESTDATA_DIR
from core.graph import DAG
from core.metrics import dicts_same
from core.bn import BN
from fileio.pandas import Pandas

TYPES = ['loglik', 'bic', 'aic', 'bde', 'k2', 'bdj', 'bds']  # scores to test
DEFAULT_PARAMS = {'iss': 1.0, 'prior': 'uniform', 'base': 'e'}


# temp file, automatically removed
@pytest.fixture(scope="function")
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.csv'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


# --- Check scores for delete networks

# A, B unconnected
@requires_r_and_bnlearn
def test_score_bnlearn_a_b_3_ok():
    data = Pandas(DataFrame({'A': ['1', '0', '1'],
                             'B': ['1', '0', '1']}, dtype='category'))
    bnlearn = bnlearn_score(dag.a_b(), data, TYPES, DEFAULT_PARAMS)
    scores = dag.a_b().score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A, B unconnected, 4 rows
@requires_r_and_bnlearn
def test_score_bnlearn_a_b_4_ok():
    data = Pandas(DataFrame({'A': ['1', '1', '1', '0'],
                             'B': ['1', '0', '1', '1']}, dtype='category'))
    bnlearn = bnlearn_score(dag.a_b(), data, TYPES, DEFAULT_PARAMS)
    scores = dag.a_b().score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A, B unconnected, 4 rows
@requires_r_and_bnlearn
def test_score_bnlearn_a_b_5_ok():
    data = Pandas(DataFrame({'A': ['1', '0', '1', '0'],
                             'B': ['1', '0', '1', '0']}, dtype='category'))
    bnlearn = bnlearn_score(dag.a_b(), data, TYPES, DEFAULT_PARAMS)
    scores = dag.a_b().score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A, B unconnected, 7 rows
@requires_r_and_bnlearn
def test_score_bnlearn_a_b_6_ok():
    data = Pandas(DataFrame({'A': ['1', '0', '1', '0', '0', '0', '0'],
                             'B': ['0', '0', '0', '1', '0', '1', '0']},
                            dtype='category'))
    bnlearn = bnlearn_score(dag.a_b(), data, TYPES, DEFAULT_PARAMS)
    scores = dag.a_b().score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> B, 2 rows
@requires_r_and_bnlearn
def test_score_bnlearn_ab_1_ok():
    data = Pandas(DataFrame({'A': ['0', '1'], 'B': ['0', '1']},
                            dtype='category'))
    bnlearn = bnlearn_score(dag.ab(), data, TYPES, DEFAULT_PARAMS)
    scores = dag.ab().score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> B, 4 rows
@requires_r_and_bnlearn
def test_score_bnlearn_ab_2_ok():
    data = Pandas(DataFrame({'A': ['0', '1', '0', '1'],
                             'B': ['0', '1', '0', '1']}, dtype='category'))
    bnlearn = bnlearn_score(dag.ab(), data, TYPES, DEFAULT_PARAMS)
    scores = dag.ab().score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> B, 3 rows
@requires_r_and_bnlearn
def test_score_bnlearn_ab_3_ok():
    data = Pandas(DataFrame({'A': ['0', '1', '1'], 'B': ['0', '1', '1']},
                            dtype='category'))
    bnlearn = bnlearn_score(dag.ab(), data, TYPES, DEFAULT_PARAMS)
    scores = dag.ab().score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> B, 4 rows
@requires_r_and_bnlearn
def test_score_bnlearn_ab_4_ok():
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1'],
                             'B': ['0', '1', '0', '1']}, dtype='category'))
    bnlearn = bnlearn_score(dag.ab(), data, TYPES, DEFAULT_PARAMS)
    scores = dag.ab().score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> C <-- B, 2 parent combos
@requires_r_and_bnlearn
def test_score_bnlearn_ac_bc_1_ok():
    data = Pandas(DataFrame({'A': ['0', '1', '1'],
                             'B': ['0', '1', '1'],
                             'C': ['0', '1', '1']}, dtype='category'))
    bnlearn = bnlearn_score(dag.ac_bc(), data, TYPES, DEFAULT_PARAMS)
    scores = dag.ac_bc().score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> C <-- B, all parent combo
@requires_r_and_bnlearn
def test_score_bnlearn_ac_bc_2_ok():
    data = Pandas(DataFrame({'A': ['0', '1', '1', '0', '1'],
                             'B': ['0', '1', '1', '1', '0'],
                             'C': ['0', '1', '1', '1', '1']},
                            dtype='category'))
    bnlearn = bnlearn_score(dag.ac_bc(), data, TYPES, DEFAULT_PARAMS)
    scores = dag.ac_bc().score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# N1 --> N2, 12 rows
@requires_r_and_bnlearn
def test_score_bnlearn_heckerman_1_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/heckerman.csv',
                       dstype='categorical')
    dag = DAG(['N1', 'N2'], [('N1', '->', 'N2')])
    bnlearn = bnlearn_score(dag, data, TYPES, DEFAULT_PARAMS)
    scores = dag.score(data, TYPES, DEFAULT_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))


# N1 _|_ N2, 12 rows
@requires_r_and_bnlearn
def test_score_bnlearn_heckerman_2_ok():
    data = Pandas.read(TESTDATA_DIR + '/simple/heckerman.csv',
                       dstype='categorical')
    dag = DAG(['N1', 'N2'], [])
    bnlearn = bnlearn_score(dag, data, TYPES, DEFAULT_PARAMS)
    scores = dag.score(data, TYPES, DEFAULT_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))


# bnlearn score for ASIA, 1K
@requires_r_and_bnlearn
def test_score_bnlearn_asia_1k():
    asia = BN.read(TESTDATA_DIR + '/asia/asia.dsc')
    data = Pandas(asia.generate_cases(1000))
    bnlearn = bnlearn_score(asia.dag, data, TYPES, DEFAULT_PARAMS)
    scores = asia.dag.score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# bnlearn score ASIA, 1000K
@requires_r_and_bnlearn
def test_score_bnlearn_asia_10k():
    asia = BN.read(TESTDATA_DIR + '/asia/asia.dsc')
    data = Pandas(asia.generate_cases(10000))
    bnlearn = bnlearn_score(asia.dag, data, TYPES, DEFAULT_PARAMS)
    scores = asia.dag.score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# ALARM subset
@requires_r_and_bnlearn
def test_score_bnlearn_alarm_press():
    dag = DAG(['INT', 'KIN', 'VEN', 'PRE'],
              [('INT', '->', 'PRE'),
               ('KIN', '->', 'PRE'),
               ('VEN', '->', 'PRE')])
    data = Pandas(DataFrame({'INT': ['0', '1', '1', '1', '1'],
                             'KIN': ['0', '0', '0', '0', '1'],
                             'VEN': ['0', '1', '0', '0', '0'],
                             'PRE': ['1', '0', '0', '1', '1']},
                            dtype='category'))
    scores = dag.score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    bnlearn = bnlearn_score(dag, data, TYPES, DEFAULT_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))


# bnlearn score ALARM, 10K
@pytest.mark.slow
@requires_r_and_bnlearn
def test_score_bnlearn_alarm_10k():
    alarm = BN.read(TESTDATA_DIR + '/alarm/alarm.dsc')
    data = Pandas(alarm.generate_cases(10000))
    bnlearn = bnlearn_score(alarm.dag, data, TYPES, DEFAULT_PARAMS)
    scores = alarm.dag.score(data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# bnlearn PATHFINDER
@pytest.mark.slow
@requires_r_and_bnlearn
def test_score_bnlearn_pathfinder_25k():
    pathfinder = BN.read(TESTDATA_DIR + '/pathfinder/pathfinder.dsc')
    data = Pandas(pathfinder.generate_cases(25000))
    bnlearn = bnlearn_score(pathfinder.dag, data, TYPES, DEFAULT_PARAMS)
    scores = pathfinder.dag.score(data, TYPES, DEFAULT_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))


# Check score for continuous networks

# score X --> Y Gaussian Network
@requires_r_and_bnlearn
def test_score_bnlearn_x_y_score():
    x_y = BN.read(TESTDATA_DIR + '/xdsl/x_y.xdsl')
    print('\n\n{}'.format(x_y.dag))
    data = Pandas(x_y.generate_cases(3))
    bnlearn = bnlearn_score(x_y.dag, data, ['bic-g'], {'k': 1})
    scores = x_y.dag.score(data, ['bic-g'], {'k': 1})
    print(bnlearn)
    print(scores)
