
import pytest
from pandas import DataFrame

from core.metrics import dicts_same
from core.score import free_params, dag_score, ENTROPY_SCORES, BAYESIAN_SCORES
from core.bn import BN
from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_score
import testdata.example_dags as dag
from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from fileio.numpy import NumPy
from fileio.oracle import Oracle

ENTROPY_PARAMS = {'base': 'e', 'k': 1.0}
BAYESIAN_PARAMS = {'iss': 1.0, 'prior': 'uniform'}


# --- Failure cases

# bad primary arg types for DAG.score
def test_graph_score_type_error_1():
    graph = dag.ab()
    with pytest.raises(TypeError):
        graph.score()
    with pytest.raises(TypeError):
        graph.score(10, 'bic', {})
    with pytest.raises(TypeError):
        graph.score(10, 'bic', {})
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '0'], 'B': ['1', '1']},
                              dtype='category'), 37)
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '0'], 'B': ['1', '1']},
                              dtype='category'), 'bic', True)


# bad score type
def test_graph_score_type_error_2():
    graph = dag.ab()
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '1'], 'B': ['1', '1']},
                              dtype='category'), [37])


# bad 'base' score param type
def test_graph_score_type_error_3():
    graph = dag.ab()
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '1'], 'B': ['1', '0']},
                              dtype='category'), 'bic', {'base': 2.2})
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '1'], 'B': ['1', '0']},
                              dtype='category'), 'bic', {'base': True})


# bad 'prior' score param type
def test_graph_score_type_error_4():
    graph = dag.ab()
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '1'], 'B': ['1', '0']},
                              dtype='category'), 'bde', {'prior': 12})


# bad 'iss' score param type
def test_graph_score_type_error_5():
    graph = dag.ab()
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '1'], 'B': ['1', '0']},
                              dtype='category'),
                    'bds', {'prior': 'uniform', 'iss': 'should be num'})


def test_graph_score_type_error_6():  # bad 'k' score param type
    graph = dag.ab()
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '1'], 'B': ['1', '0']},
                              dtype='category'), 'bic', {'k': 'should be num'})


# DAG / Data column mismatch
def test_graph_score_value_error_7():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '1'], 'C': ['0', '1']},
                            dtype='category'))
    with pytest.raises(ValueError):
        graph.score(data, 'aic')


# single-valued variables
def test_graph_score_value_error_8():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['0', '0']},
                            dtype='category'))
    with pytest.raises(ValueError):
        graph.score(data, 'aic')


# bad arg types
def test_dag_score_type_error_1():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['0', '1']},
                            dtype='category'))
    with pytest.raises(TypeError):
        dag_score({'A': ['1', '0'], 'B': ['0', '1']}, data, 'bic', {})
    with pytest.raises(TypeError):
        dag_score('graph', data, 'bic', {})
    with pytest.raises(TypeError):
        dag_score('graph', data, 37, {})
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bic', ['base'])
    with pytest.raises(TypeError):
        dag_score(graph, None, 'bic', {})


# bad base type
def test_dag_score_type_error_2():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['0', '1']},
                            dtype='category'))
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bic', {'base': []})


# bad k type
def test_dag_score_type_error_3():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['0', '1']},
                            dtype='category'))
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bic', {'k': 'should be int/float'})
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bic', {'k': {}})


# bad prior type
def test_dag_score_type_error_4():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['0', '1']},
                            dtype='category'))
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bde', {'prior': 1})
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bde', {'prior': {}})


# bad iss type
def test_dag_score_type_error_5():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['0', '1']},
                            dtype='category'))
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bds', {'iss': 'should be int/float'})
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bds', {'iss': {}})


# cannot score an oracle type
def test_dag_score_type_error_6():
    bn = BN.read(TESTDATA_DIR + '/xdsl/ab.xdsl')
    with pytest.raises(TypeError):
        dag_score(bn.dag, Oracle(bn), 'bic', {})


# unsupported score types
def test_dag_score_value_error_2():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['0', '1']},
                  dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, [], {})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'unsupported', {})
    with pytest.raises(ValueError):
        dag_score(graph, data, ['unsupported', 'bic'], {})


# single-valued data type
def test_dag_score_value_error_3():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '1']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {})


# data / dag column mismatch
def test_dag_score_value_error_4():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'C': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {})


# unknown score parameter
def test_dag_score_value_error_5():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'unsupported': 3})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'base': 2, 'unsupported': 3})


# bad "base" score param value
def test_dag_score_value_error_6():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'base': 7})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'base': '2'})


# bad "prior" score param value
def test_dag_score_value_error_7():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'prior': 'unsupported'})


# bad "iss" score param value
def test_dag_score_value_error_8():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '2']},
                  dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'iss': 0})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'iss': 0.0})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'iss': -1.0})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'iss': 1E-10})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'iss': 10000000})


# bad "k" score param value
def test_dag_score_value_error_9():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'k': 0})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'k': 0.0})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'k': -1.0})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'k': 1E-10})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'k': 10000000})


# Test for irrelevant score parameters disabled for now (error_11 & 8)

# irrelevant parameters for entropy
def xtest_dag_score_value_error_10():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'prior': 'uniform'})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'aic', {'prior': 'uniform'})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'iss': 2.0})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'aic', {'iss': 10})


# irrelevant parameters for bayesian
def xtest_dag_score_value_error_11():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bde', {'k': 4})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'k2', {'k': 4})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bde', {'k': 10.0})


# --- Successful score cases

# A --> B, 2 rows
@requires_r_and_bnlearn
def test_dag_score_ab1():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '1'], 'B': ['0', '1']},
                            dtype='category'))
    assert free_params(graph, data.as_df()) == 3
    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))
    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -3.5, 'loglik': -2, 'aic': -5})


# A --> B, 2 rows, set k = 2
@requires_r_and_bnlearn
def test_dag_score_ab2():
    graph = dag.ab()  # A --> B
    data = Pandas(DataFrame({'A': ['0', '1'], 'B': ['0', '1']},
                            dtype='category'))
    assert free_params(graph, data.as_df()) == 3
    params = dict(ENTROPY_PARAMS)
    params.update({'k': 2})
    scores = dag_score(graph, data, ENTROPY_SCORES, params)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, params)
    assert dicts_same(bnlearn, dict(scores.sum()))
    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2, 'k': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -5, 'loglik': -2, 'aic': -8})


# A --> B, 4 rows
@requires_r_and_bnlearn
def test_dag_score_ab3():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1'],
                             'B': ['0', '1', '0', '1']},
                  dtype='category'))
    assert free_params(graph, data.as_df()) == 3

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -11, 'loglik': -8, 'aic': -11})


# A --> B, 4 rows
@requires_r_and_bnlearn
def test_dag_score_ab4():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1'],
                             'B': ['0', '1', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.as_df()) == 3

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -9, 'loglik': -6, 'aic': -9})


# A --> B, 4 rows
@requires_r_and_bnlearn
def test_dag_score_ab5():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '1', '1', '1'],
                             'B': ['0', '1', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.as_df()) == 3

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -6.245112498, 'loglik': -3.245112498,
                       'aic': -6.245112498})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -1.879966188, 'loglik': -0.9768762012,
                       'aic': -3.976876201})


# A --> B, 4 rows, 3 states
@requires_r_and_bnlearn
def test_dag_score_ab6():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1'],
                             'B': ['0', '1', '1', '2']},
                            dtype='category'))

    assert free_params(graph, data.as_df()) == 5

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -13, 'loglik': -8, 'aic': -13})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -3.913389944, 'loglik': -2.408239965,
                       'aic': -7.408239965})


# A --> B, 4 rows, 3 states
@requires_r_and_bnlearn
def test_dag_score_ab7():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1', '2', '2', '2'],
                             'B': ['0', '1', '1', '2', '0', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.as_df()) == 8

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -28.88090414, 'loglik': -17.65148445,
                       'aic': -25.65148445})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    print(dict(scores.sum()))
    assert dicts_same(dict(scores.sum()),
                      {'bic': -8.694018449, 'loglik': -5.313626289,
                       'aic': -13.31362629})


# A --> B, 2 rows, Bayesian scores
@requires_r_and_bnlearn
def test_dag_score_ab8():  # Bayesian scores for A --> B, 2 rows
    graph = dag.ab()  # A --> B
    data = Pandas(DataFrame({'A': ['0', '1'], 'B': ['0', '1']},
                            dtype='category'))
    assert free_params(graph, data.as_df()) == 3
    scores = dag_score(graph, data, BAYESIAN_SCORES, BAYESIAN_PARAMS)
    bnlearn = bnlearn_score(graph, data, BAYESIAN_SCORES,
                            BAYESIAN_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))
    assert dicts_same(dict(scores.sum()),
                      {'bde': -3.465735903, 'bdj': -4.564348191,
                       'bds': -3.465735903, 'k2': -3.178053830})


# Bayesian scores for A --> B, 2 rows, ISS =5
@requires_r_and_bnlearn
def test_dag_score_ab9():
    graph = dag.ab()  # A --> B
    data = Pandas(DataFrame({'A': ['0', '1'], 'B': ['0', '1']},
                            dtype='category'))
    assert free_params(graph, data.as_df()) == 3
    params = BAYESIAN_PARAMS.copy()
    params.update({'iss': 5})
    scores = dag_score(graph, data, BAYESIAN_SCORES, params)
    bnlearn = bnlearn_score(graph, data, BAYESIAN_SCORES, params)
    assert dicts_same(bnlearn, dict(scores.sum()))
    assert dicts_same(dict(scores.sum()),
                      {'bde': -2.954910279, 'bdj': -4.564348191,
                       'bds': -2.954910279, 'k2': -3.178053830})


# Bayesian scores, A --> B, 8 rows
@requires_r_and_bnlearn
def test_dag_score_ab10():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1', '2', '2', '2'],
                             'B': ['0', '1', '1', '2', '0', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.as_df()) == 8

    scores = dag_score(graph, data, BAYESIAN_SCORES, BAYESIAN_PARAMS)
    bnlearn = bnlearn_score(graph, data, BAYESIAN_SCORES,
                            BAYESIAN_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))


# Bayesian scores, A --> B, 8 rows, ISS=10.0
@requires_r_and_bnlearn
def test_dag_score_ab11():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1', '2', '2', '2'],
                             'B': ['0', '1', '1', '2', '0', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.as_df()) == 8

    params = BAYESIAN_PARAMS.copy()
    params.update({'iss': 10.0})
    scores = dag_score(graph, data, BAYESIAN_SCORES, BAYESIAN_PARAMS)
    bnlearn = bnlearn_score(graph, data, BAYESIAN_SCORES,
                            BAYESIAN_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))


# single-valued data type allowed
@requires_r_and_bnlearn
def test_dag_score_ab12():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '1']},
                            dtype='category'))
    bic = dag_score(graph, data, 'bic', {'unistate_ok': True}).to_dict()['bic']
    assert dicts_same({'A': -1.732867951, 'B': 0}, bic)


# A --> B --> C, 7 rows
@requires_r_and_bnlearn
def test_dag_score_abc1():
    graph = dag.abc()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1', '2', '2', '2'],
                             'B': ['0', '1', '1', '2', '0', '1', '1'],
                             'C': ['0', '1', '1', '1', '0', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.as_df()) == 11

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -28.65148445, 'bic': -33.09193653,
                       'loglik': -17.65148445})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -16.31362629, 'bic': -9.961665509,
                       'loglik': -5.313626289})


# A --> B --> C, 10 rows
@requires_r_and_bnlearn
def test_dag_score_abc2():
    graph = dag.abc()
    data = Pandas(DataFrame(
           {'A': ['0', '0', '1', '1', '2', '2', '2', '3', '3', '3'],
            'B': ['0', '1', '1', '2', '0', '1', '1', '3', '3', '3'],
            'C': ['0', '1', '1', '1', '0', '1', '1', '2', '2', '1']},
           dtype='category'))

    assert free_params(graph, data.as_df()) == 23

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES,
                            ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -52.21928095, 'bic': -67.42145404,
                       'loglik': -29.21928095})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -31.79588002, 'bic': -20.29588002,
                       'loglik': -8.795880017})


# A --> C <-- B, 3 rows
@requires_r_and_bnlearn
def test_dag_score_ac_bc1():
    graph = dag.ac_bc()
    data = Pandas(DataFrame({'A': ['1', '0'],
                             'B': ['0', '1'],
                             'C': ['0', '1']}, dtype='category'))

    assert free_params(graph, data.as_df()) == 6

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -10, 'bic': -7, 'loglik': -4})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -7.204119983, 'bic': -2.107209970,
                       'loglik': -1.204119983})


# A --> C <-- B, 4 rows
@requires_r_and_bnlearn
def test_dag_score_ac_bc2():
    graph = dag.ac_bc()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1'],
                             'B': ['0', '1', '0', '1'],
                             'C': ['0', '0', '0', '1']},
                            dtype='category'))

    assert free_params(graph, data.as_df()) == 6

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -14, 'bic': -14, 'loglik': -8})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -8.408239965, 'bic': -4.214419939,
                       'loglik': -2.408239965})


# A --> C <-- B, 5 rows
@requires_r_and_bnlearn
def test_dag_score_ac_bc3():
    graph = dag.ac_bc()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1', '1'],
                             'B': ['0', '1', '0', '1', '1'],
                             'C': ['0', '0', '0', '1', '0']},
                  dtype='category'))

    assert free_params(graph, data.as_df()) == 6

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -17.70950594, 'bic': -18.67529023,
                       'loglik': -11.70950594})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -9.524912524, 'bic': -5.621822537,
                       'loglik': -3.524912524})


# A --> C <-- B, 10 rows
@requires_r_and_bnlearn
def test_dag_score_ac_bc4():
    graph = dag.ac_bc()
    data = {'A': ['0', '0', '1', '1', '2', '2', '2', '3', '3', '3'],
            'B': ['0', '1', '1', '2', '0', '1', '1', '3', '3', '3'],
            'C': ['0', '1', '1', '1', '0', '1', '1', '2', '2', '1']}
    data = Pandas(DataFrame(data, dtype='category'))

    assert free_params(graph, data.as_df()) == 38

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -78.92878689, 'bic': -104.0454207,
                       'loglik': -40.92878689})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -50.32079254, 'bic': -31.32079254,
                       'loglik': -12.32079254})


# Cancer, 4 rows
@requires_r_and_bnlearn
def test_dag_score_cancer_1():
    graph = dag.cancer()
    data = Pandas(DataFrame({'Smoker': ['no', 'no', 'yes', 'yes'],
                             'Pollution': ['low', 'high', 'low', 'high'],
                             'Cancer': ['no', 'no', 'yes', 'yes'],
                             'Dyspnoea': ['no', 'yes', 'no', 'yes'],
                             'Xray': ['clear', 'clear', 'dark', 'dark']},
                            dtype='category'))

    assert free_params(graph, data.as_df()) == 10

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -22, 'bic': -22, 'loglik': -12})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -13.61235995, 'bic': -6.622659905,
                       'loglik': -3.612359948})


# Cancer, 4 rows, k = 0.1
@requires_r_and_bnlearn
def test_dag_score_cancer_2():
    graph = dag.cancer()
    data = Pandas(DataFrame({'Smoker': ['no', 'no', 'yes', 'yes'],
                             'Pollution': ['low', 'high', 'low', 'high'],
                             'Cancer': ['no', 'no', 'yes', 'yes'],
                             'Dyspnoea': ['no', 'yes', 'no', 'yes'],
                             'Xray': ['clear', 'clear', 'dark', 'dark']},
                            dtype='category'))

    assert free_params(graph, data.as_df()) == 10

    params = ENTROPY_PARAMS.copy()
    print(params)
    scores = dag_score(graph, data, ENTROPY_SCORES, params)
    bnlearn = bnlearn_score(graph, data, ENTROPY_SCORES, params)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2, 'k': 0.1})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -13, 'bic': -13, 'loglik': -12})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10, 'k': 0.1})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -4.612359948, 'bic': -3.913389944,
                       'loglik': -3.612359948})


# Covid reference, 1K rows
@requires_r_and_bnlearn
def test_dag_score_covid_ref_1():
    data = NumPy.read(TESTDATA_DIR + '/experiments/datasets/covid.data.gz',
                      dstype='categorical', N=1000)
    print(data.as_df().tail())

    ref = BN.read(TESTDATA_DIR + '/discrete/medium/covid.dsc').dag
    params = {'unistate_ok': True, 'base': 'e'}

    scores = dag_score(ref, data, 'bic', params)
    print(scores)
    print((scores['bic'].sum()))

    bnlearn = bnlearn_score(ref, data, 'bic', params)
    print(bnlearn)
