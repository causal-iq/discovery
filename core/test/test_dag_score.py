
import pytest
from pandas import DataFrame

from core.metrics import dicts_same
from core.score import free_params, dag_score, ENTROPY_SCORES, BAYESIAN_SCORES
from core.bn import BN
from call.bnlearn import bnlearn_score
import testdata.example_dags as dag
from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from fileio.oracle import Oracle

ENTROPY_PARAMS = {'base': 'e', 'k': 1.0}
BAYESIAN_PARAMS = {'iss': 1.0, 'prior': 'uniform'}


def test_graph_score_type_error_1():  # bad primary arg types for DAG.score
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


def test_graph_score_type_error_2():  # bad score type
    graph = dag.ab()
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '1'], 'B': ['1', '1']},
                              dtype='category'), [37])


def test_graph_score_type_error_3():  # bad 'base' score param type
    graph = dag.ab()
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '1'], 'B': ['1', '0']},
                              dtype='category'), 'bic', {'base': 2.2})
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '1'], 'B': ['1', '0']},
                              dtype='category'), 'bic', {'base': True})


def test_graph_score_type_error_4():  # bad 'prior' score param type
    graph = dag.ab()
    with pytest.raises(TypeError):
        graph.score(DataFrame({'A': ['0', '1'], 'B': ['1', '0']},
                              dtype='category'), 'bde', {'prior': 12})


def test_graph_score_type_error_5():  # bad 'iss' score param type
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


def test_graph_score_value_error_7():  # DAG / Data column mismatch
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '1'], 'C': ['0', '1']},
                            dtype='category'))
    with pytest.raises(ValueError):
        graph.score(data, 'aic')


def test_graph_score_value_error_8():  # single-valued variables
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['0', '0']},
                            dtype='category'))
    with pytest.raises(ValueError):
        graph.score(data, 'aic')


def test_dag_score_type_error_1():  # bad arg types
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


def test_dag_score_type_error_2():  # bad base type
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['0', '1']},
                            dtype='category'))
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bic', {'base': []})


def test_dag_score_type_error_3():  # bad k type
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['0', '1']},
                            dtype='category'))
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bic', {'k': 'should be int/float'})
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bic', {'k': {}})


def test_dag_score_type_error_4():  # bad prior type
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['0', '1']},
                            dtype='category'))
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bde', {'prior': 1})
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bde', {'prior': {}})


def test_dag_score_type_error_5():  # bad iss type
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['1', '0'], 'B': ['0', '1']},
                            dtype='category'))
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bds', {'iss': 'should be int/float'})
    with pytest.raises(TypeError):
        dag_score(graph, data, 'bds', {'iss': {}})


def test_dag_score_type_error_6():  # cannot score an oracle type
    bn = BN.read(TESTDATA_DIR + '/xdsl/ab.xdsl')
    with pytest.raises(TypeError):
        dag_score(bn.dag, Oracle(bn), 'bic', {})


def test_dag_score_value_error_2():  # unsupported score types
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['0', '1']},
                  dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, [], {})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'unsupported', {})
    with pytest.raises(ValueError):
        dag_score(graph, data, ['unsupported', 'bic'], {})


def test_dag_score_value_error_3():  # single-valued data tyoe
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '1']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {})


def test_dag_score_value_error_4():  # data / dag column mismatch
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'C': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {})


def test_dag_score_value_error_5():  # unknown score parameter
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'unsupported': 3})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'base': 2, 'unsupported': 3})


def test_dag_score_value_error_6():  # bad "base" score param value
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'base': 7})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'base': '2'})


def test_dag_score_value_error_7():  # bad "prior" score param value
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bic', {'prior': 'unsupported'})


def test_dag_score_value_error_8():  # bad "iss" score param value
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


def test_dag_score_value_error_9():  # bad "k" score param value
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


def xtest_dag_score_value_error_10():  # irrelevant parameters for entropy
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


def xtest_dag_score_value_error_11():  # irrelevant parameters for bayesian
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '2']},
                            dtype='category'))
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bde', {'k': 4})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'k2', {'k': 4})
    with pytest.raises(ValueError):
        dag_score(graph, data, 'bde', {'k': 10.0})


def test_dag_score_ab1():
    graph = dag.ab()  # A --> B
    data = Pandas(DataFrame({'A': ['0', '1'], 'B': ['0', '1']},
                            dtype='category'))
    assert free_params(graph, data.sample) == 3
    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))
    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -3.5, 'loglik': -2, 'aic': -5})


def test_dag_score_ab2():  # set k = 2
    graph = dag.ab()  # A --> B
    data = Pandas(DataFrame({'A': ['0', '1'], 'B': ['0', '1']},
                            dtype='category'))
    assert free_params(graph, data.sample) == 3
    params = dict(ENTROPY_PARAMS)
    params.update({'k': 2})
    scores = dag_score(graph, data, ENTROPY_SCORES, params)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, params)
    assert dicts_same(bnlearn, dict(scores.sum()))
    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2, 'k': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -5, 'loglik': -2, 'aic': -8})


def test_dag_score_ab3():
    graph = dag.ab()  # A --> B
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1'],
                             'B': ['0', '1', '0', '1']},
                  dtype='category'))
    assert free_params(graph, data.sample) == 3

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -11, 'loglik': -8, 'aic': -11})


def test_dag_score_ab4():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1'],
                             'B': ['0', '1', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.sample) == 3

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -9, 'loglik': -6, 'aic': -9})


def test_dag_score_ab5():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '1', '1', '1'],
                             'B': ['0', '1', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.sample) == 3

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -6.245112498, 'loglik': -3.245112498,
                       'aic': -6.245112498})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -1.879966188, 'loglik': -0.9768762012,
                       'aic': -3.976876201})


def test_dag_score_ab6():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1'],
                             'B': ['0', '1', '1', '2']},
                            dtype='category'))

    assert free_params(graph, data.sample) == 5

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -13, 'loglik': -8, 'aic': -13})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -3.913389944, 'loglik': -2.408239965,
                       'aic': -7.408239965})


def test_dag_score_ab7():
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1', '2', '2', '2'],
                             'B': ['0', '1', '1', '2', '0', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.sample) == 8

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -28.88090414, 'loglik': -17.65148445,
                       'aic': -25.65148445})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'bic': -8.694018448, 'loglik': -5.313626289,
                       'aic': -13.3136262888})


def test_dag_score_ab8():  # Bayesian scores for A --> B, 2 rows
    graph = dag.ab()  # A --> B
    data = Pandas(DataFrame({'A': ['0', '1'], 'B': ['0', '1']},
                            dtype='category'))
    assert free_params(graph, data.sample) == 3
    scores = dag_score(graph, data, BAYESIAN_SCORES, BAYESIAN_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, BAYESIAN_SCORES,
                            BAYESIAN_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))
    assert dicts_same(dict(scores.sum()),
                      {'bde': -3.465735903, 'bdj': -4.564348191,
                       'bds': -3.465735903, 'k2': -3.178053830})


def test_dag_score_ab9():  # Bayesian scores for A --> B, 2 rows, ISS =5
    graph = dag.ab()  # A --> B
    data = Pandas(DataFrame({'A': ['0', '1'], 'B': ['0', '1']},
                            dtype='category'))
    assert free_params(graph, data.sample) == 3
    params = BAYESIAN_PARAMS.copy()
    params.update({'iss': 5})
    scores = dag_score(graph, data, BAYESIAN_SCORES, params)
    bnlearn = bnlearn_score(graph, data.sample, BAYESIAN_SCORES, params)
    assert dicts_same(bnlearn, dict(scores.sum()))
    assert dicts_same(dict(scores.sum()),
                      {'bde': -2.954910279, 'bdj': -4.564348191,
                       'bds': -2.954910279, 'k2': -3.178053830})


def test_dag_score_ab10():  # Bayesian scores, A --> B, 8 rows
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1', '2', '2', '2'],
                             'B': ['0', '1', '1', '2', '0', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.sample) == 8

    scores = dag_score(graph, data, BAYESIAN_SCORES, BAYESIAN_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, BAYESIAN_SCORES,
                            BAYESIAN_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))


def test_dag_score_ab11():  # Bayesian scores, A --> B, 8 rows, ISS=10.0
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1', '2', '2', '2'],
                             'B': ['0', '1', '1', '2', '0', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.sample) == 8

    params = BAYESIAN_PARAMS.copy()
    params.update({'iss': 10.0})
    scores = dag_score(graph, data, BAYESIAN_SCORES, BAYESIAN_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, BAYESIAN_SCORES,
                            BAYESIAN_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))


def test_dag_score_ab12():  # single-valued data type allowed
    graph = dag.ab()
    data = Pandas(DataFrame({'A': ['2', '0'], 'B': ['1', '1']},
                            dtype='category'))
    bic = dag_score(graph, data, 'bic', {'unistate_ok': True}).to_dict()['bic']
    assert dicts_same({'A': -1.732867951, 'B': 0}, bic)


def test_dag_score_abc1():
    graph = dag.abc()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1', '2', '2', '2'],
                             'B': ['0', '1', '1', '2', '0', '1', '1'],
                             'C': ['0', '1', '1', '1', '0', '1', '1']},
                            dtype='category'))

    assert free_params(graph, data.sample) == 11

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -28.65148445, 'bic': -33.09193653,
                       'loglik': -17.65148445})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -16.31362629, 'bic': -9.961665509,
                       'loglik': -5.313626289})


def test_dag_score_abc2():
    graph = dag.abc()  # A --> B --> C
    data = Pandas(DataFrame(
           {'A': ['0', '0', '1', '1', '2', '2', '2', '3', '3', '3'],
            'B': ['0', '1', '1', '2', '0', '1', '1', '3', '3', '3'],
            'C': ['0', '1', '1', '1', '0', '1', '1', '2', '2', '1']},
           dtype='category'))

    assert free_params(graph, data.sample) == 23

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES,
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


def test_dag_score_ac_bc1():
    graph = dag.ac_bc()  # A --> C <-- B
    data = Pandas(DataFrame({'A': ['1', '0'],
                             'B': ['0', '1'],
                             'C': ['0', '1']}, dtype='category'))

    assert free_params(graph, data.sample) == 6

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -10, 'bic': -7, 'loglik': -4})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -7.204119983, 'bic': -2.107209970,
                       'loglik': -1.204119983})


def test_dag_score_ac_bc2():
    graph = dag.ac_bc()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1'],
                             'B': ['0', '1', '0', '1'],
                             'C': ['0', '0', '0', '1']},
                            dtype='category'))

    assert free_params(graph, data.sample) == 6

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -14, 'bic': -14, 'loglik': -8})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -8.408239965, 'bic': -4.214419939,
                       'loglik': -2.408239965})


def test_dag_score_ac_bc3():
    graph = dag.ac_bc()
    data = Pandas(DataFrame({'A': ['0', '0', '1', '1', '1'],
                             'B': ['0', '1', '0', '1', '1'],
                             'C': ['0', '0', '0', '1', '0']},
                  dtype='category'))

    assert free_params(graph, data.sample) == 6

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -17.70950594, 'bic': -18.67529023,
                       'loglik': -11.70950594})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -9.524912524, 'bic': -5.621822537,
                       'loglik': -3.524912524})


def test_dag_score_ac_bc4():
    graph = dag.ac_bc()
    data = {'A': ['0', '0', '1', '1', '2', '2', '2', '3', '3', '3'],
            'B': ['0', '1', '1', '2', '0', '1', '1', '3', '3', '3'],
            'C': ['0', '1', '1', '1', '0', '1', '1', '2', '2', '1']}
    data = Pandas(DataFrame(data, dtype='category'))

    assert free_params(graph, data.sample) == 38

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -78.92878689, 'bic': -104.0454207,
                       'loglik': -40.92878689})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -50.32079254, 'bic': -31.32079254,
                       'loglik': -12.32079254})


def test_dag_score_cancer_1():
    graph = dag.cancer()
    data = Pandas(DataFrame({'Smoker': ['no', 'no', 'yes', 'yes'],
                             'Pollution': ['low', 'high', 'low', 'high'],
                             'Cancer': ['no', 'no', 'yes', 'yes'],
                             'Dyspnoea': ['no', 'yes', 'no', 'yes'],
                             'Xray': ['clear', 'clear', 'dark', 'dark']},
                            dtype='category'))

    assert free_params(graph, data.sample) == 10

    scores = dag_score(graph, data, ENTROPY_SCORES, ENTROPY_PARAMS)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, ENTROPY_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -22, 'bic': -22, 'loglik': -12})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -13.61235995, 'bic': -6.622659905,
                       'loglik': -3.612359948})


def test_dag_score_cancer_2():  # use 'k' as 0.1
    graph = dag.cancer()
    data = Pandas(DataFrame({'Smoker': ['no', 'no', 'yes', 'yes'],
                             'Pollution': ['low', 'high', 'low', 'high'],
                             'Cancer': ['no', 'no', 'yes', 'yes'],
                             'Dyspnoea': ['no', 'yes', 'no', 'yes'],
                             'Xray': ['clear', 'clear', 'dark', 'dark']},
                            dtype='category'))

    assert free_params(graph, data.sample) == 10

    params = ENTROPY_PARAMS.copy()
    print(params)
    scores = dag_score(graph, data, ENTROPY_SCORES, params)
    bnlearn = bnlearn_score(graph, data.sample, ENTROPY_SCORES, params)
    assert dicts_same(bnlearn, dict(scores.sum()))

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 2, 'k': 0.1})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -13, 'bic': -13, 'loglik': -12})

    scores = dag_score(graph, data, ENTROPY_SCORES, {'base': 10, 'k': 0.1})
    assert dicts_same(dict(scores.sum()),
                      {'aic': -4.612359948, 'bic': -3.913389944,
                       'loglik': -3.612359948})
