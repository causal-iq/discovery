
# Test entropy_scores function

import pytest
from numpy import array

from core.metrics import dicts_same
from core.score import entropy_scores


def test_metrics_entropy_scores_type_error_1():  # bad argument types
    with pytest.raises(TypeError):
        entropy_scores()
    with pytest.raises(TypeError):
        entropy_scores(1)
    with pytest.raises(TypeError):
        entropy_scores({'A': 1}, base=None)
    with pytest.raises(TypeError):
        entropy_scores([2], 'loglik', {'base': 'e'}, 1, 0)
    with pytest.raises(TypeError):
        entropy_scores(None, ['loglik', 'bic'], {'base': 'e'}, 1, 0)
    with pytest.raises(TypeError):
        entropy_scores(array([[1]]), True, {'base': 'e'}, 1, 0)
    with pytest.raises(TypeError):
        entropy_scores(array([[1]]), 'loglik', 'base', 1, 0)
    with pytest.raises(TypeError):
        entropy_scores(array([[1]]), 'loglik', {'base': 'e'}, True, 0)
    with pytest.raises(TypeError):
        entropy_scores(array([[1]]), 'loglik', {'base': 'e'}, 0, 3.2)


def test_metrics_entropy_scores_type_error_2():  # types not all strings
    with pytest.raises(TypeError):
        entropy_scores(array([[1]]), ['loglik', 6], {'base': 'e'}, 1, 0)


def test_metrics_entropy_scores_value_error_1():  # unsupported score parameter
    with pytest.raises(ValueError):
        entropy_scores(array([[1]]), ['loglik', 'unsupported'],
                       {'base': 'e'}, 1, 0)


def test_metrics_entropy_scores_value_error_2():  # unsupported log base
    with pytest.raises(ValueError):
        entropy_scores(array([[1]]), 'bic', {'base': 'unsupported'}, 1, 0)
    with pytest.raises(ValueError):
        entropy_scores(array([[1]]), 'bic', {'base': True}, 1, 0)
    with pytest.raises(ValueError):
        entropy_scores(array([[1]]), 'bic', {'base': 8}, 1, 0)


def test_metrics_entropy_scores_value_error_3():  # num_cases not positive
    with pytest.raises(ValueError):
        entropy_scores(array([[1]]), 'bic', {'base': 2}, 0, 0)


def test_metrics_entropy_scores_value_error_4():  # free_params negative
    with pytest.raises(ValueError):
        entropy_scores(array([[1]]), 'bic', {'base': 2}, 1, -1)


def test_metrics_entropy_scores_loglik_ok_1():
    scores = entropy_scores(array([[1]]), 'loglik', {'base': 'e'}, 1, 0)
    assert dicts_same(scores, {'loglik': 0.0}, sf=10)
    scores = entropy_scores(array([[1000000]]), 'loglik',
                            {'base': 'e'}, 1000000, 0)
    assert dicts_same(scores, {'loglik': 0.0}, sf=10)
    scores = entropy_scores(array([[1]]), 'loglik', {'base': 2}, 1, 0)
    assert dicts_same(scores, {'loglik': 0.0}, sf=10)
    scores = entropy_scores(array([[5]]), 'loglik', {'base': 10}, 5, 0)
    assert dicts_same(scores, {'loglik': 0.0}, sf=10)


def test_metrics_entropy_scores_aic_ok_1():
    scores = entropy_scores(array([[1]]), 'aic',
                            {'base': 'e', 'k': 1.0}, 1, 0)
    assert dicts_same(scores, {'aic': 0.0}, sf=10)
    scores = entropy_scores(array([[1000000]]), 'aic',
                            {'base': 'e', 'k': 1.0}, 1000000, 0)
    assert dicts_same(scores, {'aic': 0.0}, sf=10)
    scores = entropy_scores(array([[1]]), 'aic',
                            {'base': 2, 'k': 1.0}, 1, 0)
    assert dicts_same(scores, {'aic': 0.0}, sf=10)
    scores = entropy_scores(array([[5]]), 'aic',
                            {'base': 10, 'k': 1.0}, 5, 0)
    assert dicts_same(scores, {'aic': 0.0}, sf=10)


def test_metrics_entropy_scores_bic_ok_1():
    scores = entropy_scores(array([[1]]), 'bic',
                            {'base': 'e', 'k': 1.0}, 1, 0)
    assert dicts_same(scores, {'bic': 0.0}, sf=10)
    scores = entropy_scores(array([[1000000]]), 'bic',
                            {'base': 'e', 'k': 1.0}, 1000000, 0)
    assert dicts_same(scores, {'bic': 0.0}, sf=10)
    scores = entropy_scores(array([[1]]), 'bic',
                            {'base': 2, 'k': 1.0}, 1, 0)
    assert dicts_same(scores, {'bic': 0.0}, sf=10)
    scores = entropy_scores(array([[5]]), 'bic',
                            {'base': 10, 'k': 1.0}, 5, 0)
    assert dicts_same(scores, {'bic': 0.0}, sf=10)


def test_metrics_entropy_scores_loglik_ok_2():
    scores = entropy_scores(array([[1], [1]]),
                            'loglik', {'base': 'e'}, 2, 1)
    assert dicts_same(scores, {'loglik': -1.386294361}, sf=10)
    scores = entropy_scores(array([[1], [1]]),
                            'loglik', {'base': 2}, 2, 1)
    assert dicts_same(scores, {'loglik': -2.0}, sf=10)
    scores = entropy_scores(array([[10], [10]]),
                            'loglik', {'base': 2}, 20, 1)
    assert dicts_same(scores, {'loglik': -20.0}, sf=10)
    scores = entropy_scores(array([[1], [1]]),
                            'loglik', {'base': 10}, 2, 1)
    assert dicts_same(scores, {'loglik': -0.6020599913}, sf=10)
    scores = entropy_scores(array([[10], [10]]),
                            'loglik', {'base': 10}, 20, 1)
    assert dicts_same(scores, {'loglik': -6.020599913}, sf=10)


def test_metrics_entropy_scores_aic_ok_2():
    scores = entropy_scores(array([[1], [1]]),
                            'aic', {'base': 'e', 'k': 1.0}, 2, 1)
    assert dicts_same(scores, {'aic': -2.386294361}, sf=10)
    scores = entropy_scores(array([[1], [1]]),
                            'aic', {'base': 2, 'k': 1.0}, 2, 1)
    assert dicts_same(scores, {'aic': -3.0}, sf=10)
    scores = entropy_scores(array([[10], [10]]),
                            'aic', {'base': 2, 'k': 1.0}, 20, 1)
    assert dicts_same(scores, {'aic': -21.0}, sf=10)
    scores = entropy_scores(array([[1], [1]]),
                            'aic', {'base': 10, 'k': 1.0}, 2, 1)
    assert dicts_same(scores, {'aic': -1.6020599913}, sf=10)
    scores = entropy_scores(array([[10], [10]]),
                            'aic', {'base': 10, 'k': 1.0}, 20, 1)
    assert dicts_same(scores, {'aic': -7.020599913}, sf=10)


def test_metrics_entropy_scores_bic_ok_2():
    scores = entropy_scores(array([[1], [1]]),
                            'bic', {'base': 'e', 'k': 1.0}, 2, 1)
    assert dicts_same(scores, {'bic': -1.732867951}, sf=10)
    scores = entropy_scores(array([[1], [1]]),
                            'bic', {'base': 2, 'k': 1.0}, 2, 1)
    assert dicts_same(scores, {'bic': -2.5}, sf=10)
    scores = entropy_scores(array([[10], [10]]),
                            'bic', {'base': 2, 'k': 1.0}, 20, 1)
    assert dicts_same(scores, {'bic': -22.16096405}, sf=10)
    scores = entropy_scores(array([[1], [1]]),
                            'bic', {'base': 10, 'k': 1.0}, 2, 1)
    assert dicts_same(scores, {'bic': -0.7525749892}, sf=10)
    scores = entropy_scores(array([[10], [10]]),
                            'bic', {'base': 10, 'k': 1.0}, 20, 1)
    assert dicts_same(scores, {'bic': -6.671114911}, sf=10)


def test_metrics_entropy_scores_loglik_ok_3():
    scores = entropy_scores(array([[1], [2]]),
                            'loglik', {'base': 2}, 3, 1)
    assert dicts_same(scores, {'loglik': -2.754887502}, sf=10)
    scores = entropy_scores(array([[2], [1]]),
                            'loglik', {'base': 2}, 3, 1)
    assert dicts_same(scores, {'loglik': -2.754887502}, sf=10)
    scores = entropy_scores(array([[1], [2]]),
                            'loglik', {'base': 10}, 3, 1)
    assert dicts_same(scores, {'loglik': -0.8293037728}, sf=10)
    scores = entropy_scores(array([[2], [1]]),
                            'loglik', {'base': 10}, 3, 1)
    assert dicts_same(scores, {'loglik': -0.8293037728}, sf=10)
    scores = entropy_scores(array([[1], [2]]),
                            'loglik', {'base': 'e'}, 3, 1)
    assert dicts_same(scores, {'loglik': -1.909542505}, sf=10)
    scores = entropy_scores(array([[2], [1]]),
                            'loglik', {'base': 'e'}, 3, 1)
    assert dicts_same(scores, {'loglik': -1.909542505}, sf=10)


def test_metrics_entropy_scores_aic_ok_3():
    scores = entropy_scores(array([[1], [2]]),
                            'aic', {'base': 2, 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'aic': -3.754887502}, sf=10)
    scores = entropy_scores(array([[2], [1]]),
                            'aic', {'base': 2, 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'aic': -3.754887502}, sf=10)
    scores = entropy_scores(array([[1], [2]]),
                            'aic', {'base': 10, 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'aic': -1.8293037728}, sf=10)
    scores = entropy_scores(array([[2], [1]]),
                            'aic', {'base': 10, 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'aic': -1.8293037728}, sf=10)
    scores = entropy_scores(array([[1], [2]]),
                            'aic', {'base': 'e', 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'aic': -2.909542505}, sf=10)
    scores = entropy_scores(array([[2], [1]]),
                            'aic', {'base': 'e', 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'aic': -2.909542505}, sf=10)


def test_metrics_entropy_scores_bic_ok_3():
    scores = entropy_scores(array([[1], [2]]),
                            'bic', {'base': 2, 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'bic': -3.547368753}, sf=10)
    scores = entropy_scores(array([[2], [1]]),
                            'bic', {'base': 2, 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'bic': -3.547368753}, sf=10)
    scores = entropy_scores(array([[1], [2]]),
                            'bic', {'base': 10, 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'bic': -1.067864400}, sf=10)
    scores = entropy_scores(array([[2], [1]]),
                            'bic', {'base': 10, 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'bic': -1.067864400}, sf=10)
    scores = entropy_scores(array([[1], [2]]),
                            'bic', {'base': 'e', 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'bic': -2.458848649}, sf=10)
    scores = entropy_scores(array([[2], [1]]),
                            'bic', {'base': 'e', 'k': 1.0}, 3, 1)
    assert dicts_same(scores, {'bic': -2.458848649}, sf=10)


def test_metrics_entropy_scores_loglik_ok_4():
    scores = entropy_scores(array([[100], [200]]),
                            'loglik', {'base': 2}, 300, 1)
    assert dicts_same(scores, {'loglik': -275.4887502}, sf=10)
    scores = entropy_scores(array([[2000], [1000]]),
                            'loglik', {'base': 2}, 3000, 1)
    assert dicts_same(scores, {'loglik': -2754.887502}, sf=10)
    scores = entropy_scores(array([[10], [20]]),
                            'loglik', {'base': 10}, 30, 1)
    assert dicts_same(scores, {'loglik': -8.293037728}, sf=10)
    scores = entropy_scores(array([[20000], [10000]]),
                            'loglik', {'base': 10}, 30000, 1)
    assert dicts_same(scores, {'loglik': -8293.037728}, sf=10)
    scores = entropy_scores(array([[100], [200]]),
                            'loglik', {'base': 'e'}, 300, 1)
    assert dicts_same(scores, {'loglik': -190.9542505}, sf=10)
    scores = entropy_scores(array([[20], [10]]),
                            'loglik', {'base': 'e'}, 30, 1)
    assert dicts_same(scores, {'loglik': -19.09542505}, sf=10)


def test_metrics_entropy_scores_aic_ok_4():
    scores = entropy_scores(array([[100], [200]]),
                            'aic', {'base': 2, 'k': 1.0}, 300, 1)
    assert dicts_same(scores, {'aic': -276.4887502}, sf=10)
    scores = entropy_scores(array([[2000], [1000]]),
                            'aic', {'base': 2, 'k': 1.0}, 3000, 1)
    assert dicts_same(scores, {'aic': -2755.887502}, sf=10)
    scores = entropy_scores(array([[10], [20]]),
                            'aic', {'base': 10, 'k': 1.0}, 30, 1)
    assert dicts_same(scores, {'aic': -9.293037728}, sf=10)
    scores = entropy_scores(array([[20000], [10000]]),
                            'aic', {'base': 10, 'k': 1.0}, 30000, 1)
    assert dicts_same(scores, {'aic': -8294.037728}, sf=10)
    scores = entropy_scores(array([[100], [200]]),
                            'aic', {'base': 'e', 'k': 1.0}, 300, 1)
    assert dicts_same(scores, {'aic': -191.9542505}, sf=10)
    scores = entropy_scores(array([[20], [10]]),
                            'aic', {'base': 'e', 'k': 1.0}, 30, 1)
    assert dicts_same(scores, {'aic': -20.09542505}, sf=10)


def test_metrics_entropy_scores_bic_ok_4():
    scores = entropy_scores(array([[100], [200]]),
                            'bic', {'base': 2, 'k': 1.0}, 300, 1)
    assert dicts_same(scores, {'bic': -279.6031596}, sf=10)
    scores = entropy_scores(array([[2000], [1000]]),
                            'bic', {'base': 2, 'k': 1.0}, 3000, 1)
    assert dicts_same(scores, {'bic': -2760.662876}, sf=10)
    scores = entropy_scores(array([[10], [20]]),
                            'bic', {'base': 10, 'k': 1.0}, 30, 1)
    assert dicts_same(scores, {'bic': -9.031598356}, sf=10)
    scores = entropy_scores(array([[20000], [10000]]),
                            'bic', {'base': 10, 'k': 1.0}, 30000, 1)
    assert dicts_same(scores, {'bic': -8295.276289}, sf=10)
    scores = entropy_scores(array([[100], [200]]),
                            'bic', {'base': 'e', 'k': 1.0}, 300, 1)
    assert dicts_same(scores, {'bic': -193.8061417}, sf=10)
    scores = entropy_scores(array([[20], [10]]),
                            'bic', {'base': 'e', 'k': 1.0}, 30, 1)
    assert dicts_same(scores, {'bic': -20.79602374}, sf=10)


def test_metrics_entropy_scores_loglik_ok_5():
    scores = entropy_scores(array([[10], [1]]),
                            'loglik', {'base': 2}, 11, 1)
    assert dicts_same(scores, {'loglik': -4.834466856}, sf=10)
    scores = entropy_scores(array([[10], [100]]),
                            'loglik', {'base': 2}, 110, 1)
    assert dicts_same(scores, {'loglik': -48.34466856}, sf=10)
    scores = entropy_scores(array([[1], [10]]),
                            'loglik', {'base': 10}, 11, 1)
    assert dicts_same(scores, {'loglik': -1.4553195367}, sf=10)
    scores = entropy_scores(array([[1000], [100]]),
                            'loglik', {'base': 10}, 1100, 1)
    assert dicts_same(scores, {'loglik': -145.53195367}, sf=10)
    scores = entropy_scores(array([[10], [1]]),
                            'loglik', {'base': 'e'}, 11, 1)
    assert dicts_same(scores, {'loglik': -3.350997071}, sf=10)
    scores = entropy_scores(array([[100], [1000]]),
                            'loglik', {'base': 'e'}, 1100, 1)
    assert dicts_same(scores, {'loglik': -335.0997071}, sf=10)


def test_metrics_entropy_scores_aic_ok_5():
    scores = entropy_scores(array([[10], [1]]),
                            'aic', {'base': 2, 'k': 1.0}, 11, 1)
    assert dicts_same(scores, {'aic': -5.834466856}, sf=10)
    scores = entropy_scores(array([[10], [100]]),
                            'aic', {'base': 2, 'k': 1.0}, 110, 1)
    assert dicts_same(scores, {'aic': -49.34466856}, sf=10)
    scores = entropy_scores(array([[1], [10]]),
                            'aic', {'base': 10, 'k': 1.0}, 11, 1)
    assert dicts_same(scores, {'aic': -2.4553195367}, sf=10)
    scores = entropy_scores(array([[1000], [100]]),
                            'aic', {'base': 10, 'k': 1.0}, 1100, 1)
    assert dicts_same(scores, {'aic': -146.53195367}, sf=10)
    scores = entropy_scores(array([[10], [1]]),
                            'aic', {'base': 'e', 'k': 1.0}, 11, 1)
    assert dicts_same(scores, {'aic': -4.350997071}, sf=10)
    scores = entropy_scores(array([[100], [1000]]),
                            'aic', {'base': 'e', 'k': 1.0}, 1100, 1)
    assert dicts_same(scores, {'aic': -336.0997071}, sf=10)


def test_metrics_entropy_scores_bic_ok_5():
    scores = entropy_scores(array([[10], [1]]),
                            'bic', {'base': 2, 'k': 1.0}, 11, 1)
    assert dicts_same(scores, {'bic': -6.564182665}, sf=10)
    scores = entropy_scores(array([[10], [100]]),
                            'bic', {'base': 2, 'k': 1.0}, 110, 1)
    assert dicts_same(scores, {'bic': -51.73534842}, sf=10)
    scores = entropy_scores(array([[1], [10]]),
                            'bic', {'base': 10, 'k': 1.0}, 11, 1)
    assert dicts_same(scores, {'bic': -1.976015879}, sf=10)
    scores = entropy_scores(array([[1000], [100]]),
                            'bic', {'base': 10, 'k': 1.0}, 1100, 1)
    assert dicts_same(scores, {'bic': -147.0526500}, sf=10)
    scores = entropy_scores(array([[10], [1]]),
                            'bic', {'base': 'e', 'k': 1.0}, 11, 1)
    assert dicts_same(scores, {'bic': -4.549944707}, sf=10)
    scores = entropy_scores(array([[100], [1000]]),
                            'bic', {'base': 'e', 'k': 1.0}, 1100, 1)
    assert dicts_same(scores, {'bic': -338.6012398}, sf=10)


def test_metrics_entropy_scores_loglik_ok_6():
    scores = entropy_scores(array([[1], [1], [1]]),
                            'loglik', {'base': 10}, 3, 2)
    assert dicts_same(scores, {'loglik': -1.431363764}, sf=10)


def test_metrics_entropy_scores_aic_ok_6():
    scores = entropy_scores(array([[1], [1], [1]]),
                            'aic', {'base': 10, 'k': 1.0}, 3, 2)
    assert dicts_same(scores, {'aic': -3.431363764}, sf=10)


def test_metrics_entropy_scores_bic_ok_6():
    scores = entropy_scores(array([[1], [1], [1]]),
                            'bic', {'base': 10, 'k': 1.0}, 3, 2)
    assert dicts_same(scores, {'bic': -1.908485019}, sf=10)


# Scores with one parent

def test_metrics_entropy_scores_1_parents_1_ok():
    counts = array([[1, 0], [0, 1]])
    print('\n\n{}'.format(counts))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 2, 'k': 1}, 2, 2)
    assert dicts_same(scores, {'loglik': 0.0, 'bic': -1.0, 'aic': -2.0}, 10)
    print('Base 2 scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 'e', 'k': 1}, 2, 2)
    assert dicts_same(scores, {'loglik': 0.0, 'bic': -0.6931, 'aic': -2.0}, 4)
    print('Base e scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))


def test_metrics_entropy_scores_1_parents_2_ok():
    counts = array([[1, 0], [1, 0]])
    print('\n\n{}'.format(counts))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 2, 'k': 1}, 2, 2)
    assert dicts_same(scores, {'loglik': -2.0, 'bic': -3.0, 'aic': -4.0}, 10)
    print('Base 2 scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 'e', 'k': 1}, 2, 2)
    assert dicts_same(scores, {'loglik': -1.3863, 'bic': -2.0794,
                               'aic': -3.3863}, 4)
    print('Base e scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))


def test_metrics_entropy_scores_1_parents_3_ok():
    counts = array([[1, 1], [1, 1]])
    print('\n\n{}'.format(counts))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 2, 'k': 1}, 4, 2)
    assert dicts_same(scores, {'loglik': -4.0, 'bic': -6.0, 'aic': -6.0}, 4)
    print('Base 2 scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 'e', 'k': 1}, 4, 2)
    assert dicts_same(scores, {'loglik': -2.7726, 'bic': -4.1589,
                               'aic': -4.7726}, 5)
    print('Base e scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))


def test_metrics_entropy_scores_1_parents_4_ok():
    counts = array([[1, 2], [3, 4]])
    print('\n\n{}'.format(counts))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 2, 'k': 1}, 10, 2)
    assert dicts_same(scores, {'loglik': -8.7549, 'bic': -12.0768,
                               'aic': -10.7549}, 4)
    print('Base 2 scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 'e', 'k': 1}, 10, 2)
    assert dicts_same(scores, {'loglik': -6.0684, 'bic': -8.371,
                               'aic': -8.0684}, 5)
    print('Base e scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))


# Scores with two parents

def test_metrics_entropy_scores_2_parents_1_ok():
    counts = array([[1, 0, 0, 0], [0, 0, 0, 1]])
    print('\n\n{}'.format(counts))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 2, 'k': 1}, 2, 4)
    assert dicts_same(scores, {'loglik': 0.0, 'bic': -2.0, 'aic': -4.0}, 4)
    print('Base 2 scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 'e', 'k': 1}, 2, 4)
    assert dicts_same(scores, {'loglik': 0.0, 'bic': -1.3863, 'aic': -4.0}, 4)
    print('Base e scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))


def test_metrics_entropy_scores_2_parents_2_ok():
    counts = array([[1, 0, 2, 1], [1, 0, 2, 1]])
    print('\n\n{}'.format(counts))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 2, 'k': 1}, 8, 4)
    assert dicts_same(scores, {'loglik': -8.0, 'bic': -14.0, 'aic': -12.0}, 4)
    print('Base 2 scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 'e', 'k': 1}, 8, 4)
    assert dicts_same(scores, {'loglik': -5.5452, 'bic': -9.7041,
                               'aic': -9.5452}, 4)
    print('Base e scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))


def test_metrics_entropy_scores_2_parents_3_ok():
    counts = array([[0, 1, 2, 3], [4, 4, 5, 5], [2, 0, 1, 0]])
    print('\n\n{}'.format(counts))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 2, 'k': 1}, 27, 8)
    assert dicts_same(scores, {'loglik': -27.1452, 'bic': -46.1648,
                               'aic': -35.1452}, 5)
    print('Base 2 scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))

    scores = entropy_scores(counts, ['loglik', 'bic', 'aic'],
                            {'base': 'e', 'k': 1}, 27, 8)
    assert dicts_same(scores, {'loglik': -18.8157, 'bic': -31.999,
                               'aic': -26.8157}, 5)
    print('Base e scores are: {}'
          .format({s: round(v, 4) for s, v in scores.items()}))
