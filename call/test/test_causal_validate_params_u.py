
# Unit test validation of causal learn parameters

import pytest

from call.causal import _validate_learn_params


# --- Failure cases

# Mixed data types not supported
def test_mixed_data_type_value_error():
    with pytest.raises(ValueError):
        _validate_learn_params(params={}, dstype='mixed')


# Unknown data types not supported
def test_unknown_data_type_value_error():
    with pytest.raises(ValueError):
        _validate_learn_params(params={}, dstype='unknown')


# Categorical data but not bde score
def test_bad_categorical_score_value_error():
    with pytest.raises(ValueError):
        _validate_learn_params(params={'score': 'bic'}, dstype='categorical')


# Continuous data but not bic-g score
def test_bad_continuous_score_value_error():
    with pytest.raises(ValueError):
        _validate_learn_params(params={'score': 'bic'}, dstype='continuous')
    with pytest.raises(ValueError):
        _validate_learn_params(params={'score': 'bde'}, dstype='continuous')


# ISS value other than default value
def test_bad_iss_value_error():
    with pytest.raises(ValueError):
        _validate_learn_params(params={'score': 'bde', 'iss': 2},
                               dstype='categorical')


# ISS value other than default value
def test_bad_k_value_error():
    with pytest.raises(ValueError):
        _validate_learn_params(params={'score': 'bic-g', 'k': 2},
                               dstype='continuous')


# --- Successful cases - default values

# Defaults for categorical data
def test_categorical_defaults_ok():
    params = _validate_learn_params(params={}, dstype='categorical')
    assert params == {'score': 'bde', 'iss': 1, 'base': 'e'}


# Defaults for continuous data
def test_continuous_defaults_ok():
    params = _validate_learn_params(params={}, dstype='continuous')
    assert params == {'score': 'bic-g', 'k': 1, 'base': 'e'}


# --- Successful cases - set acceptable values

# params set to None
def test_params_are_none_ok():
    params = _validate_learn_params(params=None, dstype='categorical')
    assert params == {'score': 'bde', 'iss': 1, 'base': 'e'}


# Correct params for categorical data
def test_categorical_params_ok():
    params = _validate_learn_params(params={'score': 'bde', 'iss': 1,
                                            'base': 'e'}, dstype='categorical')
    assert params == {'score': 'bde', 'iss': 1, 'base': 'e'}


# Correct params for continuous data
def test_continuous_params_ok():
    params = _validate_learn_params(params={'score': 'bic-g', 'k': 1,
                                            'base': 'e'}, dstype='continuous')
    assert params == {'score': 'bic-g', 'k': 1, 'base': 'e'}
