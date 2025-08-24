
import pytest

from call.r import dispatch_r, requires_r_and_bnlearn


def test_dispatch_r_type_error():
    with pytest.raises(TypeError):
        dispatch_r()
    with pytest.raises(TypeError):
        dispatch_r(67)
    with pytest.raises(TypeError):
        dispatch_r([], 'ok')
    with pytest.raises(TypeError):
        dispatch_r('test', 32)
    with pytest.raises(TypeError):
        dispatch_r('test', 'echo', 17)


# bad package name
def test_dispatch_r_value_error_1():
    with pytest.raises(ValueError):
        dispatch_r('unsupported', 'echo')


# bad method name
def test_dispatch_r_value_error_2():
    with pytest.raises(ValueError):
        dispatch_r('test', 'unsupported')


# empty parameters
def test_dispatch_r_value_error_3():
    with pytest.raises(ValueError):
        dispatch_r('test', 'echo', {})


# Error in R code
def test_dispatch_r_runtime_error():
    with pytest.raises(RuntimeError):
        dispatch_r('test', 'error')


# Echo parameters works
@requires_r_and_bnlearn
def test_dispatch_r_runtime_test_echo_ok():
    params = {'float': 0.2, 'int': 7, 'str': 'hello', 'array': [13, 17],
              'dict': {'f': 2.9, 'i': 2, 's': 'd'}}
    check, stdout = dispatch_r('test', 'echo', params)
    assert check == params
    assert stdout == ['$float', '[1] 0.2', '',
                      '$int', '[1] 7', '',
                      '$str', '[1] "hello"', '',
                      '$array', '[1] 13 17', '',
                      '$dict', '$dict$f', '[1] 2.9', '',
                      '$dict$i', '[1] 2', '',
                      '$dict$s', '[1] "d"', '', '']
