
import pytest

from call.r import dispatch_r


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


def test_dispatch_r_value_error_1():  # bad package name
    with pytest.raises(ValueError):
        dispatch_r('unsupported', 'echo')


def test_dispatch_r_value_error_2():  # bad method name
    with pytest.raises(ValueError):
        dispatch_r('test', 'unsupported')


def test_dispatch_r_value_error_3():  # empty parameters
    with pytest.raises(ValueError):
        dispatch_r('test', 'echo', {})


def test_dispatch_r_runtime_error():  # Error in R code
    with pytest.raises(RuntimeError):
        dispatch_r('test', 'error')


def test_dispatch_r_runtime_test_echo_ok():  # Echo parameters works
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
