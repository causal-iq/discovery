
import pytest

from call.cmd import dispatch_cmd


def test_dispatch_cmd_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        dispatch_cmd()


def test_dispatch_cmd_type_error_2():  # args not a list
    with pytest.raises(TypeError):
        dispatch_cmd('ls')
    with pytest.raises(TypeError):
        dispatch_cmd(3.7)


def test_dispatch_cmd_type_error_3():  # non-string in list
    with pytest.raises(TypeError):
        dispatch_cmd(['ls', 7])
    with pytest.raises(TypeError):
        dispatch_cmd([0])


def test_dispatch_cmd_value_error_1():  # empty list
    with pytest.raises(ValueError):
        dispatch_cmd([])


def test_dispatch_cmd_runtime_error_1():  # illegal command
    with pytest.raises(RuntimeError):
        dispatch_cmd(['invalid'])
    with pytest.raises(RuntimeError):
        dispatch_cmd(['unknown.bat'])


def test_dispatch_cmd_ok_1():  # simple echo command
    stdout = dispatch_cmd(['echo', 'ABC'])
    assert stdout == ['ABC']
