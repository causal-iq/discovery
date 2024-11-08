
# Tests calling bnlearn cpdag function

import pytest

from call.bnlearn import bnlearn_cpdag
import testdata.example_pdags as ex_pdag


def test_bnlearn_cpdag_type_error1():  # bad primary arg types
    with pytest.raises(TypeError):
        bnlearn_cpdag()


def test_bnlearn_cpdag_value_error1():  # empty PDAGs not supported by bnlearn
    with pytest.raises(ValueError):
        bnlearn_cpdag(ex_pdag.empty())


def test_bnlearn_cpdag_type_a_ok1():  # A PDAG
    cpdag = bnlearn_cpdag(ex_pdag.a())
    print(cpdag)


def test_bnlearn_cpdag_type_ab_ok1():  # A -> B PDAG
    cpdag = bnlearn_cpdag(ex_pdag.ab())
    print(cpdag)
