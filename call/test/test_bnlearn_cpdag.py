
# Tests calling bnlearn cpdag function

import pytest

from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_cpdag
import testdata.example_pdags as ex_pdag


# bad primary arg types
def test_bnlearn_cpdag_type_error1():
    with pytest.raises(TypeError):
        bnlearn_cpdag()


# empty PDAGs not supported by bnlearn
def test_bnlearn_cpdag_value_error1():
    with pytest.raises(ValueError):
        bnlearn_cpdag(ex_pdag.empty())


# PDAG with single node A
@requires_r_and_bnlearn
def test_bnlearn_cpdag_type_a_ok1():
    cpdag = bnlearn_cpdag(ex_pdag.a())
    print(cpdag)


# A -> B PDAG
@requires_r_and_bnlearn
def test_bnlearn_cpdag_type_ab_ok1():
    cpdag = bnlearn_cpdag(ex_pdag.ab())
    print(cpdag)
