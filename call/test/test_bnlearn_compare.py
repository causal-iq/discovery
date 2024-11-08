
#   Test the call interface to bnlearn's compare function

import pytest

from call.bnlearn import bnlearn_compare
import testdata.example_dags as ex_dag
import testdata.example_pdags as ex_pdag
from core.graph import PDAG


def test_bnlearn_compare_type_error_1():  # no arguments supplied
    with pytest.raises(TypeError):
        bnlearn_compare()


def test_bnlearn_compare_type_error_2():  # one argument supplied
    pdag = ex_pdag.a()
    with pytest.raises(TypeError):
        bnlearn_compare(pdag)
    with pytest.raises(TypeError):
        bnlearn_compare(pdag, None)
    with pytest.raises(TypeError):
        bnlearn_compare(None, pdag)


def test_bnlearn_compare_type_error_3():  # bad arg types
    pdag = ex_pdag.a()
    with pytest.raises(TypeError):
        bnlearn_compare(pdag, 1)
    with pytest.raises(TypeError):
        bnlearn_compare(pdag, [pdag])
    with pytest.raises(TypeError):
        bnlearn_compare(-20.2, pdag)


def test_bnlearn_compare_value_error_1():  # can't compare empty graphs
    pdag = ex_pdag.empty()
    dag = ex_dag.empty()
    with pytest.raises(ValueError):
        bnlearn_compare(pdag, pdag)
    with pytest.raises(ValueError):
        bnlearn_compare(dag, pdag)
    with pytest.raises(ValueError):
        bnlearn_compare(dag, dag)


def test_bnlearn_compare_value_error_2():  # can't compare with diff nodesets
    pdag = ex_pdag.a()
    ref = ex_pdag.ab()
    with pytest.raises(ValueError):
        bnlearn_compare(pdag, ref)


def test_bnlearn_compare_a_itself_ok():  # compare A with itself
    ref = ex_dag.a()
    metrics = bnlearn_compare(ref, ref)
    assert metrics == {'tp': 0, 'fp': 0, 'fn': 0, 'shd': 0}


def test_bnlearn_compare_a_b_itself_ok():  # compare A B with itself
    ref = ex_dag.a_b()
    metrics = bnlearn_compare(ref, ref)
    assert metrics == {'tp': 0, 'fp': 0, 'fn': 0, 'shd': 0}


def test_bnlearn_compare_ab_itself_ok():  # compare A->B with itself
    ref = ex_dag.ab()
    metrics = bnlearn_compare(ref, ref)
    assert metrics == {'tp': 1, 'fp': 0, 'fn': 0, 'shd': 0}


def test_bnlearn_compare_ab3_ab_ok():  # compare A--B with A->B
    ref = ex_pdag.ab()
    pdag = ex_pdag.ab3()
    metrics = bnlearn_compare(pdag, ref)
    assert metrics == {'tp': 0, 'fp': 1, 'fn': 1, 'shd': 0}


def test_bnlearn_compare_ab_ba_ok():  # compare A->B with A<-B
    ref = ex_dag.ab()
    pdag = ex_dag.ba()
    metrics = bnlearn_compare(pdag, ref)
    assert metrics == {'tp': 0, 'fp': 1, 'fn': 1, 'shd': 0}


def test_bnlearn_compare_ab_ab3_ok():  # compare A->B with A--B
    ref = ex_pdag.ab3()
    pdag = ex_pdag.ab()
    metrics = bnlearn_compare(pdag, ref)
    assert metrics == {'tp': 0, 'fp': 1, 'fn': 1, 'shd': 0}


def test_bnlearn_compare_abc_itself_ok():  # compare A->B->C with itself
    ref = ex_pdag.abc()
    metrics = bnlearn_compare(ref, ref)
    assert metrics == {'tp': 2, 'fp': 0, 'fn': 0, 'shd': 0}


def test_bnlearn_compare_abc_to_cba_ok():  # compare A<-B<-C A->B->C

    # Note in bnlearn, wrong arc orientation adds 1 to FP and 1 to FN for DAG
    # comparison, but nothing on SHD (SHD compares CPDAGs)

    dag = ex_dag.abc3()
    ref = ex_dag.abc()
    metrics = bnlearn_compare(dag, ref)
    assert metrics == {'tp': 0, 'fp': 2, 'fn': 2, 'shd': 0}


def test_bnlearn_compare_abc_to_ab_cb_ok():  # A->B<-C with A->B<-C

    # DAG comparison has one arc-reversed, so FP and FN are 1 as is TP,
    # SHD compares A->B<-C with A--B--C giving SHD of 1

    ref = ex_pdag.abc()
    pdag = ex_pdag.ab_cb()
    metrics = bnlearn_compare(pdag, ref)
    assert metrics == {'tp': 1, 'fp': 1, 'fn': 1, 'shd': 2}


def test_bnlearn_compare_ab_cb_to_abc_ok():  # compare A->C<-B A->B->C

    # SHD is 2 here because two CPDAGs compared are A->B<-C and A--B--C
    # and bnlearn counts "edge not arc" as 1 towards SHD

    pdag = ex_pdag.ab_cb()
    ref = ex_pdag.abc()
    metrics = bnlearn_compare(pdag, ref)
    assert metrics == {'tp': 1, 'fp': 1, 'fn': 1, 'shd': 2}


def test_bnlearn_compare_abc3_to_abc_ok():  # compare A--B->C with A->B->C

    # DAG compare A--B with A->B giving 1 FP and 1 FN, SHD converts
    # both to A--B--C and so gives 0

    pdag = ex_pdag.abc3()
    ref = ex_pdag.abc()
    metrics = bnlearn_compare(pdag, ref)
    assert metrics == {'tp': 1, 'fp': 1, 'fn': 1, 'shd': 0}


def test_bnlearn_compare_abc3_to_ab_cb_ok():  # compare A--B->C with A->B<-C

    # DAG compare A--B with A->B and B->C with B<-C ==> 2FP 2FN
    # CPDAG: pdag to A--B--C, ref same as A->B<-C so SHD=2

    pdag = ex_pdag.abc3()
    ref = ex_pdag.ab_cb()
    metrics = bnlearn_compare(pdag, ref)
    assert metrics == {'tp': 0, 'fp': 2, 'fn': 2, 'shd': 2}


def test_bnlearn_compare_ab_cb_to_abc_acyclic_ok():  # A->B<-C cf A->B->C<-A

    # SHD is 3 here because two CPDAGs compared are A->B<-C and A--B--C--A
    # so 2 arcs_not_edges and 1 missing edge

    pdag = ex_pdag.ab_cb()
    ref = ex_pdag.abc_acyclic()
    metrics = bnlearn_compare(pdag, ref)
    assert metrics == {'tp': 1, 'fp': 1, 'fn': 2, 'shd': 3}


def test_bnlearn_compare_abc_to_abc_acyclic_ok():  # A->B->C cf A->B->C<-A

    # SHD is 1 here because two CPDAGs compared are A--B--C and A--B--C--A

    pdag = ex_pdag.abc()
    ref = ex_pdag.abc_acyclic()
    metrics = bnlearn_compare(pdag, ref)
    assert metrics == {'tp': 2, 'fp': 0, 'fn': 1, 'shd': 1}


def test_bnlearn_compare_and4_10_to_4_11_ok():  # 1>2>4,3>2 cf 1>2<4,3>2

    # SHD is 1 here because two CPDAGs compared are 1->2->4, 3->2 and
    # 1->2<-4, 3->2 and so one arc is reversed

    pdag = ex_pdag.and4_10()
    ref = ex_pdag.and4_11()
    metrics = bnlearn_compare(pdag, ref)
    assert metrics == {'tp': 2, 'fp': 1, 'fn': 1, 'shd': 1}
