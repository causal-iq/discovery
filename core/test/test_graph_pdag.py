
from core.graph import PDAG, NotPDAGError
import testdata.example_dags as dag
import testdata.example_pdags as pdag
import testdata.example_sdgs as sdg
import pytest


def test_graph_pdag_type_error1():  # bad argument types
    with pytest.raises(TypeError):
        PDAG()
    with pytest.raises(TypeError):
        PDAG(32)
    with pytest.raises(TypeError):
        PDAG('not', 'right')


def test_graph_pdag_type_error2():  # bad type within nodes
    with pytest.raises(TypeError):
        PDAG([1], [])


def test_graph_pdag_type_error3():  # bad type within edges
    with pytest.raises(TypeError):
        PDAG(['A', 'B'], [3])
    with pytest.raises(TypeError):
        PDAG(['A', 'B'], ['S'])
    with pytest.raises(TypeError):
        PDAG(['A', 'B'], [('A', '->')])
    with pytest.raises(TypeError):
        PDAG(['A', 'B'], [('A', '->', True)])


def test_graph_pdag_value_error1():  # empty node name
    with pytest.raises(ValueError):
        PDAG(['A', 'B', ''], [])


def test_graph_pdag_value_error2():  # duplicate node name
    with pytest.raises(ValueError):
        PDAG(['A', 'B', 'A'], [])


def test_graph_pdag_value_error3():  # cylic edge
    with pytest.raises(ValueError):
        PDAG(['A', 'B'], [('A', '->', 'A')])


def test_graph_pdag_value_error4():  # invalid edge symbol
    with pytest.raises(TypeError):
        PDAG(['A', 'B'], [('A', '?', 'B')])


def test_graph_pdag_value_error5():  # edge references unknown node
    with pytest.raises(ValueError):
        PDAG(['A', 'B'], [('A', '->', 'C')])


def test_graph_pdag_value_error6():  # duplicate edges
    with pytest.raises(ValueError):
        PDAG(['A', 'B'], [('A', 'o-o', 'B'), ('A', '->', 'B')])
    with pytest.raises(ValueError):
        PDAG(['A', 'B'], [('A', 'o-o', 'B'), ('B', '->', 'A')])


def test_graph_pdag_value_error7():  # unsupported edge types for PDAG
    with pytest.raises(NotPDAGError):
        PDAG(['A', 'B'], [('A', 'o-o', 'B')])
    with pytest.raises(NotPDAGError):
        PDAG(['A', 'B'], [('A', 'o->', 'B')])
    with pytest.raises(NotPDAGError):
        PDAG(['A', 'B'], [('A', '<->', 'B')])


def test_graph_pdag_value_error8():  # cycles present
    with pytest.raises(NotPDAGError):  # directed & cyclic
        PDAG(['A', 'B', 'C'], [('A', '->', 'B'), ('B', '->', 'C'),
                               ('C', '->', 'A')])
    with pytest.raises(NotPDAGError):  # partially directed with cycle
        PDAG(['A', 'B', 'C', 'D'], [('A', '->', 'B'), ('B', '->', 'C'),
                                    ('C', '->', 'A'), ('A', '-', 'D')])


def test_graph_pdag_empty_ok():  # empty graph
    pdag.empty(pdag.empty())


def test_graph_pdag_a_ok():  # single node DAG
    pdag.a(pdag.a())


def test_graph_pdag_ab_ok():  # A -> B chain
    pdag.ab(pdag.ab())


def test_graph_pdag_ab_2_ok():  # A -> B chain
    pdag.ab_2(pdag.ab_2())


def test_graph_pdag_ba_ok():  # B -> A chain
    pdag.ba(pdag.ba())


def test_graph_pdag_a_b_ok():  # A, B unconnected
    pdag.a_b(pdag.a_b())


def test_graph_pdag_ab3_ok():  # A - B PDAG
    pdag.ab3(pdag.ab3())


def test_graph_pdag_a_b_c_ok():  # A, B, C unconnected
    pdag.a_b_c(pdag.a_b_c())


def test_graph_pdag_ac_b_ok():  # A -> C, B
    pdag.ac_b(pdag.ac_b())


def test_graph_pdag_ac_b2_ok():  # A - C, B
    pdag.ac_b2(pdag.ac_b2())


def test_graph_pdag_abc_ok():  # A -> B -> C
    pdag.abc(pdag.abc())


def test_graph_pdag_abc2_ok():  # A -> B -> C
    pdag.abc2(pdag.abc2())


def test_graph_pdag_abc3_ok():  # A - B -> C
    pdag.abc3(pdag.abc3())


def test_graph_pdag_abc4_ok():  # A - B - C
    pdag.abc4(pdag.abc4())


def test_graph_pdag_abc5_ok():  # A -> B - C
    pdag.abc5(pdag.abc5())


def test_graph_pdag_abc6_ok():  # C - A - B
    pdag.abc6(pdag.abc6())


def test_graph_pdag_ab_ac_ok():  # B <- A -> C - common cause
    pdag.ab_ac(pdag.ab_ac())


def test_graph_pdag_ba_bc_ok():  # A <- B -> C - common cause
    pdag.ba_bc(pdag.ba_bc())


def test_graph_pdag_ac_bc_ok():  # A -> C <- B - common effect
    pdag.ac_bc(pdag.ac_bc())


def test_graph_pdag_ab_cb_ok():  # A -> B <- C - common effect
    pdag.ab_cb(pdag.ab_cb())


def test_graph_pdag_abc_acyclic_ok():  # A -> B -> C <- A
    pdag.abc_acyclic(pdag.abc_acyclic())


def test_graph_pdag_abc_acyclic2_ok():  # A -> B -> C - A
    pdag.abc_acyclic2(pdag.abc_acyclic2())


def test_graph_pdag_abc_acyclic3_ok():  # A -- B -- C, A --> C
    pdag.abc_acyclic3(pdag.abc_acyclic3())


def test_graph_pdag_abc_acyclic4_ok():  # A -- B -- C -- A
    pdag.abc_acyclic4(pdag.abc_acyclic4())


def test_graph_pdag_cancer1_OK():  # 5-node CANCER DAG as a PDAG
    pdag.cancer1(pdag.cancer1())


def test_graph_pdag_cancer2_OK():  # 5-node CANCER PDAG with 2 undirected edges
    pdag.cancer2(pdag.cancer2())


def test_graph_pdag_cancer3_OK():  # 5-node CANCER skeleton PDAG
    pdag.cancer3(pdag.cancer3())


def test_graph_pdag_asia_OK():  # Fully-orientated Asia PDAG
    pdag.asia(pdag.asia())


# Exemplar 4 node PDAGs from Andersson et al., 1995

def test_graph_pdag_and4_1_OK():  # 1  2  3  4 PDAG
    pdag.and4_1(pdag.and4_1())


def test_graph_pdag_and4_2_OK():  # 1 - 2  3  4 PDAG
    pdag.and4_2(pdag.and4_2())


def test_graph_pdag_and4_3_OK():  # 1 - 2  3 - 4 PDAG
    pdag.and4_3(pdag.and4_3())


def test_graph_pdag_and4_4_OK():  # 1 - 2 - 3  4 PDAG
    pdag.and4_4(pdag.and4_4())


def test_graph_pdag_and4_5_OK():  # 1 -> 2 <- 3  4 PDAG
    pdag.and4_5(pdag.and4_5())


def test_graph_pdag_and4_6_OK():  # 1 - 2 - 3 - 1  4 PDAG
    pdag.and4_6(pdag.and4_6())


def test_graph_pdag_and4_7_OK():  # 1 - 2 - 3 - 4 PDAG
    pdag.and4_7(pdag.and4_7())


def test_graph_pdag_and4_8_OK():  # 1 -> 2 <- 3 - 4 PDAG
    pdag.and4_8(pdag.and4_8())


def test_graph_pdag_and4_9_OK():  # 3 - 2 - 1, 2 - 4 (undirected star) PDAG
    pdag.and4_9(pdag.and4_9())


def test_graph_pdag_and4_10_OK():  # 1 -> 2 -> 4, 3 -> 2
    pdag.and4_10(pdag.and4_10())


def test_graph_pdag_and4_11_OK():  # 1 -> 2 <- 4, 3 -> 2 (star collider)
    pdag.and4_11(pdag.and4_11())


def test_graph_pdag_and4_12_OK():  # 2 - 3 - 1 - 2 - 4
    pdag.and4_12(pdag.and4_12())


def test_graph_pdag_and4_13_OK():  # 2 <- 1 - 3 -> 2 <- 4
    pdag.and4_13(pdag.and4_13())


def test_graph_pdag_and4_14_OK():  # 2 <- 1 -> 3 <- 2 <- 4
    pdag.and4_14(pdag.and4_14())


def test_graph_pdag_and4_15_OK():  # 2->4<-3, 2-1-3 (square, 1 collider)
    pdag.and4_15(pdag.and4_15())


def test_graph_pdag_and4_16_OK():  # 2->4<-3, 2->1<-3 (square colliders)
    pdag.and4_16(pdag.and4_16())


def test_graph_pdag_and4_17_OK():  # 4 - 3 - 1 - 2 - 4 - 1 (undirected square)
    pdag.and4_17(pdag.and4_17())


def test_graph_pdag_complete4_OK():  # complete skeleton
    pdag.complete4(pdag.complete4())


def test_graph_pdag_and4_inv1_OK():  # 1 -2 -3 -4 -1 (unextendable square)
    pdag.and4_inv1(pdag.and4_inv1())


# Equality and non-equality tests


def test_graph_pdag_empty_eq():  # compare identical empty PDAGs
    assert pdag.empty() == pdag.empty()
    assert (pdag.empty() != pdag.empty()) is False


def test_graph_pdag_a_eq():  # compare identical single node PDAGs
    assert pdag.a() == pdag.a()
    assert (pdag.a() != pdag.a()) is False


def test_graph_pdag_ab_eq():  # compare identical DAGs
    assert dag.ab() == dag.ab()
    assert (dag.ab() != dag.ab()) is False


def test_graph_pdag_ab_2_eq():  # identical, differently specified PDAGs
    assert pdag.ab_2() == pdag.ab()
    assert (pdag.ab_2() != pdag.ab()) is False


def test_graph_pdag_abc_eq():  # compare identical three node PDAGs (1)
    assert pdag.abc() == pdag.abc()
    assert (pdag.abc() != pdag.abc()) is False


def test_graph_pdag_abc3_eq():  # compare identical three node PDAGs (2)
    assert pdag.abc3() == pdag.abc3()
    assert (pdag.abc3() != pdag.abc3()) is False


def test_graph_pdag_abc4_eq():  # compare identical three node PDAGs (3)
    assert pdag.abc4() == pdag.abc4()
    assert (pdag.abc4() != pdag.abc4()) is False


def test_graph_pdag_sdg_ab_eq():  # compare identical SDG and PDAG
    assert pdag.ab() == sdg.ab()
    assert (pdag.ab() != sdg.ab()) is False


def test_graph_pdag_dag_ab_eq():  # compare identical DAG and PDAG
    assert pdag.ab() == dag.ab()
    assert (pdag.ab() != dag.ab()) is False


def test_graph_pdag_ne1():  # compare PDAG with non DAGs
    assert (pdag.ab() is None) is False
    assert pdag.ab() is not None
    assert (pdag.ab() == 1) is False
    assert pdag.ab() != 1
    assert (pdag.ab() == 4.7) is False
    assert pdag.ab() != 4.7
    assert (pdag.ab() == 'string') is False
    assert pdag.ab() != 'string'


def test_graph_pdag_ne2():  # compare different DAGs
    assert (pdag.a() == pdag.empty()) is False
    assert pdag.a() != pdag.empty()
    assert (pdag.ab() == pdag.empty()) is False
    assert pdag.ab() != pdag.empty()
    assert (pdag.ab() == pdag.ba()) is False
    assert pdag.ab() != pdag.ba()
    assert (pdag.ab() == pdag.a_b()) is False
    assert pdag.ab() != pdag.a_b()
    assert (pdag.ab() == pdag.abc()) is False
    assert pdag.ab() != pdag.abc()
    assert (pdag.ab_ac() == pdag.ac_bc()) is False
    assert pdag.ab_ac() != pdag.ac_bc()
