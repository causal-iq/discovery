
from core.graph import DAG, NotDAGError
import testdata.example_dags as dag
import testdata.example_pdags as ex_pdag
import testdata.example_sdgs as ex_sdg
import pytest


def test_graph_dag_type_error1():  # bad argument types
    with pytest.raises(TypeError):
        DAG()
    with pytest.raises(TypeError):
        DAG(32)
    with pytest.raises(TypeError):
        DAG('not', 'right')


def test_graph_dag_type_error2():  # bad type within nodes
    with pytest.raises(TypeError):
        DAG([1], [])


def test_graph_dag_type_error3():  # bad type within edges
    with pytest.raises(TypeError):
        DAG(['A', 'B'], [3])
    with pytest.raises(TypeError):
        DAG(['A', 'B'], ['S'])
    with pytest.raises(TypeError):
        DAG(['A', 'B'], [('A', '->')])
    with pytest.raises(TypeError):
        DAG(['A', 'B'], [('A', '->', True)])


def test_graph_dag_value_error1():  # empty node name
    with pytest.raises(ValueError):
        DAG(['A', 'B', ''], [])


def test_graph_dag_value_error2():  # duplicate node name
    with pytest.raises(ValueError):
        DAG(['A', 'B', 'A'], [])


def test_graph_dag_value_error3():  # cylic edge
    with pytest.raises(ValueError):
        DAG(['A', 'B'], [('A', '->', 'A')])


def test_graph_dag_value_error4():  # invalid edge symbol
    with pytest.raises(TypeError):
        DAG(['A', 'B'], [('A', '?', 'B')])


def test_graph_dag_value_error5():  # edge references unknown node
    with pytest.raises(ValueError):
        DAG(['A', 'B'], [('A', '->', 'C')])


def test_graph_dag_value_error6():  # duplicate edges
    with pytest.raises(ValueError):
        DAG(['A', 'B'], [('A', 'o-o', 'B'), ('A', '->', 'B')])
    with pytest.raises(ValueError):
        DAG(['A', 'B'], [('A', 'o-o', 'B'), ('B', '->', 'A')])


def test_graph_dag_value_error7():  # undirected edges
    with pytest.raises(NotDAGError):
        DAG(['A', 'B'], [('A', 'o-o', 'B')])
    with pytest.raises(NotDAGError):
        DAG(['A', 'B'], [('A', 'o->', 'B')])
    with pytest.raises(NotDAGError):
        DAG(['A', 'B'], [('A', '-', 'B')])
    with pytest.raises(NotDAGError):
        DAG(['A', 'B'], [('A', '<->', 'B')])


def test_graph_dag_value_error8():  # cycles present
    with pytest.raises(NotDAGError):
        DAG(['A', 'B', 'C'], [('A', '->', 'B'), ('B', '->', 'C'),
                              ('C', '->', 'A')])


def test_graph_dag_empty_ok():  # empty graph
    dag.empty(dag.empty())


def test_graph_dag_a_ok():  # single node DAG
    dag.a(dag.a())


def test_graph_dag_x_ok():  # single node DAG
    dag.x(dag.x())


def test_graph_dag_ab_ok():  # A -> B chain
    dag.ab(dag.ab())


def test_graph_dag_xy_ok():  # X -> Y chain
    dag.xy(dag.xy())


def test_graph_dag_yx_ok():  # X <- Y chain
    dag.yx(dag.yx())


def test_graph_dag_ba_ok():  # B -> A chain
    dag.ba(dag.ba())


def test_graph_dag_a_b_ok():  # A, B unconnected
    dag.a_b(dag.a_b())


def test_graph_dag_x_y_ok():  # X, Y unconnected
    dag.x_y(dag.x_y())


def test_graph_dag_a_b_c_ok():  # A, B, C unconnected
    dag.a_b_c(dag.a_b_c())


def test_graph_dag_ac_b_ok():  # A -> C, B
    dag.ac_b(dag.ac_b())


def test_graph_dag_ac_b2_ok():  # C -> A, B
    dag.ac_b2(dag.ac_b2())


def test_graph_dag_abc_ok():  # A -> B -> C
    dag.abc(dag.abc())


def test_graph_dag_xyz_ok():  # X -> Y -> Z
    dag.xyz(dag.xyz())


def test_graph_dag_abc_2_ok():  # A -> B -> C
    dag.abc_2(dag.abc_2())


def test_graph_dag_abc3_ok():  # A <- B <- C
    dag.abc3(dag.abc3())


def test_graph_dag_ab_ac_ok():  # B <- A -> C - common cause
    dag.ab_ac(dag.ab_ac())


def test_graph_dag_ba_bc_ok():  # A <- B -> C - common cause
    dag.ba_bc(dag.ba_bc())


def test_graph_dag_ac_bc_ok():  # A -> C <- B - common effect
    dag.ac_bc(dag.ac_bc())


def test_graph_dag_xy_zy_ok():  # X -> Y <- Z - common effect
    dag.xy_zy(dag.xy_zy())


def test_graph_dag_abc_acyclic_ok():  # A -> B -> C <- A
    dag.abc_acyclic(dag.abc_acyclic())


def test_graph_dag_abc_acyclic4_ok():  # C -> B -> A <- C
    dag.abc_acyclic4(dag.abc_acyclic4())


def test_cancer_dag():  # 5-node CANCER DAG
    dag.cancer(dag.cancer())


def test_cancer3_dag():  # 5-node CANCER DAG with Xray as root
    dag.cancer3(dag.cancer3())


def test_asia_dag():  # Correct 8-node ASIA DAG
    dag.asia(dag.asia())


def test_asia2_dag():  # Wrong 8-node ASIA DAG produced by extending Asia PDAG
    dag.asia2(dag.asia2())

# Exemplar 4 node PDAGs from Andersson et al., 1995


def test_and4_1_dag():  # 1  2  3  4 DAG
    dag.and4_1(dag.and4_1())


def test_and4_2_dag():  # 1 <- 2  3  4 DAG
    dag.and4_2(dag.and4_2())


def test_and4_3_dag():  # 1 <- 2  3 <- 4 DAG
    dag.and4_3(dag.and4_3())


def test_and4_4_dag():  # 1 <- 2 <- 3  4  DAG
    dag.and4_4(dag.and4_4())


def test_and4_5_dag():  # 1 -> 2 <- 3  4  DAG
    dag.and4_5(dag.and4_5())


def test_and4_6_dag():  # 1 <- 3 -> 2 -> 1  4  DAG
    dag.and4_6(dag.and4_6())


def test_and4_7_dag():  # 1 <- 2 <- 3 <- 4  DAG
    dag.and4_7(dag.and4_7())


def test_and4_8_dag():  # 1 -> 2 <- 3 <- 4  DAG
    dag.and4_8(dag.and4_8())


def test_and4_9_dag():  # 4 -> 2 -> 1, 2 -> 3
    dag.and4_9(dag.and4_9())


def test_and4_10_dag():  # 1 -> 2 -> 4, 3 -> 2
    dag.and4_10(dag.and4_10())


def test_and4_11_dag():  # 1 -> 2 <- 4, 3 -> 2 (star collider)
    dag.and4_11(dag.and4_11())


def test_and4_12_dag():  # 2 -> 1 <- 3 <- 2 <- 4
    dag.and4_12(dag.and4_12())


def test_and4_13_dag():  # 2 <- 1 <- 3 -> 2 <- 4
    dag.and4_13(dag.and4_13())


def test_and4_14_dag():  # 2 <- 1 -> 3 <- 2 <- 4
    dag.and4_14(dag.and4_14())


def test_and4_15_dag():  # 1->2->4<-3->1 (square, 1 collider)
    dag.and4_15(dag.and4_15())


def test_and4_16_dag():  # 4->3->1->2, 4->1, 4->2
    dag.and4_16(dag.and4_16())


def test_and4_17_dag():  # 4->3->1->2, 4->1, 4->2
    dag.and4_17(dag.and4_17())


def test_complete4_dag():  # 4->3->1->2, 4->1, 4->2
    dag.complete4(dag.complete4())


def test_gauss_dag():  # BNLearn 7-node example Gaussian DAG
    dag.gauss(dag.gauss())


# Equality and inequality tests

def test_graph_dag_empty_eq():  # compare identical DAGs
    assert dag.empty() == dag.empty()
    assert (dag.empty() != dag.empty()) is False


def test_graph_dag_a_eq():  # compare identical DAGs
    assert dag.a() == dag.a()
    assert (dag.a() != dag.a()) is False


def test_graph_dag_ab_eq():  # compare identical DAGs
    assert dag.ab() == dag.ab()
    assert (dag.ab() != dag.ab()) is False


def test_graph_dag_ab_2_eq():  # compare identical, differently specified DAGs
    assert dag.ab_2() == dag.ab()
    assert (dag.ab_2() != dag.ab()) is False


def test_graph_dag_abc_eq():  # compare identical DAGs
    assert dag.abc() == dag.abc()
    assert (dag.abc() != dag.abc()) is False


def test_graph_dag_abc_2_eq():  # compare identical, differently specified DAGs
    assert dag.abc_2() == dag.abc()
    assert (dag.abc_2() != dag.abc()) is False


def test_graph_dag_graph_ab_eq():  # compare identical SDG and DAG
    assert dag.ab() == ex_sdg.ab()
    assert (dag.ab() != ex_sdg.ab()) is False


def test_graph_dag_graph_ab2_eq():  # compare identical PDAG and DAG
    assert dag.ab() == ex_pdag.ab()
    assert (dag.ab() != ex_pdag.ab()) is False


def test_graph_dag_ne1():  # compare DAG with non DAGs
    assert (dag.ab() is None) is False
    assert dag.ab() is not None
    assert (dag.ab() == 1) is False
    assert dag.ab() != 1
    assert (dag.ab() == 4.7) is False
    assert dag.ab() != 4.7
    assert (dag.ab() == 'string') is False
    assert dag.ab() != 'string'


def test_graph_dag_ne2():  # compare different DAGs
    assert (dag.a() == dag.empty()) is False
    assert dag.a() != dag.empty()
    assert (dag.ab() == dag.empty()) is False
    assert dag.ab() != dag.empty()
    assert (dag.ab() == dag.ba()) is False
    assert dag.ab() != dag.ba()
    assert (dag.ab() == dag.a_b()) is False
    assert dag.ab() != dag.a_b()
    assert (dag.ab() == dag.abc()) is False
    assert dag.ab() != dag.abc()
    assert (dag.ab_ac() == dag.ac_bc()) is False
    assert dag.ab_ac() != dag.ac_bc()
