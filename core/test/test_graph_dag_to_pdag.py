
from core.graph import PDAG
import testdata.example_dags as ex_dag
import testdata.example_pdags as ex_pdag
import pytest


def test_graph_fromDAG_type_error():  # bad argument types
    with pytest.raises(TypeError):
        PDAG.fromDAG()
    with pytest.raises(TypeError):
        PDAG.fromDAG(32)
    with pytest.raises(TypeError):
        PDAG.fromDAG('not', 'right')


def test_graph_dag_to_pdag_empty_ok():  # empty DAG
    dag = ex_dag.empty()
    print("\nEmpty DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.empty(pdag)


def test_graph_dag_to_pdag_a_ok():  # A (single node) DAG
    dag = ex_dag.a()
    print("\nA (single-node) DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.a(pdag)

# 2 Node DAGs


def test_graph_dag_to_pdag_a_b_ok():  # A  B  DAG
    dag = ex_dag.a_b()
    print("\nA  B  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.a_b(pdag)  # A  B  PDAG


def test_graph_dag_to_pdag_ab_ok():  # A -> B DAG
    dag = ex_dag.ab()
    print("\nA -> B DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.ab3(pdag)  # A - B  PDAG


def test_graph_dag_to_pdag_ba_ok():  # B -> A DAG
    dag = ex_dag.ba()
    print("\nB -> A DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.ab3(pdag)  # A - B  PDAG

# 3 Node DAGs


def test_graph_dag_to_pdag_a_b_c_ok():  # A  B  C  DAG
    dag = ex_dag.a_b_c()
    print("\nA  B  C  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.a_b_c(pdag)  # A  B  C  PDAG


def test_graph_dag_to_pdag_ac_b_ok():  # A -> C   B  DAG
    dag = ex_dag.ac_b()
    print("\nA -> C  B  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.ac_b2(pdag)  # A - C  B  PDAG


def test_graph_dag_to_pdag_ac_b2_ok():  # C -> A   B  DAG
    dag = ex_dag.ac_b2()
    print("\nC -> A  B  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.ac_b2(pdag)  # A - C  B  PDAG


def test_graph_dag_to_pdag_abc_ok():  # A -> B -> C  DAG
    dag = ex_dag.abc()
    print("\nA -> B -> C  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.abc4(pdag)  # A - B - C  PDAG


def test_graph_dag_to_pdag_abc3_ok():  # A <- B <- C  DAG
    dag = ex_dag.abc3()
    print("\nA <- B <- C  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.abc4(pdag)  # A - B - C  PDAG


def test_graph_dag_to_pdag_ab_ac_ok():  # C <- A -> B  DAG
    dag = ex_dag.ab_ac()
    print("\nA -> B -> C  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.abc6(pdag)  # C - A - B  PDAG


def test_graph_dag_to_pdag_ac_bc_ok():  # A -> C <- B (collider)  DAG
    dag = ex_dag.ac_bc()
    print("\nA -> C <- B  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.ac_bc(pdag)  # A -> C <- B  PDAG


def test_graph_dag_to_pdag_abc_acyclic_ok():  # C <- A -> B -> C (complete) DAG
    dag = ex_dag.abc_acyclic()
    print("\nC <- A -> B -> C  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.abc_acyclic4(pdag)  # C - A - B - C  PDAG

# Exemplar 4 node PDAGs from Andersson et al., 1995


def test_graph_dag_to_pdag_and4_1_ok():  # 1  2  3  4  DAG
    dag = ex_dag.and4_1()
    print("\n1  2  3  4  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_1(pdag)  # 1  2  3  4  PDAG


def test_graph_dag_to_pdag_and4_2_ok():  # 1 <- 2  3  4  DAG
    dag = ex_dag.and4_2()
    print("\n1 <- 2  3  4  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_2(pdag)  # 1 - 2  3  4  PDAG


def test_graph_dag_to_pdag_and4_3_ok():  # 1 <- 2  3 <- 4  DAG
    dag = ex_dag.and4_3()
    print("\n1 <- 2  3 <- 4  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_3(pdag)  # 1 - 2  3 - 4  PDAG


def test_graph_dag_to_pdag_and4_4_ok():  # 1 <- 2 <- 3  4  DAG
    dag = ex_dag.and4_4()
    print("\n1 <- 2 <- 3  4  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_4(pdag)  # 1 - 2 - 3  4  PDAG


def test_graph_dag_to_pdag_and4_5_ok():  # 1 -> 2 <- 3  4  DAG
    dag = ex_dag.and4_5()
    print("\n1 -> 2 <- 3  4  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_5(pdag)  # 1 -> 2 <- 3  4  PDAG


def test_graph_dag_to_pdag_and4_6_ok():  # 1 <- 3 -> 2 -> 1  4  DAG
    dag = ex_dag.and4_6()
    print("\n1 <- 3 -> 2 -> 1  4  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_6(pdag)  # 1 - 2 - 3 - 1  4  PDAG


def test_graph_dag_to_pdag_and4_7_ok():  # 1 <- 2 <- 3 <- 4  DAG
    dag = ex_dag.and4_7()
    print("\n1 <- 2 <- 3 <- 4  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_7(pdag)  # 1 - 2 - 3 - 4  PDAG


def test_graph_dag_to_pdag_and4_8_ok():  # 1 -> 2 <- 3 <- 4  DAG
    dag = ex_dag.and4_8()
    print("\n1 -> 2 <- 3 <- 4  DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_8(pdag)  # 1 -> 2 <- 3 - 4  PDAG


def test_graph_dag_to_pdag_and4_9_ok():  # 3 -> 2 -> 1, 2 -> 4  DAG
    dag = ex_dag.and4_9()
    print("\n1 -> 2 -> 4, 3 -> 2 DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_9(pdag)  # 3 - 2 - 1, 2 - 4 (undirected star) PDAG


def test_graph_dag_to_pdag_and4_10_ok():  # 1 -> 2 -> 4, 3 -> 2
    dag = ex_dag.and4_10()
    print("\n1 -> 2 -> 4, 3 -> 2 DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_10(pdag)  # 1 -> 2 -> 4, 3 -> 2  PDAG


def test_graph_dag_to_pdag_and4_11_ok():  # 1 -> 2 <- 4, 3 -> 2 (star collider)
    dag = ex_dag.and4_11()
    print("\n1 -> 2 <- 4, 3 -> 2 (star collider) DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_11(pdag)  # 1 -> 2 <- 4, 3 -> 2 (star collider) PDAG


def test_graph_dag_to_pdag_and4_12_ok():  # 2 -> 1 <- 3 <- 2 <- 4
    dag = ex_dag.and4_12()
    print("\n1 -> 2 <- 4, 3 -> 2 (star collider) DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_12(pdag)  # 2 - 3 - 1 - 2 - 4 PDAG


def test_graph_dag_to_pdag_and4_13_ok():  # 2 <- 1 <- 3 -> 2 <- 4
    dag = ex_dag.and4_13()
    print("\n2 <- 1 <- 3 -> 2 <- 4 DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_13(pdag)  # 2 <- 1 - 3 -> 2 <- 4 PDAG


def test_graph_dag_to_pdag_and4_14_ok():  # 2 <- 1 -> 3 <- 2 <- 4
    dag = ex_dag.and4_14()
    print("\n2 <- 1 -> 3 <- 2 <- 4 DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_14(pdag)  # 2 <- 1 -> 3 <- 2 <- 4 PDAG


def test_graph_dag_to_pdag_and4_15_ok():  # 1->2->4<-3->1 (square, 1 collider)
    dag = ex_dag.and4_15()
    print("\n1->2->4<-3->1 (square, 1 collider) DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_15(pdag)  # 2->4<-3, 2-1-3 (square, 1 collider) PDAG


def test_graph_dag_to_pdag_and4_16_ok():  # 2->4<-3, 2->1<-3 (square colliders)
    dag = ex_dag.and4_16()
    print("\n2->4<-3, 2->1<-3 (square colliders) DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_16(pdag)  # 2->4<-3, 2->1<-3 (square colliders) PDAG


def test_graph_dag_to_pdag_and4_17_ok():  # 4->3->1->2, 4->1, 4->2
    dag = ex_dag.and4_17()
    print("\n4->3->1->2, 4->1, 4->2 DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.and4_17(pdag)  # 4 - 3 - 1 - 2 - 4 - 1 (undirected square)


def test_graph_dag_to_pdag_complete4_ok():  # 4 nodes, 6 edges
    dag = ex_dag.complete4()
    print("\n4 nodes, 6 edges DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.complete4(pdag)  # complete skeleton


#   Standard BNs

def test_graph_dag_to_pdag_cancer_ok():  # Cancer DAG
    dag = ex_dag.cancer()
    print("\nCancer DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.cancer1(pdag)  # fully orientated Cancer PDAG


def test_graph_dag_to_pdag_asia_ok():  # Asia DAG
    dag = ex_dag.asia()
    print("\nAsia DAG:\n{}".format(dag))
    pdag = PDAG.fromDAG(dag)
    print("\nextends PDAG:\n{}".format(pdag))
    ex_pdag.asia(pdag)  # fully orientated Cancer PDAG
