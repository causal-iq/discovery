
from core.graph import DAG
import testdata.example_dags as ex_dag
import testdata.example_pdags as ex_pdag
# import testdata.example_sdgs as sdg
import pytest


def test_graph_extendPDAG_type_error():  # bad argument types
    with pytest.raises(TypeError):
        DAG.extendPDAG()
    with pytest.raises(TypeError):
        DAG.extendPDAG(32)
    with pytest.raises(TypeError):
        DAG.extendPDAG('not', 'right')


def test_graph_extendPDAG_empty_ok():  # empty PDAG
    pdag = ex_pdag.empty()
    print("\nEmpty PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.empty(dag)


def test_graph_extendPDAG_a_ok():  # single node PDAG
    pdag = ex_pdag.a()
    print("\nSingle node PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.a(dag)


def test_graph_extendPDAG_a_b_ok():  # A  B PDAG
    pdag = ex_pdag.a_b()
    print("\nA  B PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.a_b(dag)


def test_graph_extendPDAG_ab_ok():  # A -> B PDAG
    pdag = ex_pdag.ab()
    print("\nA --> B PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.ab(dag)


def test_graph_extendPDAG_ab3_ok():  # A - B PDAG
    pdag = ex_pdag.ab3()
    print("\nA - B PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.ba(dag)


def test_graph_extendPDAG_a_b_c_ok():  # A  B  C PDAG
    pdag = ex_pdag.a_b_c()
    print("\nA  B  C PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.a_b_c(dag)


def test_graph_extendPDAG_abc_ok():  # A -> B -> C PDAG
    pdag = ex_pdag.abc()
    print("\nA -> B -> C PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.abc(dag)


def test_graph_extendPDAG_ac_bc_ok():  # A -> C <- B PDAG
    pdag = ex_pdag.ac_bc()
    print("\nA -> C -> B PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.ac_bc(dag)


def test_graph_extendPDAG_abc_acyclic4_ok():  # A - B - C - A PDAG
    pdag = ex_pdag.abc_acyclic4()
    print("\nA - B - C - A PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.abc_acyclic4(dag)  # NB Exact DAG produced is an artefact


def test_graph_extendPDAG_ac_b_ok():  # A -> C   B PDAG
    pdag = ex_pdag.ac_b()
    print("\nA -> C   B PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.ac_b(dag)


def test_graph_extendPDAG_cancer1_ok():  # fully-oriented cancer PDAG
    pdag = ex_pdag.cancer1()
    print("\nFully Orienatated Cancer PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.cancer(dag)


def test_graph_extendPDAG_cancer2_ok():  # v-structure cancer PDAG
    pdag = ex_pdag.cancer2()
    print("\nV-structure Cancer PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.cancer(dag)  # should recover correct DAG


def test_graph_extendPDAG_cancer3_ok():  # skeleton Cancer PDAG
    pdag = ex_pdag.cancer3()
    print("\nSkeleton Cancer PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.cancer3(dag)  # NB Exact DAG produced is an artefact


def test_graph_extendPDAG_asia_ok():  # fully orientated Asia PDAG
    pdag = ex_pdag.asia()
    print("\nFully orientated Asia PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.asia2(dag)  # NB Exact DAG produced is an artefact

# Exemplar 4 node PDAGs from Andersson et al., 1995


def test_graph_extendPDAG_and4_1_ok():  # 1  2  3  4 PDAG
    pdag = ex_pdag.and4_1()
    print("\n1  2  3  4 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_1(dag)  # 1  2  3  4 DAG


def test_graph_extendPDAG_and4_2_ok():  # 1 - 2  3  4 PDAG
    pdag = ex_pdag.and4_2()
    print("\n1 - 2  3  4 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_2(dag)  # 1 <- 2  3  4 DAG (artefact)


def test_graph_extendPDAG_and4_3_ok():  # 1 - 2  3 - 4 PDAG
    pdag = ex_pdag.and4_3()
    print("\n1 - 2  3 - 4 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_3(dag)  # 1 <- 2  3 <- 4 DAG (artefact)


def test_graph_extendPDAG_and4_4_ok():  # 1 - 2 - 3  4 PDAG
    pdag = ex_pdag.and4_4()
    print("\n1 - 2 - 3  4 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_4(dag)  # 1 <- 2 <- 3  4 DAG (artefact)


def test_graph_extendPDAG_and4_5_ok():  # 1 -> 2 <- 3  4 PDAG
    pdag = ex_pdag.and4_5()
    print("\n1 -> 2 <- 3  4 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_5(dag)  # 1 -> 2 <- 3  4 DAG


def test_graph_extendPDAG_and4_6_ok():  # 1 - 2 - 3 - 1  4 PDAG
    pdag = ex_pdag.and4_6()
    print("\n1 - 2 - 3 - 1  4 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_6(dag)  # 1 <- 3 -> 2 -> 1  4 DAG


def test_graph_extendPDAG_and4_7_ok():  # 1 - 2 - 3 - 4 PDAG
    pdag = ex_pdag.and4_7()
    print("\n1 - 2 - 3 - 4 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_7(dag)  # 1 <- 2 <- 3 <- 4 DAG (artefact)


def test_graph_extendPDAG_and4_8_ok():  # 1 -> 2 <- 3 - 4 PDAG
    pdag = ex_pdag.and4_8()
    print("\n1 -> 2 <- 3 - 4 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_8(dag)  # 1 -> 2 <- 3 <- 4 DAG (artefact)


def test_graph_extendPDAG_and4_9_ok():  # 3 - 2 - 1, 2 - 4 (undirected star)
    pdag = ex_pdag.and4_9()
    print("\n3 - 2 - 1, 2 - 4 (undirected star) PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_9(dag)  # 4 -> 2 -> 1, 2 -> 3  DAG (artefact)


def test_graph_extendPDAG_and4_10_ok():  # 1 -> 2 -> 4, 3 -> 2 PDAG
    pdag = ex_pdag.and4_10()
    print("\n1 -> 2 -> 4, 3 -> 2 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_10(dag)  # 1 -> 2 -> 4, 3 -> 2 DAG


def test_graph_extendPDAG_and4_11_ok():  # 1 -> 2 <- 4, 3 -> 2 (star collider)
    pdag = ex_pdag.and4_11()
    print("\n1 -> 2 <- 4, 3 -> 2 (star collider) PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_11(dag)  # 1 -> 2 <- 4, 3 -> 2 (star collider)


def test_graph_extendPDAG_and4_12_ok():  # 2 - 1 - 3 - 2 - 4
    pdag = ex_pdag.and4_12()
    print("\n2 - 1 - 3 - 2 - 4 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_12(dag)  # 2 -> 1 <- 3 <- 2 <- 4 (artefact)


def test_graph_extendPDAG_and4_13_ok():  # 2 <- 1 - 3 -> 2 <- 4
    pdag = ex_pdag.and4_13()
    print("\n2 <- 1 - 3 -> 2 <- 4 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_13(dag)  # 2 <- 1 <- 3 -> 2 <- 4 (artefact)


def test_graph_extendPDAG_and4_14_ok():  # 2 <- 1 -> 3 <- 2 <- 4
    pdag = ex_pdag.and4_14()
    print("\n2 <- 1 -> 3 <- 2 <- 4 PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_14(dag)  # 2 <- 1 -> 3 <- 2 <- 4


def test_graph_extendPDAG_and4_15_ok():  # 2->4<-3, 2-1-3 (square, 1 collider)
    pdag = ex_pdag.and4_15()
    print("\n2->4<-3, 2-1-3 (square, 1 collider) PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_15(dag)  # 1->2->4<-3->1 (square, 1 collider)


def test_graph_extendPDAG_and4_16_ok():  # 2->4<-3, 2->1<-3 (square colliders)
    pdag = ex_pdag.and4_16()
    print("\n2->4<-3, 2->1<-3 (square colliders) PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_16(dag)  # 2->4<-3, 2->1<-3 (square colliders)


def test_graph_extendPDAG_and4_17_ok():  # 4 - 3 - 1 - 2 - 4 - 1 (undir square)
    pdag = ex_pdag.and4_17()
    print("\n4 - 3 - 1 - 2 - 4 - 1 (undirected square) PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.and4_17(dag)  # 4->3->1->2, 4->1, 4->2


def test_graph_extendPDAG_and4_inv1_ok():  # 1-2-3-4-1 (unextendable square)
    pdag = ex_pdag.and4_inv1()
    print("\n1-2-3-4-1 (unextendable square PDAG:\n{}".format(pdag))
    print("This pdag is un-extendable")
    with pytest.raises(ValueError):
        DAG.extendPDAG(pdag)


def test_graph_extendPDAG_complete4_ok():  # complete skeleton
    pdag = ex_pdag.complete4()
    print("\ncomplete skeleton PDAG:\n{}".format(pdag))
    dag = DAG.extendPDAG(pdag)
    print("\nextended to:\n{}".format(dag))
    ex_dag.complete4(dag)  # complete DAG (artefact)
