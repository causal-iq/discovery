
#   Tests of completing a PDAG including checks against bnlearn

import pytest

from core.graph import PDAG
from fileio.common import TESTDATA_DIR
from core.bn import BN
from call.bnlearn import bnlearn_cpdag
import testdata.example_pdags as ex_pdag
import testdata.example_dags as ex_dag


def test_graph_complete_pdag_type_error():  # bad argument types
    with pytest.raises(TypeError):
        PDAG.toCPDAG()
    with pytest.raises(TypeError):
        PDAG.toCPDAG(32)
    with pytest.raises(TypeError):
        PDAG.toCPDAG('[not][supported]')

# Validate against small internal test PDAGs/DAGs


def test_graph_complete_pdag_empty_ok1():  # empty PDAG
    pdag = ex_pdag.empty()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\ncompleted to\n{}\n'.format(pdag, cpdag))
    assert pdag == cpdag  # empty PDAG


def test_graph_complete_pdag_empty_ok2():  # empty DAG
    dag = ex_dag.empty()
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\ncompleted to\n{}\n'.format(dag, cpdag))
    assert dag == cpdag  # empty PDAG


def test_graph_complete_pdag_a_ok1():  # single node PDAG
    pdag = ex_pdag.a()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\ncompleted to\n{}\n'.format(pdag, cpdag))
    assert pdag == cpdag  # A  PDAG


def test_graph_complete_pdag_a_ok2():  # single node DAG
    dag = ex_dag.a()
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\ncompleted to\n{}\n'.format(dag, cpdag))
    assert dag == cpdag  # A PDAG


def test_graph_complete_pdag_ab_ok1():  # A->B PDAG
    pdag = ex_pdag.ab()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\ncompleted to\n{}\n'.format(pdag, cpdag))
    assert cpdag == ex_pdag.ab3()  # A-B PDAG


def test_graph_complete_pdag_ab_ok2():  # A->B DAG
    dag = ex_dag.ab()
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\ncompleted to\n{}\n'.format(dag, cpdag))
    assert cpdag == ex_pdag.ab3()  # A-B PDAG


def test_graph_complete_pdag_abc_ok1():  # A->B->C PDAG
    pdag = ex_pdag.abc()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\ncompleted to\n{}\n'.format(pdag, cpdag))
    assert cpdag == ex_pdag.abc4()  # A-B-C PDAG


def test_graph_complete_pdag_abc_ok2():  # A->B->C DAG
    dag = ex_dag.abc()
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\ncompleted to\n{}\n'.format(dag, cpdag))
    assert cpdag == ex_pdag.abc4()  # A-B-C PDAG


def test_graph_complete_pdag_ba_bc_ok1():  # A<-B->C PDAG
    pdag = ex_pdag.ba_bc()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\ncompleted to\n{}\n'.format(pdag, cpdag))
    assert cpdag == ex_pdag.abc4()  # A-B-C PDAG


def test_graph_complete_pdag_ba_bc_ok2():  # A<-B->C DAG
    dag = ex_dag.ba_bc()
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\ncompleted to\n{}\n'.format(dag, cpdag))
    assert cpdag == ex_pdag.abc4()  # A-B-C PDAG

# Validate against small internal DAGs and bnlearn


def test_graph_complete_pdag_cancer_ok1():  # Cancer Standard DAG
    dag = (BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nCANCER completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == ex_pdag.cancer1()
    assert cpdag == bnlearn_cpdag(dag)


def test_graph_complete_pdag_cancer_ok2():  # Cancer PDAG with collider only
    pdag = ex_pdag.cancer2()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\nCANCER (collider-only) completed to\n{}\n'
          .format(pdag, cpdag))
    assert cpdag == ex_pdag.cancer1()
    assert cpdag == bnlearn_cpdag(pdag)


def test_graph_complete_pdag_asia_ok1():  # Asia Standard DAG
    dag = (BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nASIA completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == ex_pdag.asia()
    assert cpdag == bnlearn_cpdag(dag)


def test_graph_complete_pdag_sports_ok():  # Sports Standard DAG
    dag = (BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nSports completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)

# Validate medium DAGs against bnlearn


def test_graph_complete_pdag_child_ok1():  # CHILD (n=20) Standard DAG
    dag = (BN.read(TESTDATA_DIR + '/discrete/medium/child.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nCHILD completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


def test_graph_complete_pdag_insurance_ok1():  # INSURANCE (n=27) Standard DAG
    dag = (BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nINSURANCE completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


def test_graph_complete_pdag_water_ok1():  # WATER (n=32) Standard DAG
    dag = (BN.read(TESTDATA_DIR + '/discrete/medium/water.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nWATER completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


def test_graph_complete_pdag_alarm_ok1():  # ALARM (n=37) Standard DAG
    dag = (BN.read(TESTDATA_DIR + '/discrete/medium/alarm.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nALARM completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)

# Validate large DAGs against bnlearn


def test_graph_complete_pdag_hailfinder_ok1():  # HAILFINDER (n=56) Std DAG
    dag = (BN.read(TESTDATA_DIR + '/discrete/large/hailfinder.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nHAILFINDER completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


def test_graph_complete_pdag_hepar2_ok1():  # HEPAR2 (n=70) Std DAG
    dag = (BN.read(TESTDATA_DIR + '/discrete/large/hepar2.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nHAILFINDER completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


def test_graph_complete_pdag_win95pts_ok1():  # WIN95PTS (n=76) Std DAG
    dag = (BN.read(TESTDATA_DIR + '/discrete/large/win95pts.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nWIN95PTS completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)

# Validate very large DAGs against bnlearn


def test_graph_complete_pdag_pathfinder_ok1():  # PATHFINDER (n=109) Std DAG
    dag = (BN.read(TESTDATA_DIR + '/discrete/verylarge/pathfinder.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nPATHFINDER completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)
