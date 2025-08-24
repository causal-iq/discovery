
#   Tests of completing a PDAG including checks against bnlearn

import pytest

from core.graph import PDAG
from fileio.common import TESTDATA_DIR
from core.bn import BN
from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_cpdag
import testdata.example_pdags as ex_pdag
import testdata.example_dags as ex_dag


# --- Failure cases

# bad agrument types
def test_graph_complete_pdag_type_error():
    with pytest.raises(TypeError):
        PDAG.toCPDAG()
    with pytest.raises(TypeError):
        PDAG.toCPDAG(32)
    with pytest.raises(TypeError):
        PDAG.toCPDAG('[not][supported]')


# --- Validate against small internal test PDAGs/DAGs

# empty PDAG
def test_graph_complete_pdag_empty_ok1():
    pdag = ex_pdag.empty()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\ncompleted to\n{}\n'.format(pdag, cpdag))
    assert pdag == cpdag  # empty PDAG


# empty DAG
def test_graph_complete_pdag_empty_ok2():
    dag = ex_dag.empty()
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\ncompleted to\n{}\n'.format(dag, cpdag))
    assert dag == cpdag  # empty PDAG


# single node PDAG
def test_graph_complete_pdag_a_ok1():
    pdag = ex_pdag.a()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\ncompleted to\n{}\n'.format(pdag, cpdag))
    assert pdag == cpdag  # A  PDAG


# single node DAG
def test_graph_complete_pdag_a_ok2():
    dag = ex_dag.a()
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\ncompleted to\n{}\n'.format(dag, cpdag))
    assert dag == cpdag  # A PDAG


# A->B PDAG
def test_graph_complete_pdag_ab_ok1():
    pdag = ex_pdag.ab()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\ncompleted to\n{}\n'.format(pdag, cpdag))
    assert cpdag == ex_pdag.ab3()  # A-B PDAG


# A->B->C PDAG
def test_graph_complete_pdag_abc_ok1():
    pdag = ex_pdag.abc()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\ncompleted to\n{}\n'.format(pdag, cpdag))
    assert cpdag == ex_pdag.abc4()  # A-B-C PDAG


# A<-B->C PDAG
def test_graph_complete_pdag_ba_bc_ok1():
    pdag = ex_pdag.ba_bc()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\ncompleted to\n{}\n'.format(pdag, cpdag))
    assert cpdag == ex_pdag.abc4()  # A-B-C PDAG


# ---- Validate against small internal DAGs and bnlearn

# Cancer Standard DAG
@requires_r_and_bnlearn
def test_graph_complete_pdag_cancer_ok1():
    dag = (BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nCANCER completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == ex_pdag.cancer1()
    assert cpdag == bnlearn_cpdag(dag)


# Cancer PDAG with collider only
@requires_r_and_bnlearn
def test_graph_complete_pdag_cancer_ok2():
    pdag = ex_pdag.cancer2()
    cpdag = PDAG.toCPDAG(pdag)
    print('\n{}\nCANCER (collider-only) completed to\n{}\n'
          .format(pdag, cpdag))
    assert cpdag == ex_pdag.cancer1()
    assert cpdag == bnlearn_cpdag(pdag)


# Asia Standard DAG
@requires_r_and_bnlearn
def test_graph_complete_pdag_asia_ok1():
    dag = (BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nASIA completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == ex_pdag.asia()
    assert cpdag == bnlearn_cpdag(dag)


# Sports Standard DAG
@requires_r_and_bnlearn
def test_graph_complete_pdag_sports_ok():
    dag = (BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nSports completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


# --- Validate medium DAGs against bnlearn

# CHILD (n=20) Standard DAG
@requires_r_and_bnlearn
def test_graph_complete_pdag_child_ok1():
    dag = (BN.read(TESTDATA_DIR + '/discrete/medium/child.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nCHILD completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


# INSURANCE (n=27) Standard DAG
@requires_r_and_bnlearn
def test_graph_complete_pdag_insurance_ok1():
    dag = (BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nINSURANCE completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


# WATER (n=32) Standard DAG
@requires_r_and_bnlearn
def test_graph_complete_pdag_water_ok1():
    dag = (BN.read(TESTDATA_DIR + '/discrete/medium/water.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nWATER completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


# ALARM (n=37) Standard DAG
@requires_r_and_bnlearn
def test_graph_complete_pdag_alarm_ok1():
    dag = (BN.read(TESTDATA_DIR + '/discrete/medium/alarm.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nALARM completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


# --- Validate large DAGs against bnlearn

# HAILFINDER (n=56) Std DAG
@requires_r_and_bnlearn
def test_graph_complete_pdag_hailfinder_ok1():
    dag = (BN.read(TESTDATA_DIR + '/discrete/large/hailfinder.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nHAILFINDER completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


# HEPAR2 (n=70) Std DAG
@requires_r_and_bnlearn
def test_graph_complete_pdag_hepar2_ok1():
    dag = (BN.read(TESTDATA_DIR + '/discrete/large/hepar2.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nHAILFINDER completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


# WIN95PTS (n=76) Std DAG
@requires_r_and_bnlearn
def test_graph_complete_pdag_win95pts_ok1():
    dag = (BN.read(TESTDATA_DIR + '/discrete/large/win95pts.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nWIN95PTS completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)


# --- Validate very large DAGs against bnlearn

# PATHFINDER (n=109) Std DAG
@requires_r_and_bnlearn
def test_graph_complete_pdag_pathfinder_ok1():
    dag = (BN.read(TESTDATA_DIR + '/discrete/verylarge/pathfinder.dsc')).dag
    cpdag = PDAG.toCPDAG(dag)
    print('\n{}\nPATHFINDER completed to\n{}\n'.format(dag, cpdag))
    assert cpdag == bnlearn_cpdag(dag)
