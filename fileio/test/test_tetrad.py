
#   Test module to read Tetrad format graph specification files

import pytest
from random import random
from os import remove

from core.common import EdgeType
from fileio.common import TESTDATA_DIR, FileFormatError
from fileio.tetrad import read, write
from fileio.pandas import Pandas
import testdata.example_dags as ex_dag
import testdata.example_pdags as ex_pdag
from core.bn import BN


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.tetrad'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


def test_tetrad_read_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        read()


def test_tetrad_read_type_error_2():  # bad argument types
    with pytest.raises(TypeError):
        read(37)
    with pytest.raises(TypeError):
        read(None)
    with pytest.raises(TypeError):
        read(['bad type'])
    with pytest.raises(TypeError):
        read({'bad': 'type'})
    with pytest.raises(TypeError):
        read(True)


def test_tetrad_read_value_error_1():  # bad file suffix
    with pytest.raises(ValueError):
        read(TESTDATA_DIR + '/tetrad/ab.tetrad.txt')


def test_tetrad_read_value_error_2():  # unknown node in edge
    with pytest.raises(ValueError):
        read(TESTDATA_DIR + '/tetrad/ab_bad_edge_2.tetrad')


def test_tetrad_read_filenotfound_error_1():  # non existent file
    with pytest.raises(FileNotFoundError):
        read(TESTDATA_DIR + '/tetrad/nonexistent.tetrad')


def test_tetrad_read_fileformat_error_1():  # binary file
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/tetrad/null.sys.tetrad')


def test_tetrad_read_fileformat_error_2():  # bad section
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/tetrad/ab_bad_section_1.tetrad')


def test_tetrad_read_fileformat_error_3():  # bad section
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/tetrad/ab_bad_section_2.tetrad')


def test_tetrad_read_fileformat_error_4():  # bad edge
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/tetrad/ab_bad_edge_1.tetrad')


# Read DAGs OK

def test_tetrad_read_a_ok():  # read a OK
    dag = read(TESTDATA_DIR + '/tetrad/a.tetrad')
    ex_dag.a(dag)
    print(dag)


def test_tetrad_read_ab_ok():  # read ab OK
    dag = read(TESTDATA_DIR + '/tetrad/ab.tetrad')
    ex_dag.ab(dag)
    print(dag)


def test_tetrad_read_abc_ok():  # read abc OK
    dag = read(TESTDATA_DIR + '/tetrad/abc.tetrad')
    ex_dag.abc(dag)
    print(dag)


def test_tetrad_read_d7_gold_ok():  # read d7-gold OK
    dag = read(TESTDATA_DIR + '/diarrhoea/d7_gold.tetrad')
    assert len(dag.nodes) == 28
    assert len(dag.edges) == 68
    bn = BN.fit(dag, Pandas.read(TESTDATA_DIR +
                                 '/diarrhoea/IA74-0715-1156-d7a.zip',
                                 dstype='categorical'))
    bn.write(TESTDATA_DIR + '/diarrhoea/diarrhoea.dsc')


def test_tetrad_read_dag_ac_bc_ok():  # A -> C <- B
    dag = read(TESTDATA_DIR + '/tetrad/ac_bc.tetrad')
    assert dag == ex_dag.ac_bc()


def test_tetrad_read_dag_abc_acyclic_ok():  # A -> B -> C <- A
    dag = read(TESTDATA_DIR + '/tetrad/abc_acyclic.tetrad')
    assert dag == ex_dag.abc_acyclic()


def test_tetrad_read_dag_cancer_ok():  # Cancer DAG
    dag = read(TESTDATA_DIR + '/tetrad/cancer.tetrad')
    assert dag == ex_dag.cancer()


def test_tetrad_read_dag_asia_ok():  # Asia DAG
    dag = read(TESTDATA_DIR + '/tetrad/asia.tetrad')
    assert dag == ex_dag.asia()


def test_tetrad_read_dag_and4_17_ok():  # 4->3->1->2, 4->1, 4->2
    dag = read(TESTDATA_DIR + '/tetrad/and4_17.tetrad')
    assert dag == ex_dag.and4_17()


def test_tetrad_read_dag_complete4_ok():  # complete 4-node DAG
    dag = read(TESTDATA_DIR + '/tetrad/complete4.tetrad')
    assert dag == ex_dag.complete4()


# Read PDAGs OK

def test_tetrad_read_pdag_ab3_ok():  # A - B
    pdag = read(TESTDATA_DIR + '/tetrad/ab3_pdag.tetrad')
    assert pdag == ex_pdag.ab3()


def test_tetrad_read_pdag_abc3_ok():  # A - B -> C
    pdag = read(TESTDATA_DIR + '/tetrad/abc3_pdag.tetrad')
    assert pdag == ex_pdag.abc3()


def test_tetrad_read_pdag_abc_acyclic4_ok():  # A -- B -- C -- A
    pdag = read(TESTDATA_DIR + '/tetrad/abc_acyclic4_pdag.tetrad')
    assert pdag == ex_pdag.abc_acyclic4()


def test_tetrad_read_pdag_cancer1_ok():  # True Cancer DAG as PDAG
    pdag = read(TESTDATA_DIR + '/tetrad/cancer1_pdag.tetrad')
    assert pdag == ex_pdag.cancer1()


def test_tetrad_read_pdag_cancer3_ok():  # Cancer skeleton
    pdag = read(TESTDATA_DIR + '/tetrad/cancer3_pdag.tetrad')
    assert pdag == ex_pdag.cancer3()


def test_tetrad_read_pdag_asia_ok():  # Correct Asia CPDAG
    pdag = read(TESTDATA_DIR + '/tetrad/asia_pdag.tetrad')
    assert pdag == ex_pdag.asia()


def test_tetrad_read_pdag_complete4_ok():  # 4 node complete graph
    pdag = read(TESTDATA_DIR + '/tetrad/complete4_pdag.tetrad')
    assert pdag == ex_pdag.complete4()


# Check write file errors

def test_tetrad_write_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        write()


def test_tetrad_write_type_error_2():  # bad/missing pdag
    with pytest.raises(TypeError):
        write(10, TESTDATA_DIR + '/nonexistent/ab.csv')
    with pytest.raises(TypeError):
        write(None, TESTDATA_DIR + '/nonexistent/ab.csv')
    with pytest.raises(TypeError):
        write('bad type', TESTDATA_DIR + '/nonexistent/ab.csv')


def test_tetrad_write_type_error_3():  # bad/missing path
    dag = ex_dag.ab()
    with pytest.raises(TypeError):
        write(dag, None)
    with pytest.raises(TypeError):
        write(dag)
    with pytest.raises(TypeError):
        write(dag, 45)


def test_tetrad_write_filenotfounderror():
    dag = ex_dag.ab()
    with pytest.raises(FileNotFoundError):
        write(dag, TESTDATA_DIR + '/nonexistent/ab.csv')


# Check writing some DAGs

def test_tetrad_write_ab_ok_1(tmpfile):  # A -> B
    dag = ex_dag.ab()
    write(dag, tmpfile)
    assert dag == read(tmpfile)


def test_tetrad_write_abc_ok_1(tmpfile):  # A -> B -> C
    dag = ex_dag.abc()
    write(dag, tmpfile)
    assert dag == read(tmpfile)


def test_tetrad_write_ac_bc_ok_1(tmpfile):  # A -> C <- B
    dag = ex_dag.ac_bc()
    write(dag, tmpfile)
    assert dag == read(tmpfile)


def test_tetrad_write_alarm_ok_1(tmpfile):  # Alarm [37 nodes]
    dag = BN.read(TESTDATA_DIR + '/discrete/medium/alarm.dsc').dag
    write(dag, tmpfile)
    assert dag == read(tmpfile)


def test_tetrad_write_diarrhoea_ok_1(tmpfile):  # Diarrhoea [28 nodes]
    dag = BN.read(TESTDATA_DIR + '/discrete/medium/diarrhoea.dsc').dag
    write(dag, tmpfile)
    assert dag == read(tmpfile)


# Check writing some PDAGs

def test_tetrad_write_ab3_ok_1(tmpfile):  # A - B
    pdag = ex_pdag.ab3()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_tetrad_write_abc3_ok_1(tmpfile):  # A - B -> C
    pdag = ex_pdag.abc3()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_tetrad_write_abc_acyclic4_ok_1(tmpfile):  # A -- B -- C -- A
    pdag = ex_pdag.abc_acyclic4()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_tetrad_write_cancer1_ok_1(tmpfile):  # True Cancer DAG as PDAG
    pdag = ex_pdag.cancer1()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_tetrad_write_cancer3_ok_1(tmpfile):  # True Cancer skeleton
    pdag = ex_pdag.cancer3()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_tetrad_write_asia_ok_1(tmpfile):  # True Asia PDAG
    pdag = ex_pdag.asia()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_tetrad_write_complete4_ok_1(tmpfile):  # 4 node complete
    pdag = ex_pdag.complete4()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


# Test some of the COVID project

def test_tetrad_read_covid_kmeans_fges_p1_ok():
    pdag = read(TESTDATA_DIR + '/tetrad/covid/kmeans_fges_p1.tetrad')
    assert ['Deaths_with_COVID_on_certificate', 'Excess_mortality',
            'Face_masks', 'Hospital_admissions', 'Leisure_activity',
            'Lockdown', 'Majority_COVID_19_variant', 'New_infections',
            'Patients_in_MVBs', 'Patients_in_hospital', 'Positive_tests',
            'Reinfections', 'Season', 'Second_dose_uptake',
            'Tests_across_all_4_Pillars', 'Transportation_activity',
            'Work_and_school_activity'] == pdag.nodes
    assert len(pdag.edges) == 25
    assert pdag.edges[('Lockdown', 'Leisure_activity')] == EdgeType.DIRECTED
    assert (pdag.edges[('Lockdown', 'Patients_in_hospital')]
            == EdgeType.UNDIRECTED)
