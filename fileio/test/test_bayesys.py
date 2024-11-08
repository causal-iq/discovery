
#   Test reading and write Bayesys DAG specification files

import pytest
from random import random
from os import remove

from fileio.bayesys import read, write, read_constraints
from fileio.common import FileFormatError, TESTDATA_DIR
import testdata.example_dags as ex_dag
import testdata.example_pdags as ex_pdag
from core.bn import BN
from learn.knowledge import Knowledge
from learn.knowledge_rule import Rule


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.csv'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


def test_bayesys_read_typeerror1():  # fail on no arguments
    with pytest.raises(TypeError):
        read()


def test_bayesys_read_typeerror2():  # fail on bad argument types
    with pytest.raises(TypeError):
        read(1)
    with pytest.raises(TypeError):
        read(0.7)
    with pytest.raises(TypeError):
        read(False)


def test_bayesys_read_filenotfounderror():  # fail on nonexistent file
    with pytest.raises(FileNotFoundError):
        read('doesnotexist.txt')


def test_binaryfile():
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/misc/null.sys')


def test_emptyfile():
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/misc/empty.txt')


def test_missingcolumns():
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/bayesys/missing_columns.csv')


def test_badheader1():
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/bayesys/bad_header1.csv')


def test_badheader2():
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/bayesys/bad_header2.csv')


# Known bad headers should raise exception  with default strict=True

def test_badheader_known1_a():
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/bayesys/bad_header_known1.csv')


def test_badheader_known2():
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/bayesys/bad_header_known2.csv')


# Known bad headers should not raise exception with strict=False

def test_badheader_known1_notstrict():
    graph = read(TESTDATA_DIR + '/bayesys/bad_header_known1.csv',
                                strict=False)
    ex_dag.asia(graph)


def test_badheader_known2_notstrict():
    graph = read(TESTDATA_DIR + '/bayesys/bad_header_known2.csv',
                                strict=False)
    ex_dag.asia(graph)


# These perfect DAG files should not raise any exceptions

def test_asia_ok():
    graph = read(TESTDATA_DIR + '/asia/asia.csv')
    ex_dag.asia(graph)


def test_asia_all_nodes_ok():
    all_nodes = ['asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub',
                 'xray']
    graph = read(TESTDATA_DIR + '/asia/asia.csv', all_nodes)
    ex_dag.asia(graph)


def test_asia_quoted_ok():
    graph = read(TESTDATA_DIR + '/asia/asia_quoted.csv')
    ex_dag.asia(graph)


def test_triple_chain_ok():
    graph = read(TESTDATA_DIR + '/bayesys/abc.csv')
    ex_dag.abc(graph)


def test_triple_chain_all_nodes_ok():
    graph = read(TESTDATA_DIR + '/bayesys/abc.csv',
                                all_nodes=['A', 'B', 'C'])
    ex_dag.abc(graph)


def test_bayesys_read_alarm_ok():
    dag = read(TESTDATA_DIR + '/bayesys/alarm.csv')
    assert dag == BN.read(TESTDATA_DIR + '/discrete/medium/alarm.dsc').dag


def test_bayesys_read_diarrhoea_ok():
    dag = read(TESTDATA_DIR + '/bayesys/diarrhoea.csv')
    assert dag == BN.read(TESTDATA_DIR + '/discrete/medium/diarrhoea.dsc').dag


# These perfect PDAG files should not raise any exceptions

def test_bayesys_read_pdag_ab3_ok():  # A - B
    pdag = read(TESTDATA_DIR + '/bayesys/ab_pdag.csv')
    assert pdag == ex_pdag.ab3()


def test_bayesys_read_pdag_abc3_ok():  # A - B -> C
    pdag = read(TESTDATA_DIR + '/bayesys/abc3_pdag.csv')
    assert pdag == ex_pdag.abc3()


def test_bayesys_read_pdag_abc_acyclic4_ok():  # A -- B -- C -- A
    pdag = read(TESTDATA_DIR + '/bayesys/abc_acyclic4_pdag.csv')
    assert pdag == ex_pdag.abc_acyclic4()


def test_bayesys_read_pdag_cancer1_ok():  # True Cancer DAG as PDAG
    pdag = read(TESTDATA_DIR + '/bayesys/cancer1_pdag.csv')
    assert pdag == ex_pdag.cancer1()


def test_bayesys_read_pdag_cancer3_ok():  # Cancer skeleton
    pdag = read(TESTDATA_DIR + '/bayesys/cancer3_pdag.csv')
    assert pdag == ex_pdag.cancer3()


def test_bayesys_read_pdag_asia_ok():  # correct Asia CPDAG
    pdag = read(TESTDATA_DIR + '/bayesys/asia_pdag.csv')
    assert pdag == ex_pdag.asia()


def test_bayesys_read_pdag_complete4_ok():  # 4 node complete graph
    pdag = read(TESTDATA_DIR + '/bayesys/complete4_pdag.csv')
    assert pdag == ex_pdag.complete4()


# File contains nodes not in all_nodes - raise exception

def test_asia_all_nodes_incomplete():
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/asia/asia.csv',
                            all_nodes=['bronc'])


def test_triple_chain_all_nodes_incomplete():
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/bayesys/abc.csv',
                            all_nodes=['C', 'A'])


# Check can add in isolated nodes using all_nodes parameter

def test_triple_chain_add_nodes_ok1():
    graph = read(TESTDATA_DIR + '/bayesys/abc.csv',
                                all_nodes=['A', 'B', 'C', 'D'])
    ex_dag.abc(graph, nodes=['A', 'B', 'C', 'D'], number_components=2,
               compact='[A][B|A][C|B][D]')


def test_triple_chain_add_nodes_ok2():
    graph = read(TESTDATA_DIR + '/bayesys/abc.csv',
                                all_nodes=['A', 'B', 'C', 'D', 'E', 'F'])
    ex_dag.abc(graph, nodes=['A', 'B', 'C', 'D', 'E', 'F'],
               number_components=4, compact='[A][B|A][C|B][D][E][F]')

# Check writing files


def test_bayesys_write_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        write()


def test_bayesys_write_type_error_2():  # bad/missing pdag
    with pytest.raises(TypeError):
        write(10, TESTDATA_DIR + '/nonexistent/ab.csv')
    with pytest.raises(TypeError):
        write(None, TESTDATA_DIR + '/nonexistent/ab.csv')
    with pytest.raises(TypeError):
        write('bad type', TESTDATA_DIR + '/nonexistent/ab.csv')


def test_bayesys_write_type_error_3():  # bad/missing path
    dag = ex_dag.ab()
    with pytest.raises(TypeError):
        write(dag, None)
    with pytest.raises(TypeError):
        write(dag)
    with pytest.raises(TypeError):
        write(dag, 45)


def test_bayesys_write_filenotfounderror():
    dag = ex_dag.ab()
    with pytest.raises(FileNotFoundError):
        write(dag, TESTDATA_DIR + '/nonexistent/ab.csv')


# Check writing some DAGs

def test_bayesys_write_ab_ok_1(tmpfile):  # A -> B
    dag = ex_dag.ab()
    write(dag, tmpfile)
    assert dag == read(tmpfile)


def test_bayesys_write_abc_ok_1(tmpfile):  # A -> B -> C
    dag = ex_dag.abc()
    write(dag, tmpfile)
    assert dag == read(tmpfile)


def test_bayesys_write_ac_bc_ok_1(tmpfile):  # A -> C <- B
    dag = ex_dag.ac_bc()
    write(dag, tmpfile)
    assert dag == read(tmpfile)


def test_bayesys_write_alarm_ok_1(tmpfile):  # Alarm [37 nodes]
    dag = BN.read(TESTDATA_DIR + '/discrete/medium/alarm.dsc').dag
    write(dag, tmpfile)
    assert dag == read(tmpfile)


def test_bayesys_write_diarrhoea_ok_1(tmpfile):  # Diarrhoea [28 nodes]
    dag = BN.read(TESTDATA_DIR + '/discrete/medium/diarrhoea.dsc').dag
    write(dag, tmpfile)
    assert dag == read(tmpfile)


# Check writing some PDAGs

def test_bayesys_write_ab3_ok_1(tmpfile):  # A - B
    pdag = ex_pdag.ab3()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_bayesys_write_abc3_ok_1(tmpfile):  # A - B -> C
    pdag = ex_pdag.abc3()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_bayesys_write_abc_acyclic4_ok_1(tmpfile):  # A -- B -- C -- A
    pdag = ex_pdag.abc_acyclic4()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_bayesys_write_cancer1_ok_1(tmpfile):  # True Cancer DAG as PDAG
    pdag = ex_pdag.cancer1()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_bayesys_write_cancer3_ok_1(tmpfile):  # True Cancer skeleton
    pdag = ex_pdag.cancer3()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_bayesys_write_asia_ok_1(tmpfile):  # True Asia PDAG
    pdag = ex_pdag.asia()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


def test_bayesys_write_complete4_ok_1(tmpfile):  # 4 node complete
    pdag = ex_pdag.complete4()
    write(pdag, tmpfile)
    assert pdag == read(tmpfile)


# Read directed constraints file

C_FILE = TESTDATA_DIR + '/bayesys/constraintsDirected_{}.csv'


def test_read_directed_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        read_constraints()


def test_read_directed_type_error_2():  # bad path type
    with pytest.raises(TypeError):
        read_constraints(1, {'A'})
    with pytest.raises(TypeError):
        read_constraints(['file'], {'A'})


def test_read_directed_type_error_3():  # bad nodes type
    with pytest.raises(TypeError):
        read_constraints(C_FILE.format('ab_1'), ['A', 'B'])
    with pytest.raises(TypeError):
        read_constraints(C_FILE.format('ab_1'), ('A', 'B'))
    with pytest.raises(TypeError):
        read_constraints(C_FILE.format('ab_1'), 'A')


def test_read_directed_file_error_1():  # wrong number of entries on row
    with pytest.raises(FileFormatError):
        read_constraints(C_FILE.format('bad_1'), {'A', 'B'})


def test_read_directed_file_error_2():  # incompatible constraints
    with pytest.raises(FileFormatError):
        read_constraints(C_FILE.format('bad_2'), {'A', 'B'})


def test_read_directed_file_error_3():  # invalid line id
    with pytest.raises(FileFormatError):
        read_constraints(C_FILE.format('bad_3'), {'A', 'B', 'C'})


def test_read_directed_file_error_4():  # bad header
    with pytest.raises(FileFormatError):
        read_constraints(C_FILE.format('bad_4'), {'A', 'B'})


def test_read_directed_file_error_5():  # unknown name pattern
    with pytest.raises(FileFormatError):
        read_constraints(TESTDATA_DIR + '/bayesys/constraintsUnknown_ab_1.csv',
                         {'A', 'B'})


def test_read_directed_file_error_6():  # wrong number of entries on row
    with pytest.raises(FileFormatError):
        read_constraints(C_FILE.format('bad_1'), {'A', 'C'})


def test_read_directed_ab_1_ok():  # A --> B with 1 reqd
    know = read_constraints(C_FILE.format('ab_1'), {'A', 'B'})
    assert isinstance(know, Knowledge)
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '1 required and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {('A', 'B'): (True, True)}


def test_read_directed_sports_7_ok():  # sports with 8 reqd
    dag = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc').dag
    know = read_constraints(C_FILE.format('SPORTS_7'), set(dag.nodes))
    assert isinstance(know, Knowledge)
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '8 required and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == {('ATshots', 'ATshotsOnTarget'): (True, True),
                         ('ATshotsOnTarget', 'ATgoals'): (True, True),
                         ('HTshotOnTarget', 'HTgoals'): (True, True),
                         ('HTgoals', 'HDA'): (True, True),
                         ('HTshots', 'HTshotOnTarget'): (True, True),
                         ('possession', 'ATshots'): (True, True),
                         ('possession', 'HTshots'): (True, True),
                         ('ATgoals', 'HDA'): (True, True)}


def test_read_directed_covid_7_ok():  # covid with 13 required
    dag = BN.read(TESTDATA_DIR + '/discrete/medium/covid.dsc').dag
    know = read_constraints(C_FILE.format('COVID-19_7'), set(dag.nodes))
    assert isinstance(know, Knowledge)
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '13 required and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == \
        {('Face_masks', 'New_infections'): (True, True),
         ('New_infections', 'Hospital_admissions'): (True, True),
         ('New_infections', 'Patients_in_hospital'): (True, True),
         ('Lockdown', 'New_infections'): (True, True),
         ('Lockdown', 'Leisure_activity'): (True, True),
         ('Lockdown', 'Work_and_school_activity'): (True, True),
         ('Lockdown', 'Transportation_activity'): (True, True),
         ('Majority_COVID_19_variant', 'New_infections'): (True, True),
         ('Deaths_with_COVID_on_certificate',
          'Excess_mortality'): (True, True),
         ('Season', 'New_infections'): (True, True),
         ('Second_dose_uptake', 'New_infections'): (True, True),
         ('Second_dose_uptake', 'Hospital_admissions'): (True, True),
         ('Tests_across_all_4_Pillars', 'Positive_tests'): (True, True)}


def test_read_directed_diarrhoea_7_ok():  # diarrhoea with 9 required
    dag = BN.read(TESTDATA_DIR + '/discrete/medium/diarrhoea.dsc').dag

    # Correct spelling of DIA_HadDiarrhoea

    nodes = (set(dag.nodes) - {'DIA_HadDiahorrea'}) | {'DIA_HadDiarrhoea'}
    know = read_constraints(C_FILE.format('DIARRHOEA_7'), nodes)
    assert isinstance(know, Knowledge)
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '9 required and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == \
        {('BF_BreastfedMonths', 'CHI_Weight4Height'): (True, True),
         ('BF_EarlyBreastfeeding', 'CHI_Weight4Height'): (True, True),
         ('WSH_ImprovedWaterSource', 'DIA_HadDiarrhoea'): (True, True),
         ('ECO_WealthQuintile', 'WSH_ImprovedWaterSource'): (True, True),
         ('MTH_Education', 'FP_ModernMethod'): (True, True),
         ('WSH_ImprovedToilet', 'DIA_HadDiarrhoea'): (True, True),
         ('WSH_SafeStoolDisposal', 'DIA_HadDiarrhoea'): (True, True),
         ('WSH_WashWithAgent', 'DIA_HadDiarrhoea'): (True, True),
         ('WSH_WaterTreated', 'DIA_HadDiarrhoea'): (True, True)}


def test_read_directed_formed_5_ok():  # formed with 7 required
    dag = BN.read(TESTDATA_DIR + '/discrete/large/formed.dsc').dag
    know = read_constraints(C_FILE.format('FORMED_5'), set(dag.nodes))
    assert isinstance(know, Knowledge)
    assert know.rules.rules == [Rule.REQD_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Required arc" with '
                          + '7 required and expertise 1.0')
    assert know.stop == {}
    assert know.reqd == \
        {('Impulsivity', 'Violence'): (True, True),
         ('CriminalFamilyBackground', 'CriminalAttitude'): (True, True),
         ('GangMember', 'CriminalAttitude'): (True, True),
         ('PCLRfacet3', 'Violence'): (True, True),
         ('PCLRfactor1', 'Violence'): (True, True),
         ('PCLRfactor2', 'Violence'): (True, True),
         ('PriorViolentConvictions', 'Violence'): (True, True)}


# Read temporal constraints file

T_FILE = TESTDATA_DIR + '/bayesys/constraintsTemporal_{}.csv'


def test_read_temporal_file_error_1():  # bad header row
    with pytest.raises(FileFormatError):
        read_constraints(T_FILE.format('empty_1'), {'A', 'B'})


def test_read_temporal_file_error_2():  # bad header row
    with pytest.raises(FileFormatError):
        read_constraints(T_FILE.format('bad_hdr_1'), {'A', 'B'})


def test_read_temporal_file_error_3():  # bad header row
    with pytest.raises(FileFormatError):
        read_constraints(T_FILE.format('bad_hdr_2'), {'A', 'B'})


def test_read_temporal_file_error_4():  # bad header row
    with pytest.raises(FileFormatError):
        read_constraints(T_FILE.format('bad_hdr_3'), {'A', 'B'})


def test_read_temporal_file_error_5():  # bad header row
    with pytest.raises(FileFormatError):
        read_constraints(T_FILE.format('bad_hdr_4'), {'A', 'B'})


def test_read_temporal_file_error_6():  # bad row id
    with pytest.raises(FileFormatError):
        read_constraints(T_FILE.format('bad_id_1'), {'A', 'B'})


def test_read_temporal_file_error_7():  # missing tiers
    with pytest.raises(FileFormatError):
        read_constraints(T_FILE.format('miss_tier_1'), {'A', 'B'})


def test_read_temporal_file_error_8():  # bad node
    with pytest.raises(FileFormatError):
        read_constraints(T_FILE.format('ab_1'), {'A', 'C'})


def test_read_temporal_file_error_9():  # node in several tiers
    with pytest.raises(FileFormatError):
        read_constraints(T_FILE.format('dupl_node_1'), {'A', 'B'})


def test_read_temporal_ab_1_ok():  # A --> B
    know = read_constraints(T_FILE.format('ab_1'), {'A', 'B'})
    assert isinstance(know, Knowledge)
    assert know.rules.rules == [Rule.STOP_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '1 prohibited and expertise 1.0')
    assert know.reqd == {}
    assert know.stop == \
        {('B', 'A'): (True, True)}


def test_read_temporal_sports_7_ok():  # Sports
    dag = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc').dag
    know = read_constraints(T_FILE.format('SPORTS_7'), set(dag.nodes))
    assert isinstance(know, Knowledge)
    assert know.rules.rules == [Rule.STOP_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '25 prohibited and expertise 1.0')
    assert know.reqd == {}
    assert know.stop == \
        {('ATshots', 'possession'): (True, True),
         ('HTshots', 'possession'): (True, True),
         ('ATshotsOnTarget', 'possession'): (True, True),
         ('HTshotOnTarget', 'possession'): (True, True),
         ('ATshotsOnTarget', 'ATshots'): (True, True),
         ('HTshotOnTarget', 'ATshots'): (True, True),
         ('ATshotsOnTarget', 'HTshots'): (True, True),
         ('HTshotOnTarget', 'HTshots'): (True, True),
         ('ATgoals', 'possession'): (True, True),
         ('HTgoals', 'possession'): (True, True),
         ('ATgoals', 'ATshots'): (True, True),
         ('HTgoals', 'ATshots'): (True, True),
         ('ATgoals', 'HTshots'): (True, True),
         ('HTgoals', 'HTshots'): (True, True),
         ('ATgoals', 'ATshotsOnTarget'): (True, True),
         ('HTgoals', 'ATshotsOnTarget'): (True, True),
         ('ATgoals', 'HTshotOnTarget'): (True, True),
         ('HTgoals', 'HTshotOnTarget'): (True, True),
         ('HDA', 'possession'): (True, True),
         ('HDA', 'ATshots'): (True, True),
         ('HDA', 'HTshots'): (True, True),
         ('HDA', 'ATshotsOnTarget'): (True, True),
         ('HDA', 'HTshotOnTarget'): (True, True),
         ('HDA', 'ATgoals'): (True, True),
         ('HDA', 'HTgoals'): (True, True)}


def test_read_temporal_covid_7_ok():  # Sports, 3 tiers
    dag = BN.read(TESTDATA_DIR + '/discrete/medium/covid.dsc').dag
    know = read_constraints(T_FILE.format('COVID-19_7'), set(dag.nodes))
    assert isinstance(know, Knowledge)
    assert know.rules.rules == [Rule.STOP_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '68 prohibited and expertise 1.0')
    assert know.reqd == {}
    assert know.stop == \
        {('Transportation_activity', 'Lockdown'): (True, True),
         ('New_infections', 'Lockdown'): (True, True),
         ('Leisure_activity', 'Lockdown'): (True, True),
         ('Work_and_school_activity', 'Lockdown'): (True, True),
         ('Positive_tests', 'Lockdown'): (True, True),
         ('Excess_mortality', 'Lockdown'): (True, True),
         ('Transportation_activity', 'Second_dose_uptake'): (True, True),
         ('New_infections', 'Second_dose_uptake'): (True, True),
         ('Leisure_activity', 'Second_dose_uptake'): (True, True),
         ('Work_and_school_activity', 'Second_dose_uptake'): (True, True),
         ('Positive_tests', 'Second_dose_uptake'): (True, True),
         ('Excess_mortality', 'Second_dose_uptake'): (True, True),
         ('Transportation_activity', 'Season'): (True, True),
         ('New_infections', 'Season'): (True, True),
         ('Leisure_activity', 'Season'): (True, True),
         ('Work_and_school_activity', 'Season'): (True, True),
         ('Positive_tests', 'Season'): (True, True),
         ('Excess_mortality', 'Season'): (True, True),
         ('Transportation_activity',
          'Majority_COVID_19_variant'): (True, True),
         ('New_infections', 'Majority_COVID_19_variant'): (True, True),
         ('Leisure_activity', 'Majority_COVID_19_variant'): (True, True),
         ('Work_and_school_activity',
          'Majority_COVID_19_variant'): (True, True),
         ('Positive_tests', 'Majority_COVID_19_variant'): (True, True),
         ('Excess_mortality', 'Majority_COVID_19_variant'): (True, True),
         ('Transportation_activity', 'Face_masks'): (True, True),
         ('New_infections', 'Face_masks'): (True, True),
         ('Leisure_activity', 'Face_masks'): (True, True),
         ('Work_and_school_activity', 'Face_masks'): (True, True),
         ('Positive_tests', 'Face_masks'): (True, True),
         ('Excess_mortality', 'Face_masks'): (True, True),
         ('Transportation_activity',
          'Tests_across_all_4_Pillars'): (True, True),
         ('New_infections', 'Tests_across_all_4_Pillars'): (True, True),
         ('Leisure_activity', 'Tests_across_all_4_Pillars'): (True, True),
         ('Work_and_school_activity',
          'Tests_across_all_4_Pillars'): (True, True),
         ('Positive_tests', 'Tests_across_all_4_Pillars'): (True, True),
         ('Excess_mortality', 'Tests_across_all_4_Pillars'): (True, True),
         ('Transportation_activity',
          'Deaths_with_COVID_on_certificate'): (True, True),
         ('New_infections', 'Deaths_with_COVID_on_certificate'): (True, True),
         ('Leisure_activity',
          'Deaths_with_COVID_on_certificate'): (True, True),
         ('Work_and_school_activity',
          'Deaths_with_COVID_on_certificate'): (True, True),
         ('Positive_tests', 'Deaths_with_COVID_on_certificate'): (True, True),
         ('Excess_mortality',
          'Deaths_with_COVID_on_certificate'): (True, True),
         ('Hospital_admissions', 'Lockdown'): (True, True),
         ('Patients_in_hospital', 'Lockdown'): (True, True),
         ('Hospital_admissions', 'Second_dose_uptake'): (True, True),
         ('Patients_in_hospital', 'Second_dose_uptake'): (True, True),
         ('Hospital_admissions', 'Season'): (True, True),
         ('Patients_in_hospital', 'Season'): (True, True),
         ('Hospital_admissions', 'Majority_COVID_19_variant'): (True, True),
         ('Patients_in_hospital', 'Majority_COVID_19_variant'): (True, True),
         ('Hospital_admissions', 'Face_masks'): (True, True),
         ('Patients_in_hospital', 'Face_masks'): (True, True),
         ('Hospital_admissions', 'Tests_across_all_4_Pillars'): (True, True),
         ('Patients_in_hospital', 'Tests_across_all_4_Pillars'): (True, True),
         ('Hospital_admissions',
          'Deaths_with_COVID_on_certificate'): (True, True),
         ('Patients_in_hospital',
          'Deaths_with_COVID_on_certificate'): (True, True),
         ('Hospital_admissions', 'Transportation_activity'): (True, True),
         ('Patients_in_hospital', 'Transportation_activity'): (True, True),
         ('Hospital_admissions', 'New_infections'): (True, True),
         ('Patients_in_hospital', 'New_infections'): (True, True),
         ('Hospital_admissions', 'Leisure_activity'): (True, True),
         ('Patients_in_hospital', 'Leisure_activity'): (True, True),
         ('Hospital_admissions', 'Work_and_school_activity'): (True, True),
         ('Patients_in_hospital', 'Work_and_school_activity'): (True, True),
         ('Hospital_admissions', 'Positive_tests'): (True, True),
         ('Patients_in_hospital', 'Positive_tests'): (True, True),
         ('Hospital_admissions', 'Excess_mortality'): (True, True),
         ('Patients_in_hospital', 'Excess_mortality'): (True, True)}


def test_read_temporal_diarrhoea_7_ok():  # Diarrhoea, 3 tiers
    dag = BN.read(TESTDATA_DIR + '/discrete/medium/diarrhoea.dsc').dag
    nodes = (set(dag.nodes) - {'DIA_HadDiahorrea'}) | {'DIA_HadDiarrhoea'}
    know = read_constraints(T_FILE.format('DIARRHOEA_7'), nodes)
    assert isinstance(know, Knowledge)
    assert know.rules.rules == [Rule.STOP_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '35 prohibited and expertise 1.0')
    assert know.reqd == {}
    assert know.stop == \
        {('WSH_ImprovedWaterSource', 'WSH_ImprovedToilet'): (True, True),
         ('CHI_Weight4Height', 'WSH_ImprovedToilet'): (True, True),
         ('FP_ModernMethod', 'WSH_ImprovedToilet'): (True, True),
         ('WSH_ImprovedWaterSource', 'ECO_WealthQuintile'): (True, True),
         ('CHI_Weight4Height', 'ECO_WealthQuintile'): (True, True),
         ('FP_ModernMethod', 'ECO_WealthQuintile'): (True, True),
         ('WSH_ImprovedWaterSource', 'WSH_WashWithAgent'): (True, True),
         ('CHI_Weight4Height', 'WSH_WashWithAgent'): (True, True),
         ('FP_ModernMethod', 'WSH_WashWithAgent'): (True, True),
         ('WSH_ImprovedWaterSource', 'WSH_SafeStoolDisposal'): (True, True),
         ('CHI_Weight4Height', 'WSH_SafeStoolDisposal'): (True, True),
         ('FP_ModernMethod', 'WSH_SafeStoolDisposal'): (True, True),
         ('WSH_ImprovedWaterSource', 'WSH_WaterTreated'): (True, True),
         ('CHI_Weight4Height', 'WSH_WaterTreated'): (True, True),
         ('FP_ModernMethod', 'WSH_WaterTreated'): (True, True),
         ('WSH_ImprovedWaterSource', 'BF_BreastfedMonths'): (True, True),
         ('CHI_Weight4Height', 'BF_BreastfedMonths'): (True, True),
         ('FP_ModernMethod', 'BF_BreastfedMonths'): (True, True),
         ('WSH_ImprovedWaterSource', 'BF_EarlyBreastfeeding'): (True, True),
         ('CHI_Weight4Height', 'BF_EarlyBreastfeeding'): (True, True),
         ('FP_ModernMethod', 'BF_EarlyBreastfeeding'): (True, True),
         ('WSH_ImprovedWaterSource', 'MTH_Education'): (True, True),
         ('CHI_Weight4Height', 'MTH_Education'): (True, True),
         ('FP_ModernMethod', 'MTH_Education'): (True, True),
         ('DIA_HadDiarrhoea', 'WSH_ImprovedToilet'): (True, True),
         ('DIA_HadDiarrhoea', 'ECO_WealthQuintile'): (True, True),
         ('DIA_HadDiarrhoea', 'WSH_WashWithAgent'): (True, True),
         ('DIA_HadDiarrhoea', 'WSH_SafeStoolDisposal'): (True, True),
         ('DIA_HadDiarrhoea', 'WSH_WaterTreated'): (True, True),
         ('DIA_HadDiarrhoea', 'BF_BreastfedMonths'): (True, True),
         ('DIA_HadDiarrhoea', 'BF_EarlyBreastfeeding'): (True, True),
         ('DIA_HadDiarrhoea', 'MTH_Education'): (True, True),
         ('DIA_HadDiarrhoea', 'WSH_ImprovedWaterSource'): (True, True),
         ('DIA_HadDiarrhoea', 'CHI_Weight4Height'): (True, True),
         ('DIA_HadDiarrhoea', 'FP_ModernMethod'): (True, True)}


def test_read_temporal_formed_5_ok():  # Formed, 2 tiers
    dag = BN.read(TESTDATA_DIR + '/discrete/large/formed.dsc').dag
    know = read_constraints(T_FILE.format('FORMED_5'), set(dag.nodes))
    assert isinstance(know, Knowledge)
    assert know.rules.rules == [Rule.STOP_ARC]
    assert know.ref is None
    assert know.limit is False
    assert know.ignore == 0
    assert know.expertise == 1.0
    assert know.count == 0
    assert know.label == ('Ruleset "Prohibited arc" with '
                          + '14 prohibited and expertise 1.0')
    assert know.reqd == {}
    assert know.stop == \
        {('CriminalAttitude', 'Impulsivity'): (True, True),
         ('Violence', 'Impulsivity'): (True, True),
         ('CriminalAttitude', 'PCLRfacet3'): (True, True),
         ('Violence', 'PCLRfacet3'): (True, True),
         ('CriminalAttitude', 'PCLRfactor1'): (True, True),
         ('Violence', 'PCLRfactor1'): (True, True),
         ('CriminalAttitude', 'PCLRfactor2'): (True, True),
         ('Violence', 'PCLRfactor2'): (True, True),
         ('CriminalAttitude', 'PriorViolentConvictions'): (True, True),
         ('Violence', 'PriorViolentConvictions'): (True, True),
         ('CriminalAttitude', 'CriminalFamilyBackground'): (True, True),
         ('Violence', 'CriminalFamilyBackground'): (True, True),
         ('CriminalAttitude', 'GangMember'): (True, True),
         ('Violence', 'GangMember'): (True, True)}
