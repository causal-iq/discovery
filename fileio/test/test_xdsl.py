
#   Test of xdsl module to read and write DSC format BN definitions

import pytest
from os import remove
from random import random

from fileio.xdsl import read, write
from fileio.common import TESTDATA_DIR, FileFormatError
import fileio.bayesys
from core.bn import BN
import testdata.example_bns as ex_bn


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.xdsl'.format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


def test_xdsl_read_type_error1():  # missing input argument
    with pytest.raises(TypeError):
        read()


def test_xdsl_read_type_error2():  # incorrect argument types
    with pytest.raises(TypeError):
        read(1)
    with pytest.raises(TypeError):
        read(0.7)
    with pytest.raises(TypeError):
        read(False)
    with pytest.raises(TypeError):
        read(TESTDATA_DIR + '/xdsl/ab.xdsl', 37)
    with pytest.raises(TypeError):
        read(TESTDATA_DIR + '/xdsl/ab.xdsl', 'False')


def test_xdsl_read_type_error3():  # fail on non-existent file
    with pytest.raises(FileNotFoundError):
        read('doesnotexist.txt')


def test_xdsl_read_binaryfile():  # fail on binary file
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/misc/null.sys')


def test_xdsl_read_emptyfile():  # fail on empty file
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/misc/empty.txt')


def test_xdsl_read_ab_format_error1():  # fail on non-XML file
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/dsc/ab_bad_network1.dsc')


def test_xdsl_read_ab_format_error2():  # fail on invalid root
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_invalid_root.xdsl')


def test_xdsl_read_ab_format_error3():  # too many top level
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_too_many_top_level.xdsl')


def test_xdsl_read_ab_format_error4():  # no top level
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_no_top_level.xdsl')


def test_xdsl_read_ab_format_error5():  # first top level not nodes
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_not_nodes.xdsl')


def test_xdsl_read_ab_format_error6():  # second top level not extensions
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_not_extensions.xdsl')


def test_xdsl_read_ab_format_error7():  # <nodes> contains invalid child
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_not_cpt.xdsl')


def test_xdsl_read_ab_format_error8():  # <cpt> has no id attribute
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_no_id.xdsl')


def test_xdsl_read_xy_format_error8():  # <equation> has no id attribute
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_no_id.xdsl')


def test_xdsl_read_ab_format_error9():  # <cpt> has no <state>
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_no_state.xdsl')


def test_xdsl_read_ab_format_error10():  # <cpt> has only 1 <state>
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_single_state.xdsl')


def test_xdsl_read_ab_format_error11():  # <cpt> has too many <parents>
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_too_many_parents.xdsl')


def test_xdsl_read_xy_format_error11():  # <equation> has too many <parents>
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_too_many_parents.xdsl')


def test_xdsl_read_ab_format_error12():  # <cpt> has no <probabilities>
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_no_probabilities.xdsl')


def test_xdsl_read_xy_format_error12():  # <equation> has no <definition>
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_no_definition.xdsl')


def test_xdsl_read_ab_format_error13():  # <cpt> has too many <probabilities>
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_too_many_probabilities.xdsl')


def test_xdsl_read_ab_format_error14():  # <state> has no id
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_state_no_id.xdsl')


def test_xdsl_read_ab_format_error15():  # <parents> has no values
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_empty_parents.xdsl')


def test_xdsl_read_xy_format_error15():  # <parents> has no values
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_empty_parents.xdsl')


def test_xdsl_read_ab_format_error16():  # <probabilities> has no values
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_empty_probabilities.xdsl')


def test_xdsl_read_xy_format_error16():  # <definition> has no values
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_empty_definition.xdsl')


def test_xdsl_read_ab_format_error17():  # <probabilities> too many entries
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_too_many_probabilities_1.xdsl')


def test_xdsl_read_ab_format_error18():  # <probabilities> too many entries
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_too_many_probabilities_2.xdsl')


def test_xdsl_read_ab_format_error19():  # <probabilities> too few entries
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_too_few_probabilities_1.xdsl')


def test_xdsl_read_ab_format_error20():  # <probabilities> too few entries
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ab_too_few_probabilities_2.xdsl')


def test_xdsl_read_xy_format_error21():  # definition no =
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_no_equals.xdsl')


def test_xdsl_read_xy_format_error22():  # definition no LHS
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_no_lhs.xdsl')


def test_xdsl_read_xy_format_error23():  # definition no RHS
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_no_rhs.xdsl')


def test_xdsl_read_xy_format_error24():  # definition multiple =
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_multiple_equals.xdsl')


def test_xdsl_read_xy_format_error25():  # definition LHS node mismatch
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_lhs_node_mismatch.xdsl')


def test_xdsl_read_xy_format_error26():  # definition no normal present
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_no_normal.xdsl')


def test_xdsl_read_xy_format_error27():  # definition multiple normal present
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_multiple_normal.xdsl')


def test_xdsl_read_xy_format_error28():  # definition leading plus
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_leading_plus.xdsl')


def test_xdsl_read_xy_format_error29():  # definition trailing plus
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_trailing_plus.xdsl')


def test_xdsl_read_xy_format_error30():  # definition leading minus
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_trailing_minus.xdsl')


def test_xdsl_read_xy_format_error31():  # definition leading minus
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_trailing_minus.xdsl')


def test_xdsl_read_xy_format_error32():  # definition non-numeric in Normal
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_normal_neg_sd.xdsl')


def test_xdsl_read_xy_format_error33():  # definition bad coeff
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_bad_coeff_1.xdsl')


def test_xdsl_read_xy_format_error34():  # definition bad coeff
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_bad_coeff_2.xdsl')


def test_xdsl_read_xy_format_error35():  # definition bad coeff
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_bad_coeff_3.xdsl')


def test_xdsl_read_xy_format_error36():  # definition bad coeff
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_bad_coeff_4.xdsl')


def test_xdsl_read_xy_format_error37():  # definition bad coeff
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/xy_definition_bad_coeff_5.xdsl')


def test_xdsl_read_ax_format_error38():  # mixed network
    with pytest.raises(FileFormatError):
        read(TESTDATA_DIR + '/xdsl/ax_mixed_unsupported.xdsl')


def test_xdsl_read_ab_ok():  # successfully reads ab file
    bn_xdsl = BN.read(TESTDATA_DIR + '/xdsl/ab.xdsl')
    bn_dsc = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    assert bn_xdsl == bn_dsc


def test_xdsl_read_x_ok():  # successfully reads x file
    x = BN.read(TESTDATA_DIR + '/xdsl/x.xdsl')
    assert x.dag.to_string() == '[X]'
    assert x.cnds['X'].__str__() == 'Normal(0.0,1.0)'


def test_xdsl_read_xy_ok():  # successfully reads xy file
    xy = BN.read(TESTDATA_DIR + '/xdsl/xy.xdsl')
    assert xy.dag.to_string() == '[X][Y|X]'
    assert xy.cnds['X'].__str__() == 'Normal(2.0,1.0)'
    assert xy.cnds['Y'].__str__() == '1.5*X+Normal(0.5,0.5)'


def test_xdsl_read_xyz_ok():  # successfully reads xyz file
    xyz = BN.read(TESTDATA_DIR + '/xdsl/xyz.xdsl')
    assert xyz.dag.to_string() == '[X][Y|X][Z|Y]'
    assert xyz.cnds['X'].__str__() == 'Normal(0.0,1.0)'
    assert xyz.cnds['Y'].__str__() == '1.5*X+Normal(0.5,0.5)'
    assert xyz.cnds['Z'].__str__() == '-2.0*Y+Normal(-2.0,0.2)'


def test_xdsl_read_xy_zy_ok():  # successfully reads xy_zy file
    xy_zy = BN.read(TESTDATA_DIR + '/xdsl/xy_zy.xdsl')
    assert xy_zy.dag.to_string() == '[X][Y|X:Z][Z]'
    assert xy_zy.cnds['X'].__str__() == 'Normal(0.0,1.0)'
    assert xy_zy.cnds['Y'].__str__() == '1.5*X-2.2*Z+Normal(0.5,0.5)'
    assert xy_zy.cnds['Z'].__str__() == 'Normal(-2.0,0.2)'


def test_xdsl_read_gauss_ok():  # successfully reads gauss file
    gauss = BN.read(TESTDATA_DIR + '/xdsl/gauss.xdsl')
    assert gauss.dag.to_string() == '[A][B][C|A:B][D|B][E][F|A:D:E:G][G]'
    assert gauss.cnds['A'].__str__() == 'Normal(1.0,1.0)'
    assert gauss.cnds['B'].__str__() == 'Normal(2.0,3.0)'
    assert gauss.cnds['C'].__str__() == '2.0*A+2.0*B+Normal(2.0,0.5)'
    assert gauss.cnds['D'].__str__() == '1.5*B+Normal(6.0,0.33)'
    assert gauss.cnds['E'].__str__() == 'Normal(3.5,2.0)'
    assert gauss.cnds['F'].__str__() == \
        '2.0*A+1.0*D+1.0*E+1.5*G+Normal(0.0,1.0)'
    assert gauss.cnds['G'].__str__() == 'Normal(5.0,2.0)'


def test_xdsl_read_a_b_c_ok():  # successfully reads a_b_c file
    bn_xdsl = BN.read(TESTDATA_DIR + '/xdsl/a_b_c.xdsl')
    bn_dsc = BN.read(TESTDATA_DIR + '/discrete/tiny/a_b_c.dsc')
    assert bn_xdsl == bn_dsc


def test_xdsl_read_abc_ok():  # successfully reads abc file
    bn_xdsl = BN.read(TESTDATA_DIR + '/xdsl/abc.xdsl')
    bn_dsc = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    assert bn_xdsl == bn_dsc


def test_xdsl_read_ab_cb_ok():  # successfully reads ab_cb file
    bn_xdsl = BN.read(TESTDATA_DIR + '/xdsl/ab_cb.xdsl')
    bn_dsc = BN.read(TESTDATA_DIR + '/discrete/tiny/ab_cb.dsc')
    assert bn_xdsl == bn_dsc


def test_xdsl_read_and4_10_ok():  # successfully reads and4_10 file
    bn_xdsl = BN.read(TESTDATA_DIR + '/xdsl/and4_10.xdsl')
    bn_dsc = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    assert bn_xdsl == bn_dsc


def test_xdsl_read_cancer_ok():  # successfully reads cancer file
    bn_xdsl = BN.read(TESTDATA_DIR + '/xdsl/cancer.xdsl')
    bn_dsc = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    assert bn_xdsl == bn_dsc


def test_xdsl_read_asia_ok():  # successfully reads asia file
    bn_xdsl = BN.read(TESTDATA_DIR + '/xdsl/asia.xdsl')
    bn_dsc = BN.read(TESTDATA_DIR + '/asia/asia.dsc')
    assert bn_xdsl == bn_dsc


def test_xdsl_read_sports_ok():  # successfully reads sports file
    bn = BN.read(TESTDATA_DIR + '/xdsl/sports.xdsl')
    print('\n{}'.format(bn.dag))
    assert bn.free_params == 1049
    assert ['ATgoals', 'ATshots', 'ATshotsOnTarget', 'HDA', 'HTgoals',
            'HTshotOnTarget', 'HTshots', 'RDlevel',
            'possession'] == bn.dag.nodes
    assert {('RDlevel', 'HTgoals'),
            ('RDlevel', 'HTshotOnTarget'),
            ('RDlevel', 'HTshots'),
            ('RDlevel', 'possession'),
            ('RDlevel', 'ATshots'),
            ('RDlevel', 'ATshotsOnTarget'),
            ('RDlevel', 'ATgoals'),
            ('possession', 'HTshots'),
            ('possession', 'ATshots'),
            ('HTshots', 'HTshotOnTarget'),
            ('ATshots', 'ATshotsOnTarget'),
            ('HTshotOnTarget', 'HTgoals'),
            ('ATshotsOnTarget', 'ATgoals'),
            ('HTgoals', 'HDA'),
            ('ATgoals', 'HDA')} == set(bn.dag.edges)
    assert bn.cnds['HDA'].cdist({'HTgoals': 'x_', 'ATgoals': 'x_'}) == \
        {'H': 0.3234421364985163, 'D': 0.5014836795252225,
         'A': 0.1750741839762611}
    bn.write(TESTDATA_DIR + '/discrete/small/sports.dsc')


def test_xdsl_read_heartdisease_ok():  # successfully reads heart file
    bn = BN.read(TESTDATA_DIR + '/xdsl/heartdisease.xdsl')
    assert bn.dag.nodes == \
        ['Angina', 'Atherosclerosis', 'Diet', 'ECG', 'Exercise',
         'Family_History', 'Heart_Attack', 'Heart_Disease',
         'High_blood_pressure', 'High_cholestrol_level',
         'High_triglyceride_levels', 'Low_protein_concentration',
         'Obesity', 'Smoking', 'Stroke']
    assert set(bn.dag.edges) == \
        {('Exercise', 'Obesity'),
         ('High_cholestrol_level', 'Atherosclerosis'),
         ('High_blood_pressure', 'Heart_Disease'),
         ('Heart_Disease', 'Heart_Attack'),
         ('Exercise', 'Atherosclerosis'),
         ('Diet', 'Obesity'),
         ('Heart_Disease', 'ECG'),
         ('Obesity', 'High_blood_pressure'),
         ('Diet', 'High_triglyceride_levels'),
         ('Heart_Disease', 'Stroke'),
         ('Atherosclerosis', 'Heart_Disease'),
         ('Smoking', 'High_blood_pressure'),
         ('Heart_Disease', 'Angina'),
         ('Family_History', 'Heart_Disease'),
         ('Diet', 'High_cholestrol_level'),
         ('Diet', 'Low_protein_concentration'),
         ('Exercise', 'High_blood_pressure'),
         ('Low_protein_concentration', 'Heart_Disease'),
         ('High_triglyceride_levels', 'Atherosclerosis')}
    print('\nHeart Disease DAG:\n{}'.format(bn.dag))

    bn.generate_cases(1000, TESTDATA_DIR + '/tmp/heartdisease.dat')

    fileio.bayesys.write(bn.dag, TESTDATA_DIR + '/tmp/heartdisease.csv')


def test_xdsl_read_property_ok():  # successfully reads property file
    bn = BN.read(TESTDATA_DIR + '/xdsl/property.xdsl')
    print('\n{}'.format(bn.dag))
    assert bn.free_params == 3056
    assert sorted(["propertyManagement", "otherPropertyExpenses",
                   "rentalIncomeLoss", "rentalIncome",
                   "propertyPurchaseValue", "propertyExpensesGrowth",
                   "rentalGrowth", "capitalGrowth",
                   "incomeTax", "interestRate",
                   "borrowing", "otherInterestFees",
                   "actualRentalIncome", "rentalGrossYield",
                   "rentalIncomeT1", "LTV",
                   "stampDutyTaxBand", "stampDutyTax",
                   "capitalGains", "otherPropertyExpensesT1",
                   "interest", "propertyExpenses",
                   "rentalGrossProfit", "rentalNetProfitBeforeInterest",
                   "propertyValueT1", "interestTaxRelief",
                   "netProfit"]) == bn.dag.nodes
    assert {('actualRentalIncome', 'propertyExpenses'),
            ('actualRentalIncome', 'rentalGrossProfit'),
            ('actualRentalIncome', 'rentalGrossYield'),
            ('actualRentalIncome', 'rentalIncomeT1'),
            ('borrowing', 'LTV'),
            ('borrowing', 'interest'),
            ('capitalGains', 'propertyValueT1'),
            ('capitalGrowth', 'capitalGains'),
            ('incomeTax', 'rentalNetProfitBeforeInterest'),
            ('interest', 'interestTaxRelief'),
            ('interest', 'netProfit'),
            ('interestRate', 'interest'),
            ('interestTaxRelief', 'netProfit'),
            ('otherInterestFees', 'interest'),
            ('otherPropertyExpenses', 'otherPropertyExpensesT1'),
            ('otherPropertyExpenses', 'propertyExpenses'),
            ('propertyExpenses', 'rentalGrossProfit'),
            ('propertyExpensesGrowth', 'otherPropertyExpensesT1'),
            ('propertyManagement', 'propertyExpenses'),
            ('propertyPurchaseValue', 'LTV'),
            ('propertyPurchaseValue', 'capitalGains'),
            ('propertyPurchaseValue', 'propertyValueT1'),
            ('propertyPurchaseValue', 'rentalGrossYield'),
            ('propertyPurchaseValue', 'stampDutyTax'),
            ('propertyPurchaseValue', 'stampDutyTaxBand'),
            ('rentalGrossProfit', 'rentalNetProfitBeforeInterest'),
            ('rentalGrowth', 'rentalIncomeT1'),
            ('rentalIncome', 'actualRentalIncome'),
            ('rentalIncomeLoss', 'actualRentalIncome'),
            ('rentalNetProfitBeforeInterest', 'netProfit'),
            ('stampDutyTaxBand', 'stampDutyTax')} == set(bn.dag.edges)
    assert bn.cnds['interest'].cdist({'borrowing': 'biggerThan400k',
                                      'otherInterestFees': 'lessThan1k',
                                      'interestRate': 'x'}) == \
        {'biggerThan25k': 0.947238730250761, 'x0_15k': 0.0,
         'x0_25k': 0.05276126974923902, 'x5_20k': 0.0,
         'lessThan5k': 0.0, 'x_10k': 0.0}
    bn.write(TESTDATA_DIR + '/discrete/medium/property.dsc')


def test_xdsl_read_formed_ok():  # successfully reads formed file
    bn = BN.read(TESTDATA_DIR + '/xdsl/formed.xdsl', correct=True)
    assert len(bn.dag.nodes) == 88
    assert len(bn.dag.edges) == 138
    assert bn.free_params == 910  # Bayesys repo reports 912
    bn.write(TESTDATA_DIR + '/discrete/large/formed.dsc')
    print(bn.dag)


def test_xdsl_read_covid_ok():  # successfully reads covid file
    bn = BN.read(TESTDATA_DIR + '/xdsl/covid_knowledge_k-means.xdsl',
                 correct=True)
    assert len(bn.dag.nodes) == 17
    assert len(bn.dag.edges) == 37
    assert bn.free_params == 7834
    print(bn.dag)


def test_xdsl_write_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        write()


def test_xdsl_write_type_error_2():  # one argument
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    with pytest.raises(TypeError):
        write(bn)
    with pytest.raises(TypeError):
        write(filename=TESTDATA_DIR + '/tmp/ab.xdsl')


def test_xdsl_write_type_error_3():  # bn not BN object
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    with pytest.raises(TypeError):
        write('invalid', TESTDATA_DIR + '/tmp/ab.xdsl')
    with pytest.raises(TypeError):
        write(bn.dag, TESTDATA_DIR + '/tmp/ab.xdsl')


def test_xdsl_write_type_error_4():  # filename not a string
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    with pytest.raises(TypeError):
        write(bn, True)
    with pytest.raises(TypeError):
        write(bn, 17)


def test_xdsl_write_filenotfound_error_1():  # write path non-existent
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    with pytest.raises(FileNotFoundError):
        write(bn, 'nonexistent/ab.xdsl')


# Test writing out categorical networks

def test_xdsl_write_ab_ok_1(tmpfile):  # A -> B BN
    # tmpfile = TESTDATA_DIR + '/tmp/ab.xdsl'
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    write(bn, tmpfile)
    bn_xdsl = BN.read(tmpfile)
    assert bn == bn_xdsl


def test_xdsl_write_ab_ok_2(tmpfile):  # A -> B BN, set genie true
    bn = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')
    write(bn, tmpfile, genie=True)
    bn_xdsl = BN.read(tmpfile)

    # Variable values must start with letters for Genie, so write modified
    # the values - need to check the CPTs explicitly therefore

    assert bn_xdsl.dag == bn.dag
    assert bn_xdsl.cnds['A'].cdist() == {'S0': 0.75, 'S1': 0.25}
    assert bn_xdsl.cnds['B'].cdist({'A': 'S0'}) == {'S0': 0.5, 'S1': 0.5}
    assert bn_xdsl.cnds['B'].cdist({'A': 'S1'}) == {'S0': 0.25, 'S1': 0.75}


def test_xdsl_write_abc_ok_1(tmpfile):  # A -> B -> C BN
    bn = BN.read(TESTDATA_DIR + '/dsc/abc.dsc')
    write(bn, tmpfile)
    bn_xdsl = BN.read(tmpfile)
    assert bn == bn_xdsl


def test_xdsl_write_ab_cb_ok_1(tmpfile):  # A -> B <- C BN
    bn = BN.read(TESTDATA_DIR + '/dsc/ab_cb.dsc')
    write(bn, tmpfile)
    bn_xdsl = BN.read(tmpfile)
    assert bn == bn_xdsl


def test_xdsl_write_and4_10_ok_1(tmpfile):  # and4_10 BN
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/and4_10.dsc')
    write(bn, tmpfile)
    bn_xdsl = BN.read(tmpfile)
    assert bn == bn_xdsl


def test_xdsl_write_cancer_ok_1(tmpfile):  # cancer BN
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    write(bn, tmpfile)
    bn_xdsl = BN.read(tmpfile)
    assert bn == bn_xdsl

    # manually check the CPT entries

    cpt = bn_xdsl.cnds['Pollution']
    assert cpt.cdist() == {'low': 0.9, 'high': 0.1}

    cpt = bn_xdsl.cnds['Smoker']
    assert cpt.cdist() == {'True': 0.3, 'False': 0.7}

    cpt = bn_xdsl.cnds['Cancer']
    assert (cpt.cdist({'Pollution': 'low', 'Smoker': 'True'})
            == {'True': 0.03, 'False': 0.97})
    assert (cpt.cdist({'Pollution': 'high', 'Smoker': 'True'})
            == {'True': 0.05, 'False': 0.95})
    assert (cpt.cdist({'Pollution': 'low', 'Smoker': 'False'})
            == {'True': 0.001, 'False': 0.999})
    assert (cpt.cdist({'Pollution': 'high', 'Smoker': 'False'})
            == {'True': 0.02, 'False': 0.98})

    cpt = bn_xdsl.cnds['Dyspnoea']
    assert cpt.cdist({'Cancer': 'True'}) == {'True': 0.65, 'False': 0.35}
    assert cpt.cdist({'Cancer': 'False'}) == {'True': 0.3, 'False': 0.7}

    cpt = bn_xdsl.cnds['Xray']
    assert cpt.cdist({'Cancer': 'True'}) == {'positive': 0.9, 'negative': 0.1}
    assert cpt.cdist({'Cancer': 'False'}) == {'positive': 0.2, 'negative': 0.8}


def test_xdsl_write_cancer_ok_2(tmpfile):  # cancer BN, but set genie flag true
    # tmpfile = TESTDATA_DIR + '/tmp/cancer.xdsl'
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    write(bn, tmpfile, genie=True)
    bn_xdsl = BN.read(tmpfile)
    assert bn == bn_xdsl

    # manually check the CPT entries

    cpt = bn_xdsl.cnds['Pollution']
    assert cpt.cdist() == {'low': 0.9, 'high': 0.1}

    cpt = bn_xdsl.cnds['Smoker']
    assert cpt.cdist() == {'True': 0.3, 'False': 0.7}

    cpt = bn_xdsl.cnds['Cancer']
    assert (cpt.cdist({'Pollution': 'low', 'Smoker': 'True'})
            == {'True': 0.03, 'False': 0.97})
    assert (cpt.cdist({'Pollution': 'high', 'Smoker': 'True'})
            == {'True': 0.05, 'False': 0.95})
    assert (cpt.cdist({'Pollution': 'low', 'Smoker': 'False'})
            == {'True': 0.001, 'False': 0.999})
    assert (cpt.cdist({'Pollution': 'high', 'Smoker': 'False'})
            == {'True': 0.02, 'False': 0.98})

    cpt = bn_xdsl.cnds['Dyspnoea']
    assert cpt.cdist({'Cancer': 'True'}) == {'True': 0.65, 'False': 0.35}
    assert cpt.cdist({'Cancer': 'False'}) == {'True': 0.3, 'False': 0.7}

    cpt = bn_xdsl.cnds['Xray']
    assert cpt.cdist({'Cancer': 'True'}) == {'positive': 0.9, 'negative': 0.1}
    assert cpt.cdist({'Cancer': 'False'}) == {'positive': 0.2, 'negative': 0.8}


def test_xdsl_write_asia_ok_1(tmpfile):  # asia BN
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    write(bn, tmpfile)
    bn_xdsl = BN.read(tmpfile)
    assert bn == bn_xdsl


def test_xdsl_write_sports_ok_1(tmpfile):  # sports BN
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    write(bn, tmpfile)
    bn_xdsl = BN.read(tmpfile)
    assert bn == bn_xdsl


def test_xdsl_write_diarrhoea_ok_1(tmpfile):  # diarrhoea BN, genie
    # tmpfile = TESTDATA_DIR + '/tmp/diarrhoea.xdsl'
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/diarrhoea.dsc')
    print('\nDiarrhoea dag is:\n{}'.format(bn.dag))
    write(bn, tmpfile, genie=True)
    bn_xdsl = BN.read(tmpfile)
    assert bn.dag == bn_xdsl.dag


def test_xdsl_write_property_ok_1(tmpfile):  # property BN
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/property.dsc')
    write(bn, tmpfile)
    bn_xdsl = BN.read(tmpfile)
    assert bn == bn_xdsl


def test_xdsl_write_formed_ok_1(tmpfile):  # formed BN
    bn = BN.read(TESTDATA_DIR + '/discrete/large/formed.dsc')
    write(bn, tmpfile)
    bn_xdsl = BN.read(tmpfile)
    assert bn == bn_xdsl


def test_xdsl_write_sepsis_ok_1(tmpfile):  # sepsis BN, not genie
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/sepsis.dsc')
    print('\nSepsis dag is:\n{}'.format(bn.dag))
    write(bn, tmpfile)
    bn_xdsl = BN.read(tmpfile)
    assert bn == bn_xdsl

    # manually test some CPTs derived from generated Sepsis XDSL file

    cpt = bn_xdsl.cnds['Elevated.Respiratory.Rate']
    assert (cpt.cdist({'Reduced.oxygen.saturations': '0', 'Tachycardia': '0'})
            == {'0': 0.9997523245, '1': 0.0002476755})
    assert (cpt.cdist({'Reduced.oxygen.saturations': '1', 'Tachycardia': '0'})
            == {'0': 0.998978848, '1': 0.001021152})
    assert (cpt.cdist({'Reduced.oxygen.saturations': '0', 'Tachycardia': '1'})
            == {'0': 0.998598101, '1': 0.001401899})
    assert (cpt.cdist({'Reduced.oxygen.saturations': '1', 'Tachycardia': '1'})
            == {'0': 0.994550409, '1': 0.005449591})

    cpt = bn_xdsl.cnds['Cancer']
    assert (cpt.cdist({'Alcohol': '0', 'Diabetes': '0',
                       'Age_At_Start_of_Spell_SUS': '[1_12.9]', 'Sex': '1',
                       'Der_Diagnosis_Count': '[0_2.4]'})
            == {'0': 0.5, '1': 0.5})
    assert (cpt.cdist({'Alcohol': '1', 'Diabetes': '1',
                       'Age_At_Start_of_Spell_SUS': '_60.5_72.4]', 'Sex': '2',
                       'Der_Diagnosis_Count': '_2.4_4.8]'})
            == {'0': 0.94736842, '1': 0.05263158})


def test_xdsl_write_sepsis_ok_2(tmpfile):  # sepsis BN, genie
    # tmpfile = TESTDATA_DIR + '/tmp/sepsis.xdsl'
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/sepsis.dsc')
    print('\nSepsis dag is:\n{}'.format(bn.dag))
    write(bn, tmpfile, genie=True)
    bn_xdsl = BN.read(tmpfile)

    # manually test some CPTs derived from generated Sepsis XDSL file
    # which has altered node and values to conform to Genie requirements.

    cpt = bn_xdsl.cnds['Elevated_Respiratory_Rate']
    assert (cpt.cdist({'Reduced_oxygen_saturations': 'S0',
                       'Tachycardia': 'S0'})
            == {'S0': 0.9997523245, 'S1': 0.0002476755})
    assert (cpt.cdist({'Reduced_oxygen_saturations': 'S1',
                       'Tachycardia': 'S0'})
            == {'S0': 0.998978848, 'S1': 0.001021152})
    assert (cpt.cdist({'Reduced_oxygen_saturations': 'S0',
                       'Tachycardia': 'S1'})
            == {'S0': 0.998598101, 'S1': 0.001401899})
    assert (cpt.cdist({'Reduced_oxygen_saturations': 'S1',
                       'Tachycardia': 'S1'})
            == {'S0': 0.994550409, 'S1': 0.005449591})

    cpt = bn_xdsl.cnds['Cancer']
    assert (cpt.cdist({'Alcohol': 'S0', 'Diabetes': 'S0',
                       'Age_At_Start_of_Spell_SUS': 'S_1_12_9_', 'Sex': 'S1',
                       'Der_Diagnosis_Count': 'S_0_2_4_'})
            == {'S0': 0.5, 'S1': 0.5})
    assert (cpt.cdist({'Alcohol': 'S1', 'Diabetes': 'S1',
                       'Age_At_Start_of_Spell_SUS': 'S_60_5_72_4_',
                       'Sex': 'S2', 'Der_Diagnosis_Count': 'S_2_4_4_8_'})
            == {'S0': 0.94736842, 'S1': 0.05263158})


# Test writing out continuous networks

def test_xdsl_write_x_ok_1(tmpfile):  # X BN
    # tmpfile = TESTDATA_DIR + '/tmp/x.xdsl'
    bn = ex_bn.x()
    write(bn, tmpfile, True)
    bn_read = BN.read(tmpfile)
    assert bn == bn_read


def test_xdsl_write_xy_ok_1(tmpfile):  # XY BN
    # tmpfile = TESTDATA_DIR + '/tmp/xy.xdsl'
    bn = ex_bn.xy()
    write(bn, tmpfile, True)
    bn_read = BN.read(tmpfile)
    assert bn == bn_read


def test_xdsl_write_x_y_ok_1(tmpfile):  # X Y BN
    # tmpfile = TESTDATA_DIR + '/tmp/x_y.xdsl'
    bn = ex_bn.x_y()
    write(bn, tmpfile, True)
    bn_read = BN.read(tmpfile)
    assert bn == bn_read


def test_xdsl_write_xyz_ok_1(tmpfile):  # X --> Y --> Z BN
    # tmpfile = TESTDATA_DIR + '/tmp/xyx.xdsl'
    bn = ex_bn.xyz()
    write(bn, tmpfile, True)
    bn_read = BN.read(tmpfile)
    assert bn == bn_read


def test_xdsl_write_xy_zy_ok_1(tmpfile):  # X --> Y <-- Z BN
    # tmpfile = TESTDATA_DIR + '/tmp/xy_zy.xdsl'
    bn = ex_bn.xy_zy()
    write(bn, tmpfile, True)
    bn_read = BN.read(tmpfile)
    assert bn == bn_read
