
# Tests calling bnlearn impport function

import pytest
from random import random
from os import remove

from call.r import requires_r_and_bnlearn
from call.bnlearn import bnlearn_import
from fileio.common import TESTDATA_DIR, EXPTS_DIR
from core.bn import BN


# temp file, automatically removed
@pytest.fixture(scope="function")
def tmpfile():
    _tmpfile = TESTDATA_DIR + '/tmp/{}.xdsl'.format(int(random() * 10000000))
    yield _tmpfile
    try:
        remove(_tmpfile)
    except Exception:
        pass


# No argument
def test_bnlearn_type_error_1():
    with pytest.raises(TypeError):
        bnlearn_import()


# Bad arg type
def test_bnlearn_type_error_2():
    with pytest.raises(TypeError):
        bnlearn_import(False)
    with pytest.raises(TypeError):
        bnlearn_import(['invalid'])
    with pytest.raises(TypeError):
        bnlearn_import(37)


# Non-existent rda
def test_bnlearn_filenotfound_error_1():
    with pytest.raises(FileNotFoundError):
        bnlearn_import('nonexistent')


# Invalid RDA file
def test_bnlearn_value_error_1():
    with pytest.raises(ValueError):
        bnlearn_import('not_rda')


# No BN inside RDA
def test_bnlearn_value_error_2():
    with pytest.raises(ValueError):
        bnlearn_import('not_bn')


# Import gauss
@requires_r_and_bnlearn
def test_bnlearn_import_gauss_1_ok():
    bn = bnlearn_import('gauss')
    assert bn.dag.to_string() == '[A][B][C|A:B][D|B][E][F|A:D:E:G][G]'
    assert set(bn.cnds) == {'A', 'B', 'C', 'D', 'E', 'F', 'G'}
    print(type(bn.cnds['A']))
    print(bn.cnds['A'].__str__())
    assert '{}'.format(bn.cnds['A']) == 'Normal(1.0,1.0)'
    assert '{}'.format(bn.cnds['B']) == 'Normal(2.0,3.0)'
    assert '{}'.format(bn.cnds['C']) == '2.0*A+2.0*B+Normal(2.0,0.5)'
    assert '{}'.format(bn.cnds['D']) == '1.5*B+Normal(6.0,0.33)'
    assert '{}'.format(bn.cnds['E']) == 'Normal(3.5,2.0)'
    assert '{}'.format(bn.cnds['F']) == \
        '2.0*A+1.0*D+1.0*E+1.5*G+Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['G']) == 'Normal(5.0,2.0)'
    assert bn.free_params == 21
    print('\n\ngauss BN:\n{}\nNode distributions:'.format(bn.dag))
    for node, cnd in bn.cnds.items():
        print(' {}: {}'.format(node, cnd))

    ref = BN.read(TESTDATA_DIR + '/xdsl/gauss.xdsl')
    assert ref == bn


# Import building
@requires_r_and_bnlearn
def test_bnlearn_import_building_1_ok():
    bn = bnlearn_import('building')
    assert bn.dag.to_string() == \
        ('[X1|X10:X2:X9]' +
         '[X10]' +
         '[X11|X12:X21]' +
         '[X12|X21]' +
         '[X13]' +
         '[X14]' +
         '[X15|X16:X17]' +
         '[X16]' +
         '[X17]' +
         '[X18|X19]' +
         '[X19]' +
         '[X2|X3:X4:X5:X6]' +
         '[X20]' +
         '[X21]' +
         '[X22]' +
         '[X23]' +
         '[X24]' +
         '[X3|X11:X12:X21]' +
         '[X4|X24:X5:X8]' +
         '[X5|X13:X22:X7]' +
         '[X6|X23:X8]' +
         '[X7|X14:X15:X16:X17]' +
         '[X8|X18:X19:X20]' +
         '[X9|X10]')

    assert set(bn.cnds) == \
        {'X12', 'X11', 'X23', 'X1', 'X3', 'X7', 'X24', 'X14', 'X2', 'X13',
         'X16', 'X5', 'X20', 'X17', 'X10', 'X8', 'X4', 'X21', 'X6', 'X19',
         'X22', 'X9', 'X18', 'X15'}

    assert '{}'.format(bn.cnds['X1']) == \
        '0.7*X10+2.0*X2+0.3*X9+Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X10']) == 'Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X11']) == '0.3*X12+0.7*X21+Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X12']) == '0.8*X21+Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X13']) == 'Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X14']) == 'Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X15']) == '0.9*X16+0.1*X17+Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X16']) == 'Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X17']) == 'Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X18']) == '0.5*X19+Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X19']) == 'Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X2']) == \
        '0.7*X3+0.5*X4+0.7*X5+0.3*X6+Normal(0.0,0.01)'
    assert '{}'.format(bn.cnds['X20']) == 'Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X21']) == 'Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X22']) == 'Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X23']) == 'Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X24']) == 'Normal(0.0,1.0)'
    assert '{}'.format(bn.cnds['X3']) == \
        '0.7*X11+0.9*X12+0.5*X21+Normal(0.0,0.01)'
    assert '{}'.format(bn.cnds['X4']) == \
        '0.7*X24+0.3*X5+0.7*X8+Normal(0.0,0.01)'
    assert '{}'.format(bn.cnds['X5']) == \
        '0.7*X13+0.5*X22+0.9*X7+Normal(0.0,0.01)'
    assert '{}'.format(bn.cnds['X6']) == \
        '0.9*X23+0.5*X8+Normal(0.0,0.01)'
    assert '{}'.format(bn.cnds['X7']) == \
        '0.6*X14+0.6*X15+0.4*X16+0.4*X17+Normal(0.0,0.01)'
    assert '{}'.format(bn.cnds['X8']) == \
        '0.5*X18+0.9*X19+0.8*X20+Normal(0.0,0.01)'
    assert '{}'.format(bn.cnds['X9']) == '0.5*X10+Normal(0.0,1.0)'

    # print()
    # for node, cnd in bn.cnds.items():
    #     print("    assert '{{}}'.format(bn.cnds['{}']) == '{}'"
    #           .format(node, cnd))
    # return

    assert bn.free_params == 80

    print('\n\ngauss BN:\n{}\nNode distributions:'.format(bn.dag))
    for node, cnd in bn.cnds.items():
        print(' {}: {}'.format(node, cnd))

    # bn.write(EXPTS_DIR + '/bn/xdsl/building_c.xdsl')

    ref = BN.read(EXPTS_DIR + '/bn/xdsl/building_c.xdsl')
    assert ref == bn


# Import ecoli70 [n=46]
@requires_r_and_bnlearn
def test_bnlearn_import_ecoli70_1_ok():
    bn = bnlearn_import('ecoli70')

    print('\n\necoli70 BN:\n{}\nNode distributions:'.format(bn.dag))
    for node, cnd in bn.cnds.items():
        print(' {}: {}'.format(node, cnd))

    assert bn.dag.to_string() == \
        ('[aceB|icdA][asnA|ygcE][atpD|sucA:ygcE][atpG|sucA][b1191]' +
         '[b1583|lacA:lacZ:yceP][b1963|yheI][cchB|fixC][cspA|cspG][cspG]' +
         '[dnaG|ycgX:yheI][dnaJ|sucA][dnaK|yheI][eutG][fixC|b1191]' +
         '[flgD|sucA][folK|yheI][ftsJ|mopB][gltA|sucA][hupB|cspA:yfiA]' +
         '[ibpB|eutG:yceP][icdA|asnA:ygcE][lacA|asnA:cspG]' +
         '[lacY|asnA:cspG:eutG:lacA][lacZ|asnA:lacA:lacY][lpdA|yedE]' +
         '[mopB|dnaK:lacZ][nmpC|pspA][nuoM|lacY][pspA|cspG:pspB:yedE]' +
         '[pspB|cspG:yedE][sucA|eutG][sucD|sucA][tnaA|b1191:fixC:sucA]' +
         '[yaeM|cspG:lacA:lacZ][yceP|eutG:fixC][ycgX|fixC:yheI][yecO|cspG]' +
         '[yedE|cspG][yfaD|eutG:sucA:yceP][yfiA|cspA][ygbD|fixC]' +
         '[ygcE|b1191:sucA][yhdM|sucA][yheI|atpD:yedE][yjbO|fixC]')

    assert bn.free_params == 162

    assert set(bn.cnds) == \
        {'ibpB', 'yhdM', 'nmpC', 'ycgX', 'pspA', 'eutG', 'b1583', 'nuoM',
         'ygbD', 'sucA', 'atpD', 'lpdA', 'aceB', 'lacA', 'yfiA', 'ygcE',
         'yaeM', 'yfaD', 'sucD', 'dnaJ', 'folK', 'gltA', 'cspA', 'dnaK',
         'b1963', 'lacZ', 'fixC', 'lacY', 'yedE', 'flgD', 'atpG', 'hupB',
         'icdA', 'b1191', 'pspB', 'yheI', 'yceP', 'asnA', 'ftsJ', 'yjbO',
         'cchB', 'yecO', 'tnaA', 'mopB', 'dnaG', 'cspG'}

    assert '{}'.format(bn.cnds['aceB']) == \
        '1.046357576*icdA+Normal(0.1324256926,0.2921168368)'
    assert '{}'.format(bn.cnds['asnA']) == \
        '0.7974540519*ygcE+Normal(0.3494150099,0.3022863615)'
    assert '{}'.format(bn.cnds['atpD']) == \
        ('0.2602935541*sucA-0.7252201444*ygcE+' +
         'Normal(-0.04025016949,0.6427110846)')
    assert '{}'.format(bn.cnds['atpG']) == \
        '0.6179975307*sucA+Normal(-0.8907913884,0.6279110689)'
    assert '{}'.format(bn.cnds['b1191']) == 'Normal(1.272988889,0.7801025725)'
    assert '{}'.format(bn.cnds['b1583']) == \
        ('-0.2456686941*lacA+0.342180487*lacZ+0.2407019249*yceP+' +
         'Normal(1.381954853,1.052861332)')
    assert '{}'.format(bn.cnds['b1963']) == \
        '1.037575346*yheI+Normal(0.9649125245,0.6138791233)'
    assert '{}'.format(bn.cnds['cchB']) == \
        '0.6179875908*fixC+Normal(1.069522719,0.8003899053)'
    assert '{}'.format(bn.cnds['cspA']) == \
        '0.2887115211*cspG+Normal(-0.4264853304,1.239623031)'
    assert '{}'.format(bn.cnds['cspG']) == 'Normal(2.026077778,1.037081083)'
    assert '{}'.format(bn.cnds['dnaG']) == \
        '0.599217928*ycgX+0.1954802383*yheI+Normal(0.1170131737,0.3608202653)'
    assert '{}'.format(bn.cnds['dnaJ']) == \
        '-0.8084874383*sucA+Normal(0.1209770449,0.7585577314)'
    assert '{}'.format(bn.cnds['dnaK']) == \
        '1.0797222*yheI+Normal(-0.2469326394,0.5392408518)'
    assert '{}'.format(bn.cnds['eutG']) == 'Normal(1.2654,0.8313535229)'
    assert '{}'.format(bn.cnds['fixC']) == \
        '0.94060517*b1191+Normal(0.3164756254,1.063448432)'
    assert '{}'.format(bn.cnds['flgD']) == \
        '0.6361944775*sucA+Normal(-0.5166704966,0.6268254939)'
    assert '{}'.format(bn.cnds['folK']) == \
        '0.8180528338*yheI+Normal(0.5640817127,0.3666615418)'
    assert '{}'.format(bn.cnds['ftsJ']) == \
        '0.924118625*mopB+Normal(0.6737550756,0.3954748843)'
    assert '{}'.format(bn.cnds['gltA']) == \
        '0.3789963092*sucA+Normal(-0.9572130204,0.8303467585)'
    assert '{}'.format(bn.cnds['hupB']) == \
        '-0.2984208314*cspA+1.386687107*yfiA+Normal(-0.1820572328,0.340371642)'
    assert '{}'.format(bn.cnds['ibpB']) == \
        '1.447146518*eutG+0.1249285077*yceP+Normal(-0.422681345,0.6789090295)'
    assert '{}'.format(bn.cnds['icdA']) == \
        '0.522839342*asnA-1.058481385*ygcE+Normal(-0.4154791411,0.5638464642)'
    assert '{}'.format(bn.cnds['lacA']) == \
        '0.2723903753*asnA+0.2539059274*cspG+Normal(0.446869937,1.699122854)'
    assert '{}'.format(bn.cnds['lacY']) == \
        ('-0.203973524*asnA-0.2240816794*cspG+0.3536548494*eutG+' +
         '1.046207479*lacA+Normal(-0.1148750724,0.245346463)')
    assert '{}'.format(bn.cnds['lacZ']) == \
        ('-0.01611600816*asnA+1.368384389*lacA-0.4375679069*lacY+' +
         'Normal(0.199960288,0.5838168324)')
    assert '{}'.format(bn.cnds['lpdA']) == \
        '0.9609476896*yedE+Normal(-0.1007488242,0.3740323355)'
    assert '{}'.format(bn.cnds['mopB']) == \
        ('0.8957953701*dnaK-0.05902882849*lacZ+' +
         'Normal(0.03568436734,0.560236969)')
    assert '{}'.format(bn.cnds['nmpC']) == \
        '-0.7690483881*pspA+Normal(0.1688145408,0.5685209954)'
    assert '{}'.format(bn.cnds['nuoM']) == \
        '0.4082614631*lacY+Normal(-2.01756937,0.901857501)'
    assert '{}'.format(bn.cnds['pspA']) == \
        ('0.1079257748*cspG+0.1399003051*pspB-0.7455691487*yedE+' +
         'Normal(-0.1900509488,0.4151537823)')
    assert '{}'.format(bn.cnds['pspB']) == \
        ('0.2758656842*cspG-0.9886108508*yedE+' +
         'Normal(-0.2490256742,0.5201656866)')
    assert '{}'.format(bn.cnds['sucA']) == \
        '-1.089404808*eutG+Normal(0.02428839953,0.8115250209)'
    assert '{}'.format(bn.cnds['sucD']) == \
        '0.6829936121*sucA+Normal(-0.5907152508,0.6600158752)'
    assert '{}'.format(bn.cnds['tnaA']) == \
        ('-0.5926133594*b1191-0.2441912511*fixC+0.1105793275*sucA+'
         'Normal(-0.3861436116,0.3102184655)')
    assert '{}'.format(bn.cnds['yaeM']) == \
        ('1.472182243*cspG-0.7172944901*lacA+0.7203860463*lacZ+' +
         'Normal(0.3665119823,0.8100231312)')
    assert '{}'.format(bn.cnds['yceP']) == \
        '1.140867517*eutG-0.3267433436*fixC+Normal(-0.1280448633,0.4090936492)'
    assert '{}'.format(bn.cnds['ycgX']) == \
        '-0.2716238928*fixC+1.244832482*yheI+Normal(0.1583832094,0.5054621594)'
    assert '{}'.format(bn.cnds['yecO']) == \
        '0.7948663077*cspG+Normal(0.2719168154,0.4742209874)'
    assert '{}'.format(bn.cnds['yedE']) == \
        '-0.6420267279*cspG+Normal(-0.1606039138,0.5264419955)'
    assert '{}'.format(bn.cnds['yfaD']) == \
        ('0.2875756661*eutG-0.243740986*sucA+0.3177723486*yceP+' +
         'Normal(0.162825259,0.4324095859)')
    assert '{}'.format(bn.cnds['yfiA']) == \
        '0.857226184*cspA+Normal(-1.192819554,0.5617982826)'
    assert '{}'.format(bn.cnds['ygbD']) == \
        '0.6606881096*fixC+Normal(1.350435857,0.8602440657)'
    assert '{}'.format(bn.cnds['ygcE']) == \
        '1.881508069*b1191+0.6326534585*sucA+Normal(0.5239507878,0.5856136678)'
    assert '{}'.format(bn.cnds['yhdM']) == \
        '-0.7770235354*sucA+Normal(0.2084979718,0.7669167536)'
    assert '{}'.format(bn.cnds['yheI']) == \
        ('-0.9633228172*atpD+0.3381501748*yedE+'
         'Normal(-0.2137443279,0.3749967742)')
    assert '{}'.format(bn.cnds['yjbO']) == \
        '-0.07064047916*fixC+Normal(1.59101726,1.360589105)'

    print('\n\necoli70 BN:\n{}\nNode distributions:'.format(bn.dag))
    for node, cnd in bn.cnds.items():
        print(' {}: {}'.format(node, cnd))

    # bn.write(EXPTS_DIR + '/bn/xdsl/ecoli70_c.xdsl')

    ref = BN.read(EXPTS_DIR + '/bn/xdsl/ecoli70_c.xdsl')
    assert ref == bn


# Import magic-niab [n=44]
@requires_r_and_bnlearn
def test_bnlearn_import_magic_niab_1_ok(tmpfile):
    bn = bnlearn_import('magic-niab')

    # xdsl will be Genie compliant: '.' ==> '_' in node names

    # bn.write(EXPTS_DIR + '/bn/xdsl/magic-niab_c.xdsl')
    # return

    bn.write(tmpfile)
    bn = BN.read(tmpfile)

    assert bn.dag.to_string() == \
        ('[FT|G1263:G1276:G1294:G1800:G2318:G266:G775]' +
         '[FUS|G1033:G1853:G1896:G383:G832:HT][G1033][G1217][G1263]' +
         '[G1276|G599][G1294|G418][G1338][G1373|G1750][G1750][G1789|G266]' +
         '[G1800|G1217:G257:G2835:G2953][G1853][G1896|G2953][G1945][G200]' +
         '[G2208|G257][G2318][G257|G1217][G2570][G260][G261]' +
         '[G266|G1276:G1338][G2835|G418][G2920][G2953][G311|G1853]' +
         '[G383|G800][G418][G43|G311][G524][G599][G775][G795][G800][G832]' +
         '[G847][G866][G942][HT|G1896:G266:G2835:G2953:G832:G847:G942]' +
         '[MIL|G1217:G1338:G1945:G2208:G524]' +
         '[YLD|FT:G2570:G260:G2920:G832:HT:YR_GLASS]' +
         '[YR_FIELD|FT:G1373:G200:G2208:G257:G261:G418:G599:YR_GLASS]' +
         '[YR_GLASS|G1217:G1750:G311:G418:G795:G800:G866:MIL]')

    assert bn.free_params == 154

    assert set(bn.cnds) == \
        {'G1789', 'G200', 'G775', 'G866', 'G1750', 'G599', 'G1217', 'G1338',
         'YR_FIELD', 'G2208', 'G847', 'G1294', 'G1945', 'G832', 'G418', 'G260',
         'G383', 'MIL', 'G1800', 'G311', 'HT', 'FT', 'G266', 'G1853', 'G257',
         'G800', 'G1033', 'YLD', 'G1276', 'G795', 'G1263', 'G1896', 'G2920',
         'YR_GLASS', 'G2318', 'G2835', 'G1373', 'G2570', 'G524', 'G261',
         'G942', 'FUS', 'G43', 'G2953'}
    assert '{}'.format(bn.cnds['FT']) == \
        ('0.8155095681*G1263-0.4554106874*G1276-0.4364590688*G1294+' +
         '1.610580655*G1800-0.43225645*G2318-1.131290695*G266-' +
         '0.8136710431*G775+Normal(32.04049373,2.82286435)')
    assert '{}'.format(bn.cnds['FUS']) == \
        ('0.1666464963*G1033-0.1626260803*G1853-0.1480134615*G1896+' +
         '0.2565776855*G383-0.1271863492*G832-0.05374466404*HT+' +
         'Normal(8.20617744,0.775598494)')
    assert '{}'.format(bn.cnds['G1033']) == 'Normal(0.3016666667,0.705805089)'
    assert '{}'.format(bn.cnds['G1217']) == 'Normal(1.43,0.8905551531)'
    assert '{}'.format(bn.cnds['G1263']) == 'Normal(1.655,0.7462594535)'
    assert '{}'.format(bn.cnds['G1276']) == \
        '0.3644106454*G599+Normal(0.6351769345,0.9175418915)'
    assert '{}'.format(bn.cnds['G1294']) == \
        '0.332173229*G418+Normal(0.3189198415,0.9167900892)'
    assert '{}'.format(bn.cnds['G1338']) == 'Normal(0.8033333333,0.974459598)'
    assert '{}'.format(bn.cnds['G1373']) == \
        '0.196969697*G1750+Normal(1.314393939,0.8447017701)'
    assert '{}'.format(bn.cnds['G1750']) == 'Normal(0.24,0.6504653529)'
    assert '{}'.format(bn.cnds['G1789']) == \
        '0.2523659306*G266+Normal(0.2126708728,0.6720243115)'
    assert '{}'.format(bn.cnds['G1800']) == \
        ('0.07603258202*G1217+0.07399117945*G257+0.1242971675*G2835+' +
         '0.04372213969*G2953+Normal(-0.1556572323,0.2256248205)')
    assert '{}'.format(bn.cnds['G1853']) == 'Normal(1.75,0.6619897159)'
    assert '{}'.format(bn.cnds['G1896']) == \
        '0.1843797425*G2953+Normal(0.6613889888,0.9331863849)'
    assert '{}'.format(bn.cnds['G1945']) == 'Normal(0.9466666667,0.9876476947)'
    assert '{}'.format(bn.cnds['G200']) == 'Normal(0.2216666667,0.6135889296)'
    assert '{}'.format(bn.cnds['G2208']) == \
        '-0.04067778679*G257+Normal(0.08267212093,0.2559652751)'
    assert '{}'.format(bn.cnds['G2318']) == 'Normal(1.308333333,0.9423785604)'
    assert '{}'.format(bn.cnds['G257']) == \
        '-0.4593525028*G1217+Normal(1.746874079,0.8836115621)'
    assert '{}'.format(bn.cnds['G2570']) == 'Normal(1.796666667,0.5937815393)'
    assert '{}'.format(bn.cnds['G260']) == 'Normal(0.24,0.6504653529)'
    assert '{}'.format(bn.cnds['G261']) == 'Normal(1.56,0.8149684796)'
    assert '{}'.format(bn.cnds['G266']) == \
        ('0.1527166384*G1276+0.0820919137*G1338+' +
         'Normal(0.02086628174,0.6631005897)')
    assert '{}'.format(bn.cnds['G2835']) == \
        '0.04382671159*G418+Normal(0.01253357871,0.2700505696)'
    assert '{}'.format(bn.cnds['G2920']) == 'Normal(1.533333333,0.8366932821)'
    assert '{}'.format(bn.cnds['G2953']) == 'Normal(0.345,0.7304319851)'
    assert '{}'.format(bn.cnds['G311']) == \
        '0.1542857143*G1853+Normal(1.253333333,0.8314720414)'
    assert '{}'.format(bn.cnds['G383']) == \
        '-0.1629041702*G800+Normal(1.908030273,0.5875184646)'
    assert '{}'.format(bn.cnds['G418']) == 'Normal(1.463333333,0.8659707835)'
    assert '{}'.format(bn.cnds['G43']) == \
        '-0.2550872901*G311+Normal(0.5352496386,0.4586321612)'
    assert '{}'.format(bn.cnds['G524']) == 'Normal(0.4666666667,0.8306858334)'
    assert '{}'.format(bn.cnds['G599']) == 'Normal(1.728333333,0.6672875517)'
    assert '{}'.format(bn.cnds['G775']) == 'Normal(0.22,0.6100986537)'
    assert '{}'.format(bn.cnds['G795']) == 'Normal(0.7466666667,0.9401926181)'
    assert '{}'.format(bn.cnds['G800']) == 'Normal(0.745,0.9477017275)'
    assert '{}'.format(bn.cnds['G832']) == 'Normal(1.438333333,0.88175566)'
    assert '{}'.format(bn.cnds['G847']) == 'Normal(1.345,0.9277623105)'
    assert '{}'.format(bn.cnds['G866']) == 'Normal(1.245,0.9520954835)'
    assert '{}'.format(bn.cnds['G942']) == 'Normal(0.2766666667,0.6788890726)'
    assert '{}'.format(bn.cnds['HT']) == \
        ('1.749337351*G1896-0.7826928351*G266-1.396960737*G2835-' +
         '1.348789517*G2953-0.5070748074*G832+0.5766647986*G847-' +
         '0.6650741975*G942+Normal(76.65120719,3.313186192)')
    assert '{}'.format(bn.cnds['MIL']) == \
        ('0.1350137203*G1217+0.1078455052*G1338-0.1105227529*G1945+' +
         '0.4237586447*G2208+0.1127080379*G524+' +
         'Normal(2.046264991,0.6878232508)')
    assert '{}'.format(bn.cnds['YLD']) == \
        ('-0.04877691274*FT+0.1198055057*G2570+0.1085558465*G260+' +
         '0.06713049681*G2920+0.08284356283*G832+0.0211962288*HT+' +
         '0.1408304132*YR_GLASS+Normal(6.534194023,0.4393738719)')
    assert '{}'.format(bn.cnds['YR_FIELD']) == \
        ('-0.02430376508*FT-0.06294320002*G1373-0.1655098291*G200+' +
         '0.2099603729*G2208-0.06991685205*G257+0.08035431776*G261+' +
         '0.08896556334*G418+0.08358132154*G599+0.271335447*YR_GLASS+' +
         'Normal(2.322957082,0.4022332782)')
    assert '{}'.format(bn.cnds['YR_GLASS']) == \
        ('0.07919529826*G1217-0.07482120165*G1750+0.1041528177*G311+' +
         '0.2700418362*G418+0.04738049549*G795+0.05753975132*G800+' +
         '0.0460650342*G866+0.07176009608*MIL+' +
         'Normal(1.548443023,0.332856785)')

    # print('\n\n    assert set(bn.cnds) == \\\n        {}'
    #       .format(set(bn.cnds)))
    # for node, cnd in bn.cnds.items():
    #     cnd = ("    assert '{{}}'.format(bn.cnds['{}']) == '{}'"
    #            .format(node, cnd))
    #     print(cnd if len(cnd) < 80
    #           else cnd.replace('==', '== \\\n       '))

    print('\n\nmagic-niab BN:\n{}\nNode distributions:'.format(bn.dag))
    for node, cnd in bn.cnds.items():
        print(' {}: {}'.format(node, cnd))

    ref = BN.read(EXPTS_DIR + '/bn/xdsl/magic-niab_c.xdsl')
    assert ref == bn


# Import magic-irri [n=64]
@requires_r_and_bnlearn
def test_bnlearn_import_magic_irri_1_ok():
    bn = bnlearn_import('magic-irri')

    # bn.write(EXPTS_DIR + '/bn/xdsl/magic-irri_c.xdsl')

    ref = BN.read(EXPTS_DIR + '/bn/xdsl/magic-irri_c.xdsl')
    assert ref == bn

    print('\n\nmagic-irri BN:\n{}\nNode distributions:'.format(bn.dag))
    for node, cnd in bn.cnds.items():
        print(' {}: {}'.format(node, cnd))


# Import arth150 [n=107]
@requires_r_and_bnlearn
def test_bnlearn_import_arth150_1_ok(tmpfile):
    bn = bnlearn_import('arth150')

    # bn.write(EXPTS_DIR + '/bn/xdsl/arth150_c.xdsl')

    # xdsl will be Genie compliant: node names start with letters

    bn.write(tmpfile)
    bn = BN.read(tmpfile)

    ref = BN.read(EXPTS_DIR + '/bn/xdsl/arth150_c.xdsl')
    assert ref.dag.nodes == bn.dag.nodes

    print('\n\narth150 BN:\n{}\nNode distributions:'.format(bn.dag))
    for node, cnd in bn.cnds.items():
        print(' {}: {}'.format(node, cnd))
