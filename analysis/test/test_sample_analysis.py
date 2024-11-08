# Test functions which anayses sample sizes required to learn edges

import pytest
from pandas import DataFrame

from analysis.bn import SampleAnalysis
from core.bn import BN
from core.metrics import values_same
from fileio.common import TESTDATA_DIR, EXPTS_DIR
from fileio.pandas import Pandas
from fileio.oracle import Oracle


# test function which solves log x / x = c

def test_solve_log_ratiotype_error_1():  # bad arg type
    with pytest.raises(TypeError):
        SampleAnalysis.solve_log_ratio()


def test_solve_log_ratiotype_error_2():  # bad arg type
    with pytest.raises(TypeError):
        SampleAnalysis.solve_log_ratio(3)
    with pytest.raises(TypeError):
        SampleAnalysis.solve_log_ratio(0)
    with pytest.raises(TypeError):
        SampleAnalysis.solve_log_ratio('invalid')
    with pytest.raises(TypeError):
        SampleAnalysis.solve_log_ratio(None)
    with pytest.raises(TypeError):
        SampleAnalysis.solve_log_ratio([1.2])


def test_solve_log_ratiovalue_error_1():  # bad arg value
    with pytest.raises(ValueError):
        SampleAnalysis.solve_log_ratio(-1.0)
    with pytest.raises(ValueError):
        SampleAnalysis.solve_log_ratio(-1.0E-40)
    with pytest.raises(ValueError):
        SampleAnalysis.solve_log_ratio(-0.1)
    with pytest.raises(ValueError):
        SampleAnalysis.solve_log_ratio(1.0)
    with pytest.raises(ValueError):
        SampleAnalysis.solve_log_ratio(2.3)


def test_solve_log_ratio1_ok():  # very high rations return 4.0
    assert round(SampleAnalysis.solve_log_ratio(0.4)) == 4
    assert round(SampleAnalysis.solve_log_ratio(0.8)) == 4
    assert round(SampleAnalysis.solve_log_ratio(0.347)) == 4


def test_solve_log_ratio2_ok():  # very small ratios return 10**12
    assert round(SampleAnalysis.solve_log_ratio(1E-11)) == 1000000000000


def test_solve_log_ratio3_ok():  # should give 4.0 approx
    assert round(SampleAnalysis.solve_log_ratio(0.346)) == 4


def test_solve_log_ratio4_ok():  # should give 8.0 approx
    assert round(SampleAnalysis.solve_log_ratio(0.259930)) == 8


def test_solve_log_ratio5_ok():  # should give 200 approx
    assert round(SampleAnalysis.solve_log_ratio(0.0264915)) == 200


def test_solve_log_ratio6_ok():  # should give 10K approx 10K
    assert round(SampleAnalysis.solve_log_ratio(9.2103E-4)) == 10000


def test_solve_log_ratio7_ok():  # should give  approx 999999
    assert round(SampleAnalysis.solve_log_ratio(1.381552E-5)) == 999999


def test_solve_log_ratio8_ok():  # should give  approx 500M
    assert round(SampleAnalysis.solve_log_ratio(4.0060237E-8)) == 500000004


def test_solve_log_ratio9_ok():  # should give  approx 1G
    assert round(SampleAnalysis.solve_log_ratio(2.0723265E-8)) == 1000000042


def test_solve_log_ratio10_ok():  # should give  approx 10G
    assert round(SampleAnalysis.solve_log_ratio(2.302585E-9)) == 10000000422


def test_solve_log_ratio11_ok():  # should give  approx 100G
    assert round(SampleAnalysis.solve_log_ratio(2.5328436E-10)) == 100000000094


def test_solve_log_ratio12_ok():  # should give  approx 1T
    assert round(SampleAnalysis.solve_log_ratio(2.763103E-11)) == 999999666402


# test SampleAnalysis initialiser

def test_sample_analysis_type_error_1():  # no argument specified
    with pytest.raises(TypeError):
        SampleAnalysis()


def test_sample_analysis_type_error_2():  # bad bn type
    with pytest.raises(TypeError):
        SampleAnalysis(bn=False)
    with pytest.raises(TypeError):
        SampleAnalysis(bn=67)
    with pytest.raises(TypeError):
        SampleAnalysis(bn='should be BN object')
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/a.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(TypeError):
        SampleAnalysis(bn=data, data=data)


def test_sample_analysis_type_error_3():  # bad data type
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/a.dsc')
    with pytest.raises(TypeError):
        SampleAnalysis(bn=bn, data=2)
    with pytest.raises(TypeError):
        SampleAnalysis(bn=bn, data='invalid')
    with pytest.raises(TypeError):
        SampleAnalysis(bn=bn, data=bn)


def test_sample_analysis_type_error_4():  # bad data type
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/a.dsc')
    with pytest.raises(TypeError):
        SampleAnalysis(data=2)
    with pytest.raises(TypeError):
        SampleAnalysis(data='invalid')
    with pytest.raises(TypeError):
        SampleAnalysis(data=bn)


def test_sample_analysis_value_error_1():  # bad arg val (<2 bn nodes)
    with pytest.raises(ValueError):
        bn = BN.read(TESTDATA_DIR + '/discrete/tiny/a.dsc')
        SampleAnalysis(bn)


def test_sample_analysis_value_error_2():  # bad arg val (<2 data columns)
    with pytest.raises(ValueError):
        bn = BN.read(TESTDATA_DIR + '/discrete/tiny/a.dsc')
        data = bn.generate_cases(10)
        SampleAnalysis(data=data)


def test_sample_analysis_value_error_3():  # data and bn have different nodes
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/a.dsc')
    data = bn.generate_cases(10)
    with pytest.raises(ValueError):
        bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
        SampleAnalysis(bn=bn, data=data)


def test_sample_analysis_ab_1_ok():  # just bn supplied
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    analysis = SampleAnalysis(bn=bn)
    assert isinstance(analysis, SampleAnalysis)
    assert isinstance(analysis.bn, Oracle)
    assert analysis.bn.bn == bn
    assert analysis.data is None


def test_sample_analysis_ab_2_ok():  # just data supplied
    data = DataFrame({'A': ['0', '0', '1', '0'],
                      'B': ['1', '1', '0', '0']}, dtype='category')
    analysis = SampleAnalysis(data=data)
    assert isinstance(analysis, SampleAnalysis)
    assert analysis.bn is None
    assert isinstance(analysis.data, Pandas)
    assert (analysis.data.sample == data).all().all()


def test_sample_analysis_ab_3_ok():  # bn and data supplied
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(100)
    analysis = SampleAnalysis(bn=bn, data=data)
    assert isinstance(analysis, SampleAnalysis)
    assert isinstance(analysis.bn, Oracle)
    assert analysis.bn.bn == bn
    assert isinstance(analysis.data, Pandas)
    assert (analysis.data.sample == data).all().all()


def test_node_entropy_ab_1_ok():  # A-->B with no data rows
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    analysis = SampleAnalysis(bn)

    # Marginal prob for A is .75, .25 --> oracle entropy = 0.5623351446
    # Data ratio is 0.6, 0.4 --> data entropy = 0.6730116670

    entd, _, ento, fp = analysis.node_entropy('A', [])
    assert values_same(ento, 0.5623351446, sf=10)
    assert fp == 1
    assert entd is None
    print('\n\nNode A with no parent, entropy: {:.4e}, fp: {}'
          .format(ento, fp))

    # Marginal prob for B is .4375, .5625 --> oracle entropy = 0.5623351446
    # Data ratio is 0.7, 0.3 --> data entropy = 0.6730116670

    entd, _, ento, fp = analysis.node_entropy('B', [])
    assert values_same(ento, 0.6853142073, sf=10)
    assert fp == 1
    assert entd is None
    print('\n\nNode B with no parent, entropy: {:.4e}, fp: {}'
          .format(ento, fp))

    # Conditional prob for B | A is 0.5, 0.5, 0.25, 0.75

    entd, _, ento, fp = analysis.node_entropy('B', ['A'])
    assert values_same(ento, 0.6604441716, sf=10)
    assert fp == 2
    assert entd is None
    print('\n\nNode B with  A parent, entropy: {:.4e}, fp: {}'
          .format(ento, fp))


def test_node_entropy_ab_2_ok():  # A-->B with 10 data rows
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = DataFrame({'A': ['0', '1', '0', '0', '1', '0', '1', '0', '1', '0'],
                      'B': ['1', '1', '0', '0', '1', '1', '1', '1', '0', '1']},
                     dtype='category')
    analysis = SampleAnalysis(bn, data)

    # Marginal prob for A is .75, .25 --> oracle entropy = 0.5623351446
    # Data ratio is 0.6, 0.4 --> data entropy = 0.6730116670

    entd, _, ento, fp = analysis.node_entropy('A', [])
    assert values_same(ento, 0.5623351446, sf=10)
    assert fp == 1
    assert values_same(entd, 0.6730116670, sf=10)
    print('\n\nNode A with no parent, entropy: {:.4e}/{:.4e}, fp: {}'
          .format(ento, entd, fp))

    # Marginal prob for B is .4375, .5625 --> oracle entropy = 0.5623351446
    # Data ratio is 0.7, 0.3 --> data entropy = 0.6730116670

    entd, _, ento, fp = analysis.node_entropy('B', [])
    assert values_same(ento, 0.6853142073, sf=10)
    assert fp == 1
    assert values_same(entd, 0.6108643021, sf=10)
    print('\n\nNode B with no parent, entropy: {:.4e}/{:.4e}, fp: {}'
          .format(ento, entd, fp))

    # Conditional prob for B | A is 0.5, 0.5, 0.25, 0.75

    entd, _, ento, fp = analysis.node_entropy('B', ['A'])
    assert values_same(ento, 0.6604441716, sf=10)
    assert fp == 2
    assert values_same(entd, 0.6068425588, sf=10)
    print('\n\nNode B with  A parent, entropy: {:.4e}/{:.4e}, fp: {}'
          .format(ento, entd, fp))


def test_node_entropy_ab_3_ok():  # A-->B with 10 data rows, no BN
    data = DataFrame({'A': ['0', '1', '0', '0', '1', '0', '1', '0', '1', '0'],
                      'B': ['1', '1', '0', '0', '1', '1', '1', '1', '0', '1']},
                     dtype='category')
    analysis = SampleAnalysis(data=data)

    # Marginal prob for A is .75, .25 --> oracle entropy = 0.5623351446
    # Data ratio is 0.6, 0.4 --> data entropy = 0.6730116670

    entd, _, ento, fp = analysis.node_entropy('A', [])
    assert ento is None
    assert fp == 1
    assert values_same(entd, 0.6730116670, sf=10)
    print('\n\nNode A with no parent, entropy: {}/{:.4e}, fp: {}'
          .format(ento, entd, fp))

    # Marginal prob for B is .4375, .5625 --> oracle entropy = 0.5623351446
    # Data ratio is 0.7, 0.3 --> data entropy = 0.6730116670

    entd, _, ento, fp = analysis.node_entropy('B', [])
    assert ento is None
    assert fp == 1
    assert values_same(entd, 0.6108643021, sf=10)
    print('\n\nNode B with no parent, entropy: {}/{:.4e}, fp: {}'
          .format(ento, entd, fp))

    # Conditional prob for B | A is 0.5, 0.5, 0.25, 0.75

    entd, _, ento, fp = analysis.node_entropy('B', ['A'])
    assert ento is None
    assert fp == 2
    assert values_same(entd, 0.6068425588, sf=10)
    print('\n\nNode B with  A parent, entropy: {}/{:.4e}, fp: {}'
          .format(ento, entd, fp))


def test_node_entropy_ab_4_ok():  # A-->B with 10 data rows, 2 samples
    data = DataFrame({'A': ['0', '1', '0', '0', '1', '0', '1', '0', '1', '0'],
                      'B': ['1', '1', '0', '0', '1', '1', '1', '1', '0', '1']},
                     dtype='category')
    analysis = SampleAnalysis(data=data)

    entd, entsd, ento, fp = analysis.node_entropy(node='A', parents=[],
                                                  samples=(2, 5))
    assert ento is None
    assert fp == 1
    assert values_same(entd, 0.6730116670, sf=10)
    assert values_same(entsd, 0, sf=10)
    print('\n\nNode A with no parent, entropy: {:.4e}, sd {:.4e}, fp: {}'
          .format(entd, entsd, fp))

    entd, entsd, ento, fp = analysis.node_entropy('B', [], samples=(2, 5))
    assert ento is None
    assert fp == 1
    assert values_same(entd, 0.5867070453, sf=10)
    assert values_same(entsd, 0.1220531666, sf=10)
    print('\n\nNode B with no parent, entropy: {:.4e}, sd {:.4e}, fp: {}'
          .format(entd, entsd, fp))

    # Conditional prob for B | A is 0.5, 0.5, 0.25, 0.75

    entd, entsd, ento, fp = analysis.node_entropy('B', ['A'], samples=(2, 5))
    assert ento is None
    assert fp == 2
    assert values_same(entd, 0.3295836866, sf=10)
    assert values_same(entsd, 0.07399846214, sf=10)
    print('\n\nNode B with no parent, entropy: {:.4e}, sd {:.4e}, fp: {}'
          .format(entd, entsd, fp))


def test_node_entropy_ab_5_ok():  # A-->B with 1K data rows
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(1000)
    analysis = SampleAnalysis(bn, data)

    entd, _, ento, fp = analysis.node_entropy('A', [])
    assert values_same(ento, 0.5623351446, sf=10)
    assert fp == 1
    assert values_same(entd, 0.5440645960, sf=10)
    print('\n\nNode A with no parent, entropy: {:.4e}/{:.4e}, fp: {}'
          .format(ento, entd, fp))

    entd, _, ento, fp = analysis.node_entropy('B', [])
    assert values_same(ento, 0.6853142073, sf=10)
    assert fp == 1
    assert values_same(entd, 0.6849326630, sf=10)
    print('\n\nNode B with no parent, entropy: {:.4e}/{:.4e}, fp: {}'
          .format(ento, entd, fp))

    entd, _, ento, fp = analysis.node_entropy('B', ['A'])
    assert values_same(ento, 0.6604441716, sf=10)
    assert fp == 2
    assert values_same(entd, 0.6573558987, sf=10)
    print('\n\nNode B with  A parent, entropy: {:.4e}/{:.4e}, fp: {}'
          .format(ento, entd, fp))


def test_node_entropy_ab_6_ok():  # A-->B with 1K data rows, sampling
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(1000)
    analysis = SampleAnalysis(data=data)

    # Data entropy of A with and without sampling

    entd, entsd, ento, fp = analysis.node_entropy('A', [])
    assert fp == 1
    assert values_same(entd, 0.5440645960, sf=10)
    assert values_same(entsd, 0.0, sf=10)
    print('\n\n[]-->A, 1x1K, entropy: {:.4e}, sd: {:.4e}, fp: {}'
          .format(entd, entsd, fp))

    entd, entsd, ento, fp = analysis.node_entropy('A', [], samples=(10, 100))
    assert fp == 1
    assert values_same(entd, 0.5411030771, sf=10)
    assert values_same(entsd, 0.04181555085, sf=10)
    print('\n\n[]-->A, 10x100, entropy: {:.4e}, sd: {:.4e}, fp: {}'
          .format(entd, entsd, fp))

    entd, entsd, ento, fp = analysis.node_entropy('A', [], samples=(10, 20))
    assert fp == 1
    assert values_same(entd, 0.5616123942, sf=10)
    assert values_same(entsd, 0.07186102785, sf=10)
    print('\n\n[]-->A, 10x20, entropy: {:.4e}, sd: {:.4e}, fp: {}'
          .format(entd, entsd, fp))

    # Data entropy of B with and without sampling

    entd, entsd, ento, fp = analysis.node_entropy('B', [])
    assert ento is None
    assert fp == 1
    assert values_same(entd, 0.6849326630, sf=10)
    assert values_same(entsd, 0.0, sf=10)
    print('\n\n[]-->B, 1x1K, entropy: {:.4e}, sd: {:.4e}, fp: {}'
          .format(entd, entsd, fp))

    entd, entsd, ento, fp = analysis.node_entropy('B', [], samples=(10, 100))
    assert ento is None
    assert fp == 1
    assert values_same(entd, 0.6792664389, sf=10)
    assert values_same(entsd, 0.01366536276, sf=10)
    print('\n\n[]-->B, 10x100, entropy: {:.4e}, sd: {:.4e}, fp: {}'
          .format(entd, entsd, fp))

    entd, entsd, ento, fp = analysis.node_entropy('B', [], samples=(10, 20))
    assert ento is None
    assert fp == 1
    assert values_same(entd, 0.6535097343, sf=10)
    assert values_same(entsd, 0.04461704960, sf=10)
    print('\n\n[]-->B, 10x20, entropy: {:.4e}, sd: {:.4e}, fp: {}'
          .format(entd, entsd, fp))

    # Data entropy of A-->B with and without sampling

    entd, entsd, ento, fp = analysis.node_entropy('B', ['A'])
    assert ento is None
    assert fp == 2
    assert values_same(entd, 0.6573558987, sf=10)
    assert values_same(entsd, 0.0, sf=10)
    print('\n\n[A]-->B, 1x1K, entropy: {:.4e}, sd: {:.4e}, fp: {}'
          .format(entd, entsd, fp))

    entd, entsd, ento, fp = analysis.node_entropy('B', ['A'],
                                                  samples=(10, 100))
    assert ento is None
    assert fp == 2
    assert values_same(entd, 0.6488267848, sf=10)
    assert values_same(entsd, 0.02310117287, sf=10)
    print('\n\n[A]-->B, 10x100, entropy: {:.4e}, sd: {:.4e}, fp: {}'
          .format(entd, entsd, fp))

    entd, entsd, ento, fp = analysis.node_entropy('B', ['A'], samples=(10, 20))
    assert ento is None
    assert fp == 2
    assert values_same(entd, 0.5872461526, sf=10)
    assert values_same(entsd, 0.07741218243, sf=10)
    print('\n\n[A]-->B, 10x20, entropy: {:.4e}, sd: {:.4e}, fp: {}'
          .format(entd, entsd, fp))


def test_node_entropy_cancer_1_ok():  # Cancer with 20 data rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                       dstype='categorical', N=20).sample
    analysis = SampleAnalysis(bn, data)

    # Pollution with no parent

    entd, entsd, ento, fp = analysis.node_entropy('Pollution', [])
    assert values_same(ento, 0.3250829734, sf=10)
    assert fp == 1
    assert values_same(entd, 0.5004024235, sf=10)
    assert values_same(entsd, 0.0, sf=10)
    print('\n\n[]-->P, 1x20, entropy: {:.4e}/{:.4e}, sd {:.4e}, fp: {}'
          .format(ento, entd, entsd, fp))

    entd, entsd, ento, fp = analysis.node_entropy('Pollution', [],
                                                  samples=(2, 10))
    assert values_same(ento, 0.3250829734, sf=10)
    assert fp == 1
    assert values_same(entd, 0.4679736377, sf=10)
    assert values_same(entsd, 0.2020779154, sf=10)
    print('\n\n[]-->P, 2x10, entropy: {:.4e}/{:.4e}, sd {:.4e}, fp: {}'
          .format(ento, entd, entsd, fp))

    # Pollution with Dyspnoea parent, no sampling

    entd, entsd, ento, fp = analysis.node_entropy('Pollution', ['Dyspnoea'])
    assert values_same(ento, 0.3250733019, sf=10)
    assert fp == 2
    assert values_same(entd, 0.2772588722, sf=10)
    assert values_same(entsd, 0.0, sf=10)
    print('\n\n[D]-->P, 1x20, entropy: {:.4e}/{:.4e}, sd {:.4e}, fp: {}'
          .format(ento, entd, entsd, fp))

    # ... and sampling

    entd, entsd, ento, fp = analysis.node_entropy('Pollution', ['Dyspnoea'],
                                                  samples=(2, 10))
    assert values_same(ento, 0.3250733019, sf=10)
    assert fp == 2
    assert values_same(entd, 0.2637300420, sf=10)
    assert values_same(entsd, 0.1029205114, sf=10)
    print('\n\n[D]-->P, 2x10, entropy: {:.4e}/{:.4e}, sd {:.4e}, fp: {}'
          .format(ento, entd, entsd, fp))


def test_node_entropy_cancer_2_ok():  # Cancer with 1K data rows
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                       dstype='categorical', N=1000).sample
    analysis = SampleAnalysis(bn, data)

    # Pollution with no parent

    entd, entsd, ento, fp = analysis.node_entropy('Pollution', [])
    assert values_same(ento, 0.3250829734, sf=10)
    assert fp == 1
    assert values_same(entd, 0.3316250849, sf=10)
    assert values_same(entsd, 0.0, sf=10)
    print('\n\n[]-->P, 1x1K, entropy: {:.4e}/{:.4e}, sd {:.4e}, fp: {}'
          .format(ento, entd, entsd, fp))

    entd, entsd, ento, fp = analysis.node_entropy('Pollution', [],
                                                  samples=(5, 200))
    assert values_same(ento, 0.3250829734, sf=10)
    assert fp == 1
    assert values_same(entd, 0.3308495194, sf=10)
    assert values_same(entsd, 0.02947214397, sf=10)
    print('\n\n[]-->P, 5x200, entropy: {:.4e}/{:.4e}, sd {:.4e}, fp: {}'
          .format(ento, entd, entsd, fp))

    # Pollution with Dyspnoea parent, no sampling

    entd, entsd, ento, fp = analysis.node_entropy('Pollution', ['Dyspnoea'])
    assert values_same(ento, 0.3250733019, sf=10)
    assert fp == 2
    assert values_same(entd, 0.3312007698, sf=10)
    assert values_same(entsd, 0.0, sf=10)
    print('\n\n[D]-->P, 1x1K, entropy: {:.4e}/{:.4e}, sd {:.4e}, fp: {}'
          .format(ento, entd, entsd, fp))

    # ... and sampling

    entd, entsd, ento, fp = analysis.node_entropy('Pollution', ['Dyspnoea'],
                                                  samples=(5, 200))
    assert values_same(ento, 0.3250733019, sf=10)
    assert fp == 2
    assert values_same(entd, 0.3264531802, sf=10)
    assert values_same(entsd, 0.03125368391, sf=10)
    print('\n\n[D]-->P, 5x200, entropy: {:.4e}/{:.4e}, sd {:.4e}, fp: {}'
          .format(ento, entd, entsd, fp))


# Test edge stability method

def test_edge_stability_ab_1_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    ent0 = SampleAnalysis(bn=bn).node_entropy('B', [])
    ent1 = SampleAnalysis(bn=bn).node_entropy('B', ['A'])
    ratio = 2 * (ent0[2] - ent1[2]) / (ent1[3] - ent0[3])
    N = round(SampleAnalysis.solve_log_ratio(ratio))
    print('\n\nOracle entropy delta is {:.4e} requiring {}'
          .format(ent0[2] - ent1[2], N))
    data = bn.generate_cases(100000)
    analysis = SampleAnalysis(data=data)
    print()
    analysis.edge_stability(('B', 'A'))


def test_edge_stability_cancer_1_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    ent0 = SampleAnalysis(bn=bn).node_entropy('Pollution', [])
    ent1 = SampleAnalysis(bn=bn).node_entropy('Pollution', ['Dyspnoea'])
    ratio = 2 * (ent0[2] - ent1[2]) / (ent1[3] - ent0[3])
    N = round(SampleAnalysis.solve_log_ratio(ratio))
    print('\n\nOracle entropy delta is {:.4e} requiring {}'
          .format(ent0[2] - ent1[2], N))
    data = Pandas.read(EXPTS_DIR + '/datasets/cancer.data.gz',
                       dstype='categorical', N=1000000).sample
    analysis = SampleAnalysis(data=data)
    analysis.edge_stability(('Dyspnoea', 'Pollution'))


def test_edge_stability_cancer_2_ok():
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    ent0 = SampleAnalysis(bn=bn).node_entropy('Xray', [])
    ent1 = SampleAnalysis(bn=bn).node_entropy('Xray', ['Cancer'])
    ratio = 2 * (ent0[2] - ent1[2]) / (ent1[3] - ent0[3])
    N = round(SampleAnalysis.solve_log_ratio(ratio))
    print('\n\nOracle entropy delta is {:.4e} requiring {}'
          .format(ent0[2] - ent1[2], N))
    data = Pandas.read(EXPTS_DIR + '/datasets/cancer.data.gz',
                       dstype='categorical', N=1000000).sample
    analysis = SampleAnalysis(data=data)
    analysis.edge_stability(('Cancer', 'Xray'))


# Required sample analysis

def test_cps_reqd_sample_ab_1_ok():  # A-->B, no data
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    analysis = SampleAnalysis(bn)
    assert isinstance(analysis, SampleAnalysis)
    assert isinstance(analysis.bn, Oracle)
    print('\n\nSample analysis of A-->B with no data')
    for node in bn.dag.nodes:
        analysis.cps_reqd_sample(node)


def test_cps_reqd_sample_ab_2_ok():  # A-->B, 10 rows data
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = DataFrame({'A': ['0', '1', '0', '0', '1', '0', '1', '0', '1', '0'],
                      'B': ['1', '1', '0', '0', '1', '1', '1', '1', '0', '1']},
                     dtype='category')
    analysis = SampleAnalysis(bn, data)
    assert isinstance(analysis, SampleAnalysis)
    assert isinstance(analysis.bn, Oracle)
    assert isinstance(analysis.data, Pandas)
    print('\n\nSample analysis of A-->B with 10 data rows')
    for node in bn.dag.nodes:
        analysis.cps_reqd_sample(node)


def test_cps_reqd_sample_ab_3_ok():  # A-->B, 1K rows data
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = bn.generate_cases(1000)
    analysis = SampleAnalysis(bn, data)
    assert isinstance(analysis, SampleAnalysis)
    assert isinstance(analysis.bn, Oracle)
    assert isinstance(analysis.data, Pandas)
    print('\n\nSample analysis of A-->B with 1K data rows')
    for node in bn.dag.nodes:
        analysis.cps_reqd_sample(node)


def test_cps_reqd_sample_ab2_1_ok():  # A-->B (B has 3 values)
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab2.dsc')
    data = bn.generate_cases(1000)
    analysis = SampleAnalysis(bn, data)
    assert isinstance(analysis, SampleAnalysis)
    assert isinstance(analysis.bn, Oracle)
    assert isinstance(analysis.data, Pandas)
    print('\n\nSample analysis of A-->B (2) with 1K data')
    for node in bn.dag.nodes:
        analysis.cps_reqd_sample(node)


def test_cps_reqd_sample_abc_1_ok():  # A-->B-->C
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = bn.generate_cases(1000)
    analysis = SampleAnalysis(bn, data)
    assert isinstance(analysis, SampleAnalysis)
    assert isinstance(analysis.bn, Oracle)
    assert isinstance(analysis.data, Pandas)
    print('\n\nSample analysis of A-->B-->C with 1K data rows')
    for node in bn.dag.nodes:
        analysis.cps_reqd_sample(node)


def test_cps_reqd_sample_cancer_1_ok():  # Cancer 1K rows
    N = 1000
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/cancer.data.gz',
                       dstype='categorical', N=N).sample
    analysis = SampleAnalysis(bn, data)
    assert isinstance(analysis, SampleAnalysis)
    assert isinstance(analysis.bn, Oracle)
    assert isinstance(analysis.data, Pandas)
    print('\n\nSample analysis of Cancer with {} data rows'.format(N))
    for node in bn.dag.nodes:
        analysis.cps_reqd_sample(node)


def test_cps_reqd_sample_sports_1_ok():  # Sports 200 rows
    N = 200
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/sports.data.gz',
                       dstype='categorical', N=N).sample
    analysis = SampleAnalysis(bn, data)
    assert isinstance(analysis, SampleAnalysis)
    assert isinstance(analysis.bn, Oracle)
    assert isinstance(analysis.data, Pandas)
    print('\n\nSample analysis of Sports with {} data rows'.format(N))
    for node in bn.dag.nodes:
        analysis.cps_reqd_sample(node)


def test_cps_reqd_sample_sports_2_ok():  # Sports 1M rows
    N = 10000000
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = Pandas.read(EXPTS_DIR + '/datasets/sports.data.gz',
                       dstype='categorical', N=N).sample
    for N in [10, 100, 1000, 10**4, 10**5, 10**6, 10**7]:
        print('\n\nN is {}'.format(N))
        analysis = SampleAnalysis(bn, data[:N])
        analysis.cps_reqd_sample('HDA', cps=['HTgoals'])
        analysis.cps_reqd_sample('HDA', cps=['RDlevel'])
