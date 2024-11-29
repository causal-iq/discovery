
#   Test the hc hill-climbing stability feature

import pytest
from pandas import DataFrame

from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from core.bn import BN
from learn.hc import hc, set_stable_order, Stability
from learn.hc_worker import HCWorker, Prefer
from learn.knowledge import Knowledge
from learn.knowledge_rule import RuleSet


@pytest.fixture
def d_params():
    # default parameters that would be set up by hc()
    return {'score': 'bic', 'k': 1, 'stable': Stability.DEC_SCORE,
            'prefer': Prefer.NONE}


# Test checks on "stable" parameter

def test_hc_stable_type_error_1():  # stable param has bad type
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = Pandas(df=bn.generate_cases(10))
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'stable': 2})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'stable': [True]})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'stable': None})
    with pytest.raises(TypeError):
        dag, _ = hc(data, params={'stable': ['dec_score']})


def test_hc_stable_value_error_1():  # Invalid stable value
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = Pandas(df=bn.generate_cases(10))
    with pytest.raises(ValueError):
        dag, _ = hc(data, params={'stable': 'invalid'})
    with pytest.raises(ValueError):
        dag, _ = hc(data, params={'stable': 'invalid'})
    with pytest.raises(ValueError):
        dag, _ = hc(data, params={'stable': 'True'})
    with pytest.raises(ValueError):
        dag, _ = hc(data, params={'stable': 'decscore'})


def test_hc_stable_value_error_2():  # stable & knowledge incompatible
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/ab.dsc')
    data = Pandas(df=bn.generate_cases(10))
    context = {'id': 'test/hc_stable/ab_1', 'in': 'ab_1'}
    know = Knowledge(rules=RuleSet.STOP_ARC,
                     params={'stop': {('A', 'B'): True}})
    with pytest.raises(ValueError):
        dag, _ = hc(data, params={'stable': True}, context=context,
                    knowledge=know)


# Test hc stable_order()

def test_stable_order_ab_1_ok(d_params):  # A->B, same score - col order
    data = Pandas(df=DataFrame({'A': ['0', '1', '1', '1'],
                                'B': ['1', '0', '1', '1']}, dtype='category'))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'B')


def test_stable_order_ab_2_ok(d_params):  # A->B, same score - rev col order
    data = Pandas(df=DataFrame({'A': ['0', '1', '1', '1'],
                                'B': ['1', '0', '1', '1']}, dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('B', 'A')


def test_stable_order_ab_3_ok(d_params):  # A->B, same score - col order
    data = Pandas(df=DataFrame({'A': ['0', '1', '1', '1'],
                                'B': ['1', '0', '1', '1']}, dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'B')


def test_stable_order_ab_4_ok(d_params):  # A->B, same score - col order
    data = Pandas(df=DataFrame({'A': ['0', '1', '1', '1'],
                                'B': ['1', '0', '1', '1']}, dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'B')


def test_stable_order_ab_5_ok(d_params):  # A->B, A higher score so 1st
    data = Pandas(df=DataFrame({'A': ['1', '0', '1', '1'],
                                'B': ['1', '0', '0', '1']}, dtype='category'))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'B')


def test_stable_order_ab_6_ok(d_params):  # A->B, A higher score so 1st
    data = Pandas(df=DataFrame({'B': ['1', '0', '0', '1'],
                                'A': ['1', '0', '1', '1']}, dtype='category'))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'B')


def test_stable_order_ab_7_ok(d_params):  # A->B, A higher score so last
    data = Pandas(df=DataFrame({'A': ['1', '0', '1', '1'],
                                'B': ['1', '0', '0', '1']}, dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('B', 'A')


def test_stable_order_ab_8_ok(d_params):  # A->B, A higher score so last
    data = Pandas(df=DataFrame({'B': ['1', '0', '0', '1'],
                                'A': ['1', '0', '1', '1']}, dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('B', 'A')


def test_stable_order_ab_9_ok(d_params):  # A->B, A higher score so 1st
    data = Pandas(df=DataFrame({'A': ['1', '0', '1', '1'],
                                'B': ['1', '0', '0', '1']}, dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'B')


def test_stable_order_ab_10_ok(d_params):  # A->B, A higher score so 1st
    data = Pandas(df=DataFrame({'B': ['1', '0', '0', '1'],
                                'A': ['1', '0', '1', '1']}, dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'B')


def test_stable_order_abc_1_ok(d_params):  # A->B->C, = scores - col order
    data = Pandas(df=DataFrame({'A': ['0', '1', '1', '1', '0'],
                                'B': ['1', '0', '0', '0', '1'],
                                'C': ['1', '0', '0', '0', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('B', 'C', 'A')


def test_stable_order_abc_2_ok(d_params):  # A->B->C, = scores - rev col order
    data = Pandas(df=DataFrame({'A': ['0', '1', '1', '1', '0'],
                                'B': ['1', '0', '0', '0', '1'],
                                'C': ['1', '0', '0', '0', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'C', 'B')


def test_stable_order_abc_3_ok(d_params):  # A->B->C, = scores - col order
    data = Pandas(df=DataFrame({'A': ['0', '1', '1', '1', '0'],
                                'B': ['1', '0', '0', '0', '1'],
                                'C': ['1', '0', '0', '0', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('B', 'C', 'A')


def test_stable_order_abc_4_ok(d_params):  # A->B->C, = scores - col order
    data = Pandas(df=DataFrame({'A': ['0', '1', '1', '1', '0'],
                                'B': ['1', '0', '0', '0', '1'],
                                'C': ['1', '0', '0', '0', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('B', 'C', 'A')


def test_stable_order_abc_5_ok(d_params):  # A->B->C, B&C equal, so B, C
    data = Pandas(df=DataFrame({'B': ['1', '0', '0', '0', '1'],
                                'A': ['0', '1', '1', '1', '0'],
                                'C': ['1', '0', '0', '0', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('B', 'C', 'A')


def test_stable_order_abc_6_ok(d_params):  # A->B->C, B&C equal, so B, C
    data = Pandas(df=DataFrame({'B': ['1', '0', '0', '0', '1'],
                                'A': ['0', '1', '1', '1', '0'],
                                'C': ['1', '0', '0', '0', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'C', 'B')


def test_stable_order_abc_7_ok(d_params):  # A->B->C, B&C equal, so B, C
    data = Pandas(df=DataFrame({'B': ['1', '0', '0', '0', '1'],
                                'A': ['0', '1', '1', '1', '0'],
                                'C': ['1', '0', '0', '0', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('B', 'C', 'A')


def test_stable_order_abc_8_ok(d_params):  # A->B->C, B&C equal, so B, C
    data = Pandas(df=DataFrame({'B': ['1', '0', '0', '0', '1'],
                                'A': ['0', '1', '1', '1', '0'],
                                'C': ['1', '0', '0', '0', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('B', 'C', 'A')


def test_stable_order_abc_9_ok(d_params):  # A, B, C dec score order
    data = Pandas(df=DataFrame({'C': ['1', '0', '1', '0', '0', '1'],
                                'B': ['0', '1', '1', '0', '0', '0'],
                                'A': ['0', '1', '1', '1', '1', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.DEC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'B', 'C')


def test_stable_order_abc_10_ok(d_params):  # C, B, A inc score order
    data = Pandas(df=DataFrame({'C': ['1', '0', '1', '0', '0', '1'],
                                'B': ['0', '1', '1', '0', '0', '0'],
                                'A': ['0', '1', '1', '1', '1', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('C', 'B', 'A')


def test_stable_order_abc_11_ok(d_params):  # C, B, A best score order
    data = Pandas(df=DataFrame({'C': ['1', '0', '1', '0', '0', '1'],
                                'B': ['0', '1', '1', '0', '0', '0'],
                                'A': ['0', '1', '1', '1', '1', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'B', 'C')


def test_stable_order_abc_12_ok(d_params):  # C, B, A best score+ order
    data = Pandas(df=DataFrame({'C': ['1', '0', '1', '0', '0', '1'],
                                'B': ['0', '1', '1', '0', '0', '0'],
                                'A': ['0', '1', '1', '1', '1', '1']},
                               dtype='category'))
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == ('A', 'B', 'C')


def test_stable_order_cancer_1_ok(d_params):  # Cancer, CDPSX col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(10))
    assert data.get_order() == ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                                'Xray')

    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    # decreasing score order is CXPSD

    assert data.get_order() == \
        ('Cancer', 'Xray', 'Pollution', 'Smoker', 'Dyspnoea')


def test_stable_order_cancer_2_ok(d_params):  # Cancer, CDPSX col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(10))
    assert data.get_order() == ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                                'Xray')
    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)

    # increasing score order is DSPXC

    assert data.get_order() == \
        ('Dyspnoea', 'Smoker', 'Pollution', 'Xray', 'Cancer')


def test_stable_order_cancer_3_ok(d_params):  # Cancer, CDPSX col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(10))
    assert data.get_order() == ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                                'Xray')

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Cancer', 'Xray', 'Pollution', 'Smoker', 'Dyspnoea')


def test_stable_order_cancer_4_ok(d_params):  # Cancer, CDPSX col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(10))
    assert data.get_order() == ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker',
                                'Xray')

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Cancer', 'Pollution', 'Smoker', 'Dyspnoea', 'Xray')


def test_stable_order_cancer_5_ok(d_params):  # Cancer, DXSPC col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    col_order = tuple(['Dyspnoea', 'Xray', 'Smoker', 'Pollution', 'Cancer'])
    data.set_order(col_order)
    assert data.get_order() == col_order

    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_cancer_6_ok(d_params):  # Cancer, DXSPC col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    col_order = tuple(['Dyspnoea', 'Xray', 'Smoker', 'Pollution', 'Cancer'])
    data.set_order(col_order)
    assert data.get_order() == col_order

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Smoker', 'Dyspnoea', 'Xray', 'Pollution', 'Cancer')


def test_stable_order_cancer_7_ok(d_params):  # Cancer, DXSPC col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    col_order = tuple(['Dyspnoea', 'Xray', 'Smoker', 'Pollution', 'Cancer'])
    data.set_order(col_order)
    assert data.get_order() == col_order

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_cancer_8_ok(d_params):  # Cancer, DXSPC col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    col_order = tuple(['Dyspnoea', 'Xray', 'Smoker', 'Pollution', 'Cancer'])
    data.set_order(col_order)
    assert data.get_order() == col_order

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_cancer_9_ok(d_params):  # Cancer, CXPSD col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    col_order = tuple(['Cancer', 'Xray', 'Pollution', 'Smoker', 'Dyspnoea'])
    data.set_order(col_order)
    assert data.get_order() == col_order

    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_cancer_10_ok(d_params):  # Cancer, CXPSD col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    col_order = tuple(['Cancer', 'Xray', 'Pollution', 'Smoker', 'Dyspnoea'])
    data.set_order(col_order)
    assert data.get_order() == col_order

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Smoker', 'Dyspnoea', 'Xray', 'Pollution', 'Cancer')


def test_stable_order_cancer_11_ok(d_params):  # Cancer, CXPSD col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    col_order = tuple(['Cancer', 'Xray', 'Pollution', 'Smoker', 'Dyspnoea'])
    data.set_order(col_order)
    assert data.get_order() == col_order

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_cancer_12_ok(d_params):  # Cancer, CXPSD col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    col_order = tuple(['Cancer', 'Xray', 'Pollution', 'Smoker', 'Dyspnoea'])
    data.set_order(col_order)
    assert data.get_order() == col_order

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_cancer_13_ok(d_params):  # Cancer, CDPSX col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    data.randomise_names(seed=3)
    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray')

    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_cancer_14_ok(d_params):  # Cancer, CDPSX col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    data.randomise_names(seed=3)
    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray')

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)

    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Smoker', 'Dyspnoea', 'Xray', 'Pollution', 'Cancer')


def test_stable_order_cancer_15_ok(d_params):  # Cancer, CDPSX col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    data.randomise_names(seed=3)
    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray')

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)

    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_cancer_16_ok(d_params):  # Cancer, CDPSX col order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    data.randomise_names(seed=3)
    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray')

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)

    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_cancer_17_ok(d_params):  # Cancer, rand ord & names
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    data.randomise_names(seed=0)
    data.randomise_order(seed=0)
    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Pollution', 'Xray', 'Cancer', 'Smoker', 'Dyspnoea')

    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_cancer_18_ok(d_params):  # Cancer, rand ord & names
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    data.randomise_names(seed=0)
    data.randomise_order(seed=0)
    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Pollution', 'Xray', 'Cancer', 'Smoker', 'Dyspnoea')

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)

    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Smoker', 'Dyspnoea', 'Xray', 'Pollution', 'Cancer')


def test_stable_order_cancer_19_ok(d_params):  # Cancer, rand ord & names
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    data.randomise_names(seed=0)
    data.randomise_order(seed=0)
    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Pollution', 'Xray', 'Cancer', 'Smoker', 'Dyspnoea')

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)

    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_cancer_20_ok(d_params):  # Cancer, rand ord & names
    bn = BN.read(TESTDATA_DIR + '/discrete/small/cancer.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    data.randomise_names(seed=0)
    data.randomise_order(seed=0)
    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Pollution', 'Xray', 'Cancer', 'Smoker', 'Dyspnoea')

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)

    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        ('Cancer', 'Pollution', 'Xray', 'Dyspnoea', 'Smoker')


def test_stable_order_asia_1_ok(d_params):  # Asia, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('tub', 'asia', 'lung', 'either', 'xray', 'smoke', 'bronc', 'dysp')

    print('\n\nStable order for Asia: {}'.format(data.get_order()))


def test_stable_order_asia_2_ok(d_params):  # Asia, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('dysp', 'bronc', 'smoke', 'xray', 'either', 'lung', 'asia', 'tub')

    print('\n\nStable order for Asia: {}'.format(data.get_order()))


def test_stable_order_asia_3_ok(d_params):  # Asia, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('tub', 'asia', 'lung', 'either', 'xray', 'smoke', 'bronc', 'dysp')

    print('\n\nStable order for Asia: {}'.format(data.get_order()))


def test_stable_order_asia_4_ok(d_params):  # Asia, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('tub', 'asia', 'lung', 'either', 'smoke', 'xray', 'bronc', 'dysp')

    print('\n\nStable order for Asia: {}'.format(data.get_order()))


def test_stable_order_asia_5_ok(d_params):  # Asia, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/asia.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.ORDER})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('tub', 'asia', 'lung', 'either', 'smoke', 'xray', 'bronc', 'dysp')

    print('\n\nStable order for Asia: {}'.format(data.get_order()))


def test_stable_order_sports_1_ok(d_params):  # Sports, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('HDA', 'ATgoals', 'HTshotOnTarget', 'ATshots', 'ATshotsOnTarget',
         'HTshots', 'HTgoals', 'possession', 'RDlevel')

    print('\n\nStable {} order for Sports: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_sports_2_ok(d_params):  # Sports, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('RDlevel', 'possession', 'HTgoals', 'HTshots', 'ATshotsOnTarget',
         'ATshots', 'HTshotOnTarget', 'ATgoals', 'HDA')

    print('\n\nStable {} order for Sports: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_sports_3_ok(d_params):  # Sports, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('HDA', 'ATgoals', 'HTshotOnTarget', 'ATshots', 'ATshotsOnTarget',
         'HTshots', 'HTgoals', 'possession', 'RDlevel')

    print('\n\nStable {} order for Sports: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_sports_4_ok(d_params):  # Sports, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('HDA', 'HTshotOnTarget', 'ATshots', 'HTshots', 'possession',
         'RDlevel', 'ATgoals', 'ATshotsOnTarget', 'HTgoals')

    print('\n\nStable {} order for Sports: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_sports_5_ok(d_params):  # Sports, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.ORDER})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('HDA', 'HTshotOnTarget', 'ATshots', 'HTshots', 'possession',
         'RDlevel', 'ATgoals', 'ATshotsOnTarget', 'HTgoals')

    print('\n\nStable {} order for Sports: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_sachs_1_ok(d_params):  # Sachs, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sachs.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.DEC_SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('PIP2', 'Plcg', 'P38', 'Jnk', 'PKA', 'Akt', 'Mek', 'Erk', 'PKC',
         'Raf', 'PIP3')

    print('\n\nStable {} order for Sachs: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_sachs_2_ok(d_params):  # Sachs, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sachs.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.INC_SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('PIP3', 'Raf', 'PKC', 'Erk', 'Mek', 'Akt', 'PKA', 'Jnk', 'P38',
         'Plcg', 'PIP2')

    print('\n\nStable {} order for Sachs: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_sachs_3_ok(d_params):  # Sachs, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sachs.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('PIP3', 'Raf', 'PKC', 'Erk', 'Mek', 'Akt', 'PKA', 'Jnk', 'P38',
         'Plcg', 'PIP2')

    print('\n\nStable {} order for Sachs: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_sachs_4_ok(d_params):  # Sachs, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sachs.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('PIP3', 'Raf', 'Plcg', 'PKC', 'Mek', 'PIP2', 'Akt', 'PKA', 'Jnk',
         'Erk', 'P38')

    print('\n\nStable {} order for Sachs: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_sachs_5_ok(d_params):  # Sachs, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/small/sachs.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.ORDER})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('PIP3', 'Raf', 'Plcg', 'PKC', 'Mek', 'PIP2', 'Akt', 'PKA', 'Jnk',
         'Erk', 'P38')

    print('\n\nStable {} order for Sachs: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_child_1_ok(d_params):  # Child, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/child.dsc')
    data = Pandas(df=bn.generate_cases(100))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('HypDistrib', 'BirthAsphyxia', 'Grunting', 'CO2Report',
         'GruntingReport', 'LVHreport', 'LVH', 'Sick', 'CO2', 'LungParench',
         'Age', 'HypoxiaInO2', 'DuctFlow', 'CardiacMixing', 'RUQO2',
         'LowerBodyO2', 'LungFlow', 'ChestXray', 'XrayReport', 'Disease')

    print('\n\nStable {} order for Child: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_child_4_ok(d_params):  # Child, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/child.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('HypDistrib', 'BirthAsphyxia', 'CO2Report', 'LVHreport', 'LVH', 'CO2',
         'LowerBodyO2', 'LungFlow', 'ChestXray', 'Disease', 'LungParench',
         'DuctFlow', 'CardiacMixing', 'XrayReport', 'Grunting', 'Sick',
         'HypoxiaInO2', 'GruntingReport', 'Age', 'RUQO2')

    print('\n\nStable {} order for Child: {}'
          .format(d_params['stable'].value, data.get_order()))


def test_stable_order_child_5_ok(d_params):  # Child, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/child.dsc')
    data = Pandas(df=bn.generate_cases(100))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.ORDER})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('HypDistrib', 'BirthAsphyxia', 'CO2Report', 'LVHreport', 'LVH', 'CO2',
         'LowerBodyO2', 'LungFlow', 'ChestXray', 'Disease', 'LungParench',
         'DuctFlow', 'CardiacMixing', 'XrayReport', 'Grunting', 'Sick',
         'HypoxiaInO2', 'GruntingReport', 'Age', 'RUQO2')

    print('\n\nStable {} order for Child: {}'
          .format(d_params['stable'].value, data.get_order()))


@pytest.mark.slow
def test_stable_order_insurance_1_ok(d_params):  # Insurance, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    data = Pandas(df=bn.generate_cases(1000))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.DEC_SCORE})
    data, _ = set_stable_order(data, d_params)
    print(data.get_order())

    assert data.get_order() == \
        ('Theft', 'ILiCost', 'GoodStudent', 'MedCost', 'SeniorTrain',
         'OtherCarCost', 'Antilock', 'ThisCarCost', 'OtherCar', 'VehicleYear',
         'AntiTheft', 'Airbag', 'ThisCarDam', 'Accident', 'DrivingSkill',
         'DrivHist', 'Age', 'PropCost', 'RuggedAuto', 'DrivQuality',
         'SocioEcon', 'RiskAversion', 'Mileage', 'MakeModel', 'CarValue',
         'HomeBase', 'Cushioning')

    print('\n\nStable order for Insurance: {}'.format(data.get_order()))


@pytest.mark.slow
def test_stable_order_insurance_3_ok(d_params):  # Insurance, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    data = Pandas(df=bn.generate_cases(1000))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Theft', 'ILiCost', 'GoodStudent', 'MedCost', 'SeniorTrain',
         'OtherCarCost', 'Antilock', 'ThisCarCost', 'OtherCar', 'VehicleYear',
         'AntiTheft', 'Airbag', 'ThisCarDam', 'Accident', 'DrivingSkill',
         'DrivHist', 'Age', 'PropCost', 'RuggedAuto', 'DrivQuality',
         'SocioEcon', 'RiskAversion', 'Mileage', 'MakeModel', 'CarValue',
         'HomeBase', 'Cushioning')

    print('\n\nStable order for Insurance: {}'.format(data.get_order()))


@pytest.mark.slow
def test_stable_order_insurance_4_ok(d_params):  # Insurance, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    data = Pandas(df=bn.generate_cases(1000))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.SCORE_PLUS})
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('Theft', 'RuggedAuto', 'MakeModel', 'CarValue', 'Antilock',
         'ThisCarCost', 'Mileage', 'VehicleYear', 'ThisCarDam', 'Airbag',
         'Accident', 'SocioEcon', 'ILiCost', 'MedCost', 'OtherCarCost',
         'OtherCar', 'DrivingSkill', 'HomeBase', 'Cushioning', 'PropCost',
         'RiskAversion', 'SeniorTrain', 'AntiTheft', 'DrivHist', 'DrivQuality',
         'Age', 'GoodStudent')

    print('\n\nStable order for Insurance: {}'.format(data.get_order()))


@pytest.mark.slow
def test_stable_order_insurance_5_ok(d_params):  # Insurance, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/insurance.dsc')
    data = Pandas(df=bn.generate_cases(1000))

    HCWorker.init_score_cache()
    d_params.update({'stable': Stability.ORDER})
    data, _ = set_stable_order(data, d_params)
    print(data.get_order())

    assert data.get_order() == \
        ('Theft', 'RuggedAuto', 'MakeModel', 'CarValue', 'Antilock',
         'ThisCarCost', 'Mileage', 'VehicleYear', 'ThisCarDam', 'Airbag',
         'Accident', 'SocioEcon', 'ILiCost', 'MedCost', 'OtherCarCost',
         'OtherCar', 'DrivingSkill', 'HomeBase', 'Cushioning', 'PropCost',
         'RiskAversion', 'SeniorTrain', 'AntiTheft', 'DrivHist', 'DrivQuality',
         'Age', 'GoodStudent')

    print('\n\nStable order for Insurance: {}'.format(data.get_order()))


@pytest.mark.slow
def test_stable_order_property_1_ok(d_params):  # Property, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/property.dsc')
    data = Pandas(df=bn.generate_cases(100))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('stampDutyTax', 'stampDutyTaxBand', 'otherPropertyExpensesT1',
         'otherInterestFees', 'otherPropertyExpenses', 'LTV',
         'propertyExpenses', 'borrowing', 'propertyManagement', 'incomeTax',
         'rentalGrossYield', 'capitalGains', 'rentalNetProfitBeforeInterest',
         'propertyExpensesGrowth', 'rentalIncomeLoss', 'capitalGrowth',
         'rentalGrowth', 'rentalGrossProfit', 'interestTaxRelief', 'interest',
         'actualRentalIncome', 'propertyPurchaseValue', 'propertyValueT1',
         'rentalIncomeT1', 'interestRate', 'rentalIncome', 'netProfit')

    print('\n\nStable order for Property: {}'.format(data.get_order()))


@pytest.mark.slow
def test_stable_order_property_2_ok(d_params):  # Property, rev std order
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/property.dsc')
    data = Pandas(df=bn.generate_cases(100))
    print('\nStandard order starting from {}'.format(data.get_order()[0]))
    data.set_order(tuple(list(data.get_order())[::-1]))  # reverse order
    print('... reversed to start from {}'.format(data.get_order()[0]))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == \
        ('stampDutyTax', 'stampDutyTaxBand', 'otherPropertyExpensesT1',
         'otherInterestFees', 'otherPropertyExpenses', 'LTV',
         'propertyExpenses', 'borrowing', 'propertyManagement', 'incomeTax',
         'rentalGrossYield', 'capitalGains', 'rentalNetProfitBeforeInterest',
         'propertyExpensesGrowth', 'rentalIncomeLoss', 'capitalGrowth',
         'rentalGrowth', 'rentalGrossProfit', 'interest', 'interestTaxRelief',
         'actualRentalIncome', 'propertyPurchaseValue', 'propertyValueT1',
         'rentalIncomeT1', 'interestRate', 'rentalIncome', 'netProfit')

    print('\n\nStable order for Property: {}'.format(data.get_order()))


@pytest.mark.slow
def test_stable_order_diarrhoea_1_ok(d_params):  # Diarrhoea, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/diarrhoea.dsc')
    data = Pandas(df=bn.generate_cases(100))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('DIA_HadDiahorrea', 'BF_BottleFeeding', 'DEL_SmallBaby',
         'HOU_CookingFuel', 'SRV_OKAlone', 'WSH_WashWithAgent',
         'IMM_Diptheria', 'FP_ModernMethod', 'WSH_WaterTreated', 'IMM_Measles',
         'SRV_Near', 'MTH_MaternalAge', 'IMM_VitaminA1', 'KNW_WatchTV',
         'WSH_ImprovedWaterSource', 'ECO_WealthQuintile',
         'WSH_SafeStoolDisposal', 'CUL_Religion', 'HOU_ModernWallMaterial',
         'FP_BirthsLast5Yrs', 'CUL_LanguageGroup', 'CHI_Weight4Height',
         'WSH_ImprovedToilet', 'MTH_Education', 'BF_EarlyBreastfeeding',
         'BF_BreastfedMonths', 'GEO_Region', 'CHI_Age')

    print('\n\nStable order for Diarrhoea: {}'.format(data.get_order()))


@pytest.mark.slow
def test_stable_order_hailfinder_1_ok(d_params):  # Hailfinder, std order
    bn = BN.read(TESTDATA_DIR + '/discrete/large/hailfinder.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == \
        ('ScenRelAMCIN', 'WindFieldMt', 'PlainsFcst', 'N34StarFcst',
         'VISCloudCov', 'InsInMt', 'MorningBound', 'IRCloudCover',
         'CombClouds', 'MountainFcst', 'MeanRH', 'CldShadeOth', 'AMDewptCalPl',
         'Boundaries', 'AMCINInScen', 'R5Fcst', 'CapChange', 'CompPlFcst',
         'MorningCIN', 'OutflowFrMt', 'InsSclInScen', 'MidLLapse', 'CapInScen',
         'RHRatio', 'InsChange', 'CldShadeConv', 'AMInsWliScen', 'AMInstabMt',
         'LatestCIN', 'LIfr12ZDENSd', 'QGVertMotion', 'SubjVertMo',
         'MvmtFeatures', 'TempDis', 'AreaMeso_ALS', 'CombVerMo', 'LLIW',
         'SatContMoist', 'RaoContMoist', 'CombMoisture', 'LoLevMoistAd',
         'AreaMoDryAir', 'LowLLapse', 'CurPropConv', 'WindAloft',
         'WndHodograph', 'N0_7muVerMo', 'ScenRel3_4', 'ScenRelAMIns',
         'SynForcng', 'Date', 'WindFieldPln', 'SfcWndShfDis', 'Dewpoints',
         'Scenario', 'ScnRelPlFcst')

    print('\n\nStable order for Hailfiinder: {}'.format(data.get_order()))


@pytest.mark.slow
def test_stable_order_hailfinder_2_ok(d_params):  # Hailfind, rev std order
    bn = BN.read(TESTDATA_DIR + '/discrete/large/hailfinder.dsc')
    data = Pandas(df=bn.generate_cases(1000))
    print('\nStandard order starting from {}'.format(data.get_order()[0]))
    data.set_order(tuple(list(data.get_order())[::-1]))  # reverse order
    print('... reversed to start from {}'.format(data.get_order()[0]))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    # Note order is different from hailfinder 1 - still some order dependency
    # because of identical variables: CombVerMo & AreaMesoALS, and
    # Scenario and ScnRelPlFcst

    assert data.get_order() == \
        ('ScenRelAMCIN', 'WindFieldMt', 'PlainsFcst', 'N34StarFcst',
         'VISCloudCov', 'InsInMt', 'MorningBound', 'IRCloudCover',
         'CombClouds', 'MountainFcst', 'MeanRH', 'CldShadeOth', 'AMDewptCalPl',
         'Boundaries', 'AMCINInScen', 'R5Fcst', 'CapChange', 'CompPlFcst',
         'MorningCIN', 'OutflowFrMt', 'InsSclInScen', 'MidLLapse', 'CapInScen',
         'RHRatio', 'InsChange', 'CldShadeConv', 'AMInsWliScen', 'AMInstabMt',
         'LatestCIN', 'LIfr12ZDENSd', 'QGVertMotion', 'SubjVertMo',
         'MvmtFeatures', 'TempDis', 'CombVerMo', 'AreaMeso_ALS', 'LLIW',
         'SatContMoist', 'RaoContMoist', 'CombMoisture', 'LoLevMoistAd',
         'AreaMoDryAir', 'LowLLapse', 'CurPropConv', 'WindAloft',
         'WndHodograph', 'N0_7muVerMo', 'ScenRel3_4', 'ScenRelAMIns',
         'SynForcng', 'Date', 'WindFieldPln', 'SfcWndShfDis', 'Dewpoints',
         'ScnRelPlFcst', 'Scenario')

    print('\n\nStable order for Hailfiinder: {}'.format(data.get_order()))


@pytest.mark.slow
def test_stable_order_win95pts_1_ok(d_params):  # Win95pts, 100, std order
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/win95pts.data.gz',
                       dstype='categorical', N=1000)

    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    assert data.get_order() == \
        ('PrtThread', 'AppDtGnTm', 'DrvSet', 'DataFile', 'AppOK', 'PrtQueue',
         'TnrSpply', 'AppData', 'PrtStatToner', 'EPSGrphc', 'PSERRMEM',
         'PrntPrcssTm', 'DrvOK', 'AvlblVrtlMmry', 'CblPrtHrdwrOK', 'TstpsTxt',
         'PrtSel', 'HrglssDrtnAftrPrnt', 'PrtPort', 'FntInstlltn', 'REPEAT',
         'PrtCbl', 'PrntngArOK', 'PrtPaper', 'NtwrkCnfg', 'PrtStatPaper',
         'PrtPath', 'PrtStatMem', 'DskLocal', 'PrtMem', 'LclOK', 'DeskPrntSpd',
         'PgOrnttnOK', 'DS_LCLOK', 'PrtTimeOut', 'ScrnFntNtPrntrFnt',
         'PrtSpool', 'Problem2', 'EMFOK', 'GDIIN', 'GrphcsRltdDrvrSttngs',
         'NtSpd', 'PSGRAPHIC', 'NnPSGrphc', 'Problem4', 'PrtDriver', 'NnTTOK',
         'TrTypFnts', 'NtGrbld', 'GrbldOtpt', 'LclGrbld', 'CmpltPgPrntd',
         'PrtOn', 'GrbldPS', 'Problem6', 'IncmpltPS', 'Problem3',
         'PrntrAccptsTrtyp', 'PrtStatOff', 'PrtIcon', 'GDIOUT', 'Problem5',
         'TTOK', 'DSApplctn', 'PrtDataOut', 'FllCrrptdBffr', 'NetPrint',
         'PrtMpTPth', 'PC2PRT', 'PrtFile', 'PTROFFLINE', 'NetOK', 'PrtPScript',
         'Problem1', 'PrtData', 'DS_NTOK')


@pytest.mark.slow
def test_stable_order_win95pts_2_ok(d_params):  # Win95pts, 1K, rev std order
    data = Pandas.read(TESTDATA_DIR + '/experiments/datasets/win95pts.data.gz',
                       dstype='categorical', N=1000)

    print('\nStandard order starting from {}'.format(data.get_order()[0]))
    data.set_order(tuple(list(data.get_order())[::-1]))  # reverse order
    print('... reversed to start from {}'.format(data.get_order()[0]))

    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    # Note order is different from Win95pts 1 - still some order dependency
    # because of identical variables: GrbldPS & Problem6, IncmpltPS & Problem3

    assert data.get_order() == \
        ('PrtThread', 'AppDtGnTm', 'DrvSet', 'DataFile', 'AppOK', 'PrtQueue',
         'TnrSpply', 'AppData', 'PrtStatToner', 'EPSGrphc', 'PSERRMEM',
         'PrntPrcssTm', 'DrvOK', 'AvlblVrtlMmry', 'CblPrtHrdwrOK', 'TstpsTxt',
         'PrtSel', 'HrglssDrtnAftrPrnt', 'PrtPort', 'FntInstlltn', 'REPEAT',
         'PrtCbl', 'PrntngArOK', 'PrtPaper', 'NtwrkCnfg', 'PrtStatPaper',
         'PrtPath', 'PrtStatMem', 'DskLocal', 'PrtMem', 'LclOK', 'DeskPrntSpd',
         'PgOrnttnOK', 'DS_LCLOK', 'PrtTimeOut', 'ScrnFntNtPrntrFnt',
         'PrtSpool', 'Problem2', 'EMFOK', 'GDIIN', 'GrphcsRltdDrvrSttngs',
         'NtSpd', 'PSGRAPHIC', 'NnPSGrphc', 'Problem4', 'PrtDriver', 'NnTTOK',
         'TrTypFnts', 'NtGrbld', 'GrbldOtpt', 'LclGrbld', 'CmpltPgPrntd',
         'PrtOn', 'Problem6', 'GrbldPS', 'Problem3', 'IncmpltPS',
         'PrntrAccptsTrtyp', 'PrtStatOff', 'PrtIcon', 'GDIOUT', 'Problem5',
         'TTOK', 'DSApplctn', 'PrtDataOut', 'FllCrrptdBffr', 'NetPrint',
         'PrtMpTPth', 'PC2PRT', 'PrtFile', 'PTROFFLINE', 'NetOK', 'PrtPScript',
         'Problem1', 'PrtData', 'DS_NTOK')


@pytest.mark.slow
def test_stable_order_pathfinder_1_ok(d_params):  # Path.., std order
    bn = BN.read(TESTDATA_DIR + '/discrete/verylarge/pathfinder.dsc')
    data = Pandas(df=bn.generate_cases(10000))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)
    assert data.get_order() == \
        ('F72', 'F27', 'F12', 'F68', 'F69', 'F15', 'F75', 'F13', 'F59', 'F28',
         'F14', 'F37', 'F65', 'F88', 'F24', 'F66', 'F57', 'F77', 'F67', 'F1',
         'F45', 'F89', 'F87', 'F23', 'F34', 'F31', 'F95', 'F85', 'F64', 'F10',
         'F84', 'F86', 'F20', 'F36', 'F33', 'F70', 'F63', 'F54', 'F25', 'F80',
         'F32', 'F51', 'F7', 'F26', 'F8', 'F43', 'F79', 'F18', 'F92', 'F29',
         'F6', 'F55', 'F35', 'F107', 'F73', 'F53', 'F102', 'F17', 'F93', 'F56',
         'F52', 'F19', 'F11', 'F41', 'F44', 'F21', 'F76', 'F97', 'F108', 'F91',
         'F22', 'F58', 'F3', 'F5', 'F38', 'F78', 'F62', 'F100', 'F9', 'F16',
         'F4', 'F47', 'F2', 'F101', 'F90', 'F60', 'F105', 'F71', 'F61', 'F48',
         'F82', 'F106', 'F40', 'F39', 'F103', 'F104', 'F96', 'F99', 'F46',
         'F98', 'F81', 'F83', 'F74', 'F30', 'F49', 'F50', 'F42', 'F94',
         'Fault')

    print('\n\nStable order for Pathfinder: {}'.format(data.get_order()))


@pytest.mark.slow
def test_stable_order_pathfinder_2_ok(d_params):  # Path.., rev std order
    bn = BN.read(TESTDATA_DIR + '/discrete/verylarge/pathfinder.dsc')
    data = Pandas(df=bn.generate_cases(10000))
    print('\nStandard order starting from {}'.format(data.get_order()[0]))
    data.set_order(tuple(list(data.get_order())[::-1]))  # reverse order
    print('... reversed to start from {}'.format(data.get_order()[0]))
    HCWorker.init_score_cache()
    data, _ = set_stable_order(data, d_params)

    # Even though some variables are very similar in Pathfinder, they are
    # different in this test case and stable order remains the same.

    assert data.get_order() == \
        ('F72', 'F27', 'F12', 'F68', 'F69', 'F15', 'F75', 'F13', 'F59', 'F28',
         'F14', 'F37', 'F65', 'F88', 'F24', 'F66', 'F57', 'F77', 'F67', 'F1',
         'F45', 'F89', 'F87', 'F23', 'F34', 'F31', 'F95', 'F85', 'F64', 'F10',
         'F84', 'F86', 'F20', 'F36', 'F33', 'F70', 'F63', 'F54', 'F25', 'F80',
         'F32', 'F51', 'F7', 'F26', 'F8', 'F43', 'F79', 'F18', 'F92', 'F29',
         'F6', 'F55', 'F35', 'F107', 'F73', 'F53', 'F102', 'F17', 'F93', 'F56',
         'F52', 'F19', 'F11', 'F41', 'F44', 'F21', 'F76', 'F97', 'F108', 'F91',
         'F22', 'F58', 'F3', 'F5', 'F38', 'F78', 'F62', 'F100', 'F9', 'F16',
         'F4', 'F47', 'F2', 'F101', 'F90', 'F60', 'F105', 'F71', 'F61', 'F48',
         'F82', 'F106', 'F40', 'F39', 'F103', 'F104', 'F96', 'F99', 'F46',
         'F98', 'F81', 'F83', 'F74', 'F30', 'F49', 'F50', 'F42', 'F94',
         'Fault')

    print('\n\nStable order for Pathfinder: {}'.format(data.get_order()))


# Test learning with stable flag

def test_hc_stable_ab_1_ok():  # 2 variables with same entropy
    data = Pandas(df=DataFrame({'A': ['0', '1', '1', '1'],
                                'B': ['1', '0', '1', '1']}, dtype='category'))
    context = {'id': 'test/hc_stable/ab_1', 'in': 'ab_1'}
    dag, trace = hc(data, params={'stable': True}, context=context)
    print('\n\n{}\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A][B]'


def test_hc_stable_ab_2_ok():
    data = Pandas(df=DataFrame({'A': ['0', '0', '1', '1'],
                                'B': ['0', '0', '0', '1']}, dtype='category'))
    context = {'id': 'test/hc_stable/ab_1', 'in': 'ab_1'}
    dag, trace = hc(data, params={'stable': True}, context=context)
    print('\n\n{}\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B]'


def test_hc_stable_ab_3_ok():
    data = Pandas(df=DataFrame({'B': ['0', '0', '0', '1'],
                                'A': ['0', '0', '1', '1']}, dtype='category'))
    context = {'id': 'test/hc_stable/ab_2', 'in': 'ab_2'}
    dag, trace = hc(data, params={'stable': True}, context=context)
    print('\n\n{}\n\n{}'.format(trace, dag))
    assert dag.to_string() == '[A|B][B]'


def test_hc_stable_abc_1_ok(d_params):  # A->B->C, some order dependence
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = Pandas(df=bn.generate_cases(10))
    context = {'id': 'test/hc_stable/abc_1', 'in': 'abc_1'}
    dag, trace = hc(data, params={'stable': True}, context=context)
    print('\n\n{}\n\n{}'.format(trace, dag))

    # C, B have same score so still some order dependency

    assert data.get_order() == ('A', 'B', 'C')
    assert dag.to_string() == '[A][B|A][C|B]'


def test_hc_stable_abc_2_ok(d_params):  # A->B->C, some order dependence
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = Pandas(df=bn.generate_cases(10))
    data.set_order(tuple(['C', 'B', 'A']))
    context = {'id': 'test/hc_stable/abc_2', 'in': 'abc_2'}
    dag, trace = hc(data, params={'stable': True}, context=context)
    print('\n\n{}\n\n{}'.format(trace, dag))

    # C, B have same score so still some order dependency,
    # and so get different results to abc_1

    assert data.get_order() == ('A', 'C', 'B')
    assert dag.to_string() == '[A][B|C][C|A]'


def test_hc_stable_abc_3_ok(d_params):  # A->B->C, stable order
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = Pandas(df=bn.generate_cases(100))
    context = {'id': 'test/hc_stable/abc_1', 'in': 'abc_1'}
    dag, trace = hc(data, params={'stable': True}, context=context)
    print('\n\n{}\n\n{}'.format(trace, dag))

    # At higher sample size, order is stable

    assert data.get_order() == ('A', 'B', 'C')
    assert dag.to_string() == '[A][B|A][C|B]'


def test_hc_stable_abc_4_ok(d_params):  # A->B->C, stable order
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = Pandas(df=bn.generate_cases(100))
    data.set_order(tuple(['C', 'B', 'A']))
    context = {'id': 'test/hc_stable/abc_2', 'in': 'abc_2'}
    dag, trace = hc(data, params={'stable': True}, context=context)
    print('\n\n{}\n\n{}'.format(trace, dag))

    # Order is stable, result not dependent upon column order

    assert data.get_order() == ('A', 'B', 'C')
    assert dag.to_string() == '[A][B|A][C|B]'


def test_hc_stable_abc_5_ok(d_params):  # A->B->C, stable order
    bn = BN.read(TESTDATA_DIR + '/discrete/tiny/abc.dsc')
    data = Pandas(df=bn.generate_cases(100))
    data.set_order(tuple(['B', 'A', 'C']))
    context = {'id': 'test/hc_stable/abc_2', 'in': 'abc_2'}
    dag, trace = hc(data, params={'stable': True}, context=context)
    print('\n\n{}\n\n{}'.format(trace, dag))

    # Order is stable, result not dependent upon column order

    assert data.get_order() == ('A', 'B', 'C')
    assert dag.to_string() == '[A][B|A][C|B]'


def test_hc_stable_cancer_1_ok(d_params):  # Cancer, CDPSX col order
    dsc = TESTDATA_DIR + '/discrete/small/cancer.dsc'
    data = Pandas(df=BN.read(dsc).generate_cases(10000))
    context = {'id': 'test/hc_stable/cancer_1', 'in': dsc}
    dag, trace = hc(data, params={'stable': True}, context=context)
    print('\n\n{}\n\n{}'.format(trace, dag))

    # Order is stable

    assert data.get_order() == \
        tuple(['Cancer', 'Pollution', 'Xray', 'Smoker', 'Dyspnoea'])
    assert dag.to_string() == \
        ('[Cancer]' +
         '[Dyspnoea|Cancer]' +
         '[Pollution|Cancer]' +
         '[Smoker|Cancer:Pollution]' +
         '[Xray|Cancer]')


def test_hc_stable_cancer_2_ok(d_params):  # Cancer, DXSPC col order
    dsc = TESTDATA_DIR + '/discrete/small/cancer.dsc'
    data = Pandas(df=BN.read(dsc).generate_cases(10000))
    data.set_order(tuple(['Dyspnoea', 'Xray', 'Smoker',
                          'Pollution', 'Cancer']))
    context = {'id': 'test/hc_stable/cancer_2', 'in': dsc}
    dag, trace = hc(data, params={'stable': True}, context=context)
    print('\n\n{}\n\n{}'.format(trace, dag))

    # Order is stable (same order & DAG as before)

    assert data.get_order() == \
        tuple(['Cancer', 'Pollution', 'Xray', 'Smoker', 'Dyspnoea'])
    assert dag.to_string() == \
        ('[Cancer]' +
         '[Dyspnoea|Cancer]' +
         '[Pollution|Cancer]' +
         '[Smoker|Cancer:Pollution]' +
         '[Xray|Cancer]')


def test_hc_stable_cancer_3_ok(d_params):  # Cancer, XCSPD col order
    dsc = TESTDATA_DIR + '/discrete/small/cancer.dsc'
    data = Pandas(df=BN.read(dsc).generate_cases(10000))
    data.set_order(tuple(['Xray', 'Cancer', 'Smoker',
                          'Pollution', 'Dyspnoea']))
    context = {'id': 'test/hc_stable/cancer_3', 'in': dsc}
    dag, trace = hc(data, params={'stable': True}, context=context)
    print('\n\n{}\n\n{}'.format(trace, dag))

    # Order is stable (same order & DAG as before)

    assert data.get_order() == \
        tuple(['Cancer', 'Pollution', 'Xray', 'Smoker', 'Dyspnoea'])
    assert dag.to_string() == \
        ('[Cancer]' +
         '[Dyspnoea|Cancer]' +
         '[Pollution|Cancer]' +
         '[Smoker|Cancer:Pollution]' +
         '[Xray|Cancer]')


def test_hc_stable_cancer_4_ok(d_params):  # Cancer, rename columns
    dsc = TESTDATA_DIR + '/discrete/small/cancer.dsc'
    data = Pandas(df=BN.read(dsc).generate_cases(10000))
    data.randomise_names(seed=0)
    context = {'id': 'test/hc_stable/cancer_4', 'in': dsc}
    dag, trace = hc(data, params={'stable': True}, context=context)

    trace.rename(data.ext_to_orig)  # reverts names in dag too !!!
    print('\n\n{}\n\n{}'.format(trace, dag))

    # Order is stable (same order & DAG as before)

    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        tuple(['Cancer', 'Pollution', 'Xray', 'Smoker', 'Dyspnoea'])
    assert dag.to_string() == \
        ('[Cancer]' +
         '[Dyspnoea|Cancer]' +
         '[Pollution|Cancer]' +
         '[Smoker|Cancer:Pollution]' +
         '[Xray|Cancer]')


def test_hc_stable_cancer_5_ok(d_params):  # Cancer, reorder & rename columns
    dsc = TESTDATA_DIR + '/discrete/small/cancer.dsc'
    data = Pandas(df=BN.read(dsc).generate_cases(10000))
    data.set_order(tuple(['Dyspnoea', 'Cancer', 'Smoker',
                          'Pollution', 'Xray']))
    data.randomise_names(seed=1)
    context = {'id': 'test/hc_stable/cancer_5', 'in': dsc}
    dag, trace = hc(data, params={'stable': True}, context=context)

    trace.rename(data.ext_to_orig)  # reverts names in dag too !!!
    print('\n\nNodes: {}\n\n{}\n\n{}'.format(data.get_order(), trace, dag))

    # Order is stable (same order & DAG as before)

    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        tuple(['Cancer', 'Pollution', 'Xray', 'Smoker', 'Dyspnoea'])
    assert dag.to_string() == \
        ('[Cancer]' +
         '[Dyspnoea|Cancer]' +
         '[Pollution|Cancer]' +
         '[Smoker|Cancer:Pollution]' +
         '[Xray|Cancer]')


def test_hc_stable_cancer_6_ok(d_params):  # Cancer, random names and order
    dsc = TESTDATA_DIR + '/discrete/small/cancer.dsc'
    data = Pandas(df=BN.read(dsc).generate_cases(10000))
    data.randomise_order(seed=2)
    data.randomise_names(seed=2)
    context = {'id': 'test/hc_stable/cancer_6', 'in': dsc}
    dag, trace = hc(data, params={'stable': True}, context=context)

    trace.rename(data.ext_to_orig)  # reverts names in dag too !!!
    print('\n\nNodes: {}\n\n{}\n\n{}'.format(data.get_order(), trace, dag))

    # Order is stable (same order & DAG as before)

    assert tuple([data.ext_to_orig[n] for n in data.get_order()]) == \
        tuple(['Cancer', 'Pollution', 'Xray', 'Smoker', 'Dyspnoea'])
    assert dag.to_string() == \
        ('[Cancer]' +
         '[Dyspnoea|Cancer]' +
         '[Pollution|Cancer]' +
         '[Smoker|Cancer:Pollution]' +
         '[Xray|Cancer]')
