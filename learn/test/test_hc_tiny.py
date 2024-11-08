
#   Run hc hill-climbing structure learning on tiny graphs to see how
#   parameter strength affects learning outcomes

import pytest
from pandas import DataFrame, set_option

from core.bn import BN
from core.cpt import CPT
from learn.hc import hc
from analysis.trace import TraceAnalysis
import testdata.example_dags as ex_dag


@pytest.fixture
def showall():
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)


# A->B learnt correctly for 10, 100 and 1K rows

def test_hc_tiny_ab2_r1_0_ok(showall):  # A->B binary deterministic
    ab = ex_dag.ab()
    bn = BN(ab, {'A': (CPT, {'0': 0.5, '1': 0.5}),
                 'B': (CPT, [({'A': '0'}, {'0': 1.0, '1': 0.0}),
                             ({'A': '1'}, {'0': 0.0, '1': 1.0})])})
    dsc = 'A-->B 1:0'
    data = bn.generate_cases(1000)
    context = {'id': 'TINY/AB2/R1_0/N1000', 'in': dsc}
    dag, trace = hc(data, context=context, params={'score': 'loglik'})
    print('\nLearning DAG from 10 rows of A->B produces:\n{}'.format(dag))
    analysis = TraceAnalysis(trace, bn, data)
    print(analysis)
    # trace.save(EXPTS_DIR)

    # check learnt graph

    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1

    # check Oracle and data MI values reported

    item = DataFrame(analysis.trace).to_dict(orient='records')[1]
    assert item['Oracle MI'] == 0.693147  # this is ln(2)
    assert item['MI'] == 0.692635  # close to ln(2)


def test_hc_tiny_ab2_r1_1_ok(showall):  # A->B binary independent
    ab = ex_dag.ab()
    bn = BN(ab, {'A': (CPT, {'0': 0.5, '1': 0.5}),
                 'B': (CPT, [({'A': '0'}, {'0': 0.5, '1': 0.5}),
                             ({'A': '1'}, {'0': 0.5, '1': 0.5})])})
    dsc = 'A-->B 1:1'
    data = bn.generate_cases(1000)
    context = {'id': 'TINY/AB2/R1_1/N1000', 'in': dsc}
    dag, trace = hc(data, context=context, params={'score': 'loglik'})
    print('\nLearning DAG from 10 rows of A->B produces:\n{}'.format(dag))
    analysis = TraceAnalysis(trace, bn, data)
    print(analysis)
    # trace.save(EXPTS_DIR)

    # check learnt graph

    assert dag.to_string() == '[A][B|A]'
    assert dag.number_components() == 1

    # check Oracle and data MI values reported

    item = DataFrame(analysis.trace).to_dict(orient='records')[1]
    assert item['Oracle MI'] == 0  # independent so should be 0
    assert item['MI'] == 0.002011  # close to 0
