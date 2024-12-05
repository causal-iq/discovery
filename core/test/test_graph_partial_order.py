
#   Test topological partial_order function

import pytest

from core.graph import DAG
from core.bn import BN
import testdata.example_dags as ex_dag
from fileio.common import TESTDATA_DIR


def test_graph_partial_order_type_error1():
    with pytest.raises(TypeError):
        DAG.partial_order()


def test_graph_partial_order_type_error2():
    with pytest.raises(TypeError):
        DAG.partial_order(37.1)


def test_graph_partial_order_a_ok_1():
    order = DAG.partial_order({'A': []})
    assert order == [{'A'}]


def test_graph_partial_order_a_ok_2():
    order = DAG.partial_order({'A': set()})
    assert order == [{'A'}]


def test_graph_partial_order_ab_ok_1():
    order = DAG.partial_order({'A': [], 'B': ['A']})
    assert order == [{'A'}, {'B'}]


def test_graph_partial_order_ab_ok_2():
    order = DAG.partial_order({'A': set(), 'B': {'A'}})
    assert order == [{'A'}, {'B'}]


def test_graph_partial_order_ab_ok_3():
    order = DAG.partial_order({'A': {'B'}, 'B': []}, new_arc=('A', 'B'))
    assert order == [{'A'}, {'B'}]


def test_graph_partial_order_ab_ok_4():
    order = DAG.partial_order({'A': [], 'B': []}, new_arc=('A', 'B'))
    assert order == [{'A'}, {'B'}]


def test_graph_partial_order_ab_cycle_ok_1():
    order = DAG.partial_order({'A': ['B'], 'B': ['A']})
    assert order is None


def test_graph_partial_order_ab_cycle_ok_2():
    order = DAG.partial_order({'A': {'B'}, 'B': {'A'}})
    assert order is None


def test_graph_partial_order_a_b_ok_1():
    order = DAG.partial_order({'A': [], 'B': []})
    assert order == [{'A', 'B'}]


def test_graph_partial_order_a_b_ok_2():
    order = DAG.partial_order({'A': set(), 'B': set()})
    assert order == [{'A', 'B'}]


def test_graph_partial_order_abc_ok_1():
    order = DAG.partial_order({'A': set(), 'B': {'A'}, 'C': {'B'}})
    assert order == [{'A'}, {'B'}, {'C'}]


def test_graph_partial_order_abc_ok_2():
    order = DAG.partial_order({'A': [], 'B': [], 'C': {'B'}},
                              new_arc=('A', 'B'))
    assert order == [{'A'}, {'B'}, {'C'}]


def test_graph_partial_order_cba_ok_1():
    order = DAG.partial_order({'A': ['B'], 'B': {'C'}, 'C': []})
    assert order == [{'C'}, {'B'}, {'A'}]


def test_graph_partial_order_ba_bc_ok_1():
    order = DAG.partial_order({'A': {'B'}, 'C': {'B'}, 'B': []})
    assert order == [{'B'}, {'A', 'C'}]


def test_graph_partial_order_ba_bc_ok_2():
    order = DAG.partial_order({'A': ['B'], 'C': [], 'B': []},
                              new_arc=('B', 'C'))
    assert order == [{'B'}, {'A', 'C'}]


def test_graph_partial_order_ab_cb_ok_1():
    order = DAG.partial_order({'A': [], 'B': ['A', 'C'], 'C': []})
    assert order == [{'A', 'C'}, {'B'}]


def test_graph_partial_order_a_b_c_ok_1():
    order = DAG.partial_order({'A': [], 'B': [], 'C': []})
    assert order == [{'A', 'B', 'C'}]


def test_graph_partial_order_a_bc_ok_1():
    order = DAG.partial_order({'A': [], 'B': [], 'C': []}, new_arc=('B', 'C'))
    assert order == [{'A', 'B'}, {'C'}]


def test_graph_partial_order_abca_ok_1():
    order = DAG.partial_order({'A': ['C'], 'B': ['A'], 'C': ['B']})
    assert order is None


def test_graph_partial_order_abca_ok_2():
    order = DAG.partial_order({'A': ['C', 'B'], 'B': [], 'C': ['B']},
                              new_arc=('A', 'B'))
    assert order is None


def test_graph_partial_order_abcd_ok_1():
    order = DAG.partial_order({'A': [], 'B': ['A'], 'C': ['B']}, ['D'])
    assert order == [{'A', 'D'}, {'B'}, {'C'}]


def test_graph_partial_order_abcd_ok_2():
    order = DAG.partial_order({'A': [], 'B': ['A'], 'C': ['B']},
                              ['A', 'D', 'B', 'C'])
    assert order == [{'A', 'D'}, {'B'}, {'C'}]


def test_graph_partial_order_abcd_ok_3():
    order = DAG.partial_order({'A': [], 'B': ['A'], 'C': ['B']},
                              {'A', 'D'})
    assert order == [{'A', 'D'}, {'B'}, {'C'}]


def test_graph_partial_order_cancer_ok_1():  # not adding new arc
    dag = ex_dag.cancer()
    order = DAG.partial_order(dag.parents, dag.nodes)

    assert dag.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray']
    assert dag.parents == \
        {'Cancer': ['Pollution', 'Smoker'],
         'Dyspnoea': ['Cancer'],
         'Xray': ['Cancer']}
    assert order == [{'Pollution', 'Smoker'}, {'Cancer'},
                     {'Dyspnoea', 'Xray'}]


def test_graph_partial_order_cancer_ok_2():  # add S --> X, order unchanged
    dag = ex_dag.cancer()
    order = DAG.partial_order(dag.parents, dag.nodes,
                              new_arc=('Smoker', 'Xray'))

    assert dag.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray']
    assert dag.parents == \
        {'Cancer': ['Pollution', 'Smoker'],
         'Dyspnoea': ['Cancer'],
         'Xray': ['Cancer']}
    assert order == [{'Pollution', 'Smoker'}, {'Cancer'},
                     {'Dyspnoea', 'Xray'}]


def test_graph_partial_order_cancer_ok_3():  # add S --> D, order changed
    dag = ex_dag.cancer()
    order = DAG.partial_order(dag.parents, dag.nodes,
                              new_arc=('Xray', 'Dyspnoea'))

    assert dag.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray']
    assert dag.parents == \
        {'Cancer': ['Pollution', 'Smoker'],
         'Dyspnoea': ['Cancer'],
         'Xray': ['Cancer']}
    assert order == [{'Pollution', 'Smoker'}, {'Cancer'},
                     {'Xray'}, {'Dyspnoea'}]


def test_graph_partial_order_cancer_ok_4():  # add P --> S, order changed
    dag = ex_dag.cancer()
    order = DAG.partial_order(dag.parents, dag.nodes,
                              new_arc=('Pollution', 'Smoker'))

    assert dag.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray']
    assert dag.parents == \
        {'Cancer': ['Pollution', 'Smoker'],
         'Dyspnoea': ['Cancer'],
         'Xray': ['Cancer']}
    assert order == [{'Pollution'}, {'Smoker'}, {'Cancer'},
                     {'Xray', 'Dyspnoea'}]


def test_graph_partial_order_cancer_ok_5():  # Reverse P --> C, order changed
    dag = ex_dag.cancer()
    order = DAG.partial_order(dag.parents, dag.nodes,
                              new_arc=('Cancer', 'Pollution'))

    assert dag.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray']
    assert dag.parents == \
        {'Cancer': ['Pollution', 'Smoker'],
         'Dyspnoea': ['Cancer'],
         'Xray': ['Cancer']}
    assert order == [{'Smoker'}, {'Cancer'},
                     {'Dyspnoea', 'Pollution', 'Xray'}]


def test_graph_partial_order_cancer_ok_6():  # Reverse C --> X, order changed
    dag = ex_dag.cancer()
    order = DAG.partial_order(dag.parents, dag.nodes,
                              new_arc=('Xray', 'Cancer'))

    assert dag.nodes == ['Cancer', 'Dyspnoea', 'Pollution', 'Smoker', 'Xray']
    assert dag.parents == \
        {'Cancer': ['Pollution', 'Smoker'],
         'Dyspnoea': ['Cancer'],
         'Xray': ['Cancer']}
    assert order == [{'Smoker', 'Pollution', 'Xray'}, {'Cancer'},
                     {'Dyspnoea'}]


def test_graph_partial_order_asia_ok_1():
    dag = ex_dag.asia()
    order = DAG.partial_order(dag.parents, dag.nodes)
    assert order == [{'asia', 'smoke'}, {'bronc', 'lung', 'tub'}, {'either'},
                     {'dysp', 'xray'}]


def test_graph_partial_order_asia_ok_2():
    dag = ex_dag.asia()
    order = DAG.partial_order(dag.parents, dag.nodes, new_arc=('xray', 'asia'))
    assert order is None


def test_graph_partial_order_child_ok_1():
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/child.dsc')
    order = DAG.partial_order(bn.dag.parents, bn.dag.nodes)
    assert order == \
        [{'BirthAsphyxia'}, {'Disease'},
         {'Sick', 'DuctFlow', 'CardiacMixing', 'LungParench', 'LungFlow',
          'LVH'},
         {'Age', 'Grunting', 'HypDistrib', 'HypoxiaInO2', 'CO2', 'ChestXray',
          'LVHreport'},
         {'GruntingReport', 'LowerBodyO2', 'RUQO2', 'CO2Report', 'XrayReport'}]


def test_graph_partial_order_child_ok_2():
    bn = BN.read(TESTDATA_DIR + '/discrete/medium/child.dsc')
    order = DAG.partial_order(bn.dag.parents, bn.dag.nodes,
                              new_arc=('GruntingReport', 'BirthAsphyxia'))
    assert order is None

#   Test node_ordering which is built on partial_order


def test_graph_node_ordering_empty_ok():
    nodes = [n for n in ex_dag.empty().ordered_nodes()]
    assert nodes == []


def test_graph_node_ordering_ab_ok():
    nodes = [n for n in ex_dag.ab().ordered_nodes()]
    assert nodes == ['A', 'B']


def test_graph_node_ordering_abc_ok():
    nodes = [n for n in ex_dag.abc().ordered_nodes()]
    assert nodes == ['A', 'B', 'C']


def test_graph_node_ordering_cancer_ok():
    nodes = [n for n in ex_dag.cancer().ordered_nodes()]
    assert nodes == ['Pollution', 'Smoker', 'Cancer', 'Dyspnoea', 'Xray']
