
#   Test PDAG comparison metrics

import pytest

import testdata.example_pdags as ex_pdag
import testdata.example_dags as ex_dag
import testdata.example_sdgs as ex_sdg


@pytest.fixture
def expected():
    return {'arc_matched': 0, 'arc_reversed': 0, 'edge_not_arc': 0,
            'arc_not_edge': 0, 'edge_matched': 0, 'arc_extra': 0,
            'edge_extra': 0, 'arc_missing': 0, 'edge_missing': 0,
            'missing_matched': 0, 'shd': 0, 'p': None, 'r': None, 'f1': 0.0}


def test_metrics_pdag_type_error1():  # bad argument type for pdag
    with pytest.raises(TypeError):
        ex_pdag.empty().compared_to()
    with pytest.raises(TypeError):
        ex_pdag.empty().compared_to(37)
    with pytest.raises(TypeError):
        ex_pdag.empty().compared_to('bad arg type')
    with pytest.raises(TypeError):
        ex_pdag.empty().compared_to(ex_sdg.ab())


def test_metrics_pdag_type_error2():  # bad argument type for bayesys
    with pytest.raises(TypeError):
        ex_pdag.empty().compared_to(ex_pdag.empty(), False)
    with pytest.raises(TypeError):
        ex_pdag.empty().compared_to(ex_pdag.empty(), False)
    with pytest.raises(TypeError):
        ex_pdag.empty().compared_to(ex_pdag.empty(), ex_pdag.empty())


def test_metrics_pdag_value_error1():  # bad value for bayesys
    with pytest.raises(ValueError):
        ex_pdag.empty().compared_to(ex_pdag.empty(), 'unsupported')
    with pytest.raises(ValueError):
        ex_pdag.empty().compared_to(ex_pdag.empty(), 'unsupported')
    with pytest.raises(ValueError):
        ex_pdag.empty().compared_to(ex_pdag.empty(), 'bayesys1.5')


def test_metrics_pdag_value_error2():  # different node sets
    with pytest.raises(ValueError):
        ex_pdag.empty().compared_to(ex_pdag.a())
    with pytest.raises(ValueError):
        ex_pdag.empty().compared_to(ex_dag.a())
    with pytest.raises(ValueError):
        ex_pdag.asia().compared_to(ex_pdag.cancer1())


def test_metrics_pdag_empty_ok1(expected):  # compare empty PDAG empty PDAG
    metrics = ex_pdag.empty().compared_to(ex_pdag.empty())
    print('\nempty PDAG compared to empty PDAG:\n{}'.format(metrics))
    assert metrics == expected


def test_metrics_pdag_empty_ok2(expected):  # compare empty PDAG with empty DAG
    metrics = ex_pdag.empty().compared_to(ex_dag.empty())
    print('\nempty PDAG compared to empty DAG:\n{}'.format(metrics))
    assert metrics == expected


def test_metrics_pdag_empty_ok3(expected):  # compare empty DAG with empty PDAG
    metrics = ex_dag.empty().compared_to(ex_pdag.empty())
    print('\nempty DAG compared to empty PDAG:\n{}'.format(metrics))
    assert metrics == expected


def test_metrics_pdag_empty_ok4(expected):  # compare empty DAG with empty PDAG
    metrics = ex_dag.empty().compared_to(ex_dag.empty())
    print('\nempty DAG compared to empty DAG:\n{}'.format(metrics))
    assert metrics == expected

# single node comparisons


def test_metrics_pdag_a_ok1(expected):  # compare "A" PDAG with "A" PDAG
    graph = ex_pdag.a()
    reference = ex_pdag.a()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    assert metrics == expected


def test_metrics_pdag_a_ok2(expected):  # compare "A" DAG with "A" PDAG
    graph = ex_dag.a()
    reference = ex_pdag.a()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    assert metrics == expected

# two node comparisons


def test_metrics_pdag_ab_ok1(expected):  # compare A -> B PDAG  A -> B PDAG
    graph = ex_pdag.ab()
    reference = ex_pdag.ab()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'arc_matched': 1, 'p': 1.0, 'r': 1.0, 'f1': 1.0})
    assert metrics == expected


def test_metrics_pdag_ab_ok2(expected):  # compare A -> B PDAG with A <- B PDAG
    graph = ex_pdag.ab()
    reference = ex_pdag.ba()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'arc_reversed': 1, 'shd': 1, 'p': 0.0, 'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_ab_ok3(expected):  # compare A <- B PDAG  A -> B DAG
    graph = ex_pdag.ba()
    reference = ex_dag.ab()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'arc_reversed': 1, 'shd': 1, 'p': 0.0, 'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_ab_ok4(expected):  # compare A  B PDAG with A -> B PDAG
    graph = ex_pdag.a_b()
    reference = ex_pdag.ab()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'arc_missing': 1, 'shd': 1, 'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_ab_ok5(expected):  # compare A <- B DAG with A  B PDAG
    graph = ex_dag.ba()
    reference = ex_pdag.a_b()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'arc_extra': 1, 'shd': 1, 'p': 0.0})
    assert metrics == expected


def test_metrics_pdag_ab_ok6(expected):  # compare A  B PDAG with A  B PDAG
    graph = ex_pdag.a_b()
    reference = ex_pdag.a_b()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'missing_matched': 1})
    assert metrics == expected


def test_metrics_pdag_ab_ok7(expected):  # compare A - B PDAG with A - B PDAG
    graph = ex_pdag.ab3()
    reference = ex_pdag.ab3()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'edge_matched': 1, 'p': 1.0, 'r': 1.0, 'f1': 1.0})
    assert metrics == expected


def test_metrics_pdag_ab_ok8(expected):  # compare A -> B PDAG with A - B PDAG
    graph = ex_pdag.ab()
    reference = ex_pdag.ab3()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'arc_not_edge': 1, 'shd': 1, 'p': 0.0, 'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_ab_ok9(expected):  # compare A - B PDAG with A -> B DAG
    graph = ex_pdag.ab3()
    reference = ex_dag.ab()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'edge_not_arc': 1, 'shd': 1, 'p': 0.0, 'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_ab_ok10(expected):  # compare A  B DAG with A - B PDAG
    graph = ex_dag.a_b()
    reference = ex_pdag.ab3()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'edge_missing': 1, 'shd': 1, 'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_ab_ok11(expected):  # compare A - B PDAG with A  B PDAG
    graph = ex_pdag.ab3()
    reference = ex_pdag.a_b()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'edge_extra': 1, 'shd': 1, 'p': 0.0})
    assert metrics == expected

#   three node comparisons


def test_metrics_pdag_abc_ok1(expected):  # compare A B C PDAG & A B C PDAG
    graph = ex_pdag.a_b_c()
    reference = ex_pdag.a_b_c()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'missing_matched': 3})
    assert metrics == expected


def test_metrics_pdag_abc_ok2(expected):  # compare A->C B PDAG & A B C PDAG
    graph = ex_pdag.ac_b()
    reference = ex_pdag.a_b_c()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'missing_matched': 2, 'arc_extra': 1, 'shd': 1, 'p': 0.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok3(expected):  # compare A B C PDAG & C->A B PDAG
    graph = ex_pdag.a_b_c()
    reference = ex_pdag.ac_b()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'missing_matched': 2, 'arc_missing': 1, 'shd': 1,
                     'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok4(expected):  # compare C->A B PDAG & C->A B PDAG
    graph = ex_pdag.ac_b()
    reference = ex_pdag.ac_b()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'missing_matched': 2, 'arc_matched': 1,
                     'p': 1.0, 'r': 1.0, 'f1': 1.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok5(expected):  # compare A->B->C & A->B->C PDAGs
    graph = ex_pdag.abc()
    reference = ex_pdag.abc()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'missing_matched': 1, 'arc_matched': 2,
                     'p': 1.0, 'r': 1.0, 'f1': 1.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok6(expected):  # compare A->B->C & A B C PDAGs
    graph = ex_pdag.abc()
    reference = ex_pdag.a_b_c()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'missing_matched': 1, 'arc_extra': 2, 'shd': 2, 'p': 0.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok7(expected):  # compare A B C & A->B->C PDAGs
    graph = ex_pdag.a_b_c()
    reference = ex_pdag.abc()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'missing_matched': 1, 'arc_missing': 2, 'shd': 2,
                     'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok8(expected):  # compare A->B->C & A-B-C PDAGs
    graph = ex_pdag.abc()
    reference = ex_pdag.abc4()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'missing_matched': 1, 'arc_not_edge': 2, 'shd': 2,
                     'p': 0.0, 'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok9(expected):  # compare A-B-C & A->B->C PDAGs
    graph = ex_pdag.abc4()
    reference = ex_pdag.abc()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'missing_matched': 1, 'edge_not_arc': 2, 'shd': 2,
                     'p': 0.0, 'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok10(expected):  # compare A->B->C<-A & A->B->C<-A
    graph = ex_pdag.abc_acyclic()
    reference = ex_pdag.abc_acyclic()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'arc_matched': 3, 'p': 1.0, 'r': 1.0, 'f1': 1.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok11(expected):  # compare A B C & A->B->C<-A
    graph = ex_pdag.a_b_c()
    reference = ex_pdag.abc_acyclic()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'arc_missing': 3, 'shd': 3, 'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok12(expected):  # compare A->B->C<-A & A B C
    graph = ex_pdag.abc_acyclic()
    reference = ex_pdag.a_b_c()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'arc_extra': 3, 'shd': 3, 'p': 0.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok13(expected):  # compare A->B->C<-A & A-B-C
    graph = ex_pdag.abc_acyclic()
    reference = ex_pdag.abc4()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'arc_not_edge': 2, 'arc_extra': 1, 'shd': 3, 'p': 0.0,
                     'r': 0.0})
    assert metrics == expected


def test_metrics_pdag_abc_ok14(expected):  # compare A-B-C & A->B->C<-A
    graph = ex_pdag.abc4()
    reference = ex_pdag.abc_acyclic()
    metrics = graph.compared_to(reference)
    print('\n{}\ncompared to\n{}\n{}\n'.format(graph, reference, metrics))
    expected.update({'edge_not_arc': 2, 'arc_missing': 1, 'shd': 3, 'p': 0.0,
                     'r': 0.0})
    assert metrics == expected
