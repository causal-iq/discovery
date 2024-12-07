
# Test Bayesys metrics

from core.graph import PDAG
from core.metrics import dicts_same, values_same
from fileio.noisy import evaluate_noisy
from fileio.bayesys import read
from fileio.common import TESTDATA_DIR
from experiments.common import reference_bn


def test_core_metrics_bayesys_single_edge():

    expected_results = [['am', 'ar', 'ane', 'fa'],
                        ['ar', 'am', 'ane', 'fa'],
                        ['ena', 'ena', 'em', 'fe'],
                        ['ma', 'ma', 'me', 'mm']]

    expected_metrics = {
        'am':  {'arc_matched': 3, 'arc_reversed': 0, 'edge_not_arc': 0,
                'arc_extra': 0, 'edge_extra': 0, 'edge_matched': 0,
                'arc_not_edge': 0, 'arc_missing': 0, 'edge_missing': 0,
                'missing_matched': 3, 'tp-b': 3, 'tp2-b': 0.0, 'fp-b': 0,
                'tn-b': 3, 'fn-b': 0.0, 'p-b': 1.0, 'r-b': 1.0, 'f1-b': 1.0,
                'shd-b': 0.0, 'ddm': 1.0, 'bsf': 1.0, 'shd': 0, 'p': 1.0,
                'r': 1.0, 'f1': 1.0},
        'ar':  {'arc_matched': 2, 'arc_reversed': 1, 'edge_not_arc': 0,
                'arc_extra': 0, 'edge_extra': 0, 'edge_matched': 0,
                'arc_not_edge': 0, 'arc_missing': 0, 'edge_missing': 0,
                'missing_matched': 3, 'tp-b': 2, 'tp2-b': 1.0, 'fp-b': 0,
                'tn-b': 3, 'fn-b': 0.5, 'p-b': 0.833333, 'r-b': 0.833333,
                'f1-b': 0.833333, 'shd-b': 0.5, 'ddm': 0.666667,
                'bsf': 0.833333, 'shd': 1, 'p': 0.666667, 'r': 0.666667,
                'f1': 0.666667},
        'ena': {'arc_matched': 2, 'arc_reversed': 0, 'edge_not_arc': 1,
                'arc_extra': 0, 'edge_extra': 0, 'edge_matched': 0,
                'arc_not_edge': 0, 'arc_missing': 0, 'edge_missing': 0,
                'missing_matched': 3, 'tp-b': 2, 'tp2-b': 1.0, 'fp-b': 0,
                'tn-b': 3, 'fn-b': 0.5, 'p-b': 0.833333, 'r-b': 0.833333,
                'f1-b': 0.833333, 'shd-b': 0.5, 'ddm': 0.666667,
                'bsf': 0.833333, 'shd': 1, 'p': 0.666667, 'r': 0.666667,
                'f1': 0.666667},
        'ma':  {'arc_matched': 2, 'arc_reversed': 0, 'edge_not_arc': 0,
                'arc_extra': 0, 'edge_extra': 0, 'edge_matched': 0,
                'arc_not_edge': 0, 'arc_missing': 1, 'edge_missing': 0,
                'missing_matched': 3, 'tp-b': 2, 'tp2-b': 0.0, 'fp-b': 0,
                'tn-b': 3, 'fn-b': 1.0, 'p-b': 1.0, 'r-b': 0.666667,
                'f1-b': 0.8, 'shd-b': 1.0, 'ddm': 0.333333, 'bsf': 0.666667,
                'shd': 1, 'p': 1, 'r': 0.666667, 'f1': 0.8},
        'me':  {'arc_matched': 2, 'arc_reversed': 0, 'edge_not_arc': 0,
                'arc_extra': 0, 'edge_extra': 0, 'edge_matched': 0,
                'arc_not_edge': 0, 'arc_missing': 0, 'edge_missing': 1,
                'missing_matched': 3, 'tp-b': 2, 'tp2-b': 0.0, 'fp-b': 0,
                'tn-b': 3, 'fn-b': 1.0, 'p-b': 1.0, 'r-b': 0.666667,
                'f1-b': 0.8, 'shd-b': 1.0, 'ddm': 0.333333, 'bsf': 0.666667,
                'shd': 1, 'p': 1.0, 'r': 0.666667, 'f1': 0.8},
        'ane': {'arc_matched': 2, 'arc_reversed': 0, 'edge_not_arc': 0,
                'arc_extra': 0, 'edge_extra': 0, 'edge_matched': 0,
                'arc_not_edge': 1, 'arc_missing': 0, 'edge_missing': 0,
                'missing_matched': 3, 'tp-b': 3, 'tp2-b': 0.0, 'fp-b': 0,
                'tn-b': 3, 'fn-b': 0.0, 'p-b': 1.0, 'r-b': 1.0, 'f1-b': 1.0,
                'shd-b': 0.0, 'ddm': 1.0, 'bsf': 1.0, 'shd': 1, 'p': 0.666667,
                'r': 0.666667, 'f1': 0.666667},
        'em':  {'arc_matched': 2, 'arc_reversed': 0, 'edge_not_arc': 0,
                'arc_not_edge': 0, 'edge_matched': 1, 'arc_extra': 0,
                'edge_extra': 0, 'arc_missing': 0, 'edge_missing': 0,
                'missing_matched': 3, 'shd': 0, 'p': 1.0, 'r': 1.0, 'f1': 1.0,
                'tp-b': 3.0, 'tp2-b': 0.0, 'fp-b': 0.0, 'tn-b': 3.0,
                'fn-b': 0.0, 'p-b': 1.0, 'r-b': 1.0, 'f1-b': 1.0, 'shd-b': 0.0,
                'ddm': 1.0, 'bsf': 1.0},
        'fa':  {'arc_matched': 2, 'arc_reversed': 0, 'edge_not_arc': 0,
                'arc_extra': 1, 'edge_extra': 0, 'edge_matched': 0,
                'arc_not_edge': 0, 'arc_missing': 0, 'edge_missing': 0,
                'missing_matched': 3, 'tp-b': 2, 'tp2-b': 0.0, 'fp-b': 1.0,
                'tn-b': 3, 'fn-b': 0.0, 'p-b': 0.666667, 'r-b': 1.0,
                'f1-b': 0.8, 'shd-b': 1.0, 'ddm': 0.5, 'bsf': 0.75,
                'shd': 1, 'p': 0.666667, 'r': 1.0, 'f1': 0.8},
        'fe':  {'arc_matched': 2, 'arc_reversed': 0, 'edge_not_arc': 0,
                'arc_extra': 0, 'edge_extra': 1, 'edge_matched': 0,
                'arc_not_edge': 0, 'arc_missing': 0, 'edge_missing': 0,
                'missing_matched': 3, 'tp-b': 2, 'tp2-b': 0.0, 'fp-b': 1.0,
                'tn-b': 3, 'fn-b': 0.0, 'p-b': 0.666667, 'r-b': 1.0,
                'f1-b': 0.8, 'shd-b': 1.0, 'ddm': 0.5, 'bsf': 0.75, 'shd': 1,
                'p': 0.666667, 'r': 1.0, 'f1': 0.8},
        'mm':  {'arc_matched': 2, 'arc_reversed': 0, 'edge_not_arc': 0,
                'arc_extra': 0, 'edge_extra': 0, 'edge_matched': 0,
                'arc_not_edge': 0, 'arc_missing': 0, 'edge_missing': 0,
                'missing_matched': 4, 'tp-b': 2, 'tp2-b': 0.0, 'fp-b': 0,
                'tn-b': 4, 'fn-b': 0.0, 'p-b': 1.0, 'r-b': 1.0, 'f1-b': 1.0,
                'shd-b': 0.0, 'ddm': 1.0, 'bsf': 1.0, 'shd': 0, 'p': 1.0,
                'r': 1.0, 'f1': 1.0}
        }

    def _edges(bc_type):
        edges = [('A', '->', 'B'), ('C', '->', 'D')]
        if bc_type == '<-':
            edges.append(('C', '->', 'B'))
        elif bc_type is not None:
            edges.append(('B', bc_type, 'C'))
        return edges

    nodes = ['A', 'B', 'C', 'D']
    edge_types = ['->', '<-', '-', None]
    for ref_idx, ref_type in enumerate(edge_types):
        ref_graph = PDAG(nodes, _edges(ref_type))
        for idx, type in enumerate(edge_types):
            graph = PDAG(nodes, _edges(type))
            metrics = graph.compared_to(ref_graph, bayesys='v1.5+')
            expected = dict(expected_metrics[expected_results[idx][ref_idx]])
            print(expected_results[idx][ref_idx])
            print(ref_graph)
            print(graph)
            print(metrics)
            assert dicts_same(expected, metrics, sf=6) is True


def test_core_metrics_bayesys_noisy():
    print('\nstart test')
    filter = {'noise': 'N'}
    evaluate_noisy(TESTDATA_DIR + '/noisy/Graphs learned',
                   TESTDATA_DIR + '/noisy/Graphs true',
                   TESTDATA_DIR + '/noisy/Results',
                   strict=False, warnings=True, filter=filter)


def test_core_metrics_bayesys_dhs():
    d8atr_fges = read(TESTDATA_DIR + '/dhs/d8atr/d8atr-fges.csv')
    d8atr_fges3 = read(TESTDATA_DIR + '/dhs/d8atr/d8atr-fges3.csv')
    assert values_same(d8atr_fges.compared_to(d8atr_fges3,
                                              bayesys='v1.5+')['bsf'], 0.541)
    assert values_same(d8atr_fges3.compared_to(d8atr_fges,
                                               bayesys='v1.5+')['bsf'], 0.252)


def test_core_metrics_bayesys_sachs():
    dag = (reference_bn('sachs')[0]).dag
    print('\n\n{}'.format(dag))
    pdag = PDAG.fromDAG(dag)
    print(pdag)

    metrics = dag.compared_to(dag,  bayesys='v1.5+')
    print('\nDAG metrics are:\n{}'.format(metrics))
    assert metrics == \
        {'arc_matched': 17, 'arc_reversed': 0, 'edge_not_arc': 0,
         'arc_not_edge': 0, 'edge_matched': 0, 'arc_extra': 0, 'edge_extra': 0,
         'arc_missing': 0, 'edge_missing': 0, 'missing_matched': 38, 'shd': 0,
         'p': 1.0, 'r': 1.0, 'f1': 1.0, 'tp-b': 17.0, 'tp2-b': 0.0,
         'fp-b': 0.0, 'tn-b': 38.0, 'fn-b': 0.0, 'p-b': 1.0, 'r-b': 1.0,
         'f1-b': 1.0, 'shd-b': 0.0, 'ddm': 1.0, 'bsf': 1.0}

    metrics = pdag.compared_to(pdag,  bayesys='v1.5+')
    print('\nPDAG metrics are:\n{}'.format(metrics))
    assert metrics == \
        {'arc_matched': 0, 'arc_reversed': 0, 'edge_not_arc': 0,
         'arc_not_edge': 0, 'edge_matched': 17, 'arc_extra': 0,
         'edge_extra': 0, 'arc_missing': 0, 'edge_missing': 0,
         'missing_matched': 38, 'shd': 0, 'p': 1.0, 'r': 1.0, 'f1': 1.0,
         'tp-b': 17.0, 'tp2-b': 0.0, 'fp-b': 0.0, 'tn-b': 38.0, 'fn-b': 0.0,
         'p-b': 1.0, 'r-b': 1.0, 'f1-b': 1.0, 'shd-b': 0.0, 'ddm': 1.0,
         'bsf': 1.0}
