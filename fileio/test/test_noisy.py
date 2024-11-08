
import fileio.noisy as noisy
from fileio.common import FileFormatError, TESTDATA_DIR
import pytest
import testdata.example_dags as dag

TEST_CASE = {'algorithm': 'HC', 'network': 'ASIA', 'size': '100', 'noise': 'N'}


def test_get_true_file_invalid_arg1():
    with pytest.raises(TypeError):
        noisy._get_true_graph()


def test_get_true_file_invalid_arg2():
    with pytest.raises(TypeError):
        noisy._get_true_graph(1)


def test_get_true_file_invalid_arg3():
    with pytest.raises(TypeError):
        noisy._get_true_graph({})


def test_get_true_file_invalid_arg4():
    with pytest.raises(TypeError):
        noisy._get_true_graph(TEST_CASE)


def test_get_true_file_invalid_arg5():
    with pytest.raises(TypeError):
        noisy._get_true_graph(TEST_CASE, 3)


def test_get_true_graph():
    graph, _ = noisy._get_true_graph(TEST_CASE,
                                     TESTDATA_DIR + '/noisy/Graphs true')
    dag.asia(graph)


def test_get_results_file_bad_args():
    with pytest.raises(TypeError):
        noisy._get_recorded_metrics()
    with pytest.raises(TypeError):
        noisy._get_recorded_metrics('a', 2)
    with pytest.raises(TypeError):
        noisy._get_recorded_metrics('a', {'wrong': 1})
    with pytest.raises(TypeError):
        noisy._get_recorded_metrics('nonexistent',
                                    {'noise': '?!', 'algorithm': 'HC',
                                     'network': 'ASIA', 'size': '1'})


def test_get_results_file_bad_path():
    with pytest.raises(FileNotFoundError):
        noisy._get_recorded_metrics('nonexistent',
                                    {'noise': 'N', 'algorithm': 'GS',
                                     'network': 'ASIA', 'size': '1'})


def test_get_results_file_bad_case():
    with pytest.raises(FileFormatError):
        noisy._get_recorded_metrics(TESTDATA_DIR + '/noisy/Results',
                                    {'noise': 'N', 'algorithm': 'GS',
                                     'network': 'WRONG', 'size': '1'})
    with pytest.raises(FileFormatError):
        noisy._get_recorded_metrics(TESTDATA_DIR + '/noisy/Results',
                                    {'noise': 'N', 'algorithm': 'GS',
                                     'network': 'SPORTS', 'size': '?'})


def test_get_results_file_ok():
    assert noisy._get_recorded_metrics(TESTDATA_DIR + '/noisy/Results',
                                       {'noise': 'N', 'algorithm': 'HC',
                                        'network': 'ASIA', 'size': '1'}) == \
        {'edges': 7.0, 'fragments': 2.0, 'p-b': 1.0, 'r-b': 0.875,
         'f1-b': 0.933, 'shd-b': 1.0, 'bsf': 0.875}
    assert noisy._get_recorded_metrics(TESTDATA_DIR + '/noisy/Results',
                                       {'noise': 'N', 'algorithm': 'GS',
                                        'network': 'FORMED',
                                        'size': '1000'}) == \
        {'edges': 63.0, 'fragments': 31.0, 'p-b': 0.651, 'r-b': 0.297,
         'f1-b': 0.408, 'shd-b': 102.0, 'bsf': 0.296}
