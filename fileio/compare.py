
from os import listdir
from os.path import sep

from core.common import BAYESYS_VERSIONS
from fileio.common import is_valid_path
from fileio.bayesys import read


def compare_all(dir, metric, bayesys):
    """
        Compare graphs (every one with each other) in specified folder using
        specified metric

        :param [dir]: directory holding graphs to be compared
        :type [dir]: str
        :param [metric]: metric used for comparison e.g. 'f1', 'bsf'
        :type [metric]: str
        :param [bayesys]: score & comparison bayesys to use e.g. 'bnlearn'
        :type [bayesys]: str

        :raises [TypeError]: if args not of correct type
        :raises [ValueError]: if args have invalid values

        :returns: mean of metric
        :rtype: float
    """
    if not isinstance(dir, str) or not isinstance(metric, str) \
            or not isinstance(bayesys, str):
        raise TypeError('Bad arg type for compare_all')

    if bayesys not in BAYESYS_VERSIONS:
        raise ValueError('Bad bayesys value for compare_all')

    is_valid_path(dir, is_file=False)

    # generate graphs from all the files

    graphs = {f: read(dir + sep + f) for f in listdir(dir)}

    # ensure all graphs have same set of nodes unless v1.3 bayesys

    if bayesys != 'v1.3':
        all_nodes = None
        for _, graph in graphs.items():
            if graph.number_components() == 1:
                all_nodes = graph.nodes
                break
        if all_nodes is None:
            raise ValueError('No single fragment graphs')
        graphs = {f: read(dir + sep + f, all_nodes=all_nodes)
                  for f in listdir(dir)}

    # compare each graph with every other graph

    num_values = 0
    total_metric = 0.0
    for file1, graph1 in graphs.items():
        for file2, graph2 in graphs.items():
            if file1 == file2:
                continue
            metrics = graph1.compared_to(graph2, bayesys=bayesys)
            if metric not in metrics:
                raise ValueError('Bad metric value for compare_all')
            print('{} of {} compared to {} is {:.3f}'
                  .format(metric, file1, file2, metrics[metric]))
            num_values += 1
            total_metric += metrics[metric]

    mean = total_metric / num_values
    print('\nMean value for {} is {:.3f}'.format(metric, mean))

    return mean
