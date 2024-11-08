
# Naive exhaustive score-based search with no attempt at optimisation

from pandas import DataFrame
from itertools import combinations

from fileio.data import Data
from fileio.oracle import Oracle
from core.graph import DAG, NotDAGError


def exhaustive(data, types=['bic', 'loglik', 'bde'], params={'base': 2},
               normalise=False):
    """
        Do an exhaustive search of all possible DAGs using nodes defined
        in data, and scoring against the data

        :param Data data: data to generate & score graphs from
        :param list types: scores required e.g. ['loglik', 'aic', 'bde']
        :param dict params: parameters for scores e.g. logarithm base
        :param bool normalise: whether to normalise to unconnected graph scores

        :raises TypeError: if bad argument types used
        :raises FileNotFoundError: if non-existent file specified
        :raises FileFormatError: if specified file has invalid format
        :raises ValueError: if data supplied unacceptable

        :return DataFrame: index is DAG string, columns are scores
    """
    if not isinstance(data, Data) or isinstance(data, Oracle):
        raise TypeError('bad arg types for exhaustive')

    if len(data.nodes) > 6:
        raise ValueError('Data must have no more than six variables')

    results = []
    for edges in AllEdgeLists(data.nodes):
        try:
            dag = DAG(list(data.nodes), edges)
            scores = dict(dag.score(data, types, params).sum())
            scores['dag'] = dag.to_string()
            results.append(scores)
        except (NotDAGError):
            pass

    sort_by = types[0] if isinstance(types, list) else types
    results = DataFrame(results).set_index('dag') \
        .sort_values(sort_by, ascending=False)

    if normalise:
        normaliser = results.loc['[' + (']['.join(data.nodes)) + ']']
        results = results.subtract(normaliser) \
            .append(normaliser.rename('normalisation'))

    return results


class AllEdgeLists():
    """
        Iterable over all possible lists of edges for specified nodes

        :param nodes list: nodes used in edges
    """
    def __init__(self, nodes):
        self.nodes = nodes

    def __iter__(self):
        """
            Returns the initialised iterator

            :returns AllEdgeLists: the iterator
        """
        self.edges = list(combinations(self.nodes, 2))
        self.edge_types = [0] * len(self.edges)
        self.edge_types[0] = -1
        return self

    def __next__(self):
        """
            Generate the next list of edges

            :raises StopIteration: when all lists of edge sets returned

            :returns list: next list of edges
        """
        self.edge_types[0] += 1
        for i in range(0, len(self.edge_types)):
            if self.edge_types[i] == 3:
                if i == len(self.edge_types) - 1:
                    raise StopIteration
                self.edge_types[i] = 0
                self.edge_types[i + 1] += 1
            else:
                break
        edges = []
        for edge, type in zip(self.edges, self.edge_types):
            if type == 1:
                edges.append((edge[0], '->', edge[1]))
            elif type == 2:
                edges.append((edge[1], '->', edge[0]))
        return edges
