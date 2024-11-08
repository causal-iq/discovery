
# Concrete subclass of Data which implements an oracle data source

from numpy import float64

from fileio.data import Data
from fileio.common import VariableType


class Oracle(Data):
    def __init__(self, bn):
        if type(bn).__name__ != 'BN':
            raise TypeError('Oracle() bad arg type')

        self.bn = bn
        self.nodes = tuple(bn.dag.nodes)
        self.order = tuple(i for i in range(len(self.nodes)))
        self.ext_to_orig = {n: n for n in self.nodes}
        self.orig_to_ext = {n: n for n in self.nodes}
        self.node_types = {n: VariableType.CATEGORY
                           if self.bn.cnds[n].__class__.__name__ == 'CPT'
                           else VariableType.FLOAT32 for n in self.nodes}
        self._set_dstype()
        self.N = 1

    def set_N(self, N, seed=None):
        """
            Set current working sample size.

            :param int N: current working sample size
            :param int/None seed: seed for row order randomisation if reqd.

            :raises TypeError: if bad argument type
            :raises ValueError: if bad argument value
        """
        if not isinstance(N, int) or isinstance(N, bool) or seed is not None:
            raise TypeError('Data.set_N() bad arg type')

        if N < 1:
            raise ValueError('Data.set_N() bad arg value')

        self.N = N

    def marginals(self, node, parents, values_reqd=False):
        """
            Return marginal counts for a node and its parents.

            :param str node: node for which marginals required.
            :param dict parents: {node: parents} parents of non-orphan nodes
            :param bool values_reqd: whether parent and child values required

            :raises TypeError: for bad argument types

            :returns tuple: of counts, and optionally, values:
                            - ndarray counts: 2D, rows=child, cols=parents
                            - int maxcol: maximum number of parental values
                            - tuple rowval: child values for each row
                            - tuple colval: parent combo (dict) for each col
        """
        if (not isinstance(node, str) or not isinstance(parents, dict)
                or not all([isinstance(p, list) for p in parents.values()])
                or not isinstance(values_reqd, bool)):
            raise TypeError('Oracle.marginals() bad arg type')

        # obtain marginals as a DataFrame

        nodes = [node] + parents[node] if node in parents else [node]
        marginals = self.bn.marginals(nodes).apply(lambda x: self.N * x)

        # Convert DataFrame to NumPy format

        counts = marginals.to_numpy(dtype=float64, copy=True)
        maxcol = len(marginals.columns)
        rowval = None
        colval = None
        if values_reqd is True:
            rowval = tuple(marginals.index)
            if node in parents:
                colval = tuple(dict(zip(marginals.columns.names,
                                        (col,) if isinstance(col, str)
                                        else col))
                               for col in marginals.columns)
        marginals = None

        return (counts, maxcol, rowval, colval)

    def values(self, nodes):
        """
            Return the (float) values for the specified set of nodes.

            :param tuple nodes: nodes for which data required

            :raises TypeError: always raised as not implemented for Oracle
        """
        raise TypeError('Oracle.values() not implemented')

    def randomise_names(self, seed):
        """
            Randomises the node names that the learning algorithm uses
            (so sensitivity to these names can be assessed).

            :param int seed: randomisation seed

            :raises NotImplementedError: always as not implemented for subclass
        """
        raise NotImplementedError('Data.randomise_names() n/a for Oracle')

    def as_df(self):
        """
            Return the data as a Pandas dataframe with current sample size
            and column order.

            :returns DataFrame: data as Pandas
        """
        raise NotImplementedError('Data.df() n/a for Oracle')
