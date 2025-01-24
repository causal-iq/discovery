#
#   Bayesian Network class
#

from pandas import DataFrame, MultiIndex

from core.common import ln, set_random_seed
from core.graph import DAG
from core.score import bn_score
from core.cnd import CND
from core.cpt import CPT, NodeValueCombinations
from core.lingauss import LinGauss
import fileio.dsc as dsc
import fileio.xdsl as xdsl
from fileio.pandas import Pandas


class BN():
    """
        Base class for Bayesian Networks - which have a DAG and
        an associated probability distribution defined by CPTs

        :param DAG dag: DAG for the Bayesian Network
        :param dict cnd_specs: specification of each conditional node dist.
        :param dict estimated_pmfs: number of PMFs that had to be estimated
                                    for each node

        :ivar DAG dag: BN's DAG
        :ivar dict cnds: conditional distributions for each node {node: CND}
        :ivar int free_params: total number of free parameters in BN
        :ivar list estimated_pmfs: number of estimated pmfs for each node

        :raises TypeError: if arguments have invalid types
        :raises ValueError: if arguments have invalid values
    """
    def __init__(self, dag, cnd_specs, estimated_pmfs={}):

        if not isinstance(dag, DAG) or not isinstance(cnd_specs, dict):
            raise TypeError('BN() bad arg type      ')

        self.dag = dag

        if sorted(self.dag.nodes) != sorted(list(cnd_specs.keys())):
            raise ValueError('Different nodes in DAG and cnd_specs')

        self.cnds = {}
        self.free_params = 0
        self.estimated_pmfs = estimated_pmfs
        for node in self.dag.nodes:
            self.cnds[node] = (cnd_specs[node][0])(cnd_specs[node][1])
            self.free_params += self.cnds[node].free_params

        self.cached_marginals = MarginalsCache()

        CND.validate_cnds(self.dag.nodes, self.cnds, self.dag.parents)

    @classmethod
    def fit(self, dag, data):
        """
            Alternative instantiation of BN using data to
            implicitly define the conditional probability data

            :param DAG dag: DAG for the Bayesian Network
            :param Data data: data to fit CPTs to

            :raises TypeError: if arguments have invalid types
            :raises ValueError: if arguments have invalid values
        """
        if not isinstance(dag, DAG) or not isinstance(data, Pandas):
            raise TypeError('bn.fit() arguments have invalid types')

        if sorted(list(data.nodes)) != dag.nodes:
            raise ValueError('data empty, col mismatch or missing data')

        if any([(len(data.sample[c].unique()) == 1) for c in data.nodes]):
            raise ValueError('Some variables have only one value')

        cnd_specs = {}
        estimated_pmfs = {}
        for node in dag.nodes:
            parents = tuple(dag.parents[node]) if node in dag.parents else None
            cnd = CPT if data.node_types[node] == 'category' else LinGauss
            cnd_specs[node], estimated_pmfs[node] = cnd.fit(node, parents,
                                                            data)
        estimated_pmfs = {n: c for n, c in estimated_pmfs.items()
                          if c is not None and c > 0}

        return self(dag, cnd_specs, estimated_pmfs)

    @classmethod
    def read(self, path, correct=False):
        """
            Instantiate BN from a DSC or XDSL format file specification

            :param str path: path to DSC/XDSL file
            :param bool correct: whether to correct probabilities that do not
                                 sum to 1 (XDSL files only)

            :raises TypeError: if path is not a string
            :raises ValueError: if path suffix is not "dsc" or "xdsl"
            :raises FileNotFoundError: if file does not exist
            :raises FileFormatError: if file contents not valid

            :returns BN: Bayesian Network specified in file
        """
        if not isinstance(path, str) or not isinstance(correct, bool):
            raise TypeError('BN.read() bad arg type')

        suffix = path.split('.')[-1]
        if suffix.lower() == 'dsc':
            nodes, edges, cnd_specs = dsc.read(path)
        elif suffix.lower() == 'xdsl':
            nodes, edges, cnd_specs = xdsl.read(path, correct)
        else:
            raise ValueError('BN.read() invalid file suffix')

        return self(DAG(nodes, edges), cnd_specs)

    def rename(self, name_map):
        """
            Nodes renamed in place according to name map.

            :param dict name_map: name mapping {name: new name}

            :raises TypeError: with bad arg type
            :raises ValueError: with bad arg values e.g. unknown node names
        """
        def _map(odict):  # rename and re-sort dict keys
            ndict = {name_map[k] if k in name_map else k: v
                     for k, v in odict.items()}
            return {k: ndict[k] for k in sorted(ndict)}

        # rename variables in DAG - which checks validity of name_map

        old_names = self.dag.nodes
        self.dag.rename(name_map)

        # Generate CND specifications with renamed nodes

        cnd_specs = {}
        for node in old_names:
            cnd = self.cnds[node]
            cnd_specs.update({node: (type(cnd), cnd.to_spec(name_map))})

        # Rename and re-order the keys of the dict of {node: cnd}

        cnd_specs = _map(cnd_specs)

        # re-instantiate BN with new DAG and CPT data

        self = self.__init__(self.dag, cnd_specs)

    def write(self, path):
        """
            Write BN to a DSC or XDSL format file

            :param str path: path to file

            :raises ValueError: if suffix not ".dsc" or ".xdsl"
            :raises FileNotFoundError: if file location nonexistent
        """
        suffix = path.split('.')[-1].lower()
        if suffix == 'dsc':
            dsc.write(self, path)
        elif suffix == 'xdsl':
            xdsl.write(self, path, genie=True)
        else:
            raise ValueError('BN.read() invalid file suffix')

    def score(self, N, types, params={}):
        """
            Generate 'oracle' score for BN using specified dataset size
            and CPT entries.

            :param int N: dataset size to base score on
            :param str/list types: type(s) of score e.g. 'loglik', 'bic', 'bde'
            :param dict params: score parameters e.g. log base

            :raises TypeError: if arguments have bad type
            :raises ValueError: if arguments have bad values

            :returns DataFrame: requested score types (col) for each node (row)
        """
        return bn_score(self, N, types, params)

    def global_distribution(self):
        """
            Generate the global probability distribution for the BN

            :returns DataFrame: global distribution in descending probability
                                (and then by ascending values)
        """

        # Generate possible values at every node {node: [poss values]}

        node_values = {n: c.node_values() for n, c in self.cnds.items()}

        # Loop over all possible combinations of node values (i.e. a "case")
        # and collect the probability of each one

        values = {n: [] for n in self.dag.nodes}
        probs = []
        for case in NodeValueCombinations(node_values):
            for node, value in case.items():
                values[node].append(value)
            lnprob = self.lnprob_case(case)
            probs.append(0.0 if lnprob is None else 10 ** lnprob)

        # return DataFrame with correct dtypes and sorted by descending
        # probability, and then ascending value order

        return DataFrame(values, dtype='category') \
            .join(DataFrame({'': probs}, dtype='float64')) \
            .sort_values([''] + self.dag.nodes, ignore_index=True,
                         ascending=[False] + [True] * len(self.dag.nodes))

    def marginal_distribution(self, node, parents=None):
        """
            Generate a marginal probability distribution for a specified node
            and its parents in same format returned by Panda crosstab function.

            :param str node: node for which distribution required
            :param list parents: parents of node

            :returns DataFrame: marginal distribution with parental value
                                combos as columns, and node values as rows.
        """
        if not isinstance(node, str) or \
                (not isinstance(parents, list) and parents is not None) or \
                (parents is not None and
                 any([not isinstance(p, str) for p in parents])):
            raise TypeError('marginal_distribution bad arg types')

        if node not in self.dag.nodes or \
                (parents is not None and (
                 node in parents
                 or any([p not in self.dag.nodes for p in parents])
                 or len(parents) != len(set(parents)))):
            raise ValueError('marginal_distribution bad node value')

        # Generate possible values at every node {node: [poss values]}

        node_values = {n: c.node_values() for n, c in self.cnds.items()}

        # Loop through every possible combination of all variable values,
        # get its probability of occurrence and add it in to the running
        # marginal probability for that value of parental values and node value

        marginals = {}
        for case in NodeValueCombinations(node_values):
            lnprob = self.lnprob_case(case)
            if lnprob is None:
                continue  # ignore cases with zero possibility
            node_value = case[node]
            pvs = frozenset([(p, case[p])
                            for p in parents]) if parents else node
            if pvs not in marginals:
                marginals[pvs] = {v: 0.0 for v in node_values[node]}
            marginals[pvs][node_value] += 10 ** lnprob

        # reconfigure the marginal probabilities into the Dataframe format
        # produced by Pandas crosstab so compatible with rest of code base

        if parents is None:
            return DataFrame([[v, marginals[node][v]]
                              for v in node_values[node]],
                             columns=[node, '']).set_index(node)

        columns = []  # list of tuples of each parental value combo
        probs = []  # marg. probs for each pvs for each node value
        for pvs, pmf in marginals.items():
            pvs = {t[0]: t[1] for t in pvs}
            columns.append(tuple([pvs[p] for p in parents]))
            probs.append([pmf[v] for v in node_values[node]])
        return DataFrame(data=[list(i) for i in zip(*probs)],  # transpose
                         columns=MultiIndex.from_tuples(columns,
                                                        names=parents),
                         index=node_values[node]).rename_axis(node)

    def _dist(self, dist, required, node, cpt):
        """
            Merge a node's CPT into marginal distribution.

            :param list dist: current marginal distribution, format is
                              [({n: v, ....}, pr), ...]
            :param set required: nodes to include in distribution
            :param str node: node being added to distribution
            :param CPT cpt: CPT for node being added

            :returns list: updated marginal distribution with node in
        """
        # print('_dist: dist={}, required={}, node={}'
        #       .format(dist, required, node))
        parents = cpt.parents()
        result = {}
        for entry in dist:  # Loop over entries in current marginal

            # extract parental values for this entry

            parent_values = None if parents is None else \
                {n: v for n, v in entry[0].items() if n in parents}

            # loop over items in PMF for this entry's parental values
            # getting this node's value and associated probability

            for value, prob in cpt.cdist(parent_values).items():

                # construct a new marginal entry key which contains
                # all required nodes including current node

                values = frozenset({(n, v) for n, v in entry[0].items()
                                    if n in required} | {(node, value)})

                # Accumulate the probabilities for these new marginal
                # entries - the new probability is the old marginal entry
                # probability x the probability of this node's PMF entry

                if values not in result:
                    result.update({values: 0.0})
                result[values] += prob * entry[1]

        result = [({e[0]: e[1] for e in v}, p) for v, p in result.items()]
        return result

    def marginals(self, nodes):
        """
            Return marginal distribution for specified nodes

            :param list nodes: nodes for which marginal distribution required.

            :raises TypeError: if arguments have bad type
            :raises ValueError: if arguments contain bad values

            :returns DataFrame: marginal distribution in same format returned
                                by Pandas crosstab function.
        """
        if not isinstance(nodes, list):
            raise TypeError('bn.marginal_distribution() bad arg types')

        if not len(nodes) or len(nodes) != len(set(nodes)) \
                or not all([n in self.dag.nodes for n in nodes]):
            raise ValueError('bn.marginal_distribution() bad arg values')

        # Construct a topological ordering of all the nodes

        nodes = list(nodes)  # nodes we are interested in

        dist = self.cached_marginals.get(nodes)
        # print('Cache {} for {}'
        #       .format('MISS' if dist is None else 'HIT', nodes))
        if not dist:

            dag = self.dag
            parents = {n: dag.parents[n] if n in dag.parents else []
                       for n in dag.nodes}
            order = [n for g in dag.partial_order(parents) for n in g]

            # Remove entries in order which are not ancestors of required nodes

            ancestors = set()
            children = {n: set() for n in dag.nodes}  # each node's children
            for i in range(len(order) - 1, -1, -1):  # work up the order
                node = order[i]
                if node in nodes or node in ancestors:

                    # node is in reqd distribution, or is an ancestor of one,
                    # so add it and its parents to ancestors

                    ancestors = ancestors | {node} | set(parents[node])
                    for p in parents[node]:
                        children[p] = children[p] | {node}
                else:
                    #   node is not required, nor an ancestor so disregard it
                    order.pop(i)
                    children.pop(node)

            # print('Order: {}, children: {}'.format(order, children))

            # Now move forward through order building up distribution but
            # marginalising out variables not needed further down the order

            required = set()    # running set of nodes of interested
            dist = [({}, 1.0)]  # marginal distribution built here

            for node in order:  # go down the order
                required = required | {node}

                # children updated to include only those further down order

                children = {n: c - {node} for n, c in children.items()}

                # marginalise are those nodes we wish to marginalise out here,
                # and remove them from set of nodes of interest

                marginalise = {n for n in required
                               if n not in nodes and len(children[n]) == 0}
                required -= marginalise

                # update the marginal distribution with current node pmfs,
                # but marginalising out those nodes no longer required

                dist = self._dist(dist, required, node, self.cnds[node])

                self.cached_marginals.put(dist)  # cache entries down order

        if len(nodes) == 1:
            dist = DataFrame(sorted([[(e[0][(nodes[0])]), e[1]]
                                     for e in dist]),
                             columns=[nodes[0], '']).set_index(nodes[0])
        else:
            index_node = nodes.pop(0)
            node_values = self.cnds[index_node].node_values()
            row_index = {node_values[i]: i for i in range(len(node_values))}
            matrix = {}
            for entry in dist:
                values = tuple([entry[0][n] for n in nodes])
                if values not in matrix:
                    matrix[values] = [None] * len(node_values)
                matrix[values][row_index[entry[0][index_node]]] = entry[1]
            columns = [k for k in matrix.keys()]
            probs = [matrix[k] for k in matrix.keys()]
            dist = DataFrame(data=[list(i) for i in zip(*probs)],  # transpose
                             columns=MultiIndex.from_tuples(columns,
                                                            names=nodes),
                             index=node_values).rename_axis(index_node)
        return dist

    def lnprob_case(self, case_values, base=10):
        """
            Return log of probability of set of node values (case) occuring

            :param dict case_values: value for each node {node: value}
            :param int/str base: logarithm base to use - 2, 10 or 'e'

            :raises TypeError: if arguments wrong type
            :raises ValueError: if arguments have invalid values

            :returns float|None: log of probability of case occuring, or None
                                 if case has zero probability
        """
        if (not isinstance(base, int) and not isinstance(base, str)) \
                or not isinstance(case_values, dict):
            raise TypeError('bad arg type for lnprob_case')

        if sorted(list(case_values.keys())) != self.dag.nodes \
                or base not in [2, 10, 'e']:
            raise ValueError('bad arg values for lnprob_case')

        lnprob = 0.0
        for node in self.dag.ordered_nodes():
            pvs = None if node not in self.dag.parents \
                else {p: case_values[p] for p in self.dag.parents[node]}
            try:
                prob = self.cnds[node].cdist(pvs)[case_values[node]]
                if prob == 0.0:
                    return None
                lnprob += ln(prob, base)
            except (KeyError):
                raise ValueError('Bad case value in lnprob_case')

        return float(lnprob)

    def generate_cases(self, n, outfile=None, pseudo=True):
        """
            Generate specified number of random data cases for this BN

            :param int n: number of cases to generate
            :param str outfile: name of file to write instance to
            :param bool pseudo: if pseudo-random (i.e. repeatable cases) to be
                                produced, otherwise truly random

            :raises TypeError: if arguments not of correct type
            :raises ValueError: if invalid number of rows requested
            :raises FileNotFoundError: if outfile in nonexistent folder

            :returns DataFrame: of random data cases
        """
        if not isinstance(n, int) or isinstance(n, bool) \
                or (outfile is not None and not isinstance(outfile, str)) \
                or not isinstance(pseudo, bool):
            raise TypeError('generate_cases called with bad arg types')

        if n < 1 or n > 100000000:
            raise ValueError('generate_cases called with bad n')

        set_random_seed(1234 if pseudo else None)  # set pseudo-random or not

        cases = {node: [] for node in self.dag.nodes}
        for count in range(0, n):
            for node in self.dag.ordered_nodes():
                pvs = None if node not in self.dag.parents \
                    else {p: cases[p][count] for p in self.dag.parents[node]}
                cases[node].append(self.cnds[node].random_value(pvs))

        dtype = {n: 'category' if isinstance(cnd, CPT) else 'float32'
                 for n, cnd in self.cnds.items()}
        cases = DataFrame(cases).astype(dtype=dtype)

        if outfile is not None:
            Pandas(df=cases).write(outfile)

        return cases

    def remove_single_valued(self, data):
        """
            Remove nodes from BN that just contain a single value in data.
            This can be useful when syntheticaly generated data is used for
            testing or structure learning as it may have variables which only
            contain one value.

            :param DataFrame data: data for BN which may contain single-valued
                                   or zero-valued variables which should be
                                   removed.

            :raises TypeError: if data is not a Pandas dataframe.
            :raises ValueError: if less than 2 multi-valued variables

            :returns tuple: (BN, DataFrame, list) BN and data with offending
                            variables removed, and list of removed variables.
        """
        if not isinstance(data, DataFrame):
            raise TypeError('BN.remove_single_valued() bad arg type')

        remove = sorted([col for col, count in data.nunique().items()
                         if count < 2])
        if len(data.columns) - len(remove) < 2:
            raise ValueError('BN.remove_single_valued() - <2 multi-valued')

        if not len(remove):  # nothing to do
            return (self, data, remove)

        # Drop single-valued variables from nodes, edges and data

        data = data.drop(labels=remove, axis='columns').astype('category')
        nodes = list(set(self.dag.nodes) - set(remove))
        edges = [(e[0], t.value[3], e[1]) for e, t in self.dag.edges.items()
                 if e[0] not in remove and e[1] not in remove]

        data = Pandas(df=data)
        return (BN.fit(DAG(nodes, edges), data), data.sample, remove)

    def __eq__(self, other):
        """
            Compare another BN with this one

            :param BN other: the other BN to compare with this one

            :returns bool: True, if other BN is same as this one
        """
        return isinstance(other, BN) and self.dag.nodes == other.dag.nodes \
            and self.dag.edges == other.dag.edges and self.cnds == other.cnds


class MarginalsCache():

    MAX_NODES = 3  # limit on number of nodes for cached marginals

    def __init__(self):
        self.cache = {}
        self.stats = {'get.ok': 0, 'get.miss': 0, 'get.big': 0,
                      'put.ok': 0, 'put.dupl': 0, 'put.big': 0}

    def get(self, nodes):
        if len(nodes) > self.MAX_NODES:
            self.stats['get.big'] += 1
            return None
        key = frozenset(nodes)
        if key in self.cache:
            self.stats['get.ok'] += 1
            # print('Cache hit for {}'.format(nodes))
            return self.cache[key]
        else:
            self.stats['get.miss'] += 1
            return None

    def put(self, dist):
        nodes = set(dist[0][0].keys())
        if len(nodes) > self.MAX_NODES:
            self.stats['put.big'] += 1
            return None
        key = frozenset(nodes)
        if key in self.cache:
            self.stats['put.dupl'] += 1
            return None
        else:
            self.stats['put.ok'] += 1
            self.cache.update({key: dist})
            # print('Cache put for {}'.format(nodes))
            return None

    def __str__(self):
        return '{}'.format(self.stats)
