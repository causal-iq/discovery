#
#   Base level classes for graph hierarchy
#

from numpy import zeros
from pandas import DataFrame

from core.metrics import pdag_compare
from core.score import dag_score
from core.common import EdgeType, BAYESYS_VERSIONS
from fileio.data import Data
from fileio.oracle import Oracle


class SDG():
    """
        Base class for simple dependency graphs (one edge between vertices)

        :param list nodes: nodes present in the graph
        :param list edges: edges which define the graph connections as list
                           of tuples: (node1, dependency symbol, node2)

        :ivar list nodes: graph nodes in alphabetical order
        :ivar dict edges: graph edges {(node1, node2): EdgeType}
        :ivar bool is_directed: graph only has directed (causal) edges
        :ivar bool is_partially_directed: graph is partially directed
        :ivar dict parents: parents of node {node: [parents]}
        :ivar bool has_directed_cycles: contains any directed cycles

        :raises TypeError: if nodes and edges not both lists
        :raises ValueError: node or edge invalid
    """

    def __init__(self, nodes, edges):
        """
            Constructor
        """

        if type(nodes) is not list or type(edges) is not list:
            raise TypeError('graph edges and nodes not both lists')

        # Validate nodes specified

        self.nodes = []
        for node in nodes:
            if type(node) is not str:
                raise TypeError('graph has non-string node name')
            if not node:
                raise ValueError('graph has empty node name')
            if node in self.nodes:
                raise ValueError('graph has duplicate node names')
            self.nodes.append(node)
        self.nodes.sort()

        # Validate edges specified

        self.edges = {}
        edge_types = {e.value[3]: e for e in EdgeType}
        self.is_directed = True
        self.is_partially_directed = True
        self.parents = {}
        self.has_directed_cycles = False

        for edge in edges:
            if type(edge) is not tuple or len(edge) != 3:
                raise TypeError('graph has non-triple edge')
            for part in edge:
                if type(part) is not str:
                    raise TypeError('graph has non-string edge triple')
            if edge[0] == edge[2]:
                raise ValueError('graph has cyclic edge')
            if edge[1] not in edge_types:
                raise TypeError('graph edge has unknown type symbol: "{}"'
                                .format(edge[1]))
            if edge[0] not in self.nodes or edge[2] not in self.nodes:
                raise ValueError('graph edge has unknown node')
            if (edge[0], edge[2]) in self.edges or \
                    (edge[2], edge[0]) in self.edges:
                raise ValueError('graph has duplicate edges')

            # Check if graph directed or partially directed
            edge_type = edge_types[edge[1]]
            if edge_type != EdgeType.DIRECTED:
                self.is_directed = False
                if edge_type != EdgeType.UNDIRECTED:
                    self.is_partially_directed = False

            if edge_type == EdgeType.DIRECTED:

                # collect parents for directional links

                if edge[2] in self.parents:
                    self.parents[edge[2]].append(edge[0])
                else:
                    self.parents[edge[2]] = [edge[0]]

            elif edge_type in (EdgeType.UNDIRECTED, EdgeType.BIDIRECTED,
                               EdgeType.NONDIRECTED) and edge[0] > edge[2]:

                # ensure nodes in alphabetical order for non-directional edges

                edge = (edge[2], edge[1], edge[0])

            # parents held in alphabetical order

            self.parents = {k: sorted(v) for k, v in self.parents.items()}

            # store the edge in dictionary {(node1, node2): edge_type}

            self.edges[(edge[0], edge[2])] = edge_type

            # see whether has directed cycles

            self.has_directed_cycles = True if \
                self.partial_order(self.parents, self.nodes) is None else False

    def rename(self, name_map):
        """
            Nodes renamed in place according to name map.

            :param dict name_map: name mapping {name: new name}
                                  - must have mapping for every node

            :raises TypeError: with bad arg type
            :raises ValueError: with bad arg values e.g. unknown node names
        """
        if (not isinstance(name_map, dict)
            or not all([isinstance(k, str) and isinstance(v, str)
                        for k, v in name_map.items()])):
            raise TypeError('SDG.rename() bad arg types')

        if set(name_map.keys()) != set(self.nodes):
            raise ValueError('SDG.raname() bad arg values')

        # Change node names and node names in edges

        nodes = [name_map[n] if n in name_map else n for n in self.nodes]
        edges = [((name_map[e[0]] if e[0] in name_map else e[0]),
                  t.value[3],
                  (name_map[e[1]] if e[1] in name_map else e[1]))
                 for e, t in self.edges.items()]

        # Re-instantiate object with new names

        self = self.__init__(nodes, edges)

    @classmethod
    def partial_order(self, parents, nodes=None, new_arc=None):
        """
            Return partial topological ordering for the directed part of a
            graph which is specified by list of parents for each node.

            :param dict parents: parents of each node {node: [parents]}
            :param list/None nodes: optional complete list of nodes including
                                    parentless ones for use if parents argument
                                    doesn't include them already.
            :param tuple new_arc: a new arc (n1, n2) to be added before order
                                  is evaluated. If the opposing arc is implied
                                  in parents then it is removed so that arc
                                  reversal is also supported. This argument
                                  facilitates seeing whether an arc addition or
                                  reversal would create a cycle

            :returns list/None: nodes in a partial topological order as list of
                                sets or None if there is no ordering which
                                means the graph is cyclic
        """
        if not isinstance(parents, dict) \
                or (nodes is not None and not isinstance(nodes, list)
                    and not isinstance(nodes, set)):
            raise TypeError('DAG.partial_order bad arg type')

        parents_copy = {n: set(p) for n, p in parents.items()}
        if nodes is not None:
            parents_copy.update({n: set() for n in nodes if n not in parents})
        if new_arc is not None:
            parents_copy[new_arc[0]] = parents_copy[new_arc[0]] - {new_arc[1]}
            parents_copy[new_arc[1]] = parents_copy[new_arc[1]] | {new_arc[0]}

        order = []
        while len(parents_copy):
            roots = {n for n, p in parents_copy.items() if not len(p)}
            if not len(roots):
                return None
            order.append(roots)
            for root in roots:
                del parents_copy[root]
            parents_copy = {n: p - roots for n, p in parents_copy.items()}
        return order

    def is_DAG(self):
        """
            Returns whether graph is a Directed Acyclic Graph (DAG)

            :returns bool: whether graph is a DAG
        """
        return self.is_directed and self.has_directed_cycles is False

    def is_PDAG(self):
        """
            Returns whether graph is a Partially Directed Acyclic Graph (PDAG)

            :returns bool: whether graph is a PDAG
        """
        if self.is_directed:
            return self.is_DAG()

        if not self.is_partially_directed:
            return False

        # Check if there are any cycles in directed part of graph

        arcs = [(a[0], '->', a[1]) for a, t in self.edges.items()
                if t == EdgeType.DIRECTED]
        return not SDG(self.nodes, arcs).has_directed_cycles

    def undirected_trees(self):
        """
            Return undirected trees present in graph

            :returns list: of trees, each tree a set of tuples representing
                           edges in tree (n1, n2) or a single isolated node
                           (n1, None)
        """
        trees = []
        edges = {(e[0], e[1]) if e[1] > e[0] else (e[1], e[0])
                 for e in self.edges.keys()}
        isolated = set(self.nodes)
        while len(edges):
            tree = {list(edges)[0]}
            t_nodes = {list(tree)[0][0], list(tree)[0][1]}
            for edge in edges:
                if ((edge[0] in t_nodes and edge[1] not in t_nodes) or
                        (edge[1] in t_nodes and edge[0] not in t_nodes)):
                    t_nodes = t_nodes | {edge[0], edge[1]}
                    tree = tree | {edge}
            edges = edges - tree
            isolated = isolated - t_nodes
            trees.append(tree)
        trees.extend([{(n, None)} for n in isolated])
        return trees

    def components(self):
        """
            Return components present in graph. Uses tree search algorithm
            to span the undirected graph to identify nodes in individual
            trees which are the spanning tree of each component.

            :returns list: of lists, each a list of sorted nodes in component
        """
        components = []
        edges = {(e[0], e[1]) if e[1] > e[0] else (e[1], e[0])
                 for e in self.edges.keys()}  # skeleton of graph
        nodes = set(self.nodes)  # nodes yet to be included in components

        while len(nodes):
            c_nodes = {list(nodes)[0]}  # put 1st arbitrary node in component
            growing = True
            while growing:
                growing = False
                for edge in edges:  # look for edges across component boundary
                    if ((edge[0] in c_nodes and edge[1] not in c_nodes) or
                            (edge[1] in c_nodes and edge[0] not in c_nodes)):
                        c_nodes = c_nodes | {edge[0], edge[1]}
                        growing = True

            # component identified, remove its nodes from consideration

            nodes = nodes - c_nodes
            components.append(sorted(c_nodes))

        return sorted(components)

    def number_components(self):
        """
            Return number of components (including unconnected nodes) in graph

            :returns int: number of components
        """
        return len(self.components())

    def to_adjmat(self):
        """
            Returns an adjacency matrix representation of the graph

            :returns numpy.array: adjacency matrix
        """
        size = len(self.nodes)
        adjmat = zeros(shape=(size, size), dtype='int8')
        adjmat = DataFrame(adjmat, columns=self.nodes)
        adjmat[''] = self.nodes
        adjmat.set_index('', inplace=True)
        for nodes, type in self.edges.items():
            # print('{}: {}'.format(nodes, type.value[0]))
            adjmat.loc[nodes[0], nodes[1]] = type.value[0]

        return adjmat

    def __str__(self):
        """
            A human-readable description of the graph

            :returns str: description of graph
        """
        if not self.nodes:
            return 'Empty graph'

        number_components = self.number_components()
        kind = 'DAG' if self.is_directed else \
            ('PDAG' if self.is_partially_directed else 'SDG')
        desc = '{} ({}), {} node'.format(kind, type(self), len(self.nodes)) + \
            ('' if len(self.nodes) == 1 else 's') +\
            ', {} edge'.format(len(self.edges)) + \
            ('' if len(self.edges) == 1 else 's') + \
            ' and {} component'.format(number_components) + \
            ('' if number_components == 1 else 's')

        for node in self.nodes:
            children = [t.value[3] + str(e[1]) for e, t
                        in self.edges.items() if e[0] == node]
            desc += '\n{}: {}'.format(node, ' '.join(children))
        return desc

    def __eq__(self, other):
        """
            Test if graph is identical to this one

            :param Graph other: graph to compare with self

            :returns bool: True if other is identical to self
        """
        return isinstance(other, SDG) \
            and self.nodes == other.nodes and self.edges == other.edges


class PDAG(SDG):
    """
        Partially directed acyclic graph (PDAG)

        :param list nodes: nodes present in the graph
        :param list edges: edges which define the graph connections as list
                           of tuples: (node1, dependency symbol, node2)

        :ivar list nodes: graph nodes in alphabetical order
        :ivar dict edges: graph edges {(node1, node2): EdgeType}
        :ivar bool is_directed: graph only has directed (causal) edges
        :ivar dict parents: parents of node {node: [parents]}

        :raises TypeError: if nodes and edges not both lists
        :raises ValueError: node or edge invalid
    """
    def __init__(self, nodes, edges):

        SDG.__init__(self, nodes, edges)

        if not self.is_PDAG():
            raise NotPDAGError("graph is not a PDAG")

    @classmethod
    def fromDAG(self, dag):
        """
            Generates PDAG representing equivalence class DAG belongs to.

            Uses the algorithm in "A Transformational Characterization of
            Equivalent Bayesian Network Structures", Chickering, 1995. Step
            numbers in comments refer to algorithm step numbers in paper.

            :param DAG dag: DAG whose PDAG is required.

            :raises TypeError: if dag is not of type DAG

            :returns PDAG: PDAG for equivalence class that dag belongs to
        """

        def _process_x(x, y, edges):  # process incoming edges to x

            if not len(parents[x]):  # nothing to do if no parents of x
                # print('#5 no incoming to {} so no op'.format(x))
                return (edges, False)

            for w in parents[x]:  # Step 5 - Loop over inbound edges to x ...
                if (w, '->', x) in edges:  # ... that are compelled
                    if w not in parents[y]:

                        # Step 6 - if w is not a parent of y then label all
                        #          inbound to y as compelled and exit

                        # print(('#6 {}->{} compelled, not parent of {}' +
                        #        ', so compel all inbound to {}')
                        #       .format(w, x, y, y))
                        return ([(e[0], '->', e[2]) if e[2] == y else e
                                 for e in edges], True)
                    else:

                        # Step 7 - if w is parent of y then compel w -> y

                        # print('#7 {}->{} so compel it'.format(w, y))
                        edges = [(e[0], '->', e[2]) if e[0] == w and e[2] == y
                                 else e for e in edges]
            return (edges, False)

        def _process_y(x, y, edges):  # process incoming edges to y

            # Step 8 - if there is an edge z -> y where z is not x or a parent
            #          of x, label x -> y all unknown inbound to y as compelled

            for z in parents[y]:
                if z != x and z not in parents[x]:
                    # print('#8 {} not -> {} so compel all inbound to {}'
                    #       .format(z, x, y))
                    return [(e[0], '->', e[2]) if (e[0] == x or e[1] == '?')
                            and e[2] == y else e for e in edges]

            # Step 9 - set x -> y and unknown inbound arcs to y to reversible

            # print('#9 set inbound to {} as "-"'.format(y))
            return [(e[0], '-', e[2]) if (e[0] == x or e[1] == '?')
                    and e[2] == y else e for e in edges]

        if not isinstance(dag, DAG):
            raise TypeError("dag arg in fromDAG not a DAG")

        nodes = [n for n in dag.ordered_nodes()]  # nodes in topological order
        parents = {n: [p for p in nodes if p in  # node parents in topo order
                       (dag.parents[n] if n in dag.parents else [])]
                   for n in nodes}
        edges = [(p, '?', n) for n in reversed(nodes) for p in parents[n]]
        edges = [e for e in reversed(edges)]
        # print('fromDAG: reversed ordered edges are: {}'.format(edges))

        while any([t == '?'for (_, t, _) in edges]):  # 3 some edges unknown
            for i, (x, _, y) in enumerate(edges):
                if edges[i][1] != '?':  # 4 dynamic lowest unknown edge
                    continue
                # print('#4 processing {} ? {} edge in {}'.format(x, y, edges))

                edges, restart = _process_x(x, y, edges)  # 5-7 incoming to x
                if restart:
                    break

                edges = _process_y(x, y, edges)  # 8&9, edges incoming to y

        return self(dag.nodes, edges)

    @classmethod
    def toCPDAG(self, pdag):
        """
            Generates a completed PDAG (CPDAG) from supplied PDAG

            :param PDAG pdag: PDAG to be completed

            :raises TypeError: if pdag is not of type PDAG
            :raises ValueError: if pdag is non-extendable

            :returns PDAG/None: CPDAG corresponding to pdag
        """
        dag = DAG.extendPDAG(pdag)
        return PDAG.fromDAG(dag) if dag is not None else None

    def is_CPDAG(self):
        """
            Whether the PDAG is a Completed PDAG (CPDAG)

            :raises ValueError: if PDAG is not extendable

            return bool: True if CPDAG, otherwise False
        """
        return self.toCPDAG(self) == self

    def compared_to(self, reference, bayesys=None, identify_edges=False):
        """
            Compare a graph with a reference graph

            :param PDAG reference: reference graph for comparison
            :param str/None bayesys: version of Bayesys metrics to return,
                                     or None if Bayesys metrics not required
            :param bool identify_edges: whether edges in each low level
                                        category (e.g. arc_missing) are to be
                                        included in metrics returned.

            :raises TypeError: if reference is not a Dependency Graph
            :raises ValueError: if both graphs don't contain same nodes

            :returns dict: Bayesys comparison metrics
        """
        if not isinstance(reference, PDAG) \
                or (not isinstance(bayesys, str) and bayesys is not None):
            raise TypeError('bad arg type for compared_to')

        if bayesys is not None and bayesys not in BAYESYS_VERSIONS:
            raise ValueError('bad bayesys value for compared_to')

        if self.nodes != reference.nodes and bayesys != 'v1.3':
            raise ValueError('comparing two graphs with different nodes')

        return pdag_compare(self, reference, bayesys, identify_edges)

    def edge_reversible(self, edge):
        """
            Returns whether specified edge is in CPDAG and is reversible

            :param tuple edge: edge to examine, (node1, node2)

            :returns bool: whether present and reversible, or not
        """
        if not isinstance(edge, tuple) or not len(edge) == 2 or \
                not isinstance(edge[0], str) or not isinstance(edge[1], str):
            raise TypeError('PDAG.edge_reversible() bad arg type')

        e = (min(edge), max(edge))
        return e in self.edges and self.edges[e] == EdgeType.UNDIRECTED


class DAG(PDAG):
    """
        Directed Acyclic Graph (DAG)

        :param list nodes: nodes present in the graph
        :param list edges: edges which define the graph connections as list
                           of tuples: (node1, dependency symbol, node2)

        :ivar list nodes: graph nodes in alphabetical order
        :ivar dict edges: graph edges {(node1, node2): EdgeType}
        :ivar bool is_directed: always True for DAGs
        :ivar dict parents: parents of node {node: [parents]}

        :raises TypeError: if nodes and edges not both lists
        :raises ValueError: node or edge invalid
    """

    def __init__(self, nodes, edges):

        try:
            PDAG.__init__(self, nodes, edges)
        except NotPDAGError:
            raise NotDAGError("graph is not a DAG")

        if not self.is_DAG():
            raise NotDAGError("graph is not a DAG")

    @classmethod
    def extendPDAG(self, pdag):
        """
            Generates a DAG which extends a PDAG (i.e. is a member of the
            equivalence class the PDAG represents)

            Uses the algorithm in "A simple algorithm to construct a
            consistent extension of a partially oriented graph",
            Dor and Tarsi, 1992

            :param PDAG pdag: PDAG from which DAG derived

            :raises TypeError: if pdag is not of type PDAG
            :raises ValueError: if pdag is not extendable (example is
                                an undirected square PDAG)

            :returns DAG: extension of pdag
        """

        def _adj(n, s, pc=True):
            et = [EdgeType.UNDIRECTED, EdgeType.DIRECTED] if pc \
                else [EdgeType.UNDIRECTED]
            return {e[0] if e[1] == n else e[1] for e, t in s.edges.items()
                    if (e[0] == n or e[1] == n) and t in et}

        def _valid_x(s):

            # Looking for a node in s which satisfies properties a and b

            for x in s.nodes:

                # a. x is a sink node, i.e. no outbound directed edges

                is_sink = not any([e[0] == x and t == EdgeType.DIRECTED
                                   for e, t in s.edges.items()])
                if not is_sink:
                    continue
                # print('{} is{} a sink'.format(x, '' if is_sink else ' not'))

                # b. - all nodes, y, attached to n by an undirected edge
                #      are adjacent to all neighbours (nb) of node n

                cond_b = True
                adj_x = _adj(x, s)
                # print('Neighbours of {} are {}'.f#ormat(x, adj_x))
                # print('Peers of {} are {}'.format(x, _adj(x, s, False)))
                for y in _adj(x, s, False):
                    adj_y = _adj(y, s) - {x}
                    # print('{} are adjacent to {}'.format(adj_y, y))
                    # print('{} is a subset of {}: {}'
                    #       .format(adj_x - {y}, adj_y,
                    #               (adj_x - {y}).issubset(adj_y)))
                    if not (adj_x - {y}).issubset(adj_y):
                        cond_b = False
                        break

                if cond_b:  # found node n that has properties a and b
                    return x

            return None  # no node found that has properties a and b

        if not isinstance(pdag, PDAG):
            raise TypeError("pdag arg in extendPDAG not a PDAG")

        if pdag.is_directed:  # if already directed just return as DAG class
            return DAG(pdag.nodes, [(e[0], '->', e[1])
                                    for e in pdag.edges.keys()])

        # Clone pdag as dag. The DAG will be created in "dag", matching G' in
        # paper, "pdag" will act as graph A in the paper

        dag = PDAG(pdag.nodes, [(e[0], '->' if t == EdgeType.DIRECTED else '-',
                                 e[1]) for e, t in pdag.edges.items()])

        while len(pdag.edges):
            x = _valid_x(pdag)
            # print("Processing eligible node is {}".format(x))
            if x is None:  # means pdag is not extendable
                raise ValueError('pdag is not extendable')

            # orientate all edges incident to x in pdag to go into x in dag

            edges = [(e[0] if e[1] == x else e[1], '->', x)
                     if (e[0] == x or e[1] == x) and e in pdag.edges else
                     # if (e[0] == x or e[1] == x) else
                     (e[0], '->' if t == EdgeType.DIRECTED else '-', e[1])
                     for e, t in dag.edges.items()]
            dag = PDAG(dag.nodes, edges)
            # print('DAG is:\n{}'.format(dag))

            # remove x and its incident edges from pdag for next iteration

            nodes = list(set(pdag.nodes) - {x})
            edges = [(e[0], '->' if t == EdgeType.DIRECTED else '-', e[1])
                     for e, t in pdag.edges.items() if e[0] != x and e[1] != x]
            pdag = PDAG(nodes, edges)
            # print('PDAG is:\n{}\n\n'.format(pdag))

        # return dag as DAG object

        return DAG(dag.nodes, [(e[0], '->' if t == EdgeType.DIRECTED else '?',
                                e[1]) for e, t in dag.edges.items()])

    def score(self, data, types, params={}):
        """
            Score a particular data set against the DAG.

            :param Data data: data(file) to score against graph
            :param str/list types: type(s) of score e.g. 'loglik', 'bic', 'bde'
            :param dict params: score parameters e.g. log base

            :raises TypeError: if arguments have bad type
            :raises ValueError: if arguments have bad values
            :raises FileNotFoundError: if data file not found

            :returns DataFrame: requested score types (col) for each node (row)
        """
        if not isinstance(data, Data) or isinstance(data, Oracle):
            raise TypeError('DAG.score() bad arg types')

        return dag_score(self, data, types, params)

    def ordered_nodes(self):
        """
            Generator which returns nodes in a topological order

            :returns str: next node in topological order
        """
        for group in self.partial_order(self.parents, self.nodes):
            for node in sorted(list(group)):
                yield node

    def to_string(self):
        """
            Compact (bnlearn) string representation of DAG e.g. [A][B][C|A:B]

            :returns str: description of graph
        """
        if not self.nodes:
            return ''

        str = ''
        for node in self.nodes:
            str += '[{}'.format(node)
            str += '|' + ':'.join(self.parents[node]) if node in self.parents \
                else ''
            str += ']'
        return str


class NotDAGError(Exception):
    """
        Indicates graph is not a DAG when one is expected
    """
    pass


class NotPDAGError(Exception):
    """
        Indicates graph is not a PDAG when one is expected
    """
    pass
