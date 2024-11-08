
# Conditional Node Distribution at a node which will have concrete
# implementations as CPT or LinearGuassian etc.

from abc import ABC, abstractmethod


class CND(ABC):
    """
        Conditional Node Distribution for a node conditional on parantal
        values. Concrete subclasses support specific kinds of distributions,
        for example, CPT (multinomial), LinearGaussian etc.

        :ivar bool has_parents: whether CND is for a node with parents
        :ivar int free_params: number of free params in CND
    """
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def fit(self, node, parents, data, autocomplete=True):
        """
            Constructs a CND (Conditional Node Distribution) from data.

            :param str node: node that CND applies to
            :param tuple parents: parents of node
            :param Data data: data to fit CND to
            :param bool autocomplete: whether complete CPT tables

            :returns tuple: (cnd_spec, estimated_pmfs) where
                             cnd_spec is (CPT class, cpt_spec for CPT())
                             estimated_pmfs int/None - only for CPTs
        """
        pass

    @abstractmethod
    def cdist(self, parental_values=None):
        """
            Return conditional distribution for specified parental values.

            :param dict/None parental_values: parental values for which dist.
                                              required for non-orphans

            :raises TypeError: if args are of wrong type
            :raises ValueError: if args have invalid or conflicting values
        """
        pass

    @abstractmethod
    def random_value(self, pvs):
        """
            Generate a random value for a node given the value of its parents.

            :param dict/None pvs: parental values, {parent1: value1, ...}

            :return str/float: random value for node.
        """

    @abstractmethod
    def parents(self):
        """
            Return parents of node CND relates to

            :returns list: parent node names in alphabetical order
        """
        pass

    @abstractmethod
    def to_spec(self, name_map):
        """
            Returns external specification format of CND, renaming nodes
            according to a name map.

            :param dict name_map: map of node names {old: new}

            :returns dict: CND specification with renamed nodes
        """
        pass

    @abstractmethod
    def __str__(self):
        """
            Human-friendly description of the contents of the CND
        """
        pass

    @abstractmethod
    def __eq__(self, other):
        """
            Return whether two CNDs are the same allowing for probability
            rounding errors

            :param CND other: CND to compared to self

            :returns bool: whether CPTs are PRACTICALLY the same
        """
        pass

    @abstractmethod
    def validate_parents(self, node, parents, node_values):
        """
            Checks every CND's parents and (categorical) parental values are
            consistent with the other relevant CNDs and the DAG structure.

            :param str node: name of node
            :param dict parents: parents of all nodes {node: parents}
            :param dict node_values: values of each cat. node {node: values}
        """
        pass

    @classmethod
    def validate_cnds(self, nodes, cnds, parents):
        """
            Checks that all CNDs in graph are consistent with one another and
            with graph structure

            :param list nodes: BN nodes
            :param dict cnds: set of CNDs for the BN, {node: cnd}
            :param dict parents: parents of non-orphan nodes, {node: parents}

            :raises TypeError: if invalid types used in arguments
            :raises ValueError: if any inconsistent values found
        """

        # check 1:1 mapping between node and CNDs keys

        if sorted(list(cnds.keys())) != sorted(nodes):
            raise ValueError('CND.validate_cnds() bad/missing nodes in cnds')

        # collect values (states) for all categorical nodes

        values = {}
        for node, cnd in cnds.items():
            if cnd.__class__.__name__ == 'CPT':
                values[node] = cnd.values

        # check each node's CPT consistent with parents and parent values

        for node, cnd in cnds.items():
            cnd.validate_parents(node, parents, values)

    @classmethod
    def _map_keys(self, odict, name_map):
        """
            Renames some keys in a dict, re-ordering it by new key names.

            :param dict odict: some of keys of this dict will be renamed.
            :param dict name_map: name mapping for some jeys in odict

            :returns dict: with keys renamed, and re-ordered.
        """
        ndict = {name_map[k] if k in name_map else k: v
                 for k, v in odict.items()}
        return {k: ndict[k] for k in sorted(ndict)}
