#
#   Conditional Probability Table for discrete distribution
#

from itertools import combinations

from core.common import random_generator
from core.metrics import values_same
from core.cnd import CND


class CPT(CND):
    """
        Base class for conditional probability tables

        :param dict pmfs: a pmf of {value: prob} for parentless nodes OR
                          list of tuples ({parent: value}, {value: prob})
        :param int estimated: how many PMFs were estimated

        :ivar dict cpt: Internal representation of the CPT
                        ({node_values: prob} for parentless node, otherwise
                         {parental_vales as frozenset: {node_values: prob}}
        :ivar int estimated: number of PMFs that were estimated
        :ivar set values: values which node can take

        :raises TypeError: if arguments are of wrong type
        :raises ValueError: if arguments have invalid or conflicting values
    """
    def __init__(self, pmfs, estimated=0):

        def _check_pmf(pmf):  # check an individual PMF
            if len(pmf) < 2:
                raise ValueError('PMF too few entries')
            for prob in pmf.values():
                if not isinstance(prob, float) or prob < 0.0 or prob > 1.0:
                    raise ValueError('Bad PMF probability')
            total = sum(pmf.values())
            if 0.999999 > total or total > 1.000001:
                raise ValueError('PMF probability does not sum to 1')

        def _check_parental_values(p_vals):  # check parental values specified
            if not len(p_vals) or \
                    not all([isinstance(v, str) for v in p_vals.values()]):
                raise ValueError('bad parental values')

        self.estimated = estimated
        if ((not isinstance(pmfs, dict) and not isinstance(pmfs, list))
                or not isinstance(estimated, int)
                or isinstance(estimated, bool)):
            raise TypeError('bad arg type for CPT')
        if (estimated < 0
                or (isinstance(pmfs, list) and estimated > len(pmfs))
                or (isinstance(pmfs, dict) and estimated > 1)):
            raise ValueError('CPT has bad estimated value')

        if isinstance(pmfs, dict):  # parentless node with simple CPT
            _check_pmf(pmfs)
            self.cpt = pmfs
            self.has_parents = False
            self.free_params = len(pmfs) - 1
            self.values = set(self.cpt.keys())

        else:  # node with parents - multi-entry CPT
            if len(pmfs) < 2:
                raise ValueError('CPT has fewer than two entries')
            prev_parents = None
            prev_values = None
            self.cpt = {}
            self.has_parents = True
            for entry in pmfs:

                # check types of variables in CPT entry

                if not isinstance(entry, tuple) or len(entry) != 2 \
                        or not isinstance(entry[0], dict) \
                        or not isinstance(entry[1], dict):
                    raise ValueError('Invalid CPT entry types')

                # check values in each CPT entry

                _check_parental_values(entry[0])
                _check_pmf(entry[1])

                # check parents and node values in each CPT entry are the same

                if prev_parents is None:
                    prev_parents = frozenset(entry[0].keys())
                elif prev_parents != frozenset(entry[0].keys()):
                    raise ValueError('Parents in CPT entries vary')
                if prev_values is None:
                    prev_values = frozenset(entry[1].keys())
                elif prev_values != frozenset(entry[1].keys()):
                    raise ValueError('Values in CPT entries vary')

                # make parental values a frozenset so it can be a dict key

                pvs = frozenset([(k, v) for k, v in entry[0].items()])
                self.cpt[pvs] = entry[1]

            # compute number of free params

            self.free_params = (len(prev_values) - 1) * len(self.cpt)
            self.values = prev_values

    @classmethod
    def fit(self, node, parents, data, autocomplete=True):
        """
            Constructs a CPT (Conditional Probability Table) from data.

            :param str node: node that CPT applies to
            :param tuple/None parents: parents of node
            :param Data data: data to fit CPT to
            :param bool autocomplete: whether to ensure CPT data contains
                                      entries for combinations of parental
                                      values that don't occur in the data

            :returns tuple: (cnd_spec, estimated_pmfs) where
                             cnd_spec is (CPT class, cpt_spec for CPT())
                             estimated_pmfs is int, # estimated pmfs
        """
        estimated_pmfs = 0
        if parents is not None:

            # node has parents, so this will be a multi-entry CPT
            # parent_data are the individual values for each parent node
            # marginals() returns the number of instances for
            # each value of the node for each parental value combination

            cptdata = []
            _p = {node: list(parents)} if len(parents) > 0 else {}
            counts, _, rowval, colval = data.marginals(node, _p, True)

            # for each parental value combination (combo) construct the CPT
            # entry in the format used by the CPT constructor, that is:
            # ( {parent1: value1, ...} {node_value1: prob1, ...} )

            for j in range(counts.shape[1]):
                N_ij = counts[:, j].sum()  # total count for jth parent combo
                cond_pmf = {rowval[k]: counts[k, j] / N_ij
                            for k in range(counts.shape[0])}
                cptdata.append((colval[j], cond_pmf))

            # It is unlikely that all possible parental combinations will
            # actually be present in the data - if autocomplete is True,
            # then add in the missing CPT entries, using the observed
            # frequencies across all the observed combinations

            if autocomplete:
                pmf = {k: v/data.N for k, v in data.node_values[node].items()}
                parent_values = {p: list(data.node_values[p].keys())
                                 for p in parents}
                pvs_in_data = [c[0] for c in cptdata]
                for pvs in NodeValueCombinations(parent_values):
                    if pvs not in pvs_in_data:
                        cptdata.append((pvs, pmf))
                        estimated_pmfs += 1
        else:

            # This is a parentless node so a simple CPT based on observed
            # frequencies of the node's values

            cptdata = {k: v/data.N for k, v in data.node_values[node].items()}

        return ((CPT, cptdata), estimated_pmfs)

    def cdist(self, parental_values=None):
        """
            Return conditional probabilities of node values for
            specified parental values.

            :param dict/None parental_values: parental values for which pmf
                                              required for non-orphans

            :raises TypeError: if args are of wrong type
            :raises ValueError: if args have invalid or conflicting values
        """
        if ((not self.has_parents and parental_values is not None) or
                (self.has_parents and not isinstance(parental_values, dict))):
            raise TypeError('CPT.cdist() CPT/parental values mismatch')

        if not self.has_parents:
            return self.cpt
        else:
            pvs = frozenset([(k, v) for k, v in parental_values.items()])
            if pvs not in self.cpt:
                raise ValueError('Unknown parental values for cdist call')
            return self.cpt[pvs]

    def random_value(self, pvs):
        """
            Generate a random value for a node given the value of its parents.

            :param dict/None pvs: parental values, {parent1: value1, ...}

            :return str/float: random value for node.
        """
        pmf = self.cdist(pvs)
        values = list(pmf.keys())
        random = random_generator().random()
        cum_prob = 0.0
        for idx, value in enumerate(values):
            cum_prob += pmf[value]
            if random <= cum_prob:
                return value
        return values[-1]

    def node_values(self):
        """
            Return node values (states) of node CPT relates to

            :returns: node values in alphabetical order
            :rtype: list of str
        """
        return sorted(list(self.cpt.keys())) if not self.has_parents else \
            sorted(list(self.cpt[(list(self.cpt.keys())[0])].keys()))

    def parents(self):
        """
            Return parents of node CPT relates to

            :returns: parent node names in alphabetical order
            :rtype: list of str or None
        """
        if not self.has_parents:
            return None
        else:
            return sorted([t[0] for t in (list(self.cpt.keys()))[0]])

    def to_spec(self, name_map):
        """
            Returns external specification format of CPT, renaming nodes
            according to a name map.

            :param dict name_map: map of node names {old: new}

            :raise TypeError: if bad arg type
            :raise ValueError: if bad arg value, e.g. coeff keys not in map

            :returns dict: CPT specification with renamed nodes
        """
        if (not isinstance(name_map, dict)
                or not all([isinstance(k, str) for k in name_map])
                or not all([isinstance(v, str) for v in name_map.values()])):
            raise TypeError('CPT.to_spec() bad arg type')

        if self.has_parents:
            print({nv[0] for nv in tuple(self.cpt)[0]})

        if self.has_parents and len({nv[0] for nv in tuple(self.cpt)[0]}
                                    - set(name_map)) != 0:
            raise ValueError('CPT.to_spec() bad arg value')

        if self.has_parents is False:
            spec = self.cpt
        else:
            spec = [(self._map_keys({t[0]: t[1] for t in pvs}, name_map), pmf)
                    for pvs, pmf in self.cpt.items()]
        return spec

    def param_ratios(self):
        """
            Returns distribution of parameter ratios across all parental
            values for each combination of possible node values.

            :returns dict: {(node value pair): (param ratios across parents)
        """
        pairs = tuple(combinations(iterable=self.node_values(), r=2))

        ratios = {}
        for pvs, pmf in (self.cpt if self.has_parents
                         else {None: self.cdist()}).items():
            _ratios = {p: pmf[p[0]] / pmf[p[1]] for p in pairs}
            ratios[pvs] = pmf
            print('*** {} --> {}: ratios {}'.format(pvs, pmf, _ratios))

    def __str__(self):
        """
            Human-friendly description of the contents of the CPT
        """
        if not self.has_parents:
            return '{}'.format({v: round(p, 6) for v, p in self.cpt.items()})
        else:
            str = ['{} -> {}'.format({t[0]: t[1] for t in pvs},
                                     {v: round(p, 6) for v, p in pmf.items()})
                   for pvs, pmf in self.cpt.items()]
            return '\n'.join(str)

    def __eq__(self, other):
        """
            Return whether two CPTs are the same allowing for probability
            rounding errors

            :param other: CPT to compared to self
            :type other: CPT

            :returns: whether CPTs are PRACTICALLY the same
            :rtype: bool
        """

        def _pmfs_same(pmf1, pmf2):  # compare two PMFs to 6 sig. figures
            return pmf1.keys() == pmf2.keys() and \
                all([values_same(v, pmf2[k], sf=6) for k, v in pmf1.items()])

        if self.has_parents != other.has_parents \
                or self.free_params != other.free_params \
                or self.cpt.keys() != other.cpt.keys():
            return False

        return _pmfs_same(self.cpt, other.cpt) if not self.has_parents else \
            all([_pmfs_same(v, other.cpt[k]) for k, v in self.cpt.items()])

    def validate_parents(self, node, parents, node_values):
        """
            Checks every CPT's parents and parental values are consistent
            with the other relevant CPTs and the DAG structure.

            :param str node: name of node
            :param dict parents: parents of all nodes {node: parents}
            :param dict node_values: values of each cat. node {node: values}
        """

        # Check parents defined in CPT keys match those defined in parents
        # arg (i.e. tha DAG)

        if ((node not in parents and self.has_parents is True)
            or (node in parents and self.has_parents is False)
            or (node in parents and self.has_parents is True
                and set(parents[node]) != set(self.parents()))):
            raise ValueError('CPT.validate_parents() parent mismatch')

        if self.has_parents:

            # check values in CPT keys are valid for the parents specified

            for pvs, pmf in self.cpt.items():
                for parent, value in {t[0]: t[1] for t in pvs}.items():
                    if value not in node_values[parent]:
                        raise ValueError('CPT.validate_parents()' +
                                         ' bad parent value')

            # Check CPT has expected number of entries - one for each possible
            # combination of parental Values

            cpt_num_rows = 1
            for parent in parents[node]:
                cpt_num_rows *= len(node_values[parent])
            if len(self.cpt) != cpt_num_rows:
                raise ValueError('CPT.validate_parents() non-orphan' +
                                 ' has missing parental value combos')


class NodeValueCombinations():
    """
        Iterable over all combinations of node values

        :param dict node_values: allowed values for each node {node: [values]}
        :param bool sort: whether to sort node names and values into
                          alphabetic order
    """
    def __init__(self, node_values, sort=True):

        # store node values, optionally sorting them

        self.node_values = {n: sorted(s) if sort else s
                            for n, s in node_values.items()}

        # store node names, optionally sorting them

        self.nodes = sorted(list(node_values.keys())) if sort \
            else list(node_values.keys())

        # store number of possible values for each node

        self.cards = [len(node_values[n]) for n in self.nodes]

    def __iter__(self):
        """
            Returns the initialised iterator

            :returns NodeValueCombinations: the iterator
        """

        # Initialises indices which define current combination of values -
        # each index value is a pointer into the values in node_values

        self.indices = len(self.cards) * [0]
        self.indices[len(self.cards) - 1] = -1
        return self

    def __next__(self):
        """
            Generate the next node value combination

            :raises StopIteration: when all combinations have been returned

            :returns dict: next node value combination {node: value}
        """

        # move on to next combination by incrementing lowest order index
        self.indices[len(self.cards) - 1] += 1

        # if lowest order index has gone past number of possible states then
        # perform a "carry-over" operation by zeroing it and incrementing the
        # next highest order index.
        # Repeat if necessary up the index order (just like in arithmetic)

        for j in range(len(self.cards) - 1, -1, -1):
            if self.indices[j] == self.cards[j]:

                # perform "carry-over" operation

                if j == 0:

                    # if we're attempting a carry over from highest order
                    # index then we've exhausted possible combinations
                    raise StopIteration

                self.indices[j] = 0
                self.indices[j - 1] += 1
            else:
                break

        # Convert the indices to a dictionary {parent: value} and return

        return {self.nodes[p]: self.node_values[self.nodes[p]][v]
                for p, v in enumerate(self.indices)}
