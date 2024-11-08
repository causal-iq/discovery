
# Code responsible for checking Knowledge parameters are consistent

from math import floor
from itertools import permutations

from core.common import RandomIntegers, stable_random, init_stable_random
from core.bn import BN, DAG
from core.graph import NotDAGError
from learn.knowledge_rule import Rule

KNOWLEDGE_PARAMS = {'limit': (int, float, bool),  # limit on AL requests
                    'ref': BN,  # reference BN used by 'expert' in AL
                    'ignore': int,  # number of initial AL requests to ignore
                    'expertise': float,  # proportion of expert answers correct
                    'stop': (dict, float, int),  # prohibited arcs
                    'reqd': (dict, float, int),  # required _setup_arc_lists
                    'initial': (DAG, bool),  # initial DAG
                    'partial': bool,  # expert supplies orientation info only?
                    'threshold': float,  # sensitivity for some AL criteria
                    'earlyok': bool,  # initial expert responses correct
                    'nodes': (int, float),  # nodes used in TIERS rule
                    'sequence': tuple,  # decision sequence
                    'pause': bool,  # whether to pause at end of sequence
                    'order': tuple,  # complete node order
                    'nodeset': set}  # complete set of node names


class KnowledgeParams():
    """
        Helper class of static methods used to check Knowledge parameters
    """

    @classmethod
    def check(self, params, rules, sample):
        """
            Check Knowledge parameters correct for RuleSet

            :param dict params: Knowledge parameters to check
            :param RuleSet rules: Knowledge rules
            :param int sample: sample number for Knowledge object

            :raises TypeError: if bad parameter types
            :raises ValueError: if bad parameter values
        """
        # Check no unknown parameters

        if len(set(params) - set(KNOWLEDGE_PARAMS)):
            raise ValueError('Knowledge(): unknown parameter')

        # Check specified parameters are right type

        for param, type in params.items():
            if not isinstance(type, KNOWLEDGE_PARAMS[param]):
                raise TypeError('Knowledge(): bad parameter type')

        # Check 'limit' value and compute integer in case limit initially
        # specified as a fractional size of number of refereence arcs

        if 'limit' in params and params['limit'] is not False:
            limit = params['limit']
            if isinstance(limit, float):
                if limit <= 0.0 or limit >= 1.0 or 'ref' not in params:
                    raise ValueError('Knowledge(): bad parameter value')
                params['limit'] = max(1, round(limit *
                                               len(params['ref'].dag.nodes)))
                print('Fractional limit set to {}'.format(params['limit']))
            elif limit is True or limit < 1:
                raise ValueError('Knowledge(): bad parameter value')

        # Set expertise for stop/reqd knowledge where number/fractional
        # amount of arcs has been specified but expertise has not. This has the
        # effect of sampling across arcs regardless of whether they are in the
        # reference graph or not.

        if ((('stop' in params and not isinstance(params['stop'], dict)) or
             ('reqd' in params and not isinstance(params['reqd'], dict)))
                and 'expertise' not in params and 'ref' in params):
            print('\n\n** set expertise\n')
            n_nodes = len(params['ref'].dag.nodes)
            n_edges = len(params['ref'].dag.edges)
            params['expertise'] = n_edges / (n_nodes * (n_nodes - 1))

        # Check ignore, expertise, initial & threshold have valid values

        if (('ignore' in params and params['ignore'] < 0)
            or ('expertise' in params
                and (params['expertise'] < 0.0 or params['expertise'] > 1.0))
                or ('initial' in params and
                    isinstance(params['initial'], bool)
                    and params['initial'] is True)
                or ('threshold' in params and
                    (params['threshold'] < 0.0
                     or params['threshold'] > 1.0))):
            raise ValueError('Knowledge(): bad parameter value')

        # Some specific checks for particular rules
        if Rule.EQUIV_SEQ in rules.rules:
            self.equiv_seq(params)
        if Rule.TIERS in rules.rules:
            self.tiers(params, sample)
        if Rule.STOP_ARC in rules.rules:
            self.check_arc_params('stop', params, sample)
        if Rule.REQD_ARC in rules.rules:
            self.check_arc_params('reqd', params, sample)

        _params = {} if params is None else params

        if (len(set(rules.rules) & {Rule.EQUIV_ADD, Rule.BIC_UNSTABLE})
                and 'ref' not in _params):
            raise ValueError('Knowledge(): missing ref parameter')
        if ('threshold' in params
            and len(set(rules.rules) & {Rule.BIC_UNSTABLE, Rule.MI_CHECK,
                                        Rule.HI_LT5, Rule.LO_DELTA}) == 0):
            raise ValueError('Knowledge(): extraneous threshold parameter')
        if 'earlyok' in params and 'expertise' not in params:
            raise ValueError('Knowledge(): earlyok specified, expertise not')
        if ('earlyok' in params
            and len(set(rules.rules) &
                    {Rule.BIC_UNSTABLE, Rule.MI_CHECK, Rule.HI_LT5,
                     Rule.EQUIV_ADD, Rule.POS_DELTA}) == 0):
            raise ValueError('Knowledge(): extraneous threshold parameter')
        if ('partial' in params and params['partial'] is True
                and not len(set(rules.rules) |
                            {Rule.EQUIV_ADD, Rule.BIC_UNSTABLE})):
            raise ValueError('Knowledge(): extraneous partial parameter')

    @classmethod
    def equiv_seq(self, params):
        """
            Check parameters relating to EQUIV_SEQ rule

            :param dict params: Knowledge parameters to check
            :param RuleSet rules: Knowledge rules

            :raises TypeError: if bad parameter types
            :raises ValueError: if bad parameter values
        """
        if 'sequence' not in params:
            raise ValueError('Knowledge(): missing sequence parameter')

        sequence = params['sequence']
        if (not len(sequence)
                or any([not isinstance(v, bool) for v in sequence])):
            raise TypeError('Knowledge(): bad sequence element type')

    @classmethod
    def tiers(self, params, sample):
        """
            Check Knowledge params relating to tiers.

            :param dict params: all knowledge parameters
            :param int sample: sample number for randomisation

            :raises ValueError: for bad parameter values

            :returns None: but modifies the params argument to set up
                           prohibited arcs
        """

        # Check relevant parameter values are present and correct

        if 'nodes' not in params or 'ref' not in params:
            raise ValueError('Knowledge(): bad parameter values')

        ref = params['ref']
        num_nodes = params['nodes']
        if (isinstance(num_nodes, int) and
                (num_nodes < 2 or num_nodes > len(ref.dag.nodes)) or
                (isinstance(num_nodes, float) and
                 (num_nodes <= 0.0 or num_nodes > 1.0))):
            raise ValueError('Knowledge(): bad parameter values')

        if isinstance(num_nodes, float):
            num_nodes = max(2, round(num_nodes * len(ref.dag.nodes)))
        print('\n\n{} nodes assigned to tiers'.format(num_nodes))

        # randomly chose num_nodes nodes which will be placed in tiers

        print('Init tiers randomisation with {}'.format(sample))
        nodes = [ref.dag.nodes[i] for i in
                 RandomIntegers(len(ref.dag.nodes), sample)][:num_nodes]
        print('Tier nodes are {}'.format(nodes))

        # Obtain correct topological ordering from ref graph, and if expertise
        # is less than 1.0, move proportion of nodes to the wrong tier.

        order = ref.dag.partial_order(ref.dag.parents, ref.dag.nodes)
        if 'expertise' in params and params['expertise'] < 1.0:
            print('Need to place some nodes in incorrect tiers')
            for node in nodes:
                if stable_random() > params['expertise']:
                    frm = [i for i, t in enumerate(order) if node in t][0]
                    to = len(order) - 2
                    to = min(to, round(to * stable_random()))
                    to = to if to < frm else to + 1
                    print('Move node {}: {} --> {}'.format(node, frm, to))
                    order[frm] = order[frm] - {node}
                    order[to] = order[to] | {node}
                else:
                    print('Do not move node {}'.format(node))

        print('Order is {}'.format(order))

        # Now convert tier specifications into prohibited arcs

        if 'stop' not in params:
            params.update({'stop': {}})
        ancestors = set()
        for group in order:
            group = group & set(nodes)  # only consider selected nodes

            # prohibit arcs from current tier to any higher tiers

            for node in group:
                for a in ancestors:
                    print('Prohibit {} --> {}'.format(node, a))
                    params['stop'].update({(node, a): (False if (node, a)
                                                       in ref.dag.edges
                                                       else True)})

            # prohibit arcs within a tier

            for p in permutations(group, 2):
                print('Prohibit {} --> {}'.format(p[0], p[1]))
                params['stop'].update({p: (False if p in ref.dag.edges
                                           else True)})

            ancestors = ancestors | group  # nodes in higher tiers

    @classmethod
    def check_arc_params(self, type, params, sample):
        """
            Check Knowledge params relating to required or prohibited arcs.

            :param str type: type to check - 'stop' or 'reqd'
            :param dict params: all knowledge parameters
            :param int sample: sample number for randomisation

            :raises TypeError: if types in explicit arc lists incorrect
            :raises ValueError: if invalid parameter values
        """
        if type not in params:
            raise ValueError('Knowledge(): {} param missing'
                             .format(type))
        value = params[type]  # value of stop or reqd param

        if isinstance(value, (int, float)):

            # Fraction or number specified, so sample that fraction or number
            # from reference graph arcs or non-arcs ('reqd' and 'stop')

            if ((isinstance(value, float) and value <= 0.0)
                    or (isinstance(value, int) and value < 1)
                    or 'ref' not in params):
                raise ValueError('Knowledge(): bad {} values'.format(type))

            print('Init arc randomisation with {}'.format(sample))
            init_stable_random(sample)

            self.setup_arc_lists(type, params)

        elif (any([not isinstance(k, tuple) or len(k) != 2 or
                   not isinstance(k[0], str) or not isinstance(k[1], str)
                   for k in value.keys()])
              or any([not isinstance(v, bool) for v in value.values()])):
            raise TypeError('Knowledge() bad stop/reqd type')

        if type == 'reqd' and 'initial' not in params:
            raise ValueError('Knowledge(): initial DAG not specified')

    @classmethod
    def setup_arc_lists(self, type, params):
        """
            Generate stop/reqd arc lists

            :param str type: type to generate - 'stop' or 'reqd'
            :param dict params: all knowledge parameters
        """
        # Generate lists of arcs present in and absent from the reference DAG

        ref = params['ref'].dag
        present = [a for a in ref.edges]  # will be non-arcs for stop
        absent = [(n1, n2) for n1 in ref.nodes for n2 in ref.nodes
                  if n1 != n2 and (n1, n2) not in present]
        # print('\n{} are in graph, {} are not'.format(present, len(absent)))

        # Specify right and wrong lists to pick from depending on list type

        right = present if type == 'reqd' else absent
        wrong = absent if type == 'reqd' else present
        num_reqd = (params[type] if isinstance(params[type], int)
                    else max(1, round(params[type] * len(ref.nodes))))
        max_reqd = round(len(right) if 'expertise' not in params else
                         params['expertise'] * len(right) +
                         (1.0 - params['expertise']) * len(wrong))
        print('\n\nnum_reqd: {} max_reqd {}'.format(num_reqd, max_reqd))
        num_reqd = max_reqd if num_reqd > max_reqd else num_reqd
        initial = None

        # Loop finding num_reqd randomly chosen arcs

        params[type] = {}
        while len(params[type]) < num_reqd:

            # Obtain stable random number used to pick which arc to include,
            # and possibly also whether a correct or incorrect choice needed

            rando = stable_random()
            correct = (True if 'expertise' not in params or
                       # (10 * rando - floor(10 * rando)) < params['expertise']
                       rando < params['expertise']
                       else False)
            _from = right if correct else wrong
            elem = min(floor(rando * len(_from)), len(_from) - 1)
            print('{}orrect elem {}: {}'
                  .format(('C' if correct else 'Inc'), elem, _from[elem]))

            # If type is reqd then need to construct matching initial graph.
            # If incorrect arcs are being included it is possible the new arc
            # might create a cycle so check for this - if so, ignore this
            # arc and continue onto next randomly chosen one

            if type == 'reqd' and 'initial' not in params:
                try:
                    arcs = [(a[0], '->', a[1]) for a in params[type]]
                    arcs.append((_from[elem][0], '->', _from[elem][1]))
                    nodes = list({n for a in arcs for n in a if n != '->'})
                    initial = DAG(nodes, arcs)
                except (NotDAGError, ValueError):
                    # print('*** cycle/duplicate adding {} to:\n{}'
                    #       .format(_from[elem], initial))
                    continue

            params[type].update({_from[elem]: correct})
            del _from[elem]

        if type == 'reqd' and 'initial' not in params:
            params['initial'] = initial
