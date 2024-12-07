
#   Class analysing characteristics of a graph

from pandas import DataFrame, Series
from math import log, exp, floor
from statistics import mean, stdev
from scipy.optimize import root_scalar
from itertools import chain, combinations

from core.graph import DAG, PDAG
from core.bn import BN
from core.cpt import CPT
from core.metrics import kl
from core.score import node_score
from fileio.pandas import Pandas
from fileio.oracle import Oracle


BN_PROPERTIES = {  # Properties of analysed BN
                 'n': 'Number of nodes',
                 '|A|': 'Number of arcs',
                 'in-avg': 'Mean node in-degree',
                 'in-max': 'Maximum node in-degree',
                 'deg-avg': 'Mean node degree',
                 'deg-max': 'Maximum node degree',
                 'mb-avg': 'Mean Markov Blanket size',
                 'mb-max': 'Max. Markov Blanket size',
                 'card.avg': 'Mean node cardinality',
                 'card.max': 'Maximum node cardinality',
                 'free.avg': 'Mean free parameters per node',
                 'free.max': 'Max. free parameters per node',
                 'k-l.avg': 'Mean node K-L variability',
                 'k-l.max': 'Max. node K-L variability',
                 'reversible': 'Fraction of edges reversible',
                 'aligned': 'Fraction of edges aligned'
                 }


class SampleAnalysis():
    """
        Analyse sample size needed to learn each edge in the network

        :param BN bn: Bayesian Network to perform sample analysis on
        :param DataFrame/None data: data to optionally analyse

        :ivar BN bn: Bayesian network being analysed
        :ivar DataFrame/None data: dataset for network

        :throws TypeError: for bad argument type
        :throws ValueError: if BN only has one node
    """
    __bounds = [log(10**N) / 10**N if N > 0  # search bounds used for approx.
                else 0.477 for N in range(0, 13)]  # solution of r = log(x) / x

    __LOG_E_10 = log(10)

    def __init__(self, bn=None, data=None):

        if ((bn is None and data is None) or
                (bn is not None and not isinstance(bn, BN)) or
                (data is not None and not isinstance(data, DataFrame))):
            raise TypeError('SampleAnalysis() bad arg type')

        if ((bn is not None and len(bn.dag.nodes) < 2) or
                (data is not None and len(data.columns) < 2) or
                (data is not None and bn is not None and
                 set(data.columns) != set(bn.dag.nodes))):
            raise ValueError('SampleAnalysis() bad arg value')

        self.bn = Oracle(bn=bn) if bn is not None else None
        self.data = Pandas(df=data) if data is not None else None

    def node_entropy(self, node, parents, samples=None):
        """
            Returns entropy (oracle and/or from data) and number of free
            parameters for node with specified parents

            :param str node: node for which entropy required
            :param list parents: parents of that node
            :param tuple/None samples: number of samples, sample size

            :returns tuple: data entropy, SD of data entropy (if sampling)
                            oracle entropy and number of free parameters
        """
        ent_o = ent_d = ent_sd = fp = None
        _parents = {node: parents} if parents else {}

        # If bn is present determine oracle entropy - use N=100 and log base
        # 10 so that difference between LL and BIC is number of free parameters

        if self.bn is not None:
            self.bn.set_N(100)
            scores = node_score(node=node, parents=_parents,
                                types=['loglik', 'bic'],
                                params={'base': 10, 'k': 1}, data=self.bn)
            fp = round(scores['loglik'] - scores['bic'])
            ent_o = -0.01 * self.__LOG_E_10 * scores['loglik']

        # if data is present determine entropy in data

        if self.data is not None:
            samples = (1, self.data.N) if samples is None else samples
            offset = floor(self.data.N / samples[0])
            size = samples[1]
            ent_d = []
            for sample in range(samples[0]):
                subset = Pandas(df=self.data.sample[sample * offset:
                                                    (sample * offset) + size])
                scores = node_score(node=node, parents=_parents,
                                    types=['loglik', 'bic'],
                                    params={'base': 'e', 'k': 1},
                                    data=subset)
                ent_d.append(scores['loglik'] * -1.0 / size)
                # print('Entropy from {}-{} to {:.6e}'
                #       .format(sample * offset, (sample * offset) + size - 1,
                #               ent_d[-1]))
                if fp is None:
                    fp = round(2.0 * (scores['loglik'] - scores['bic'])
                               / log(subset.N))
            ent_sd = 0.0 if samples[0] == 1 else stdev(ent_d)
            ent_d = mean(ent_d)

        # convert LL to base e and scale by -N (making it an entropy value
        # and return it together with free params

        return (ent_d, ent_sd, ent_o, fp)

    def cps_reqd_sample(self, node, limit=1, cps=None):
        """
            Compute sample size to establish edges in all possible parent sets
            for a specified node

            :param str node: node which will be child
            :param int limit: limit on number of parents
            :param list/None cps: candidate parent set, if None all other nodes
                                  are considered candidate parent sets
        """
        cps = (sorted(list(set(self.bn.nodes) - {node}))
               if cps is None else cps)

        # First compute entropy and free parameters for orphan node

        entd0, _, ent0, fp0 = self.node_entropy(node, [])
        print('\nOrphan {} with fp = {} has entropy {:.2e}/{:.2e}'
              .format(node, fp0, ent0, 0 if entd0 is None else entd0))

        limit = len(cps) if limit is None else limit
        for parents in chain.from_iterable(combinations(cps, r)
                                           for r in range(1, limit + 1)):
            entd, _, ent, fp = self.node_entropy(node, list(parents))
            N = round(self.solve_log_ratio(2 * (ent0 - ent) / (fp - fp0)))
            Nd = (round(self.solve_log_ratio(2 * (entd0 - entd) / (fp - fp0))
                        ) if entd is not None else 0)
            print(('{} --> {} with fp = {} has entropy {:.2e}/{:.2e}, ' +
                   'requiring N = {}/{}')
                  .format(parents, node, fp, ent, 0 if entd is None else entd,
                          N, Nd))
            # print('Ratio was {:.6f}'.format(ratio))

    @classmethod
    def solve_log_ratio(self, ratio):
        """
            Solves the transcendental equation log x / x = ratio approximately

            :param float ratio: required ratio of log x to x

            :raises TypeError: if ratio not a float
            :raises ValueError: if ratio not between 0.0 and 1.0

            :returns float: value of x which solves the equation
        """
        def _equation(x, ratio):  # equation to solve: e**(x*ratio) - x
            """
                Compute exp(ratio * x) - x and its derivative

                :param float ratio: value of ratio
                :param float x: value of x

                :throws ValueError: if x * ratio is too large

                :returns tuple: (function value, derivatve value)
            """
            # Raising e to power above 710 creates a floating point number
            # larger than Python supports. See following URL:
            # https://math.stackexchange.com/questions/4257765/
            # taylor-approximation-of-expx-function-for-large-x
            if x * ratio > 710:
                raise OverflowError('SampleAnalysis.solve_log_ratio overflow')
            # print('Calling _equation with x={:.6f}'.format(x))
            return exp(ratio * x) - x, ratio * exp(ratio * x) - 1

        if not isinstance(ratio, float):
            raise TypeError('SampleAnalysis.solve_log_ratopm bad arg type')

        if ratio >= 1.0:
            raise ValueError('SampleAnalysis.solve_log_ratopm bad arg value')

        # Only support returning N between 4 and 10**12

        if ratio > 0.3465:
            return 4.0
        elif ratio < 2.763102116E-11:
            return 1.0e12

        # now find range of x to try using __bounds class variable

        i = 0
        while i < len(self.__bounds) and self.__bounds[i] > ratio:
            i += 1
        bracket = tuple([0.99 * 10**(i-1) if i > 1 else 3.0, 1.5 * 10**(i)])
        # print('Bracket is ({:.2e}, {:.2e})'.format(bracket[0], bracket[1]))

        x = root_scalar(_equation, args=(ratio, ), bracket=bracket,
                        x0=0.5*(bracket[0] + bracket[1]), xtol=1e-2,
                        fprime=True, maxiter=1000)
        # print(x)
        return x.root if x.converged is True else None

    def edge_stability(self, arc):
        """
            Compute LL stability for an arc at different sample sizes.

            :param tuple arc: arc for which stability required.

            :returns list: of ent, sd at different subsample sizes
        """
        NUM_SAMPLES = 10
        for power in range(1, 6):
            size = 10 ** power
            if NUM_SAMPLES * size > self.data.N:
                break
            ents = self.node_entropy(arc[0], [arc[1]],
                                     samples=(NUM_SAMPLES, size))
            print('Edge stability at 10**{}: {:.4e}, {:3f}'
                  .format(power, ents[0], ents[1]/ents[0]))


class DAGAnalysis():
    """
        Encapsulates analysis of a DAG

        :param dag DAG: DAG to be analysed

        :ivar DataFrame nodes: analysis of each node
        :ivar DataGrame/None arcs: analysis of each arc

        :raises TypeError: if arguments have invalid types
        :raises RuntimeError: if sanity check on arc reversibility fails
    """
    def __init__(self, dag):

        def _mb(node, parents, children):  # return Markov Blanket of node
            spouses = {p for c in children[node] for p in parents[c]} - {node}
            return spouses.union(parents[node], children[node])

        if not isinstance(dag, DAG):
            raise TypeError('DAGAnalysis bad arg type')

        parents = {p: dag.parents[p] if p in dag.parents else []
                   for p in dag.nodes}
        node_pos = {p: pos for pos, ps in enumerate(dag.partial_order(parents))
                    for p in ps}
        children = {p: [] for p in dag.nodes}
        pdag = PDAG.fromDAG(dag)

        if len(dag.edges):
            self.arcs = []
            for e in dag.edges:
                children[e[0]].append(e[1])
                aligned = ((e[0] < e[1] and node_pos[e[0]] < node_pos[e[1]]) or
                           (e[0] > e[1] and node_pos[e[0]] > node_pos[e[1]]))
                self.arcs.append({'from': e[0], 'to': e[1],
                                  'reversible': pdag.edge_reversible(e),
                                  'aligned': aligned})
            self.arcs = DataFrame(self.arcs) \
                .set_index(['from', 'to']) \
                .sort_index()
        else:
            self.arcs = None

        nodes = [{'node': n, 'in': len(parents[n]), 'out': len(children[n]),
                 'mb': len(_mb(n, parents, children))} for n in dag.nodes]
        nodes = DataFrame(nodes).set_index('node')
        nodes['deg'] = nodes['in'] + nodes['out']
        self.nodes = nodes


class BNAnalysis(DAGAnalysis):
    """
        Encapsulates analysis of a BN

        :param BN bn: Bayesian Network to be analysed

        :raises TypeError: if arguments have invalid types
    """
    def __init__(self, bn):

        if not isinstance(bn, BN):
            raise TypeError('BNAnalysis bad arg type')

        super().__init__(bn.dag)

        nodes = []
        for node, cnd in bn.cnds.items():
            if isinstance(cnd, CPT):
                cpt = cnd
                parents = cpt.parents()
                node_values = cpt.node_values()
                # print('Node {}, CPT:\n{}'.format(node, cpt))
                if parents is not None:
                    mean_pmf = DataFrame(cpt.cpt.values()).mean()
                    # print('Mean PMF is {}'.format(mean_pmf.to_dict()))
                    total_kl = 0.0
                    for pmf in cpt.cpt.values():
                        pmf = Series(pmf)
                        pmf_kl = kl(pmf, mean_pmf)
                        # print(
                        # 'PMF {} has K-L {}'
                        #       .format(pmf.to_dict(), pmf_kl))
                        total_kl += pmf_kl
                    pmf_kl = total_kl / len(cpt.cpt)
                else:
                    unif_pmf = Series({v: 1.0/len(cpt.cpt) for v in cpt.cpt})
                    # print('Uniform pmf is {}'.format(unif_pmf.to_dict()))
                    pmf_kl = kl(Series(cpt.cpt), unif_pmf)
                    # print('PMF {} has K-L {}'.format(cpt.cpt, pmf_kl))

                nodes.append({'node': node,
                              'card': len(node_values),
                              'free': cpt.free_params,
                              'k-l': pmf_kl})

            else:
                nodes.append({'node': node, 'card': None, 'free': None,
                              'k-l': None})

        nodes = DataFrame(nodes).set_index('node')
        self.nodes = self.nodes.join(nodes)
