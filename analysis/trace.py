
#   Analysis of structure learning traces

from enum import Enum
from pandas import DataFrame

from core.common import BAYESYS_VERSIONS, EdgeType
from core.graph import DAG, PDAG
from core.bn import BN
from core.indep import indep
from fileio.common import EXPTS_DIR
from learn.trace import Trace, Activity


class ArcStatus(Enum):  # Arc status compared to reference graph after change
    CORRECT = 'ok'  # matches arc in reference graph
    EQUIVALENT = 'eqv'  # wrong orientation but in a reversible arc
    REVERSED = 'rev'  # wrong orientation in a non-reversible arc
    EXTRA = 'ext'  # arc not present in reference graph
    MISSING = 'mis'  # removed arc present in reference graph
    ABSENT = 'abs'  # removed arc so now agrees with reference


class DeltaDiff(Enum):  # Categorise difference between best and next delta
    NONE = 'no'  # no difference, so change is arbitrary
    SMALL = 'sm'  # small difference in deltas (<= 5%)
    LARGE = 'lg'  # large difference in deltas (> 5%)


class TraceAnalysis():
    """
        Encapsulates detailed analysis of a structure learning trace including
        comparison with reference graph.

        :param Trace trace: raw trace to analyse
        :param DAG/BN ref: reference DAG to compare trace against, or BN if
                           Oracle MI values required
        :param DataFrame/None data: data graph was learnt from - needed for
                                    Mutual Information or loglik values
                                    NB This argument SHOULD NOT BE df[:n] if
                                    correct loglik values are to be created.

        :ivar float start: start time of learning copied from trace
        :ivar Trace trace: trace from structure learning, supplemented by
                           status and margin, and scores normalised
        :ivar dict context: context information copied from trace
        :ivar DAG/PDAG result: learnt DAG or PDAG
        :ivar dict summary: summary statistics for trace and learnt graph
        :ivar dict edges: details of missing, extra etc. edges in result

        :raises TypeError: if arguments have invalid types
        :raises ValueError: if trace and ref have different node sets
    """
    def __init__(self, trace, ref, data=None):

        if (not isinstance(trace, Trace)
            or (not isinstance(ref, DAG) and not isinstance(ref, BN))
                or (data is not None and not isinstance(data, DataFrame))):
            raise TypeError('TraceAnalysis() bad arg type')

        bn = ref if isinstance(ref, BN) else None
        ref = ref.dag if bn is not None else ref

        if trace.result is None or trace.result.nodes != ref.nodes:
            raise ValueError('TraceAnalysis() trace/ref diff nodes')

        if data is not None and set(data.columns) != set(ref.nodes):
            raise ValueError('TraceAnalysis() data/ref diff nodes')

        self.trace = dict(trace.trace)
        self.context = trace.context
        self.start = trace.start
        self.result = trace.result
        self.edges = {}

        ref_parents = {n: set(ref.parents[n]) if n in ref.parents else set()
                       for n in ref.nodes}
        ref_pdag = PDAG.fromDAG(ref)

        t = self.trace
        status = []
        margin = []
        mi = []  # Mutual Information from data
        omi = []  # Oracle Multual Information from BN
        loglik = None
        N = self.context['N']
        key = self.context['id'].split('/')[-1]
        sample = int(key.split('_')[-1]) if '_' in key else None
        self.summary = {'N': N, 'sample': sample, 'iter': len(t['time']) - 2}
        pretime = self.context['pretime'] if 'pretime' in self.context else 0.0
        loglik = (round(self.context['loglik'] / N, 5)
                  if 'loglik' in self.context else None)

        #   Loop over entries in trace, further categorising each change,
        #   and accumulating counts of each category of change for summary

        for i in range(len(t['time'])):
            if t['arc'][i] is not None:
                _status = self._arc_status(t['activity'][i], t['arc'][i],
                                           ref_parents, ref_pdag)
                best = t['delta/score'][i] / N
                if t['delta_2'][i]:
                    next = t['delta_2'][i] / N
                    _margin = (0 if best - next < 1E-6 or next > best
                               or best == 0
                               else round(100 * (best - next) / best, 3))
                    t['delta_2'][i] = round(t['delta_2'][i] / N, 6)
                else:
                    _margin = 0.0
                margin.append(_margin)
                status.append(_status.value if _status is not None else None)
                _omi = (round(indep(t['arc'][i][0], t['arc'][i][1], None, None,
                              bn, N)['mi']['statistic'] / (2 * N), 6)
                        if bn is not None else None)
                _mi = (round(indep(t['arc'][i][0], t['arc'][i][1], None, data)
                             ['mi']['statistic'] / (2 * N), 6)
                       if data is not None else None)
                omi.append(_omi)
                mi.append(_mi)
            else:
                status.append(None)
                margin.append(None)
                omi.append(None)
                mi.append(None)
                if t['activity'][i] == Activity.STOP.value:
                    self.summary.update({'time': round(t['time'][i]
                                                       + pretime, 1)})
                    self.summary.update({'score': round(t['delta/score'][i]
                                                        / N, 5)})
                self.summary['loglik'] = loglik
            t['delta/score'][i] = round(t['delta/score'][i] / N, 6)

        # add in change status and delta margin, and if available,
        # Oracle MI and MI to trace

        self.trace.update({'status': status, 'margin': margin})
        if bn is not None:
            self.trace.update({'Oracle MI': omi})
        if data is not None:
            self.trace.update({'MI': mi})

        # check type of learnt graph

        graph = trace.result
        metrics = graph.compared_to(ref, bayesys=BAYESYS_VERSIONS[-1],
                                    identify_edges=True)
        if isinstance(graph, DAG):
            graph_type = 'DAG'
            pdag = PDAG.fromDAG(graph)  # will always be a CPDAG
        elif graph.is_PDAG():
            try:
                graph_type = 'CPDAG' if graph.is_CPDAG() else 'PDAG'
            except ValueError:
                graph_type = 'NONEX'
            pdag = PDAG.toCPDAG(graph) if graph_type == 'PDAG' else graph
        else:
            graph_type = 'MIXED'
            pdag is None

        ref_cpdag = PDAG.fromDAG(ref)
        equiv_metrics = (pdag.compared_to(ref_cpdag,
                                          bayesys=BAYESYS_VERSIONS[-1],
                                          identify_edges=True)
                         if pdag is not None else None)

        # Compute Log-Likelihood if data provided

        stats = {'type': graph_type,
                 'n': len(ref.nodes),
                 '|A|': len(ref.edges),
                 '|E|': len(graph.edges),
                 'shd': metrics['shd'] if metrics else None,
                 'shd-s': (None if not len(ref.edges) or not metrics else
                           round(metrics['shd'] / len(ref.edges), 2)),
                 'shd-e': (None if equiv_metrics is None else
                           equiv_metrics['shd']),
                 'shd-es': (None if equiv_metrics is None else
                            round(equiv_metrics['shd'] / len(ref.edges), 2)),
                 'shd-b': (None if not len(ref.edges) or not metrics else
                           round(metrics['shd-b'] / len(ref.edges), 2)),
                 'a-ok': metrics['arc_matched'] if metrics else None,
                 'a-rev': metrics['arc_reversed'] if metrics else None,
                 'a-eqv': None,
                 'a-non': None,
                 'a-ext': metrics['arc_extra'] if metrics else None,
                 'a-mis': metrics['arc_missing'] if metrics else None,
                 'p': (round(metrics['p'], 3) if metrics
                       and metrics['p'] is not None else None),
                 'r': (round(metrics['r'], 3) if metrics
                       and metrics['r'] is not None else None),
                 'f1': (round(metrics['f1'], 3) if metrics
                        and metrics['f1'] is not None else None),
                 'f1-b': (round(metrics['f1-b'], 3)
                          if metrics else None),
                 'bsf': round(metrics['bsf'], 3) if metrics else None,
                 'bsf-e': (None if equiv_metrics is None else
                           round(equiv_metrics['bsf'], 3)),
                 'f1-e': (round(equiv_metrics['f1'], 3)
                          if equiv_metrics['f1'] is not None else None),
                 'e-ori': (None if equiv_metrics is None else
                           equiv_metrics['arc_reversed'] +
                           equiv_metrics['edge_not_arc'] +
                           equiv_metrics['arc_not_edge']),
                 'loglik': loglik}
        self.summary.update(stats)

        self.edges['result'] = metrics['edges'] if metrics else None
        if graph_type == 'DAG':
            self.reversed_equivalent(ref_cpdag)

    def reversed_equivalent(self, ref_cpdag):
        """
            Categorise reversed (i.e. wrongly orientated) arcs as to whether
            they are compatible with the equivalence class (a-eqv)
            or not (a-non).

            :param PDAG ref_cpdag: CPDAG representing equivalence class of
                                   reference DAG
        """
        self.summary['a-eqv'] = 0
        self.summary['a-non'] = 0
        self.edges['result'].update({'arc_equivalent': set(),
                                     'arc_nonequivalent': set()})
        for arc in list(self.edges['result']['arc_reversed']):
            cpdag_type = (ref_cpdag.edges[arc] if arc in ref_cpdag.edges
                          else ref_cpdag.edges[(arc[1], arc[0])])
            if cpdag_type == EdgeType.DIRECTED:
                self.summary['a-non'] += 1
                self.edges['result']['arc_nonequivalent'].add(arc)
            else:
                self.summary['a-eqv'] += 1
                self.edges['result']['arc_equivalent'].add(arc)

    @classmethod
    def select(self, series, network, root_dir=EXPTS_DIR,
               bn_dir=EXPTS_DIR + '/bn'):
        """
            Select the trace for a specified series and network with the
            highest F1 score.

            :param str series: experiment series e.g. "HC_N_1"
            :param str network: BN name e.g. "alarm"
            :param str root_dir: where traces located - useful for testing
            :param str bn_dir: where ref BNs located - useful for testing

            :raises TypeError: if bad argument types
            :raises ValueError: if series or network doesn't exist

            :returns TraceAnalysis: for the selected trace
        """
        if not isinstance(series, str) or not isinstance(network, str) or \
                not isinstance(root_dir, str) or not isinstance(bn_dir, str):
            raise TypeError('TraceAnalysis.select() bad arg types')

        try:
            ref = BN.read(bn_dir + '/' + network + '.dsc').dag
        except FileNotFoundError:
            raise ValueError('TraceAnalysis.select() unknown series/network')

        selected = None
        traces = Trace.read(series + '/' + network, root_dir)
        if traces is None:
            raise ValueError('TraceAnalysis.select() unknown series/network')

        for trace in traces.values():
            analysis = TraceAnalysis(trace, ref)
            if analysis.summary['f1'] is not None and \
                    (selected is None or
                     analysis.summary['f1'] > selected.summary['f1']):
                selected = analysis

        return selected

    @classmethod
    def _arc_status(self, activity, arc, ref_parents, ref_pdag):
        """
            Determine status of an arc (e.g. whether correct, extra or
            reversed etc. compared to reference graph) following an activity.

            :param Activity activity: arc addition, deletion or reversal
            :param tuple arc: arc changed (node1, node2)
            :param dict ref_parents: the reference DAG specified in terms of
                                     parents of each node {node: {parents}}
            :param PDAG ref_pdag: PDAG representing equivalence class of
                                  reference DAG

            :returns ArcStatus: status of arc after activity
        """

        #   Identify arc frm --> to involved after change made

        to = arc[1] if activity != Activity.REV.value else arc[0]
        frm = arc[0] if activity != Activity.REV.value else arc[1]

        if activity == Activity.NONE.value:
            status = None

        elif activity == Activity.DEL.value:
            status = ArcStatus.MISSING if frm in ref_parents[to] \
                or to in ref_parents[frm] else ArcStatus.ABSENT

        elif frm in ref_parents[to]:
            status = ArcStatus.CORRECT

        elif to in ref_parents[frm]:
            status = ArcStatus.EQUIVALENT if ref_pdag.edge_reversible(arc) \
                else ArcStatus.REVERSED
        else:
            status = ArcStatus.EXTRA

        return status

    def __str__(self):
        """
            Details of trace analysis in human-readable printable format.

            :returns str: TraceAnalysis in printable form
        """
        return (Trace.context_string(self.context, self.start)
                + '\n\n{}'.format(DataFrame(self.trace)))
