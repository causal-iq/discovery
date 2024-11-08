
# Class which undertakes hill-climbing search

from copy import deepcopy
from time import asctime, localtime

from core.common import EnumWithAttrs
from core.score import node_score
from core.metrics import values_same
from core.graph import DAG
from fileio.pandas import Pandas
from learn.trace import Trace
from learn.tabulist import TabuList
from learn.trace import Activity, Detail
from learn.dagchange import DAGChange, BestDAGChanges
from learn.knowledge import Knowledge


class HCWorker():
    """
        Performs iterations of the hill-climbing search.

        :cvar dict score_cache: cache of scores for node/parent combos

        :ivar Data data: data to learn graph from
        :ivar dict params: parameters for algorithm e.g. score to use
        :ivar Knowledge/bool knowledge: human/pre-existing knowledge
        :ivar Trace/None: learning trace
        :ivar dict parents: parents of each node (i.e. defines DAG) in format
                            {node1: {parent1, ...}, node2 ...}
        :ivar float score: overall score of the current DAG
        :ivar float score2: score used for comparisons which may include some
                            lookahead iterations
        :ivar TabuList/None tabulist: DAGs recently visited in Tabu learning

        :ivar int iter: current interation number
        :ivar BestDAGChanges best: best and 2nd best DAG change in iteration
        :ivar int num_noinc: number of iterations where score not increased
        :ivar dict best_parents: highest scoring parents (i.e. DAG) so far
        :ivar float best_parents_score: score of highest scoring DAG so far
        :ivar bool paused: whether learning has paused
    """

    score_cache = {}  # score level cache shared across all workers

    def __init__(self, data, params, knowledge, context, init_cache):
        """
            Initialise the HC process, including trace, DAG, Tabu list,

            :param Data data: data to learn graph from
            :param dict params: parameters for algorithm e.g. score to use
            :param Knowledge/bool knowledge: human/pre-existing knowledge
            :param dict/None context: context information for trace if reqd
            :param bool init_cache: whether to (re)initialise the score cache
        """
        self.data = data
        self.params = params
        self.knowledge = knowledge
        if init_cache is True:
            HCWorker.init_score_cache()
        self.params['zero'] = (0.0 if isinstance(self.data, Pandas)
                               else self.data.N * 10**-6)

        # Initialise parents to represent empty DAG and compute its score.

        self.parents = {n: set() for n in self.data.get_order()}
        # print('\nStart scoring empty graph ...')
        self.score = sum([HCWorker.nscore(n, set(), self.data, self.params)[0]
                          for n in self.parents.keys()])
        # print('\nEnd scoring empty graph')

        # Initialise the tabulist of Tabu learning requested

        self.tabulist = (TabuList(params['tabu']) if 'tabu' in params
                         and params['tabu'] is not None else None)
        if self.tabulist is not None:
            self.tabulist.add(self.parents)
            if 'noinc' not in params:
                params['noinc'] = params['tabu']
        else:
            self.params['noinc'] = 0

        #   Order of edges in deltas defines processing order in algorithm and
        #   it is derived from the ordered specified in the Data object

        order = self.data.get_order()
        self.deltas = {(n1, n2): [self._delta((n1, n2))]
                       for n1 in order for n2 in order if n1 != n2}

        # add arcs defined in any initial knowledge graph - using
        # apply_change_to_dag to do this updates the parents and delta arrays
        # correctly. Also update total score with delta for each add so
        # get correct score for initial DAG.

        if self.knowledge is not False and self.knowledge.initial is not None:
            for arc in self.knowledge.initial.edges:
                self.score += self.deltas[arc][0][0]
                self._apply_change_to_dag(DAGChange(Activity.ADD, arc, 0.0,
                                          {}))
            print('\nInitial parents: {}\n'.format(self.parents))

        # Initialise trace and add first entry (if context not None)

        self._init_trace(context)

        # Initialise learning state variables

        self.iter = 0
        self.num_noinc = 0
        self.best_parents = deepcopy(self.parents)
        self.best_parents_score = self.score
        self.score2 = round(self.score, 10)
        self.best = None
        self.paused = None

    def run(self):
        """
            Run iterations of hill-climbing learning until it either completes
            at a local score maximum or a PAUSE is signalled.
        """
        debug = False
        self.paused = False
        self.best = BestDAGChanges(None, None)

        while (self.best.top is None or self.best.top.activity
                not in {Activity.STOP, Activity.PAUSE}):

            self.best = BestDAGChanges()  # initialise best change
            self.iter += 1

            # Find two highest scoring arc changes consistent with knowledge

            for arc, delta in self.deltas.items():
                self._check_possible_arc_changes_against_best(arc, delta)

            # Maybe alter the best change according to knowledge

            if self.knowledge is not False:
                self.best, know_event = \
                    self.knowledge.hc_best(self.best, 6, self.data,
                                           self.parents)
            else:
                know_event = None

            # Having identified best change, update total score with its delta,
            # and if the delta is non-positive, then increment num_noinc

            if (self.best.top.activity not in {Activity.NONE, Activity.PAUSE}
                    and self.best.top.delta is not None):
                self.score += self.best.top.delta  # increment DAG score
                if self.best.top.delta <= 0.0:
                    self.num_noinc += 1
                    if debug:
                        print('{} iterations without improvement'
                              .format(self.num_noinc))

            # if maxiter is reached or we have exceeded the limit on
            # non-positive deltas then stop the learning process

            if ((self.params and 'maxiter' in self.params
                 and self.iter > self.params['maxiter'])
                    or self.num_noinc > self.params['noinc']):
                self.best.top = DAGChange(Activity.STOP, None, 0.0, {})

            if self.best.top.activity not in {Activity.STOP, Activity.NONE,
                                              Activity.PAUSE}:

                # Apply the change to the DAG updating the parents and deltas
                # to take account of the change, and add DAG to Tabu list

                self._apply_change_to_dag(self.best.top)
                if self.tabulist is not None:
                    self.tabulist.add(self.parents)

                if self.score > self.best_parents_score:

                    if debug:
                        print('Score has increased')

                    # Also, reset the count of non-positive deltas if running
                    # in bnlearn identical mode and if the new score is
                    # "significantly" better - this prevents
                    # infinite loops because of numerical drift.

                    if ('bnlearn' in self.params
                        and self.params['bnlearn'] is True
                        and not values_same(self.score,
                                            self.best_parents_score, sf=10)):
                        self.num_noinc = 0
                        if debug:
                            print('Resetting no improvement counter')

                    # record if this is highest scoring DAG so far

                    self.best_parents = deepcopy(self.parents)
                    self.best_parents_score = self.score

                if debug:
                    print(('    change at iter {} ({}) is {} of {} with ' +
                           'delta {:.3f} and total score {:.3f}\n')
                          .format(self.iter, asctime(localtime()),
                                  self.best.top.activity.value,
                                  self. best.top.arc,
                                  self.best.top.delta / self.data.N,
                                  self.score / self.data.N))

            elif self.best.top.activity == Activity.STOP:
                self.parents = deepcopy(self.best_parents)
                self.score = self.best_parents_score

            self.score2 = round(self.score, 10)
            if self.best.top.activity == Activity.PAUSE:
                self.iter -= 1
                self.paused = True
            elif self.trace is not None:
                self._add_trace_entry(know_event)

        return self

    def clone(self, sequence=None, pause=None):
        """
            Makes a "clone" of an HCWorker object which references the *same*
            data and parameters, but has its own copies of run time variables
            such as deltas, iteration number, knowledge, trace etc.

            :param tuple/None sequence: decision sequence for clone.
            :param bool/None pause: pause flag

            :raises TypeError: for bad arg types
            :raises ValueError: for bad arg values
        """
        _clone = HCWorker.__new__(HCWorker)

        # Copy references for data, params so they are shared between clones

        _clone.data = self.data
        _clone.params = self.params

        # these run time variables are simple types so initiallu they
        # will reference the same values, but they will diverge as the run()
        # method proceeds on the clone

        _clone.score = self.score
        _clone.score2 = round(self.score)
        _clone.iter = self.iter
        _clone.num_noinc = self.num_noinc
        _clone.best_parents_score = self.best_parents_score
        _clone.paused = self.paused

        # use deep copies for other variables as these
        # will be different as each clone runs

        _clone.knowledge = deepcopy(self.knowledge)
        _clone.parents = deepcopy(self.parents)
        _clone.deltas = deepcopy(self.deltas)
        _clone.tabulist = deepcopy(self.tabulist)
        _clone.trace = deepcopy(self.trace)
        _clone.best = deepcopy(self.best)
        _clone.best_parents = deepcopy(self.best_parents)

        # update sequence information if specified using set_sequence which
        # also validates arguments given.

        if sequence is not None or pause is not None:
            if not isinstance(_clone.knowledge, Knowledge):
                raise ValueError('HCWorker.clone() bad arg values')
            _seq = (sequence if sequence is not None
                    else _clone.knowledge.sequence)
            _pause = (pause if pause is not None
                      else _clone.knowledge.pause)
            _clone.knowledge.set_sequence(_seq, _pause)

        return _clone

    @classmethod
    def init_score_cache(self):
        """
            Initialises the score cache - should be done when sample size,
            network or score related parameters change.
        """
        # print('\n+++ HCWorker score cache [{}] initialised\n'
        #       .format(id(HCWorker.score_cache)))
        HCWorker.score_cache = {}

    def _check_possible_arc_changes_against_best(self, arc, delta):
        """
            Identifies the best change for a particular arc and updates
            the current best changes if it is an improvement on them.

            :param tuple arc: (n1, n2) arc for which best change required
            :param list delta: deltas for the arc.
            :param BestDAGChanges best: current best changes

            :returns DAGChange: change giving highest delta
        """
        top = None  # highest scoring change for arc's delta (if any)
        second = None  # 2nd highest scoring change for arc's delta (if any)
        best = self.best  # 2 highest-scoring DAG change in iteration so far

        # First, identify scores of possible changes to this arc

        if len(delta) == 1:

            # arc doesn't exist so can only be add arc - check if possible

            top = DAGChange(Activity.ADD, arc, delta[0][0], delta[0][1])
            top = self._change_permitted(top)

        elif len(delta) > 1:

            # check if delete a possibility

            delete = DAGChange(Activity.DEL, arc, delta[0][0], delta[0][1])
            delete = self._change_permitted(delete)

            # ... and check if reverse possble

            reverse = DAGChange(Activity.REV, arc, delta[1][0], delta[1][1])
            reverse = self._change_permitted(reverse)

            # delete and reverse both possible, so rank according to precedence

            if delete is not None and reverse is not None:
                top = (delete if delta[0][0] > delta[1][0]
                       or values_same(delta[0][0], delta[1][0]) else reverse)
                second = reverse if top == delete else delete
            elif reverse is not None:
                top = reverse
            elif delete is not None:  # change so now deletes have to be valid
                top = delete

        # Secondly, check if the possible changes for this arc, are higher
        # scoring than the highest-scoring changes found so far - is so,
        # update the highest-scoring changes accordingly.

        if top is not None:
            if (self._is_better(top, best.top, self.params['prefer'])
                and ((self.tabulist is None
                      and top.delta > self.params['zero'])
                     or (self.tabulist is not None))):

                # proposed is better than current top and is not prohibited by
                # tabulist (if defined)

                best.second = (best.top if best.top.activity != Activity.STOP
                               else best.second)
                if (second is not None
                    and (best.second is None
                         or second.delta > best.second.delta)):
                    best.second = second
                best.top = top

            elif (self._is_better(top, best.second, self.params['prefer'])
                  and top.activity != Activity.STOP):

                # top is not better than current best, so now see if it is
                # better than current second best - if so update that

                best.second = top

        self.best = best

    def _change_permitted(self, change):
        """
            Checks whether a change to an arc is allowed or not, applying the
            following checks in this order for maximum compatability with
            bnlearn:
                1) does change create a cycle
                2) is it blocked by any tabulist
                3) is change allowed by required and prohibited arcs

            :param DAGChange change: change to check

            :returns DAGChange/None: change if allowed, otherwise None
        """
        activity = change.activity
        arc = change.arc

        # Check Add or Reverse don't create a cycle

        if activity in [Activity.ADD, Activity.REV]:
            new_arc = arc if activity == Activity.ADD else (arc[1], arc[0])
            if DAG.partial_order(self.parents, new_arc=new_arc) is None:
                return None

        # Check against a Tabu list if defined

        if (self.tabulist is not None and
            self.tabulist.hit(self.parents, change,
                              trace=self._is_better(change, self.best.top,
                                                    self.params['prefer']))):
            return None

            # Check change is not prohibited by initial or active knowledge

        if (self.knowledge is not False and
            self.knowledge.blocked(change, (True if self.best.top.delta is None
                                            or (change.delta >
                                                self.best.top.delta)
                                            else False))):
            return None

        return change

    def _is_better(self, proposed, current, prefer):
        """
            Is a proposed change better than a current one.

            :param DAGChange proposed: proposed change
            :param DAGChange best: current best change
            :param prefer Prefer: whether connected or unconnected arcs
                                  preferred for add or neither.

            :returns bool: True iff proposed change better than current best
        """
        def _connected(arc):
            connected = ({c for c in self.parents if len(self.parents[c])} |
                         {p for c in self.parents for p in self.parents[c]})
            return arc[0] in connected or arc[1] in connected

        # if no current best arc then proposed is automatically better!

        if current is None or current.delta is None:
            return True

        # If there is a preference for connected arcs then reject any proposed
        # disconnected arc if the best is connected regardless of score. Apply
        # the reverse logic if disconnected arcs are preferred.

        if (prefer in [Prefer.CONN, Prefer.UNCO]
            and proposed.activity == Activity.ADD
            and current.activity == Activity.ADD
            and ((prefer == Prefer.CONN
                  and _connected(current.arc)
                  and not _connected(proposed.arc)) or
                 (prefer == Prefer.UNCO
                  and not _connected(current.arc)
                  and _connected(proposed.arc)))):
            print('*** Reject {}connected arc {}'
                  .format(('dis' if prefer == Prefer.CONN else ''),
                          proposed.arc))
            return False

        # see if the two score improvements are the same (to 6 s.f.)

        draw = (values_same(proposed.delta, current.delta, sf=6)
                if proposed is not None and current is not None
                and current.delta is not None else False)

        # proposed arc is better if it has a higher score, or it has the same
        # score but a higher priority (to mimic bnlearn behaviour)

        return ((not draw and proposed.delta > current.delta) or
                (draw and proposed.activity.priority >
                 current.activity.priority))

    def _apply_change_to_dag(self, change):
        """
            Make change to DAG, adusting parents and deltas appropriately

            :param DAGChange change: the arc change being made
            :param dict parents: current parents of each node
            :param dict deltas: possible score changes for arcs
        """
        activity = change.activity
        frm, to = change.arc

        #  Change the parents of the nodes affected by the arc being changed.
        #  Also, set the correct length of the deltas for the arc being
        # changed to indicate whether that arc now exists, does not exist, or
        # is blocked from existing because the opposing arc exists. Dummy zero
        # delta values are used at this point - they will be updated to their
        # correct values in the subsequent _update_deltas calls

        if activity == Activity.ADD:
            self.parents[to] = self.parents[to] | {frm}
            self.deltas[(frm, to)] = [(0.0, {}), (0.0, {})]
            self.deltas[(to, frm)] = []
        elif activity == Activity.DEL:
            self.parents[to] = self.parents[to] - {frm}
            self.deltas[(frm, to)] = [(0.0, {})]
            self.deltas[(to, frm)] = [(0.0, {})]
        else:
            self.parents[to] = self.parents[to] - {frm}
            self.parents[frm] = self.parents[frm] | {to}
            self.deltas[(frm, to)] = []
            self.deltas[(to, frm)] = [(0.0, {}), (0.0, {})]

        # Changing an arc always affects the parents of the "to" node.
        # Therefore the deltas of all current and potential inbound arcs to
        # "to" must be recomputed. The deltas of current outbound arcs from
        # "to" must also be recomputed since their reversal deltas are also
        # affected.

        self._update_deltas(to, True)
        self._update_deltas(to, False)

        #   Deltas of arcs incident to the "frm" arc must also be recomputed in
        #   deletion and reversal changes

        if activity != Activity.ADD:
            self._update_deltas(frm, True)
            self._update_deltas(frm, False)

        # print('Made change: {} {}->{} giving new score: {}\n'
        #       .format(activity.value.strip(), frm, to, self.score))

    def _update_deltas(self, target, inbound):
        """
            Update deltas for arcs incident to specified target node.

            :param str target: update deltas of arcs incident this target node
            :param bool inbound: considering inbound or outbound arcs
        """
        for node in self.parents.keys():

            # determine arc to process depending upon inbound or not

            arc = (node, target) if inbound else (target, node)

            # ignore 'self-arcs', arcs blocked because opposing arc exists,
            # and arc additions when doing outbound arcs as these cannot
            # change target node score.

            if node == target or not len(self.deltas[arc]) or \
                    (inbound is False and len(self.deltas[arc]) == 1):
                continue

            if len(self.deltas[arc]) == 1:

                # arc not in graph currently, so just consider arc addition

                self.deltas[arc] = [self._delta(arc)]
            else:

                # arc is in graph currently, so must update deltas for
                # deletion and reversal, latter being combination of deletion
                # delta and delta for addition of arc in reverse direction.

                del_delta = self._delta(arc)
                add_delta = self._delta((arc[1], arc[0]))
                rev_counts = {k: 0.5 * (del_delta[1][k] + add_delta[1][k])
                              for k in del_delta[1].keys()}
                rev_delta = (0.0 if self.parents[arc[1]] - {arc[0]} ==
                             self.parents[arc[0]]
                             else del_delta[0] + add_delta[0], rev_counts)
                self.deltas[arc] = [del_delta, rev_delta]

                # print('     update {} delta to {}'.format(arc, deltas[arc]))

    def _delta(self, arc):
        """
            Generate delta and counts info for the possible change to an
            individual arc. The possible type of change is determined according
            to whether the arc currently exists or not - that is, a deletion
            or addition respectively.

            :param tuple arc: arc (n1, n2) being added or deleted

            :returns tuple: (delta, counts) for the possible change
        """

        # Work out new parents of arc[1] depending upon whether this is an arc
        # addition or deletion indicated by whether arc[0] is a parent of
        # arc[1] currently

        new_parents = (self.parents[arc[1]] - {arc[0]}
                       if arc[0] in self.parents[arc[1]]
                       else self.parents[arc[1]] | {arc[0]})

        # Determine score and counts delta based on, before and after change

        before = HCWorker.nscore(arc[1], self.parents[arc[1]], self.data,
                                 self.params)
        after = HCWorker.nscore(arc[1], new_parents, self.data, self.params)

        # Average counts from before and after change

        counts = {k: 0.5 * (before[1][k] + after[1][k])
                  for k in before[1].keys()}

        return (round(after[0] - before[0], 10), counts)

    def _score(self, node, parents):
        """
            Returns score, and counts on which that score is based, for a
            node with a specified set of parents. This method caches
            scores and counts.

            :param str node: node score required for
            :param set parents: parents of node for this score

            :returns tuple: (score, counts)
        """

        # If score for this node parent combination is not in the cache,
        # compute it and place it in the cache

        key = (node, tuple(parents))
        parents = {node: list(parents)} if len(parents) else {}
        if key not in HCWorker.score_cache:
            # action = 'MISS'
            _params = dict(self.params)
            scores = [_params.pop('score')]
            score, counts = node_score(node, parents, scores, _params,
                                       self.data, counts_reqd=True)
            HCWorker.score_cache.update({key: (score[self.params['score']],
                                        counts)})
        # else:
            # action = ' HIT'

        # print('     cache {} for {}:{} of {}'
        #       .format(action, node,
        #               parents[node] if node in parents else '[]',
        #               HCWorker.score_cache[key]))

        return HCWorker.score_cache[key]  # return cached score & counts

    @classmethod
    def nscore(self, node, parents, data, params):
        """
            Returns score, and counts on which that score is based, for a
            node with a specified set of parents. This method caches
            scores and counts.

            :param str node: node score required for
            :param set parents: parents of node for this score
            :param Data data: data learning from
            :param str params: learning params incl. score type, log base

            :returns tuple: (score, counts)
        """

        # If score for this node parent combination is not in the cache,
        # compute it and place it in the cache

        key = (node, tuple(parents))
        parents = {node: list(parents)} if len(parents) else {}
        if key not in HCWorker.score_cache:
            # action = 'MISS'
            _params = dict(params)
            scores = [_params.pop('score')]
            score, counts = node_score(node, parents, scores, _params, data,
                                       counts_reqd=True)
            HCWorker.score_cache.update({key: (score[params['score']],
                                               counts)})
        # else:
            # action = ' HIT'

        # print('     cache {} for {}:{} of {}'
        #       .format(action, node,
        #               parents[node] if node in parents else '[]',
        #               HCWorker.score_cache[key]))

        return HCWorker.score_cache[key]  # return cached score & counts

    def _init_trace(self, context):
        """
            Initialises the trace if context is provided.

            :param dict/None context: context information for trace, or no
                                      trace required if context is None.
        """
        if context is None:
            self.trace = None
        else:
            context = context.copy()
            context.update({'algorithm': 'HC', 'N': self.data.N,
                            'params': self.params,
                            'dataset': isinstance(self.data, Pandas)})
            if self.knowledge is not False:
                context.update({'knowledge': self.knowledge.label,
                                'initial': self.knowledge.initial})
            self.trace = Trace(context)

            blocked = None if self.tabulist is None else []
            self.trace.add(Activity.INIT, {Detail.DELTA: self.score,
                                           Detail.BLOCKED: blocked})

    def _add_trace_entry(self, know_event):
        """
            Add an entry to trace including information about the change
            made (best) and the second best change.

            :param KnowledgeOutcome/None know_event: knowledge event detail
        """
        def _rounded_count(key):
            return round(self.best.top.counts[key], 1)

        details = {Detail.DELTA: (self.score
                                  if self.best.top.activity == Activity.STOP
                                  else self.best.top.delta),
                   Detail.ARC: self.best.top.arc}

        if self.best.second is not None:
            details.update({Detail.ACTIVITY_2: self.best.second.activity.value,
                            Detail.ARC_2: self.best.second.arc,
                            Detail.DELTA_2: self.best.second.delta})

        if self.best.top.counts is not None and 'min' in self.best.top.counts:
            details.update({Detail.MIN_N: _rounded_count('min'),
                            Detail.MEAN_N: _rounded_count('mean'),
                            Detail.MAX_N: _rounded_count('max'),
                            Detail.LT5: _rounded_count('lt5'),
                            Detail.FPA: _rounded_count('fpa')})

        # Obtain details of any changes blocked by a tabulist in this iteration
        # (this has effect of clearing the list of blocked changes too).
        # Only report blocked changes with non-positive score if change made
        # for that iteration has a non-negative score and is not when the limit
        # on the number of iterations with no score increase has beem exceeded.
        # This matches the blocks reported by bnlearn's implementation of tabu.

        blocked = ([e for e in self.tabulist.blocked() if e[2] > 0 or
                    (details[Detail.DELTA] <= 0.0 and self.num_noinc <=
                     len(self.tabulist.tabu))]
                   if self.tabulist is not None else None)
        details.update({Detail.BLOCKED: blocked})

        know_event = (None if know_event is None else
                      (know_event.rule.value, know_event.correct,
                       know_event.outcome.value, know_event.arc))
        details.update({Detail.KNOWLEDGE: know_event})

        self.trace.add(self.best.top.activity, details)


class Prefer(EnumWithAttrs):
    """
        Defines preference for kind of arc to add - connected, unconnected or
        none

        :ivar str value: short string code for preference
        :ivar str label: human-readable label for preference
    """
    CONN = "conn", "connected arc"
    UNCO = "unco", "unconnected arc"
    NONE = "none", "no preference"
