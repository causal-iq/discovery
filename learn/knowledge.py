
# Encapsulates knowledge to aid structure learning

from copy import deepcopy

from core.common import stable_random, init_stable_random
from learn.trace import Activity
from learn.dagchange import BestDAGChanges, DAGChange
from learn.knowledge_rule import Rule, RuleSet, KnowledgeOutcome, \
    KnowledgeEvent
from learn.knowledge_params import KnowledgeParams


class Knowledge():
    """
        Knowledge for structure learning

        :param RuleSet rules: knowledge rules that will be applied
        :param dict params: knowledge parameters e.g. limit, ref
        :param int sample: sample number for Knowledge object

        :ivar RuleSet rules: knowledge rules that will be applied
        :ivar BN ref: reference BN for dataset
        :ivar int limit: limit on number of times knowledge can be applied
        :ivar int count: number of times knowledge was requested
        :ivar int ignore: how many knowledge request to ignore initially
        :ivar float expertise: expertise of simulated expert
        :ivar bool partial: expert supplies partial or complete knowledge?
        :ivar float threshold: threshold threshold for triggers
        :ivar tuple sequence: decision sequence for EQUIV_SEQ rule

        :ivar str label: description of the knowledge
        :ivar dict stop: prohibited arcs: {arc: (correct, initial)}
        :ivar dict reqd: required arcs: {arc: (correct, initial)}
        :ivar KnowledgeEvent event: knowledge event with highest delta
        :ivar float event_delta: delta for event
        :ivar DAG initial: initial DAG

        :raises TypeError: if arguments have bad types
        :raises ValueError:  if arguments have bad values
    """

    def __init__(self, rules, params=None, sample=0):

        if (not isinstance(rules, RuleSet) or not isinstance(sample, int)
                or (params is not None and not isinstance(params, dict))):
            raise TypeError('Knowledge(): bad arg types')

        if sample < 0 or sample > 100:
            raise ValueError('Knowledge(): bad arg value')

        params = {} if params is None else params

        # Check parameter values, and set up reqd or stop arcs if required

        KnowledgeParams.check(params, rules, sample)

        self.rules = rules
        self.ref = params['ref'] if 'ref' in params else None
        self.limit = (params['limit'] if 'limit' in params else False)
        self.ignore = (params['ignore'] if 'ignore' in params else 0)
        self.expertise = (params['expertise'] if 'expertise' in params
                          else 1.0)
        self.earlyok = params['earlyok'] if 'earlyok' in params else None
        self.partial = params['partial'] if 'partial' in params else False
        self.sequence = params['sequence'] if 'sequence' in params else None
        self.pause = params['pause'] if 'pause' in params else False
        self.order = params['order'] if 'order' in params else None
        self.threshold = (params['threshold'] if 'threshold' in params
                          else 0.05)
        self.count = 0
        self.stop = ({} if 'stop' not in params
                     else {a: (v, True) for a, v in params['stop'].items()})
        self.reqd = ({} if 'reqd' not in params
                     else {a: (v, True) for a, v in params['reqd'].items()})
        self.event = None
        self.event_delta = None
        self.max_abs_delta = None
        self.initial = (params['initial'] if 'initial' in params and
                        params['initial'] is not False else None)
        self.label = self._label(rules)

        # re-seed stable random numbers for decision randomisation

        if self.expertise != 1.0:
            print('Init decision randomisation with {}'.format(sample))
            init_stable_random(sample)

    def set_sequence(self, sequence, pause):
        """
            Set the sequence of decisions used by the EQUIV_SEQ rule to the
            specified sequence of booleans.

            :param tuple sequence: new value for true/false sequence.
            :param bool pause: whether to pause when past end of sequence

            :raises TypeError: if bad arg type for sequence
            :raises ValueError: if Knowledge doesn't include EQUIV_SEQ rule
        """
        if (not isinstance(sequence, tuple) or not len(sequence)
                or not all([isinstance(b, bool) for b in sequence])
                or not isinstance(pause, bool)):
            raise TypeError('Knowledge.set_sequence() bad arg types')

        if Rule.EQUIV_SEQ not in self.rules.rules:
            raise ValueError('Knowledge.set_sequence() only for EQUIV_SEQ')

        self.sequence = deepcopy(sequence)
        self.pause = pause

    def _label(self, rules):
        """
            Descriptive label for Knowledge

            :param RuleSet rules: rule set being used
            :param dict params: (original) knowledge parameter values

            :returns str: human-readable description of Knowledge
        """
        if rules == RuleSet.EQUIV_ADD:
            label = 'limit {}, ignore {}, partial {} and expertise {}' \
                    .format(self.limit, self.ignore, self.partial,
                            round(self.expertise, 3))
        elif rules == RuleSet.BIC_UNSTABLE:
            label = 'limit {}, threshold {}, partial {} and expertise {}' \
                    .format(self.limit, round(self.threshold, 3),
                            self.partial, round(self.expertise, 3))
        elif rules == RuleSet.STOP_ARC:
            label = '{} prohibited and expertise {}' \
                    .format(len(self.stop), round(self.expertise, 3))
        elif rules == RuleSet.REQD_ARC:
            label = '{} required and expertise {}' \
                    .format(len(self.reqd), round(self.expertise, 3))
        elif rules == RuleSet.TIERS:
            label = '{} prohibited and expertise {}' \
                    .format(len(self.stop), round(self.expertise, 3))
        elif rules == RuleSet.EQUIV_SEQ:
            label = 'sequence of length {} then {}pause' \
                    .format(len(self.sequence),
                            '' if self.pause is True else 'no ')
        else:
            label = 'no parameters'

        return 'Ruleset "{}" with {}'.format(rules.label, label)

    def get_arc_knowledge(self, arc, trigger):
        """
            Simulates obtaining knowledge about a possible arc from an expert.

            :param tuple arc: arc to obtain knowledge for

            :returns tuple: (answer: arc expert says exists between the nodes,
                             correct: whether this a correct)
        """

        # See if equest can be serviced from cached knowledge

        opp = (arc[1], arc[0])
        if arc in self.reqd:
            return (arc, self.reqd[arc][0])
        elif opp in self.reqd:
            return (opp, self.reqd[opp][0])
        elif arc in self.stop and opp in self.stop:
            return (None, self.stop[arc][0])

        self.count += 1
        print("\nKnowledge request {} for {} made".format(self.count, arc))

        # whether knowledge provided by simulated expert should be correct is
        # based on random number and level of "expertise". If earlyok flag is
        # True we ensure early AL decisions are correct.

        expertise = ((1.0 if self.count <= round(0.5 * self.limit)
                      else max(2 * self.expertise - 1.0, 0.0))
                     if self.earlyok is True else self.expertise)
        correct = expertise == 1.0 or stable_random() <= expertise

        # Obtain the reference arc between these two nodes (can be None)

        if arc in self.ref.dag.edges:
            ref = arc
        elif opp in self.ref.dag.edges:
            ref = opp
        else:
            ref = None

        if correct:

            # Return ref if answer should be correct, unless ref is None
            # and partial true in which case randomly choose orientation

            print('Ref is {}'.format(ref))
            if self.partial and ref is None:
                answer = arc if stable_random() > 0.5 else opp
                correct = False
            else:
                answer = ref

        else:

            # Remove ref (correct) arc and if partial is True, None from
            # consideration. Randomly choose if there are still 2 options.

            wrong = [None, arc, opp]
            wrong.remove(ref)
            if self.partial and None in wrong:
                wrong.remove(None)
            if len(wrong) == 2:
                answer = wrong[0] if stable_random() > 0.5 else wrong[-1]
            else:
                answer = wrong[0]

        print('   --- Expert {}correctly says arc is {}'
              .format(('' if correct is True else 'in'), answer))
        if answer is not None:
            print('   --- Dynamically requiring arc {}'.format(answer))
            self.reqd.update({answer: (correct, False)})
        else:
            print('   --- Dynamically prohibiting edge {}'.format(arc))
            self.stop.update({arc: (correct, False)})
            self.stop.update({opp: (correct, False)})

        return (answer, correct)

    def new_best(self, trigger, know_arc, correct, best):
        """
            Update best changes according to answer given by expert.

            :param Rule trigger: rule that triggered knowledge request
            :param tuple know_arc: arc presence/direction from knowledge
            :param bool correct: whether knowledge correct or not
            :param BestDAGChange best: current best changes

            :raises RuntimeError: if inconsistent knowledge detected
        """
        activity = best.top.activity
        arc = best.top.arc
        opp_arc = (best.top.arc[1], best.top.arc[0])

        if ((activity == Activity.ADD and know_arc == arc)
                or (activity == Activity.DEL and know_arc is None)
                or (activity == Activity.REV and know_arc == opp_arc)):

            # knowledge agrees with proposed change so return NO_OP which
            # means the change will go ahead

            new_best = BestDAGChanges(best.top, best.second)
            outcome = KnowledgeOutcome.NO_OP

        elif (trigger == Rule.EQUIV_ADD and know_arc == opp_arc):

            # special case of an equivalent add where knowledge returns the
            # opposite arc - in this case only, we swap best and second best

            new_best = BestDAGChanges(best.second, best.top)
            outcome = KnowledgeOutcome.SWAP_BEST

        else:

            # in all other cases, the proposed change does not agree with
            # knowledge and we can't simply swap best and second best. Signal
            # the appropriate STOP_XXX outcome and modify the best activity to
            # NONE to force abort of this iteration in HC main loop.

            new_best = BestDAGChanges(best.top, best.second)
            print('   --- Aborting {} of {}'
                  .format(best.top.activity, best.top. arc))
            new_best.top.activity = Activity.NONE  # signal no activity
            if know_arc is None and activity == Activity.ADD:
                outcome = KnowledgeOutcome.EXT_ADD
            elif know_arc is None and activity == Activity.REV:
                outcome = KnowledgeOutcome.EXT_REV
            elif activity == Activity.ADD:
                outcome = KnowledgeOutcome.STOP_ADD
            elif activity == Activity.DEL:
                outcome = KnowledgeOutcome.STOP_DEL
            elif activity == Activity.REV:
                outcome = KnowledgeOutcome.STOP_REV
            else:
                raise RuntimeError('new_best() called with bad activity')

        return (new_best, outcome)

    def hc_best(self, best, sf, data, parents):
        """
            Advise best hill-climbing change on basis of knowledge

            :param BestDAGChanges best: best DAG changes
            :param int sf: number of s.f. used when checking for score equality
            :param Data data: data learning structure from
            :param dict parents: {node: {parents}} at start of this iteration

            :returns tuple: (best: possibly modified best changes
                             event: knowledge event
        """
        trigger = self.rules.match(best.top, best.second, sf, data, parents,
                                   self)
        orig_arc = best.top.arc

        if (trigger is not None and not
                (self.partial is True and best.top.activity == Activity.DEL)):

            if trigger == Rule.EQUIV_SEQ:

                # Do SWAP_BEST or not according to element in sequence

                outcome = KnowledgeOutcome.NO_OP
                if self.count < len(self.sequence):
                    if self.sequence[self.count] is True:
                        best = BestDAGChanges(best.second, best.top)
                        outcome = KnowledgeOutcome.SWAP_BEST
                    self.count += 1
                    correct = True
                else:
                    correct = None
                    if self.pause is True:
                        new_top = DAGChange(Activity.PAUSE, best.top.arc,
                                            best.top.delta, best.top.counts)
                        best = BestDAGChanges(new_top, best.second)

            elif ((self.count < self.ignore) or
                    (self.limit is not False
                     and self.count >= self.limit + self.ignore)):

                # too many requests have been made or not enough requests
                # have been ignored yet (as determined by the limit and ignore)
                # attributes.

                self.count += 1
                print("\nKnowledge request {} for {} of {} {}"
                      .format(self.count, best.top.activity, best.top.arc,
                              ('ignored' if self.count <= self.ignore
                               else 'over limit')))
                correct = None
                outcome = KnowledgeOutcome.NO_OP

            else:

                # request limit not reached, and any requests to be ignored
                # have already been ignored so request knowledge from
                # simulated expert (answer may be correct or incorrect)

                know_arc, correct = self.get_arc_knowledge(best.top.arc,
                                                           trigger)

                # ... now possibly adjust the best change for this iteration on
                # the basis of the supplied knowledge.

                best, outcome = self.new_best(trigger, know_arc, correct, best)

                print(".. {} {}correctly"
                      .format(outcome.value, '' if correct is True else 'in'))

            event = KnowledgeEvent(trigger, correct, outcome, orig_arc)

        else:

            # No trigger was matched OR it was a delete with partial expert

            # return any rule that was triggered within this iteration
            # which blocked a change with a higher score than the current
            # proposed change.

            # print('self.event_delta is {}, best.top.event_delta is {}'
            #       .format(self.event_delta, best.top.delta))
            event = (self.event if self.event is not None
                     and (best.top.delta is None or
                          self.event_delta >= best.top.delta) else None)

        # clear record of in-iteration events ready for next iteration
        # record the highest delta used

        self.event = None
        self.event_delta = None
        if (best.top.activity != Activity.NONE and best.top.delta is not None
            and (self.max_abs_delta is None
                 or abs(best.top.delta) > self.max_abs_delta)):
            self.max_abs_delta = abs(best.top.delta)
            # print('Max delta recorded: {:.3f}'.format(self.max_abs_delta))

        return (best, event)

    def blocked(self, change, trace_event=True):
        """
            Whether proposed change is blocked by knowledge.

            :param DAGChange change: proposed change to check whether blocked
            :param bool trace_event: whether block event should be traced

            :returns KnowledgeEvent/None: event which blocks change, if any
        """

        activity = change.activity
        arc = change.arc
        opp = (arc[1], arc[0])
        rule = None

        if activity == Activity.ADD and arc in self.stop:
            outcome = KnowledgeOutcome.STOP_ADD
            correct = self.stop[arc][0]
            rule = Rule.STOP_ARC if self.stop[arc][1] else Rule.ACT_CACHE

        elif activity == Activity.ADD and opp in self.reqd:
            outcome = KnowledgeOutcome.STOP_ADD
            correct = self.reqd[opp][0]
            rule = Rule.REQD_ARC if self.reqd[opp][1] else Rule.ACT_CACHE

        elif activity == Activity.DEL and arc in self.reqd:
            outcome = KnowledgeOutcome.STOP_DEL
            correct = self.reqd[arc][0]
            rule = Rule.REQD_ARC if self.reqd[arc][1] else Rule.ACT_CACHE

        elif activity == Activity.DEL and opp in self.reqd:
            outcome = KnowledgeOutcome.STOP_DEL
            correct = self.reqd[opp][0]
            rule = Rule.REQD_ARC if self.reqd[opp][1] else Rule.ACT_CACHE

        elif activity == Activity.REV and arc in self.reqd:
            outcome = KnowledgeOutcome.STOP_REV
            correct = self.reqd[arc][0]
            rule = Rule.REQD_ARC if self.reqd[arc][1] else Rule.ACT_CACHE

        elif activity == Activity.REV and opp in self.stop:
            outcome = KnowledgeOutcome.STOP_REV
            correct = self.stop[opp][0]
            rule = Rule.STOP_ARC if self.stop[opp][1] else Rule.ACT_CACHE

        # If change is blocked and it has a bigger delta than any previous
        # blocked changes (in this iteration) then record it for possible
        # output to this iteration's trace entry (if trace_event is True)

        if rule is not None:
            event = KnowledgeEvent(rule, correct, outcome, arc)
            # print('   --- blocking: {} {} {:.1f} with rule {}'.format
            #       (change.activity.value, arc, change.delta, event))
            if (trace_event is True and
                    (self.event is None or (change.delta > self.event_delta))):
                self.event = event
                self.event_delta = change.delta

        else:
            event = None

        return event
