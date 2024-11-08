
# Encapsulates knowledge rules

from core.common import EnumWithAttrs, ln
from core.metrics import values_same
from core.indep import indep
from core.score import node_score
from learn.trace import Activity


class KnowledgeOutcome(EnumWithAttrs):
    """
        Describes the outcome of a knowledge event

        :ivar str value: short string code describing outcome
        :ivar str label: human-readable label for outcome
    """
    NO_OP = 'no_op', 'No operation'
    SWAP_BEST = 'swap_best', 'Swap best and 2nd best'
    STOP_ADD = 'stop_add', 'Stop add misorientated'
    STOP_DEL = 'stop_del', 'Stop delete true arc'
    STOP_REV = 'stop_rev', 'Stop reverse true arc'
    STOP_EDGE = 'stop_edge', 'Stop edge (obsolete)'
    EXT_ADD = 'ext_add', 'Stop add extra'
    EXT_REV = 'ext_rev', 'Stop reverse extra'


class Rule(EnumWithAttrs):
    """
        Defines types of Knowledge rule.

        :ivar str value: short string code for rule
        :ivar str label: human-readable label for rule
    """
    EQUIV_ADD = "equiv_add", "Choose equivalent add"
    EQUIV_SEQ = "equiv_seq", "Equivalent add sequence"
    STOP_ARC = "stop_arc", "Prohibited arc"
    REQD_ARC = "reqd_arc", "Required arc"
    TIERS = "tiers", "Topological tiers"
    ACT_CACHE = "act_cache", "Active cache"
    MI_CHECK = "mi_check", "MI check"
    BIC_UNSTABLE = "bic_unstable", "BIC unstable"
    POS_DELTA = "pos_delta", "Positive score delta"
    LO_DELTA = "lo_delta", "Low score delta"
    HI_LT5 = "hi_lt5", "High <5 cell counts"

    # ignore the first param since it's set by __new__ in superclass
    def __init__(self, _: str, label: str):
        self._label_ = label

    def match(self, best, second, date, parents, knowledge, sf):
        """
            Test if the rule is matched.

            :param DAGChange best: best proposed change
            :param DAGChange second: second best proposed change
            :param Knowledge knowledge: Knowledge object including params etc.
            :param Data date: data learning structure from
            :param dict parents: {node: {parents}} at start of this iteration
            :param int sf: number of s.f. to use for equality comparisons

            :returns Rule/None: Rule.POS_DELTA if rule applies else None
        """
        if self.value in {'equiv_add', 'equiv_seq'}:
            return self.test_equiv_add(best, second, sf)

        elif self.value == 'mi_check':
            return self.test_mi_discrepancy(best, second, sf, date)

        elif self.value == 'bic_unstable':
            return self.test_bic_unstable(best.activity, best.arc,
                                          knowledge.threshold, date, parents)

        elif self.value == 'hi_lt5':
            return self.test_hi_lt5(best.activity, best.counts,
                                    knowledge.threshold)

        elif self.value == 'pos_delta':
            return self.test_pos_delta(best.activity, best.delta)

        elif self.value == 'lo_delta':
            return self.test_lo_delta(best.delta, knowledge.max_abs_delta,
                                      knowledge.threshold)
        else:
            return None

    def test_equiv_add(self, best, second, sf):
        """
            Test if the equivalent add/sequence rule applies.

            :param DAGChange best: best proposed change
            :param DAGChange second: second best proposed change
            :param int sf: number of s.f. used when checking for score equality

            :returns Rule/None: Rule.EQUIV_ADD if rule applies else None
        """
        return (self if
                best.activity == Activity.ADD
                and second.activity == Activity.ADD
                and best.arc == (second.arc[1], second.arc[0])
                and values_same(best.delta, second.delta, sf) else None)

    def test_mi_discrepancy(self, best, second, sf, date):
        """
            Test if the MI discrepancy rule applies.

            :param DAGChange best: best proposed change
            :param DAGChange second: second best proposed change
            :param int sf: number of s.f. used when checking for score equality
            :param Data date: data learning structure from

            :returns Rule/None: Rule.MI_CHECK if rule applies else None
        """
        if (best.activity != Activity.ADD
                or values_same(best.delta, second.delta, sf)):
            return None

        # compute mutual information (MI) between two nodes

        N = date.N
        mi = round(indep(best.arc[0], best.arc[1], None,
                         date.sample)['mi']['statistic'] / (2 * N), 6)
        delta = round(best.delta / N, 6)

        # take log of ratio of MI to score delta as "relative MI" ... low
        # values of MI can be indicative of extraneous edges

        rel_mi = (0.0 if mi == 0.0 else
                  (1.0 if delta == 0.0 else
                   min(max(1.0 + 0.1 * ln(mi / delta, 2), 0.0), 1.0)))
        # print('\n*** MI_CHECK: mi={:.5f}, norm delta={:.5f}, rel_mi={:.5f}'
        #       .format(mi, delta, rel_mi))

        return Rule.MI_CHECK if rel_mi < self.threshold else None

    def test_bic_unstable(self, activity, arc, threshold, data, parents):
        """
            Check if the BIC score is unstable by comparing value for the
            two halves of the dataset.

            :param Activity activity: activity being performed e.g. add
            :param tuple arc: arc change being made to
            :param float threshold: sensitivity threshold
            :param Data date: data learning structure from
            :param dict parents: {node: {parents}} at start of this iteration

            :returns Rule/None: Rule.BIC_UNSTABLE if rule applies else None
        """

        # may trigger for add, delete or reverse

        if activity not in {Activity.ADD, Activity.DEL, Activity.REV}:
            return None

        # determine nodes whose parents change with this activity

        changed = {}
        frm, to = arc
        if activity == Activity.ADD:
            changed[to] = list(parents[to] | {frm})
        elif activity == Activity.DEL:
            changed[to] = list(parents[to] - {frm})
        elif activity == Activity.REV:
            changed[to] = list(parents[to] - {frm})
            changed[frm] = list(parents[frm] | {to})

        # Obtain LL score for full and half sample to estimate the stability of
        # the log likelihood score for nodes with changed parents. Trigger
        # rule if instability above specified threshold.

        N = data.N
        half = round(data.N / 2)
        for node, parents in changed.items():

            # This line ensures parents is in correct format, that is
            # {node: [parents]} for checks in the score functions.
            # This version sets the key to be 'node' rather than the correct
            # node value so that the bug where parents are effectively
            # ignored persists because these are results used in paper.

            parents = {} if len(parents) == 0 else {'node': parents}
            ll = node_score(node, parents, ['loglik'], {'base': 'e'},
                            data)['loglik']
            data.set_N(half)
            ll_h = node_score(node, parents, ['loglik'], {'base': 'e'},
                              data)['loglik']
            data.set_N(N)
            instability = (abs(round((ll - 2 * ll_h) / ll, 3)) if ll < 0.0
                           else 0.0)
            print('\n*** BIC INSTABILITY: {} {}: {:.3f} vs {:.3f} -> {}'
                  .format(node, parents, (ll - ll_h), ll_h, instability))
            if instability > threshold:
                return Rule.BIC_UNSTABLE

        return None

    def test_hi_lt5(self, activity, counts, threshold):
        """
            Check if the LT5 count is high - the LT5 count is a measure of how
            may parental value combinations have less than 5 supporting
            instances.

            :param Activity activity: activity being performed e.g. add
            :param dict counts: counts associated with this activity
            :param float threshold: test threshold

            :returns Rule/None: Rule.HI_LT5 if rule applies else None
        """
        print('\n*** HI_LT5: {}'.format(counts['lt5']))

        return (Rule.HI_LT5 if counts['lt5'] is not None
                and counts['lt5'] > threshold else None)

    def test_lo_delta(self, delta, max_abs_delta, threshold):
        """
            Test if the score delta is low - that is below a threshold
            fraction of the highest (i.e. in hill climbing, initial) delta.

            :param float delta: delta associated with change
            :param float max_abs_delta: maximum delta so far
            :param float threshold: test threshold ValueError

            :returns Rule/None: Rule.LO_DELTA if rule applies else None
        """
        print('\n*** LO_DELTA: {}'.format(delta))
        return (self if max_abs_delta is not None and
                abs(delta) < max_abs_delta * threshold else None)

    def test_pos_delta(self, activity, delta):
        """
            Test if the score delta is positive - since it always is in
            hill-climbing, this rule triggers on every iteration and
            represents a baseline.

            :param Activity activity: activity being performed e.g. add
            :param float/None delta: delta for the change

            :returns Rule/None: Rule.POS_DELTA if rule applies else None
        """
        # print('\n*** POS_DELTA: {}'.format(delta))
        return (Rule.POS_DELTA
                if activity in {Activity.ADD, Activity.DEL, Activity.REV}
                and delta is not None and delta > 0.0 else None)


class RuleSet(EnumWithAttrs):
    """
        Defines set of Knowledge rules.

        :ivar str value: short string code for rule set
        :ivar str label: human-readable label for rule set
        :ivar list rules: rules in this set
    """
    EQUIV_ADD = "equiv_add", "Choose equivalent add", [Rule.EQUIV_ADD]
    EQUIV_SEQ = "equiv_seq", "Equivalent add sequence", [Rule.EQUIV_SEQ]
    STOP_ARC = "stop_arc", "Prohibited arc", [Rule.STOP_ARC]
    REQD_ARC = "reqd_arc", "Required arc", [Rule.REQD_ARC]
    TIERS = "tiers", "Topological tiers", [Rule.TIERS]
    MIX_ARC = "rqsp_arc", "Required arc", [Rule.REQD_ARC, Rule.STOP_ARC]
    MI_CHECK = "mi_check", "Check MI", [Rule.MI_CHECK]
    EQUIV_LT5 = "equiv_lt5", "Equiv add and high LT5", [Rule.EQUIV_ADD,
                                                        Rule.HI_LT5]
    BIC_UNSTABLE = "bic_unstable", "BIC unstable", [Rule.BIC_UNSTABLE]
    POS_DELTA = "pos_delta", "Positive score delta", [Rule.POS_DELTA]
    LO_DELTA = "lo_delta", "Low score delta", [Rule.LO_DELTA]
    HI_LT5 = "hi_lt5", "High <5 cell counts", [Rule.HI_LT5]

    # ignore the first param since it's already set by __new__
    def __init__(self, _: str, label: str, rules: list = None):
        self._label_ = label
        self._rules_ = rules

    # this makes sure that the rules are read-only
    @property
    def rules(self):
        return self._rules_

    def match(self, best, second, sf, date, parents, knowledge):
        """
            Returns first rule that has match otherwise None

            :param DAGChange best: best proposed change
            :param DAGChange second: second best proposed change
            :param int sf: number of s.f. used when checking for score equality
            :param Data date: data learning structure from
            :param Knowledge knowledge: knowledge including parameters
            :param dict parents: {node: {parents}} at start of this iteration

            :returns Rule: Triggering rule if any, otherwise None
        """
        match = None
        for rule in self.rules:
            match = rule.match(best, second, date, parents, knowledge, sf)

            if match is not None:
                break

        return match


class KnowledgeEvent():
    """
        Describes a Knowledge event: the rule that triggered it, whether
        knowledge supplied was correct or not, and the outcome.

        :param Rule rule: rule that triggered the event
        :param bool/None correct: whether the knowledge supplied was correct or
                                  None if no knowledge requested
        :param KnowledgeOutcome: the outcome of the event
        :param tuple arc: arc which event affects

        :raises TypeError: if arguments have incorrect types
        :raises ValueError: if arguments have incompatible values
    """
    def __init__(self, rule: Rule, correct: bool, outcome: KnowledgeOutcome,
                 arc: tuple = ('?', '?')):

        if (not isinstance(rule, Rule)
                or (correct is not None and not isinstance(correct, bool))
                or not isinstance(outcome, KnowledgeOutcome)
                or not isinstance(arc, tuple) or len(arc) != 2
                or not isinstance(arc[0], str) or not isinstance(arc[1], str)):
            raise TypeError('KnowledgeEvent() bad arg type')

        if correct is None and outcome != KnowledgeOutcome.NO_OP:
            raise ValueError('KnowledgeEvent() incompatible values')

        self.rule = rule
        self.correct = correct
        self.outcome = outcome
        self.arc = arc

    def __eq__(self, other):
        """
            Is other KnowledgeEvent equal to this one
        """
        return (isinstance(other, KnowledgeEvent)
                and self.rule == other.rule
                and self.correct == other.correct
                and self.outcome == other.outcome
                and self.arc == other.arc)

    def __str__(self):
        """
            Friendly description of KnowledgeEvent

            :returns str: human readable description.
        """
        return ('{} ({}) {} {}'
                .format(self.rule,
                        ('T' if self.correct is True else
                         ('F' if self.correct is not None else 'None')),
                        self.outcome, self.arc))
