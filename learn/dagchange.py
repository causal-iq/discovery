
# Encapsulates changes to a DAG made during structure learning

from learn.trace import Activity


class DAGChange():
    """
        Defines change made to a DAG

        :param Activity activity: type of change: add, delete etc
        :param tuple arc: arc to which change applies
        :param float delta: change in score associated with change
        :param dict counts: statistics associated with change

        :ivar Activity activity: type of change: add, delete etc
        :ivar tuple arc: arc to which change applies
        :ivar float delta: change in score associated with change
        :ivar dict counts: statistics associated with change

        :raises TypeError: if initialiser has bad arg types
    """
    def __init__(self, activity, arc=None, delta=None, counts=None):
        if (not isinstance(activity, Activity) or
                (arc is not None and not isinstance(arc, tuple)) or
                (delta is not None and not isinstance(delta, float)) or
                (counts is not None and not isinstance(counts, dict))):
            raise TypeError('DAGChange bad arg type')
        pass

        self.activity = activity
        self.arc = arc
        self.delta = delta
        self.counts = counts

    def __iter__(self):
        return iter([self.activity, self.arc, self.delta, self.counts])

    # def __getitem__(self, key):
    #     return list(self.__iter__())[key]

    def __str__(self):
        """
            Return textual description of chnage

            :returns str: textual description of change
        """
        return ('{}, score={}'.format(self.activity, self.delta)
                if self.activity in [Activity.INIT, Activity.STOP] else
                '{} {}, score={}, count={}'.format(self.activity, self.arc,
                                                   self.delta, self.counts))

    def __eq__(self, other):
        """
            Test if pther DAGChange is identical to this one

            :param DAGChange other: DAG change to compare with self

            :returns bool: True if other is identical to self
        """
        return (isinstance(other, DAGChange)
                and self.activity == other.activity
                and self.arc == other.arc
                and self.delta == other.delta
                and self.counts == other.counts)


class BestDAGChanges():
    """
        Holds the best and second best DAG change.

        :param DAGChange top: current best change
        :param DAGChange second: second best change

        :ivar DAGChange top: current best change
        :ivar DAGChange second: second best change

        :raises TypeError: if bad arg types
    """
    def __init__(self, top=DAGChange(Activity.STOP), second=None):
        if ((top is not None and not isinstance(top, DAGChange)) or
                (second is not None and not isinstance(second, DAGChange))):
            raise TypeError('BestDAGChanges() bad arg type')

        self.top = (DAGChange(top.activity, top.arc, top.delta, top.counts)
                    if top is not None else None)
        self.second = (DAGChange(second.activity, second.arc, second.delta,
                                 second.counts) if second is not None
                       else None)

    def __eq__(self, other):
        """
            Check if other identical to self.

            :param BestDAGChanges other: other instance to compare

            :returns bool: true if identical otherwise false
        """
        return (isinstance(other, BestDAGChanges)
                and self.top == other.top and self.second == other.second)

    def __str__(self):
        """
            Human-readable description of object

            :returns str: textual description of best DAG changes.
        """
        return 'top change is {} and second {}'.format(self.top, self.second)
