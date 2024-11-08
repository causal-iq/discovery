
# Implement a Tabu list which can be used by HC algorithm

from copy import deepcopy

from learn.trace import Activity
from learn.dagchange import DAGChange


class TabuList():
    """
        Implement a Tabu list of DAGs.

        :param int tabu_len: length of tabu list

        :ivar list tabu: the circular Tabu list
        :ivar int ptr: pointer into list where next entry will be placed
        :ivar list blocked: list of changes blocked by tabu list
    """
    def __init__(self, tabu_len):
        if isinstance(tabu_len, bool) or not isinstance(tabu_len, int):
            raise TypeError('TabuList() bad arg type')

        if tabu_len < 1 or tabu_len > 100:
            raise ValueError('TabuList bad arg value')

        self.tabu = [None] * tabu_len
        self.ptr = 0
        self._blocked = []

    def add(self, parents):
        """
            Adds a new DAG to list at current pointer position.

            :param dict parents: parents of each node which defines DAG
                                 {node: {parent1, parent2, ...}}

            :raises TypeError: if parents not of correct type
        """
        if (not isinstance(parents, dict)
                or not all([isinstance(s, set) for s in parents.values()])):
            raise TypeError('Tabulist.add() bad arg type')
        # print('{} added at elem {}'.format(parents, self.ptr + 1))

        self.tabu[self.ptr] = deepcopy(parents)
        self.ptr = self.ptr + 1 if len(self.tabu) - self.ptr > 1 else 0

    def hit(self, parents, proposed, trace=True):
        """
            Checks whether DAG with proposed change is present in the Tabu list

            :param dict parents: parents of each node which defines DAG
                                 {node: {parent1, parent2, ...}}
            :param DAGChange proposed: proposed changed to DAG
            :param bool record: whether any block would be traced

            :raises TypeError: if bad args type

            returns int/None: position in list of hit, otherwise None. Note
                              elements numbered from 1 to be compatible with
                              bnlearn.
        """
        if (not isinstance(parents, dict)
            or not all([isinstance(s, set) for s in parents.values()])
                or not isinstance(proposed, DAGChange)):
            raise TypeError('Tabulist.hit() bad arg type')

        # Apply proposed change to copy of parents (i.e. to the DAG)

        _parents = deepcopy(parents)
        frm = proposed.arc[0]
        to = proposed.arc[1]
        if proposed.activity == Activity.ADD:
            _parents[to] = _parents[to] | {frm}
        elif proposed.activity == Activity.DEL:
            _parents[to].discard(frm)
        elif proposed.activity == Activity.REV:
            _parents[frm] = _parents[frm] | {to}
            _parents[to].discard(frm)

        # see if DAG with proposed change is in Tabu list

        hit = None
        for i in range(len(self.tabu)):
            if self.tabu[i] is not None and self.tabu[i] == _parents:
                hit = i + 1
                change = (proposed.activity.value, proposed.arc,
                          round(proposed.delta, 6), {'elem': hit})
                if trace and change not in self._blocked:
                    self._blocked.append(change)
                break

        # if trace and hit is not None:
        #     print('Tabulist blocked change: {}'.format(self._blocked[-1]))

        return hit

    def blocked(self):
        """
            Returns latest list of changes that have been blocked by Tabu list.

            :returns list: of blocked changes.
        """
        blocked = deepcopy(self._blocked)
        self._blocked = []
        return blocked

    def __len__(self):
        """
            Return length of the tabu list.

            :returns int: length of the tabu list
        """
        return len(self.tabu)
