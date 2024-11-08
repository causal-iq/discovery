
# Common definitions and classes required by other learn modules

from copy import deepcopy

from core.common import EnumWithAttrs


class Output(EnumWithAttrs):
    """
        Defines type of output required

        :ivar str value: short string code for output type
        :ivar str label: human-readable label for output type
    """
    DAG = 'dag', "highest-scoring DAG"  # default
    DAGS = 'dags', 'all DAGs found'
    DAGX = 'dagx', 'stable DAG'  # will replace DAG


class TreeStats():
    """
        Encapsulates statistics about tree search

        :ivar dict sequences: statistics keyed on decision sequence
    """
    def __init__(self):
        """
            Initialises the tree statistics
        """
        self.sequences = {}

    def add(self, hcw, depth):
        """
            Add details of this new sequence into statistics

            :param HCWorker hcw: Worker adding stats about
            :param int depth: subtree depth
        """
        key = self.key(hcw)
        score2 = hcw.score2 / hcw.data.N
        previous = key[:-depth]
        # print('{} [{}] has score {:.4f}'.format(key, previous, score2))

        stats = (deepcopy(self.sequences[previous]) if len(previous)
                 else {'score2': []})
        stats.update({'state': ('P' if hcw.paused else 'C')})
        stats['score2'] += [score2]
        self.sequences.update({key: stats})

    def delete(self, hcw):
        """
            Remove details of the sequence associated with this HCWorker
            (since its details have been incorporated into extensions of the
             sequence
        """
        self.sequences.pop(self.key(hcw))

    def update_state(self, hcw, state):
        """
            Update sequence status - used to identify pruned sequences.

            :param HCWorker hcw: Worker associated with sequence

            :param str state: new state value e.g. XS = subtree prune
        """
        self.sequences[self.key(hcw)].update({'state': state})

    def update_ranks(self, hcws):
        """
            Update ranks of unpruned sequences.

            :params list hcws: list of HCWorks IN RANK ORDER
        """
        for i, hcw in enumerate(hcws):
            key = self.key(hcw)
            if 'rank' in self.sequences[key]:
                self.sequences[key]['rank'] += [i + 1]
            else:
                self.sequences[key]['rank'] = [i + 1]

    @classmethod
    def key(self, hcw):
        """
            Return HCWorker decision sequesce as a "TFTT..." string.

            :returns str: decision sequence for HCWorker as "TFTT..."
        """
        return ''.join(['T' if b is True else 'F'
                       for b in hcw.knowledge.sequence])

    def to_string(self, states={'C', 'P', 'XS', 'XP', 'XD'}):
        """
            Returns statistics as print-friendly string

            :param set states: include only sequences with these states

            :returns str: print friendly string, 1 sequence per line
        """
        # return '{}'.format(self.sequences)
        return '\n'.join(['{}: [{}] {:.4f} [{}]'
                          .format(k,
                                  (','.join([str(r) for r in s['rank']])
                                   if 'rank' in s else ''),
                                  s['score2'][-1], s['state'])
                          for k, s in self.sequences.items()
                          if s['state'] in states])

    def __str__(self):
        """
            Returns statistics for active workers as print-friendly string

            :returns str: print friendly string, 1 sequence per line
        """
        return self.to_string(states={'C', 'P'})
