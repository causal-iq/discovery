
#   Implements code for detailed tracing of structure learning

from enum import Enum
from time import time, localtime, asctime
from pandas import DataFrame
from re import compile
import pickle
from compress_pickle import dump, load
from gzip import BadGzipFile
from os import makedirs

from core.common import SOFTWARE_VERSION, environment, Randomise, \
                        EnumWithAttrs
from core.metrics import values_same
from core.graph import SDG, DAG
from learn.common import TreeStats
from fileio.common import EXPTS_DIR, is_valid_path

CONTEXT_FIELDS = {'id': str, 'algorithm': str, 'params': dict, 'in': str,
                  'N': int, 'dataset': bool, 'external': str,
                  'knowledge': str, 'score': float,
                  'randomise': (Randomise, list),
                  'var_order': list, 'initial': DAG, 'pretime': float}

ID_PATTERN = compile(r'^[A-Za-z0-9\/\ \_\.\-]+$')
ID_ANTIPATTERN1 = compile(r'.*[\/|\ |\_|\.|\-][\/|\ |\_|\.|\-].*')
ID_ANTIPATTERN2 = compile(r'^[\/|\ |\_|\.|\-].*')
ID_ANTIPATTERN3 = compile(r'.*[\/|\ |\_|\.|\-]$')


class Detail(Enum):  # details that can be provided on a Trace entry
    ARC = ('arc', tuple)  # Arc that was changed
    DELTA = ('delta/score', float)  # Delta as result of arc changed
    ACTIVITY_2 = ('activity_2', str)  # Arc change with second highest delta
    ARC_2 = ('arc_2', tuple)  # Arc changed in second highest delta
    DELTA_2 = ('delta_2', float)  # second highest delta
    MIN_N = ('min_N', float)  # minimum count in contingency tables' cells
    MEAN_N = ('mean_N', float)  # mean count in contingency tables' cells
    MAX_N = ('max_N', float)  # max count in contingency tables' cells
    LT5 = ('lt5', float)  # number of cells with count <5 in contingency tables
    FPA = ('free_params', float)  # number of free params in contingency tables
    KNOWLEDGE = ('knowledge', tuple)  # Knowledge used in iteration
    BLOCKED = ('blocked', list)  # list of blocked changes
# for PC delete in p-value or use score? some of MIN_N to FPA still relevant?
# need new field for conditioning set
# could arc and arc2 to defined v-structure


class Activity(EnumWithAttrs):
    """
        Defines set of Activities than can recorded in trace

        :ivar str value: short string code for activity
        :ivar str label: human-readable label for activity
        :ivar set mandatory: mandatory items for activity
        :ivar set priority: priority order for this activity
    """
    INIT = 'init', 'initialise', {Detail.DELTA}, 0
    ADD = 'add', 'add arc', {Detail.ARC, Detail.DELTA}, 3
    DEL = 'delete', 'delete arc', {Detail.ARC, Detail.DELTA}, 2
    REV = 'reverse', 'reverse arc', {Detail.ARC, Detail.DELTA}, 1
    STOP = 'stop', 'stop search', {Detail.DELTA}, 4
    PAUSE = 'pause', 'pause search', {Detail.DELTA}, 6
    NONE = 'none', 'no change', {Detail.ARC, Detail.DELTA}, 5

    # ignore the first param since it's already set by __new__
    def __init__(self, _: str, label: str, mandatory: set, priority: int):
        self._label_ = label
        self._mandatory_ = mandatory
        self._priority_ = priority

    # this makes sure that mandatory is read-only
    @property
    def mandatory(self):
        return self._mandatory_

    # this makes sure that priority is read-only
    @property
    def priority(self):
        return self._priority_

# for PC delete used for removing arc, v-struct needed for v-struct,
# and orientate for arc orientation


class DiffType(Enum):  # different kinds of trace entry difference
    MINOR = 'minor'  # difference in secondary score or counts basis
    SCORE = 'score'  # difference in score or delta
    MAJOR = 'major'  # difference in operation (operation = activity and arc)
    ORDER = 'order'  # same operation but at different iteration
    OPPOSITE = 'opposite'  # same operation but on opposite orientation arc
    EXTRA = 'extra'  # operation in trace but not in reference
    MISSING = 'missing'  # operation in reference but not in trace


class Trace():
    """
        Class encapsulating detailed structure learning trace

        :param dict context: description of learning context

        :ivar dict context: learning context
        :ivar list trace: iteration by iteration structure learning trace
        :ivar float start: time at which tracing started
        :ivar SDG result: learnt graph
        :ivar TreeStats: treestats: statistics from a tree search

        :raises TypeError: if arguments have invalid types
        :raises ValueError: if invalid context fields provided
    """

    def __init__(self, context=None):

        if context is not None and not isinstance(context, dict):
            raise TypeError('Trace() bad arg type')

        if context is None:
            context = {}

        if not set(context.keys()).issubset(set(CONTEXT_FIELDS.keys())):
            raise ValueError('Trace() invalid context keys')

        if any([v is not None and not isinstance(v, CONTEXT_FIELDS[k])
                for k, v in context.items()]):
            raise TypeError('Trace() invalid context value type')
        if ('randomise' in context and isinstance(context['randomise'], list)
                and not
                all([isinstance(r, Randomise) for r in context['randomise']])):
            raise TypeError('Trace() invalid context randomise type')

        if 'id' in context and (not ID_PATTERN.match(context['id'])
                                or ID_ANTIPATTERN1.match(context['id'])
                                or ID_ANTIPATTERN2.match(context['id'])
                                or ID_ANTIPATTERN3.match(context['id'])):
            raise ValueError('Trace() invalid id')

        context.update(environment())
        context.update({'software_version': SOFTWARE_VERSION})

        self.context = context
        self.trace = {'time': [], 'activity': []}
        self.trace.update({d.value[0]: [] for d in Detail})
        self.start = time()
        self.result = None
        self.treestats = None

    def rename(self, name_map):
        """
            Nodes in trace renamed in place according to name map.

            :param dict name_map: name mapping {name: new name}

            :raises TypeError: with bad arg type
        """
        def _map(arc):  # maps names in a tuple representing an arc
            return (None if arc is None else
                    (name_map[arc[0]] if arc[0] in name_map else arc[0],
                     name_map[arc[1]] if arc[1] in name_map else arc[1]))

        if (not isinstance(name_map, dict)
            or not all([isinstance(k, str) and isinstance(v, str)
                        for k, v in name_map.items()])):
            raise TypeError('Trace.rename() bad arg types')

        # Rename nodes in the result graph if present

        if self.result is not None:
            self.result.rename(name_map)

        # modify node names for those trace elements which contain arcs

        self.trace['arc'] = [_map(a) for a in self.trace['arc']]
        self.trace['arc_2'] = [_map(a) for a in self.trace['arc_2']]
        self.trace['knowledge'] = [((k[0], k[1], k[2], _map(k[3]))
                                    if k is not None else k)
                                   for k in self.trace['knowledge']]
        self.trace['blocked'] = [[(e[0], _map(e[1]), e[2], e[3]) for e in b]
                                 if b is not None else b
                                 for b in self.trace['blocked']]

    @classmethod
    def read(self, partial_id, root_dir=EXPTS_DIR):
        """
            Reads set of Traces matching partial_id from serialised file.

            :param str id: partial_id of Trace
            :param str root_dir: root directory holding trace files

            :raises TypeError: if arguments are not strings
            :raises FileNotFoundError: if root_dir doesn't exist
            :raises ValueError: if partial_id is entry or serialised file is
                                not a dictionary of traces

            :returns dict: {key: Trace} of traces matching partial id
        """
        if not len(partial_id):
            raise ValueError("Trace.read() empty partial id")

        _, _, _, traces = Trace._read_file(partial_id + '/*', root_dir)

        traces = {id: t._upgrade() for id, t in traces.items()}

        return traces if traces != {} else None

    def add(self, activity, details):
        """
            Add an entry to the structure learning trace

            :param Activity activity: action e.g. initialisation, add arc
            :param dict details: supplementary details relevant to activity

            :raises TypeError: if arguments have invalid types

            :return Trace: returns trace after entry added
        """
        if not isinstance(activity, Activity) \
                or not isinstance(details, dict) or not len(details) \
                or not all(isinstance(k, Detail) for k in details.keys()) \
                or not all([isinstance(v, k.value[1]) or v is None
                            for k, v in details.items()]):

            raise TypeError('Trace.add() bad arg type')

        if not (activity.mandatory).issubset(set(details.keys())):
            raise ValueError('Trace.add() mandatory details not provided')

        self.trace['activity'].append(activity.value)
        self.trace['time'].append(time() - self.start)
        for d in Detail:
            self.trace[d.value[0]].append(details[d] if d in details else None)
        return self

    def _upgrade(self):
        """
            Upgrades earlier versions of Trace to latest version by:
             - making sure knowledge properties are included
             - making sure blocked properties are included
             - ensure randomise in context returned as a list
        """
        if 'knowledge' not in self.trace:
            self.trace.update({'knowledge': [None] * len(self.trace['time'])})
        if 'blocked' not in self.trace:
            self.trace.update({'blocked': [None] * len(self.trace['time'])})
        if ('randomise' in self.context and
                isinstance(self.context['randomise'], Randomise)):
            self.context['randomise'] = [self.context['randomise']]
        return self

    def get(self):
        """
            Returns the trace information

            :returns DataFrame: trace as Pandas data frame
        """
        return DataFrame(self.trace)

    def set_result(self, result):
        """
            Sets the result of the learning activity.

            :param SDG result: graph result from learning activity

            :raises TypeError: is result argument is not a SDG

            :returns Trace: current Trace to support chaining
        """
        if not isinstance(result, SDG):
            raise TypeError('Trace.set_result() bad arg type')

        self.result = result
        return self

    def set_treestats(self, treestats):
        """
            Sets the statistics of a tree learning activity.

            :param TreeStats treestats: statistics from tree learning activity

            :raises TypeError: is treestats argument is incorrect type

            :returns Trace: current Trace to support chaining
        """
        if not isinstance(treestats, TreeStats):
            raise TypeError('Trace.set_treestats() bad arg type')

        self.treestats = treestats
        return self

    @classmethod
    def _read_file(self, id, root_dir):
        """
            Read a composite trace file.

            :param str id: Trace id
            :param str root_dir: root directory for trace files

            :raises TypeError: if bad argument types
            :raises ValueError: if pkl file has bad format
            :raises FileNotFoundError: if root_dir does not exist

            :returns tuple: (path of pickle file, pickle file name,
                             key for entry, current traces in pickle file)
        """
        if not isinstance(id, str) or not isinstance(root_dir, str):
            raise TypeError("Trace._open() bad arg type")

        is_valid_path(root_dir, False)  # raises FileNotFoundError if invalid

        parts = id.split('/')
        if len(parts) < 2:
            raise ValueError('Trace._read_file() invalid id')
        key = parts.pop()
        file_name = parts.pop() + '.pkl.gz'
        path = root_dir + '/' + '/'.join(parts)
        # print("Path: {}, file: {}, key: {}".format(path, file_name, key))

        # Try and open pkl file, checking its format

        try:
            with open(path + '/' + file_name, 'rb') as fh:
                traces = load(fh, compression="gzip",
                              set_default_extension=False)
                if not isinstance(traces, dict) or not \
                        all([isinstance(v, Trace) for v in traces.values()]):
                    raise ValueError()
        except FileNotFoundError:
            traces = {}
        except (pickle.UnpicklingError, EOFError, ValueError, BadGzipFile):
            raise ValueError('Trace._read_file() bad .pkl.gz file')

        return (path, file_name, key, traces)

    def save(self, root_dir=EXPTS_DIR):
        """
            Saves the trace to a composite serialised (pickle) file

            :param str root_dir: root directory under which pickle files saved

            :raises TypeError: if bad argument types
            :raises ValueError: if no id defined for trace
            :raises FileNotFoundError: if root_dir does not exist
        """
        if 'id' not in self.context:
            raise ValueError('Trace.save() called with undefined id')

        path, file_name, key, traces = self._read_file(self.context['id'],
                                                       root_dir)

        traces.update({key: self})

        try:
            is_valid_path(path, False)
        except FileNotFoundError:
            makedirs(path, exist_ok=True)

        with open(path + '/' + file_name, "wb") as file:
            dump(traces, file, compression="gzip", set_default_extension=False)

    @classmethod
    def _nums_diff(self, s1, s2, strict):
        """
            Determine if two numeric values are similar if strict is False
            or exactly the same (if strict is True) to 10 d.p.

            :param int/float s1: first numeric value
            :param int/float s2: second numeric value
            :param bool strict: exact or less struct comparison

            :returns bool: True if (approximately) the same
        """
        return (not strict and not values_same(s1, s2, sf=4) and
                (s1 > s2 + 0.5 or s1 < s2 - 0.5)) or \
               (strict and not values_same(s1, s2, sf=10) and
                (s1 > s2 + 1.0e-8 or s1 < s2 - 1.0e-8)) \
            if (isinstance(s1, float) or isinstance(s2, float)) else s1 != s2

    @classmethod
    def _blocked_same(self, entry, ref, strict):
        """
            Compare the blocked field from two trace entries. The items in the
            blocked list are sorted so that adds come first, then deletes and
            finally reverses.

            :param list/None entry: blocked entry to be compared against ...
            :param list/None ref: reference blocked field
            :param bool strict: whether floats are tested to be strictly the
                                same or just reasonably similar

            :returns bool
        """
        def _sorted(blocked):
            result = [b for b in blocked if b[0] == Activity.ADD.value]
            result.extend([b for b in blocked if b[0] == Activity.DEL.value])
            result.extend([b for b in blocked if b[0] == Activity.REV.value])
            return result

        same = None
        if entry is None and ref is None:
            same = True

        elif entry is None or ref is None or len(entry) != len(ref):
            same = False

        else:
            same = True
            for _entry, _ref in zip(_sorted(entry), _sorted(ref)):
                if (_entry[0] != _ref[0]
                        or _entry[1] != _ref[1]
                        or self._nums_diff(_entry[2], _ref[2], strict)
                        or _entry[3] != _ref[3]):
                    same = False
                    break

        return same

    @classmethod
    def _compare_entry(self, entry, ref, strict, ignore):
        """
            Compare an individual entry from trace with one from reference
            trace.

            :param dict entry: trace entry to be compared against ...
            :param dict ref: ... entry from reference trace
            :param bool strict: whether floats are tested to be strictly the
                                same or just reasonably similar
            :param set ignore: features to ignore in comparison

            :returns DiffType/None: type of difference - major (arc or
                                    activity), score, other or None
        """

        # compare entries arc and activity fields - difference -> MAJOR

        if (('arc' in entry and 'arc' in ref and entry['arc'] != ref['arc'])
                or entry['activity'] != ref['activity']):
            return DiffType.MAJOR

        # compare delta/score to reqd accuracy - difference -> SCORE

        if self._nums_diff(entry['delta/score'], ref['delta/score'], strict):
            print('Different scores are {} and {}'
                  .format(entry['delta/score'], ref['delta/score']))
            return DiffType.SCORE

        # compare other numeric fields - difference -> MINOR

        for key in list(set(ref.keys()) - {'arc', 'activity', 'delta/score',
                                           'blocked', 'knowledge'}):
            if self._nums_diff(entry[key], ref[key], strict):
                print('*** Diff for {}: {}, {}'
                      .format(key, entry[key], ref[key]))
                return DiffType.MINOR

        # compare blocked fields if reqd - difference -> MINOR

        if ('blocked' in entry and 'blocked' in ref
                and 'blocked' not in ignore and not
                self._blocked_same(entry['blocked'], ref['blocked'], strict)):
            return DiffType.MINOR

        # compare knowledge fields - difference -> MINOR

        if ('knowledge' in ref and 'knowledge' in entry and 
                entry['knowledge'] != ref['knowledge']):
            print(ignore)
            if 'act_cache' not in ignore:
                return DiffType.MINOR

            ref_none = ref['knowledge'] is None
            ent_none = entry['knowledge'] is None
            ref_a_c = not ref_none and ref['knowledge'][0] == 'act_cache'
            ent_a_c = not ent_none and entry['knowledge'][0] == 'act_cache'

            return None if ((ref_a_c and ent_none) or (ref_none and ent_a_c)
                            or (ref_a_c and ent_a_c)) else DiffType.MINOR

        return None

    @classmethod
    def _update_diffs(self, iter, trace, ref, diffs):
        """
            Update dictionary of differences keyed on activity and arc with a
            difference for a specific iteration.

            :param int iter: iteration where this difference occurred
            :param dict trace: trace entry at this iteration
            :param dict ref: reference trace entry at this iteration
            :param dict diffs: dictionary of differences keyyed on actvity and
                               arc

            :returns dict: differences dictionary with this difference included
        """
        if trace:
            key = (trace['activity'], trace['arc'] if 'arc' in trace else None)
            if key not in diffs:
                diffs[key] = ([], [])
            diffs[key][0].append(iter)
        if ref:
            key = (ref['activity'], ref['arc'] if 'arc' in ref else None)
            if key not in diffs:
                diffs[key] = ([], [])
            diffs[key][1].append(iter)
        return diffs

    @classmethod
    def _merge_opposites(self, activity, diffs):
        """
            Merge activities done on opposing arcs, extra add A --> B and
            missing add A <-- B is merged into opposite add A --> B.

            :param str activity: activity being merged - add, delete or reverse
            :param dict diffs: trace differences before merging

            :returns dict: trace differences after merging done
        """
        extra = (activity, DiffType.EXTRA.value)  # key for extra entry
        missing = (activity, DiffType.MISSING.value)  # key for missing entry

        if extra in diffs and missing in diffs:

            # Merging only possible if there are some extra and some missing
            # activities. Loop over extra activities in trace looking for
            # opposing arc missing from trace

            for arc, iters in diffs[extra].copy().items():
                opp_arc = (arc[1], arc[0])
                if opp_arc in diffs[missing]:

                    # set up key for opposites in diffs and add new entry

                    opposite = (activity, DiffType.OPPOSITE.value)
                    if opposite not in diffs:
                        diffs[opposite] = {}
                    ref_iters = diffs[missing][opp_arc]
                    diffs[opposite].update({arc: [iters[0], ref_iters[1]]})

                    # remove matching entry in extra and missing

                    del diffs[extra][arc]
                    del diffs[missing][opp_arc]
        return diffs

    def diffs_from(self, ref, strict=True):
        """
            Differences of trace from reference trace

            :param Trace ref: reference trace to compare this one to
            :param bool strict: whether floats are tested to be strictly the
                                same or just reasonably similar

            :raises TypeError: if ref is not of type Trace
            :raises ValueError: if either trace is invalid

            :returns tuple/None: (major differences, minor differences,
                                  textual summary) or None if identical
        """
        if not isinstance(ref, Trace):
            raise TypeError('Trace.diffs_from() bad arg type')

        # hc_worker processing changed at v177 so determine if which
        # trace properties/entries have to be ignored.

        ignore = (set() if (self.context['software_version'] - 176.5) *
                  (ref.context['software_version'] - 176.5) > 0.0
                  else {'blocked', 'act_cache'})

        #   Will only compare columns which have some values in them in BOTH
        #   trace and reference, and will not compare time column

        ref = ref.get()
        trace = self.get()
        compare = set(trace.columns[trace.notnull().any()]) \
            .intersection(set(ref.columns[ref.notnull().any()])) \
            - {'time'} | {'arc', 'activity', 'delta/score'}

        ref = ref[compare].to_dict('records')  # list of ref entries (dicts)
        trace = trace[compare].to_dict('records')  # list of trace entries

        if len(ref) < 2 or len(trace) < 2 or \
                'activity' not in ref[0] or 'delta/score' not in ref[0] \
                or trace[0]['activity'] != 'init' \
                or ref[0]['activity'] != 'init' \
                or trace[-1]['activity'] != 'stop' \
                or ref[-1]['activity'] != 'stop':
            raise ValueError('Trace.diffs_from() bad trace format')

        #   Loop through trace entries detecting differences with ref

        _diffs = {}
        minor = []
        for iter in range(0, len(trace)):
            if iter >= len(ref):
                # print(iter, trace[iter], None)
                _diffs = self._update_diffs(iter, trace[iter], None, _diffs)
            else:
                diff = self._compare_entry(trace[iter], ref[iter], strict,
                                           ignore)
                if diff in [DiffType.MAJOR, DiffType.SCORE]:
                    # print(iter, trace[iter], ref[iter])
                    _diffs = self._update_diffs(iter, trace[iter], ref[iter],
                                                _diffs)
                elif diff is not None:
                    minor.append(iter)

        #   Loop through any extra entries in ref

        for iter in range(len(trace), len(ref)):
            # print(iter, None, ref[iter])
            _diffs = self._update_diffs(iter, None, ref[iter], _diffs)

        # if no major, score or minor errors, traces are identical

        if not len(_diffs) and not len(minor):
            return None

        # Reorganise differences to be keyed by (activity, difference type)

        diffs = {}
        for diff, iters in _diffs.items():
            if len(iters[0]) > len(iters[1]):
                type = DiffType.EXTRA
            elif len(iters[0]) < len(iters[1]):
                type = DiffType.MISSING
            elif iters[0] != iters[1]:
                type = DiffType.ORDER
            else:
                type = DiffType.SCORE
            key = (diff[0], type.value)
            if key not in diffs:
                diffs[key] = {}
            diffs[key].update({diff[1]: (iters[0][0] if iters[0] else None,
                                         iters[1][0] if iters[1] else None)})
            # print(type, diff, iters)

        # Merge activities on opposing arcs

        for activity in ['add', 'delete', 'reverse']:
            diffs = self._merge_opposites(activity, diffs)

        final_score_diff = (self._compare_entry(trace[-1], ref[-1],
                                                strict, ignore)
                            == DiffType.SCORE)
        summary = self._diffs_summary(diffs, final_score_diff,
                                      len(trace) - 1, len(ref) - 1)

        return (diffs, minor, summary)

    @classmethod
    def _diffs_summary(self, diffs, final_score_diff, trace_iters, ref_iters):
        """
            Return a human readable summary of the trace differences

            :param dict diffs: differences keyed by activity and diff type
            :param bool final_score_diff: if difference in final score
            :param int trace_iters: number of iterations in trace
            :param int ref_iters: number of iterations in reference trace

            :returns str: a human readable summary of the differences
        """
        summary = 'Trace has {} initial score' \
                  .format('different' if ('init', DiffType.SCORE.value)
                          in diffs else 'same') + \
                  ', {} final score' \
                  .format('different' if final_score_diff else 'same') + \
                  ' in {} versus {} iterations' \
                  .format(trace_iters, ref_iters)

        for activity in ['add', 'delete', 'reverse']:
            for type in ['extra', 'missing', 'opposite', 'score']:
                key = (activity, type)
                if key in diffs:
                    summary += '\n{} {} {}(s): {}' \
                        .format(len(diffs[key]), type, activity, diffs[key])

        return summary

    @classmethod
    def context_string(self, context, start):
        """
            Returns a trace context as a human readable string.

            :param dict context: individual context information

            :returns str: context information in readable form
        """
        r_str = (' randomising {}\n'
                 .format(', '.join([r.value for r in context['randomise']]))
                 if 'randomise' in context and context['randomise'] is not
                 False and context['randomise'] is not None else '\n')
        return 'Trace for {}'.format(context['id']) \
            + ' run at {}\n'.format(asctime(localtime(start))) \
            + 'Learning from {}'.format('dataset' if 'dataset' in context
                                        else 'distribution') \
            + ' with {} rows'.format(context['N']) \
            + ' from {}{}'.format(context['in'], r_str) \
            + ('Variable order is: {}{}\n'
               .format(', '.join(context['var_order'][:20]),
                       '' if len(context['var_order']) <= 20 else ' ...')
               if 'var_order' in context else '') \
            + '{}{} algorithm' \
            .format(context['external'] + '-' if 'external' in context else '',
                    context['algorithm']) \
            + (' with parameters {}{}{}\n\n'
               .format(context['params'], '\nKnowledge: '
                       + context['knowledge'] if 'knowledge' in context
                       else '',
                       (' with reference score {}'.format(context['score']
                        if 'score' in context else ''))
                       )) \
            + '(Using bnbench v{}'.format(context['software_version']) \
            + ', Python {}'.format(context['python']) \
            + ' and {}\n'.format(context['os']) \
            + 'on {}'.format(context['cpu']) \
            + ' with {} GB RAM)'.format(context['ram'])

    def __eq__(self, other):
        """
            Test if other Trace is identical to this one

            :param Trace other: trace to compare with self

            :returns bool: True if other is identical to self
        """
        return isinstance(other, Trace) and other.diffs_from(self) is None

    def __str__(self):
        """
            Details of Trace in human-readable printable format.

            :returns str: Trace in printable form
        """
        treestats = ('\n\nTree stats:\n{}'.format(self.treestats)
                     if hasattr(self, 'treestats')
                     and self.treestats is not None else '')

        return self.context_string(self.context, self.start) \
            + '\n\n{}'.format(self.get()) + treestats
