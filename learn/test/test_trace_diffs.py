
#   Test the method to compare Traces

import pytest

from learn.trace import Trace, Activity, Detail


def test_trace_diffs_type_error_1():  # bad arg type for diffs_from
    with pytest.raises(TypeError):
        Trace().diffs_from()
    with pytest.raises(TypeError):
        Trace().diffs_from(-400.001)
    with pytest.raises(TypeError):
        Trace().diffs_from(True)


def test_trace_diffs_value_error_1():  # both traces empty
    with pytest.raises(ValueError):
        Trace().diffs_from(Trace())


def test_trace_diffs_value_error_2():  # trace empty
    trace = Trace()
    ref = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                 .add(Activity.STOP, {Detail.DELTA: 10.0})
    with pytest.raises(ValueError):
        trace.diffs_from(ref)


def test_trace_diffs_value_error_3():  # reference empty
    trace = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                   .add(Activity.STOP, {Detail.DELTA: 10.0})
    ref = Trace()
    with pytest.raises(ValueError):
        trace.diffs_from(ref)


def test_trace_diffs_value_error_4():  # both traces too short
    trace = Trace().add(Activity.INIT, {Detail.DELTA: 0.0})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: 0.0})
    with pytest.raises(ValueError):
        trace.diffs_from(ref)


def test_trace_diffs_value_error_5():  # trace doesn't start with init
    trace = Trace().add(Activity.STOP, {Detail.DELTA: 0.0}) \
                   .add(Activity.STOP, {Detail.DELTA: 10.0})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                 .add(Activity.STOP, {Detail.DELTA: 10.0})
    with pytest.raises(ValueError):
        trace.diffs_from(ref)


def test_trace_diffs_value_error_6():  # trace doesn't start with init
    trace = Trace().add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 0.0}) \
                   .add(Activity.STOP, {Detail.DELTA: 10.0})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                 .add(Activity.STOP, {Detail.DELTA: 10.0})
    with pytest.raises(ValueError):
        trace.diffs_from(ref)


def test_trace_diffs_value_error_7():  # ref doesn't start with init
    trace = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                   .add(Activity.STOP, {Detail.DELTA: 10.0})
    ref = Trace().add(Activity.STOP, {Detail.DELTA: 0.0}) \
                 .add(Activity.STOP, {Detail.DELTA: 10.0})
    with pytest.raises(ValueError):
        trace.diffs_from(ref)


def test_trace_diffs_value_error_8():  # ref doesn't start with init
    trace = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                   .add(Activity.STOP, {Detail.DELTA: 10.0})
    ref = Trace().add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 0.0}) \
                 .add(Activity.STOP, {Detail.DELTA: 10.0})
    with pytest.raises(ValueError):
        trace.diffs_from(ref)


def test_trace_diffs_value_error_9():  # trace doesn't end with stop
    trace = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                   .add(Activity.INIT, {Detail.DELTA: 10.0})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                 .add(Activity.STOP, {Detail.DELTA: 10.0})
    with pytest.raises(ValueError):
        trace.diffs_from(ref)


def test_trace_diffs_value_error_10():  # trace doesn't end with stop
    trace = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                   .add(Activity.STOP, {Detail.DELTA: 10.0}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'C'),
                                       Detail.DELTA: 1.0})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                 .add(Activity.INIT, {Detail.DELTA: 10.0})
    with pytest.raises(ValueError):
        trace.diffs_from(ref)


def test_trace_diffs_value_error_11():  # ref doesn't end with stop
    trace = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                   .add(Activity.STOP, {Detail.DELTA: 10.0})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                 .add(Activity.INIT, {Detail.DELTA: 10.0})
    with pytest.raises(ValueError):
        trace.diffs_from(ref)


def test_trace_diffs_value_error_12():  # ref doesn't end with stop
    trace = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                   .add(Activity.INIT, {Detail.DELTA: 10.0})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                 .add(Activity.STOP, {Detail.DELTA: 10.0}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'C'),
                                     Detail.DELTA: 1.0})
    with pytest.raises(ValueError):
        trace.diffs_from(ref)


def test_trace_diffs_same_ok_1():  # two entries identical
    trace = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                   .add(Activity.STOP, {Detail.DELTA: 10.0})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                 .add(Activity.STOP, {Detail.DELTA: 10.0})
    assert trace.diffs_from(ref) is None
    assert trace == ref


def test_trace_diffs_same_ok_2():  # three entries identical
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 1.0}) \
                   .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 1.0}) \
                 .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    assert trace.diffs_from(ref) is None
    assert trace == ref


def test_trace_diffs_same_ok_3():  # three identical entries
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                       Detail.DELTA: 1.0}) \
                   .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                     Detail.DELTA: 1.0}) \
                 .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    assert trace.diffs_from(ref) is None
    assert trace == ref


def test_trace_diffs_same_ok_4():  # seven identical entries
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 3.0}) \
                   .add(Activity.ADD, {Detail.ARC: ('C', 'D'),
                                       Detail.DELTA: 2.0}) \
                   .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                       Detail.DELTA: 1.0}) \
                   .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 0.5}) \
                   .add(Activity.DEL, {Detail.ARC: ('C', 'D'),
                                       Detail.DELTA: 0.1}) \
                   .add(Activity.STOP, {Detail.DELTA: -3.52345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 3.0}) \
                 .add(Activity.ADD, {Detail.ARC: ('C', 'D'),
                                     Detail.DELTA: 2.0}) \
                 .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                     Detail.DELTA: 1.0}) \
                 .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 0.5}) \
                 .add(Activity.DEL, {Detail.ARC: ('C', 'D'),
                                     Detail.DELTA: 0.1}) \
                 .add(Activity.STOP, {Detail.DELTA: -3.52345678})
    assert trace.diffs_from(ref) is None
    assert trace == ref


def test_trace_diffs_same_ok_5():  # small score differences
    trace = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                   .add(Activity.STOP, {Detail.DELTA: 10.0001})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: 0.00003}) \
                 .add(Activity.STOP, {Detail.DELTA: 10.0})

    diffs = trace.diffs_from(ref)  # strict comparisons
    assert diffs[0] == {('init', 'score'): {None: (0, 0)},
                        ('stop', 'score'): {None: (1, 1)}}
    assert trace != ref

    diffs = trace.diffs_from(ref, strict=False)  # non-strict comparisons
    assert diffs is None


def test_trace_diffs_same_ok_6():  # small score differences
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -20002.1}) \
                   .add(Activity.STOP, {Detail.DELTA: -1800.3})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -20002.2}) \
                 .add(Activity.STOP, {Detail.DELTA: -1800.4})

    diffs = trace.diffs_from(ref)  # strict comparisons
    assert diffs[0] == {('init', 'score'): {None: (0, 0)},
                        ('stop', 'score'): {None: (1, 1)}}
    assert trace != ref

    diffs = trace.diffs_from(ref, strict=False)  # non-strict comparisons
    assert diffs is None


def test_trace_diffs_same_ok_7():  # small score differences
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                       Detail.DELTA: 1.00001}) \
                   .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                     Detail.DELTA: 1.0}) \
                 .add(Activity.STOP, {Detail.DELTA: -9.12346678})

    diffs = trace.diffs_from(ref)  # strict comparisons
    assert diffs[0] == {('add', 'score'): {('B', 'C'): (1, 1)},
                        ('stop', 'score'): {None: (2, 2)}}
    assert trace != ref

    diffs = trace.diffs_from(ref, strict=False)  # non-strict comparisons
    assert diffs is None


def test_trace_diffs_same_ok_8():  # small score differences
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 3.0}) \
                   .add(Activity.ADD, {Detail.ARC: ('C', 'D'),
                                       Detail.DELTA: 2.0}) \
                   .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                       Detail.DELTA: 1.0000001}) \
                   .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 0.5}) \
                   .add(Activity.DEL, {Detail.ARC: ('C', 'D'),
                                       Detail.DELTA: 0.1}) \
                   .add(Activity.STOP, {Detail.DELTA: -3.52345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 3.0}) \
                 .add(Activity.ADD, {Detail.ARC: ('C', 'D'),
                                     Detail.DELTA: 2.0}) \
                 .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                     Detail.DELTA: 1.0}) \
                 .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 0.5}) \
                 .add(Activity.DEL, {Detail.ARC: ('C', 'D'),
                                     Detail.DELTA: 0.2}) \
                 .add(Activity.STOP, {Detail.DELTA: -3.62345678})

    diffs = trace.diffs_from(ref)  # strict comparisons
    assert diffs[0] == {('add', 'score'): {('B', 'C'): (3, 3)},
                        ('delete', 'score'): {('C', 'D'): (5, 5)},
                        ('stop', 'score'): {None: (6, 6)}}
    print(diffs[2])
    assert trace != ref

    diffs = trace.diffs_from(ref, strict=False)  # non-strict comparisons
    assert diffs is None


def test_trace_diffs_same_ok_9():  # ignore unmatched columns in trace
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                       Detail.DELTA: 1.0,
                                       Detail.ARC_2: ('A', 'B')}) \
                   .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                     Detail.DELTA: 1.0}) \
                 .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    assert trace.diffs_from(ref) is None
    assert trace == ref


def test_trace_diffs_same_ok_10():  # ignore unmatched columns in ref
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                       Detail.DELTA: 1.0}) \
                   .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                     Detail.DELTA: 1.0,
                                     Detail.ARC_2: ('A', 'B')}) \
                 .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    assert trace.diffs_from(ref) is None
    assert trace == ref


def test_trace_diffs_same_ok_11():  # accept None values
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                        Detail.ARC: None}) \
                   .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                       Detail.DELTA: 1.0}) \
                   .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                     Detail.DELTA: 1.0}) \
                 .add(Activity.STOP, {Detail.DELTA: -9.12345678,
                                      Detail.ARC: None})
    assert trace.diffs_from(ref) is None
    assert trace == ref


def test_trace_diffs_major_ok_1():  # trace has additional entry, same final
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                        Detail.ARC: None}) \
                   .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                       Detail.DELTA: 1.0}) \
                   .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.STOP, {Detail.DELTA: -9.12345678,
                                      Detail.ARC: None})
    diffs = trace.diffs_from(ref)
    assert set(diffs[0].keys()) == {('add', 'extra'), ('stop', 'order')}
    assert diffs[0][('add', 'extra')] == {('B', 'C'): (1, None)}
    assert diffs[1] == []
    assert trace != ref
    print('\n{}'.format(diffs[2]))


def test_trace_diffs_major_ok_2():  # trace has additional entry, diff final
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                        Detail.ARC: None}) \
                   .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                       Detail.DELTA: 1.0}) \
                   .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.STOP, {Detail.DELTA: -10.12345678,
                                      Detail.ARC: None})
    diffs = trace.diffs_from(ref)
    assert set(diffs[0].keys()) == {('add', 'extra'), ('stop', 'order')}
    assert diffs[0][('add', 'extra')] == {('B', 'C'): (1, None)}
    assert diffs[1] == []
    assert trace != ref
    print('\n{}'.format(diffs[2]))


def test_trace_diffs_major_ok_3():  # trace has missing entry, diff final
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.STOP, {Detail.DELTA: -10.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                     Detail.DELTA: 1.0}) \
                 .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    diffs = trace.diffs_from(ref)
    assert set(diffs[0].keys()) == {('add', 'missing'), ('stop', 'order')}
    assert diffs[0][('add', 'missing')] == {('B', 'C'): (None, 1)}
    assert diffs[1] == []
    assert trace != ref
    print('\n{}'.format(diffs[2]))


def test_trace_diffs_major_ok_4():  # different arcs added
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('B', 'C'),
                                     Detail.DELTA: 1.0}) \
                 .add(Activity.STOP, {Detail.DELTA: -9.12345678})
    diffs = trace.diffs_from(ref)
    assert set(diffs[0].keys()) == {('add', 'missing'), ('add', 'extra'),
                                    ('stop', 'score')}
    assert diffs[0][('add', 'missing')] == {('B', 'C'): (None, 1)}
    assert diffs[0][('add', 'extra')] == {('A', 'B'): (1, None)}
    assert diffs[1] == []
    assert trace != ref
    print('\n{}'.format(diffs[2]))

# Compared blocked fields


def test_trace_diffs_blocked_ok_1():  # both blocked are None
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0,
                                       Detail.BLOCKED: None}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 2.0,
                                     Detail.BLOCKED: None}) \
                 .add(Activity.STOP, {Detail.DELTA: -8.12345678})
    diffs = trace.diffs_from(ref)
    assert diffs is None


def test_trace_diffs_blocked_ok_2():  # both blocked are []
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0,
                                       Detail.BLOCKED: []}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 2.0,
                                     Detail.BLOCKED: []}) \
                 .add(Activity.STOP, {Detail.DELTA: -8.12345678})
    diffs = trace.diffs_from(ref)
    assert diffs is None


def test_trace_diffs_blocked_ok_3():  # one None, one [] - no diffs

    # BLOCKED is all None for trace and so BLOCKED is not compared

    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0,
                                       Detail.BLOCKED: None}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 2.0,
                                     Detail.BLOCKED: []}) \
                 .add(Activity.STOP, {Detail.DELTA: -8.12345678})
    diffs = trace.diffs_from(ref)
    assert diffs is None


def test_trace_diffs_blocked_ok_4():  # one None, one [] - diffs

    # BLOCKED has values in each trace so comparison is made which
    # flags minor difference on entry 1.

    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0,
                                       Detail.BLOCKED: None}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                        Detail.BLOCKED: []})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 2.0,
                                     Detail.BLOCKED: []}) \
                 .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                      Detail.BLOCKED: []})
    diffs = trace.diffs_from(ref)
    assert diffs[0] == {}  # no major differences
    assert diffs[1] == [1]  # minor difference on entry 1


def test_trace_diffs_blocked_ok_5():  # no blocking on any entry

    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                        Detail.BLOCKED: []}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0,
                                       Detail.BLOCKED: []}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                        Detail.BLOCKED: []})

    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                      Detail.BLOCKED: []}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 2.0,
                                     Detail.BLOCKED: []}) \
                 .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                      Detail.BLOCKED: []})

    diffs = trace.diffs_from(ref)
    assert diffs is None


def test_trace_diffs_blocked_ok_6():  # same block on add

    block = (Activity.DEL.value, ('A', 'B'), -2.0, {'elem': 1})

    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                        Detail.BLOCKED: []}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0,
                                       Detail.BLOCKED: []}) \
                   .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 0.0,
                                       Detail.BLOCKED: [block]}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                        Detail.BLOCKED: []})

    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                      Detail.BLOCKED: []}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 2.0,
                                     Detail.BLOCKED: []}) \
                 .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 0.0,
                                     Detail.BLOCKED: [block]}) \
                 .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                      Detail.BLOCKED: []})
    diffs = trace.diffs_from(ref)
    assert diffs is None


def test_trace_diffs_blocked_ok_7():  # difference in block activity

    block = (Activity.DEL.value, ('A', 'B'), -2.0, {'elem': 1})
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                        Detail.BLOCKED: []}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0,
                                       Detail.BLOCKED: []}) \
                   .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 0.0,
                                       Detail.BLOCKED: [block]}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                        Detail.BLOCKED: []})

    block = (Activity.REV.value, ('A', 'B'), -2.0, {'elem': 1})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                      Detail.BLOCKED: []}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 2.0,
                                     Detail.BLOCKED: []}) \
                 .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 0.0,
                                     Detail.BLOCKED: [block]}) \
                 .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                      Detail.BLOCKED: []})
    diffs = trace.diffs_from(ref)
    assert diffs[0] == {}  # no major differences
    assert diffs[1] == [2]  # minor difference on entry 2


def test_trace_diffs_blocked_ok_8():  # difference in block arc

    block = (Activity.DEL.value, ('A', 'B'), -2.0, {'elem': 1})
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                        Detail.BLOCKED: []}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0,
                                       Detail.BLOCKED: []}) \
                   .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 0.0,
                                       Detail.BLOCKED: [block]}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                        Detail.BLOCKED: []})

    block = (Activity.DEL.value, ('B', 'A'), -2.0, {'elem': 1})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                      Detail.BLOCKED: []}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 2.0,
                                     Detail.BLOCKED: []}) \
                 .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 0.0,
                                     Detail.BLOCKED: [block]}) \
                 .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                      Detail.BLOCKED: []})
    diffs = trace.diffs_from(ref)
    assert diffs[0] == {}  # no major differences
    assert diffs[1] == [2]  # minor difference on entry 2


def test_trace_diffs_blocked_ok_9():  # significant difference in block score

    block = (Activity.DEL.value, ('A', 'B'), -2.0, {'elem': 1})
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                        Detail.BLOCKED: []}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0,
                                       Detail.BLOCKED: []}) \
                   .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 0.0,
                                       Detail.BLOCKED: [block]}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                        Detail.BLOCKED: []})

    block = (Activity.DEL.value, ('A', 'B'), -1.9, {'elem': 1})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                      Detail.BLOCKED: []}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 2.0,
                                     Detail.BLOCKED: []}) \
                 .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 0.0,
                                     Detail.BLOCKED: [block]}) \
                 .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                      Detail.BLOCKED: []})
    diffs = trace.diffs_from(ref)
    assert diffs[0] == {}  # no major differences
    assert diffs[1] == [2]  # minor difference on entry 2


def test_trace_diffs_blocked_ok_10():  # insignif. difference in block score

    block = (Activity.DEL.value, ('A', 'B'), -2.0, {'elem': 1})
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                        Detail.BLOCKED: []}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0,
                                       Detail.BLOCKED: []}) \
                   .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 0.0,
                                       Detail.BLOCKED: [block]}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                        Detail.BLOCKED: []})

    block = (Activity.DEL.value, ('A', 'B'), -1.9999999, {'elem': 1})
    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                      Detail.BLOCKED: []}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 2.0,
                                     Detail.BLOCKED: []}) \
                 .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 0.0,
                                     Detail.BLOCKED: [block]}) \
                 .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                      Detail.BLOCKED: []})
    diffs = trace.diffs_from(ref, strict=False)
    assert diffs is None


def test_trace_diffs_blocked_ok_11():  # ordering diff in blocks

    block1 = (Activity.DEL.value, ('A', 'B'), -2.0, {'elem': 1})
    block2 = (Activity.REV.value, ('A', 'B'), 0.0, {'elem': 1})
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                        Detail.BLOCKED: []}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 2.0,
                                       Detail.BLOCKED: []}) \
                   .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 0.0,
                                       Detail.BLOCKED: [block1, block2]}) \
                   .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                        Detail.BLOCKED: []})

    ref = Trace().add(Activity.INIT, {Detail.DELTA: -10.12345678,
                                      Detail.BLOCKED: []}) \
                 .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 2.0,
                                     Detail.BLOCKED: []}) \
                 .add(Activity.REV, {Detail.ARC: ('A', 'B'),
                                     Detail.DELTA: 0.0,
                                     Detail.BLOCKED: [block2, block1]}) \
                 .add(Activity.STOP, {Detail.DELTA: -8.12345678,
                                      Detail.BLOCKED: []})
    diffs = trace.diffs_from(ref, strict=False)
    assert diffs is None
