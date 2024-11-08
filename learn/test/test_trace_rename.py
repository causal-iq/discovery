
# Test BN rename function

import pytest
from copy import deepcopy

from learn.trace import Trace
from fileio.common import TESTDATA_DIR


def test_trace_rename_type_error_1():  # bad argument types
    trace = Trace.read('TABU/STD/asia', TESTDATA_DIR + '/experiments')['N1000']
    with pytest.raises(TypeError):
        trace.rename()


def test_trace_rename_type_error_2():  # name_map not a dictionary
    trace = Trace.read('TABU/STD/asia', TESTDATA_DIR + '/experiments')['N1000']
    with pytest.raises(TypeError):
        trace.rename(name_map=None)
    with pytest.raises(TypeError):
        trace.rename(name_map=True)
    with pytest.raises(TypeError):
        trace.rename(name_map=37)
    with pytest.raises(TypeError):
        trace.rename(name_map=[{'A': 'B'}])


def test_trace_rename_type_error_3():  # name_map has non-string keys
    trace = Trace.read('TABU/STD/asia', TESTDATA_DIR + '/experiments')['N1000']
    with pytest.raises(TypeError):
        trace.rename(name_map={1: 'B'})
    with pytest.raises(TypeError):
        trace.rename(name_map={('A',): 'B'})


def test_trace_rename_type_error_4():  # name_map has non-string values
    trace = Trace.read('TABU/STD/asia', TESTDATA_DIR + '/experiments')['N1000']
    with pytest.raises(TypeError):
        trace.rename(name_map={'A': 1})
    with pytest.raises(TypeError):
        trace.rename(name_map={'A': ['B']})


def test_trace_rename_value_error_1():  # keys that are not current node name
    trace = Trace.read('TABU/STD/asia', TESTDATA_DIR + '/experiments')['N1000']
    with pytest.raises(ValueError):
        trace.rename(name_map={'unknown': 'fail'})


def test_trace_rename_value_error_2():  # a value is a current name
    trace = Trace.read('TABU/STD/asia', TESTDATA_DIR + '/experiments')['N1000']
    with pytest.raises(ValueError):
        trace.rename(name_map={'lung': 'x0lung', 'tub': 'lung'})


def test_trace_rename_asia_1_ok():  # valid tabu trace, rename 1 node
    trace = Trace.read('TABU/STD/asia', TESTDATA_DIR + '/experiments')['N1000']
    orig_trace = deepcopy(trace)

    trace.rename({'asia': 'x0asia', 'bronc': 'bronc', 'dysp': 'dysp',
                  'either': 'either', 'lung': 'lung', 'smoke': 'smoke',
                  'tub': 'tub', 'xray': 'xray'})

    assert isinstance(trace, Trace)
    assert trace.trace['arc'] == \
        [None, ('bronc', 'dysp'), ('either', 'lung'), ('either', 'xray'),
         ('bronc', 'smoke'), ('either', 'tub'), ('lung', 'tub'),
         ('dysp', 'either'), ('lung', 'smoke'), ('bronc', 'either'),
         ('bronc', 'dysp'), ('bronc', 'either'), ('dysp', 'either'),
         ('dysp', 'bronc'), ('bronc', 'smoke'), ('either', 'bronc'),
         ('either', 'lung'), ('either', 'tub'), ('lung', 'tub'),
         ('lung', 'smoke'), ('smoke', 'bronc'), ('either', 'x0asia'),
         ('bronc', 'smoke'), None]
    assert trace.trace['arc_2'] == \
        [None, ('dysp', 'bronc'), ('lung', 'either'), ('xray', 'either'),
         ('smoke', 'bronc'), ('tub', 'either'), ('tub', 'lung'),
         ('either', 'dysp'), ('either', 'smoke'), ('bronc', 'dysp'),
         ('dysp', 'either'), ('lung', 'tub'), ('lung', 'tub'),
         ('either', 'lung'), ('either', 'bronc'), ('either', 'lung'),
         ('either', 'xray'), ('lung', 'smoke'), ('lung', 'smoke'),
         ('either', 'x0asia'), ('either', 'x0asia'), ('lung', 'x0asia'),
         ('smoke', 'tub'), ('smoke', 'tub')]
    assert trace.trace['knowledge'] == \
        [None, None, None, None, None, None, None, None, None, None, None,
         None, None, None, None, None, None, None, None, None, None, None,
         None, None]
    assert trace.trace['blocked'] == \
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [('reverse', ('bronc', 'smoke'), 0.0, {'elem': 10})],
         [('delete', ('either', 'x0asia'), 2.724381, {'elem': 1})],
         [('delete', ('either', 'x0asia'), 2.724381, {'elem': 10})]]
    assert trace.result.to_string() == \
        ('[bronc|smoke]' +
         '[dysp|bronc:either]' +
         '[either|lung:tub]' +
         '[lung]' +
         '[smoke|lung][tub]' +
         '[x0asia][xray|either]')
    print(trace)

    trace.rename({'x0asia': 'asia', 'bronc': 'bronc', 'dysp': 'dysp',
                  'either': 'either', 'lung': 'lung', 'smoke': 'smoke',
                  'tub': 'tub', 'xray': 'xray'})
    assert orig_trace == trace


def test_trace_rename_asia_2_ok():  # valid tabu trace, rename all nodes
    trace = Trace.read('TABU/STD/asia', TESTDATA_DIR + '/experiments')['N1000']
    orig_trace = deepcopy(trace)

    trace.rename({'asia': 'x3asi', 'smoke': 'x1smo', 'xray': 'x2xra',
                  'either': 'x0eit', 'bronc': 'x4bro', 'lung': 'x5lun',
                  'tub': 'x6tub', 'dysp': 'x7dys'})

    assert isinstance(trace, Trace)
    assert trace.trace['arc'] == \
        [None, ('x4bro', 'x7dys'), ('x0eit', 'x5lun'), ('x0eit', 'x2xra'),
         ('x4bro', 'x1smo'), ('x0eit', 'x6tub'), ('x5lun', 'x6tub'),
         ('x7dys', 'x0eit'), ('x5lun', 'x1smo'), ('x4bro', 'x0eit'),
         ('x4bro', 'x7dys'), ('x4bro', 'x0eit'), ('x7dys', 'x0eit'),
         ('x7dys', 'x4bro'), ('x4bro', 'x1smo'), ('x0eit', 'x4bro'),
         ('x0eit', 'x5lun'), ('x0eit', 'x6tub'), ('x5lun', 'x6tub'),
         ('x5lun', 'x1smo'), ('x1smo', 'x4bro'), ('x0eit', 'x3asi'),
         ('x4bro', 'x1smo'), None]
    assert trace.trace['arc_2'] == \
        [None, ('x7dys', 'x4bro'), ('x5lun', 'x0eit'), ('x2xra', 'x0eit'),
         ('x1smo', 'x4bro'), ('x6tub', 'x0eit'), ('x6tub', 'x5lun'),
         ('x0eit', 'x7dys'), ('x0eit', 'x1smo'), ('x4bro', 'x7dys'),
         ('x7dys', 'x0eit'), ('x5lun', 'x6tub'), ('x5lun', 'x6tub'),
         ('x0eit', 'x5lun'), ('x0eit', 'x4bro'), ('x0eit', 'x5lun'),
         ('x0eit', 'x2xra'), ('x5lun', 'x1smo'), ('x5lun', 'x1smo'),
         ('x0eit', 'x3asi'), ('x0eit', 'x3asi'), ('x5lun', 'x3asi'),
         ('x1smo', 'x6tub'), ('x1smo', 'x6tub')]
    assert trace.trace['knowledge'] == \
        [None, None, None, None, None, None, None, None, None, None, None,
         None, None, None, None, None, None, None, None, None, None, None,
         None, None]
    assert trace.trace['blocked'] == \
        [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
         [], [], [], [], [('reverse', ('x4bro', 'x1smo'), 0.0, {'elem': 10})],
         [('delete', ('x0eit', 'x3asi'), 2.724381, {'elem': 1})],
         [('delete', ('x0eit', 'x3asi'), 2.724381, {'elem': 10})]]
    assert trace.result.to_string() == \
        ('[x0eit|x5lun:x6tub]' +
         '[x1smo|x5lun]' +
         '[x2xra|x0eit]' +
         '[x3asi]' +
         '[x4bro|x1smo]' +
         '[x5lun]' +
         '[x6tub]' +
         '[x7dys|x0eit:x4bro]')
    print(trace)

    trace.rename({'x3asi': 'asia', 'x1smo': 'smoke', 'x2xra': 'xray',
                  'x0eit': 'either', 'x4bro': 'bronc', 'x5lun': 'lung',
                  'x6tub': 'tub', 'x7dys': 'dysp'})
    assert orig_trace == trace


def test_trace_rename_asia_3_ok():  # tabu with knowledge, replace all
    trace = Trace.read('TABU/EQV/L012/asia',
                       TESTDATA_DIR + '/experiments')['N1000_0']
    orig_trace = deepcopy(trace)

    trace.rename({'asia': 'x3asi', 'smoke': 'x1smo', 'xray': 'x2xra',
                  'either': 'x0eit', 'bronc': 'x4bro', 'lung': 'x5lun',
                  'tub': 'x6tub', 'dysp': 'x7dys'})

    assert isinstance(trace, Trace)
    assert trace.trace['arc'] == \
        [None, ('x4bro', 'x7dys'), ('x5lun', 'x0eit'), ('x0eit', 'x2xra'),
         ('x6tub', 'x0eit'), ('x1smo', 'x4bro'), ('x1smo', 'x5lun'),
         ('x0eit', 'x7dys'), ('x0eit', 'x3asi'), ('x6tub', 'x1smo'),
         ('x0eit', 'x3asi'), ('x6tub', 'x1smo'), ('x0eit', 'x3asi'),
         ('x1smo', 'x2xra'), ('x1smo', 'x6tub'), ('x0eit', 'x3asi'),
         ('x5lun', 'x3asi'), ('x1smo', 'x2xra'), ('x5lun', 'x3asi'),
         ('x0eit', 'x3asi'), ('x6tub', 'x1smo'), ('x0eit', 'x3asi'),
         ('x6tub', 'x1smo'), ('x0eit', 'x3asi'), None]
    assert trace.trace['arc_2'] == \
        [None, ('x7dys', 'x4bro'), ('x0eit', 'x5lun'), ('x5lun', 'x2xra'),
         ('x1smo', 'x4bro'), ('x4bro', 'x1smo'), ('x5lun', 'x1smo'),
         ('x5lun', 'x7dys'), ('x5lun', 'x3asi'), ('x1smo', 'x6tub'),
         ('x6tub', 'x1smo'), ('x5lun', 'x3asi'), ('x5lun', 'x3asi'),
         ('x0eit', 'x4bro'), ('x0eit', 'x3asi'), ('x6tub', 'x1smo'),
         ('x6tub', 'x1smo'), ('x6tub', 'x1smo'), ('x6tub', 'x1smo'),
         ('x4bro', 'x3asi'), ('x5lun', 'x6tub'), ('x1smo', 'x2xra'),
         ('x5lun', 'x3asi'), ('x5lun', 'x3asi'), ('x0eit', 'x4bro')]
    assert trace.trace['knowledge'] == \
        [None, ('equiv_add', True, 'swap_best', ('x7dys', 'x4bro')),
         ('equiv_add', True, 'no_op', ('x5lun', 'x0eit')), None, None,
         ('equiv_add', True, 'no_op', ('x1smo', 'x4bro')),
         ('equiv_add', True, 'swap_best', ('x5lun', 'x1smo')), None,
         ('act_cache', True, 'stop_rev', ('x1smo', 'x5lun')),
         ('equiv_add', None, 'no_op', ('x6tub', 'x1smo')), None, None,
         ('act_cache', True, 'stop_rev', ('x1smo', 'x5lun')),
         ('act_cache', True, 'stop_rev', ('x1smo', 'x5lun')), None, None,
         ('act_cache', True, 'stop_rev', ('x1smo', 'x5lun')), None, None,
         ('act_cache', True, 'stop_rev', ('x1smo', 'x5lun')),
         ('act_cache', True, 'stop_rev', ('x1smo', 'x5lun')), None, None,
         ('act_cache', True, 'stop_rev', ('x1smo', 'x5lun')),
         ('act_cache', True, 'stop_rev', ('x1smo', 'x5lun'))]
    assert trace.trace['blocked'] == \
        [[], [], [], [], [], [], [], [], [],
         [('delete', ('x0eit', 'x3asi'), 2.724381, {'elem': 8})],
         [('delete', ('x6tub', 'x1smo'), 2.890381, {'elem': 9})],
         [('delete', ('x6tub', 'x1smo'), 2.890381, {'elem': 8})],
         [('delete', ('x1smo', 'x6tub'), 2.890381, {'elem': 8}),
          ('reverse', ('x1smo', 'x6tub'), 0.0, {'elem': 1})],
         [('delete', ('x0eit', 'x3asi'), 2.724381, {'elem': 2}),
          ('delete', ('x1smo', 'x6tub'), 2.890381, {'elem': 9}),
          ('reverse', ('x1smo', 'x6tub'), 0.0, {'elem': 10})],
         [('delete', ('x1smo', 'x2xra'), 5.347522, {'elem': 3})],
         [('delete', ('x1smo', 'x2xra'), 5.347522, {'elem': 9})],
         [('add', ('x0eit', 'x3asi'), -2.724381, {'elem': 5}),
          ('delete', ('x1smo', 'x2xra'), 5.347522, {'elem': 8})],
         [('delete', ('x5lun', 'x3asi'), 2.863944, {'elem': 6})], [],
         [('add', ('x5lun', 'x3asi'), -2.863944, {'elem': 8}),
          ('add', ('x6tub', 'x1smo'), -2.890381, {'elem': 1})],
         [('delete', ('x0eit', 'x3asi'), 2.724381, {'elem': 9})],
         [('delete', ('x6tub', 'x1smo'), 2.890381, {'elem': 10})],
         [('delete', ('x6tub', 'x1smo'), 2.890381, {'elem': 9})],
         [('delete', ('x1smo', 'x6tub'), 2.890381, {'elem': 9}),
          ('reverse', ('x1smo', 'x6tub'), 0.0, {'elem': 2})],
         [('delete', ('x0eit', 'x3asi'), 2.724381, {'elem': 3}),
          ('delete', ('x1smo', 'x6tub'), 2.890381, {'elem': 10})]]
    assert trace.result.to_string() == \
        ('[x0eit|x5lun:x6tub]' +
         '[x1smo]' +
         '[x2xra|x0eit]' +
         '[x3asi]' +
         '[x4bro|x1smo]' +
         '[x5lun|x1smo]' +
         '[x6tub]' +
         '[x7dys|x0eit:x4bro]')
    print(trace)

    trace.rename({'x3asi': 'asia', 'x1smo': 'smoke', 'x2xra': 'xray',
                  'x0eit': 'either', 'x4bro': 'bronc', 'x5lun': 'lung',
                  'x6tub': 'tub', 'x7dys': 'dysp'})
    assert orig_trace == trace
