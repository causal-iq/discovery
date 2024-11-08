
# Tests HCWorker functionality

import pytest

from learn.hc_worker import HCWorker, Prefer
from core.bn import BN
from core.metrics import values_same
from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from fileio.oracle import Oracle
from learn.knowledge import Knowledge
from learn.knowledge_rule import RuleSet


@pytest.fixture(scope='module')
def ab1():  # A-->B, plain HC parameters
    dsc = TESTDATA_DIR + '/discrete/tiny/ab.dsc'
    bn = BN.read(dsc)
    df = Pandas(df=bn.generate_cases(100))
    bn = Oracle(bn=bn)
    bn.set_N(100)
    params = {'score': 'bic', 'k': 1, 'prefer': Prefer.NONE}
    context = {'id': 'test/hc_worker/ab', 'in': dsc}
    return {'bn': bn, 'df': df, 'pa': params, 'co': context}


@pytest.fixture(scope='module')
def abc1():  # A-->B-->C, Tabu parameters
    dsc = TESTDATA_DIR + '/discrete/tiny/abc.dsc'
    bn = BN.read(dsc)
    df = Pandas(df=bn.generate_cases(1000))
    bn = Oracle(bn=bn)
    bn.set_N(1000)
    params = {'score': 'bic', 'k': 1, 'tabu': 10, 'prefer': Prefer.NONE}
    context = {'id': 'test/hc_worker/abc', 'in': dsc}
    return {'bn': bn, 'df': df, 'pa': params, 'co': context}


@pytest.fixture(scope='module')
def asia1():  # A-->B-->C, Tabu parameters
    dsc = TESTDATA_DIR + '/discrete/small/asia.dsc'
    bn = BN.read(dsc)
    df = Pandas(df=bn.generate_cases(1000))
    bn = Oracle(bn=bn)
    bn.set_N(1000)
    params = {'score': 'bic', 'k': 1, 'tabu': 10, 'prefer': Prefer.NONE}
    context = {'id': 'test/hc_worker/asia', 'in': dsc}
    return {'bn': bn, 'df': df, 'pa': params, 'co': context}


# Test constructor

def test_hc_worker_ab_1_ok(ab1):  # using AB Dataframe and plain HC params
    HCWorker.score_cache = {'initial': 'shoud be deleted'}
    hcw = HCWorker(data=ab1['df'], params=ab1['pa'], knowledge=False,
                   context=ab1['co'], init_cache=True)

    assert (hcw.data.sample == ab1['df'].sample).any().any()
    assert hcw.data.get_order() == ('A', 'B')
    assert hcw.params == {'score': 'bic', 'k': 1, 'zero': 0, 'noinc': 0,
                          'prefer': Prefer.NONE}
    assert hcw.knowledge is False
    assert tuple(HCWorker.score_cache.keys()) == \
        (('A', ()), ('B', ()), ('B', ('A',)), ('A', ('B',)))
    assert hcw.tabulist is None
    assert list(hcw.deltas.keys()) == [('A', 'B'), ('B', 'A')]

    assert hcw.trace.context['id'] == ab1['co']['id']
    assert hcw.trace.context['in'] == ab1['co']['in']
    assert hcw.trace.context['algorithm'] == 'HC'
    assert hcw.trace.context['N'] == 100
    assert hcw.trace.context['params'] == hcw.params
    assert hcw.trace.trace['activity'] == ['init']
    assert hcw.paused is None


def test_hc_worker_ab_2_ok(ab1):  # using AB BN and plain HC params
    HCWorker.score_cache = {'initial': 'shoud be deleted'}
    hcw = HCWorker(data=ab1['bn'], params=ab1['pa'], knowledge=False,
                   context=ab1['co'], init_cache=True)

    assert hcw.data.bn == ab1['bn'].bn
    assert hcw.data.get_order() == ('A', 'B')
    assert {k: v for k, v in hcw.params.items() if k != 'zero'} == \
        {'score': 'bic', 'k': 1, 'noinc': 0, 'prefer': Prefer.NONE}
    assert values_same(hcw.params['zero'], 1E-4, sf=10)
    assert hcw.knowledge is False
    assert tuple(HCWorker.score_cache.keys()) == \
        (('A', ()), ('B', ()), ('B', ('A',)), ('A', ('B',)))
    assert hcw.tabulist is None
    assert list(hcw.deltas.keys()) == [('A', 'B'), ('B', 'A')]

    assert hcw.trace.context['id'] == ab1['co']['id']
    assert hcw.trace.context['in'] == ab1['co']['in']
    assert hcw.trace.context['algorithm'] == 'HC'
    assert hcw.trace.context['N'] == 100
    assert hcw.trace.context['params'] == hcw.params
    assert hcw.trace.trace['activity'] == ['init']
    assert hcw.paused is None


def test_hc_worker_abc_1_ok(abc1):  # using ABC Dataframe and Tabu params
    HCWorker.score_cache = {'initial': 'shoud be deleted'}
    hcw = HCWorker(data=abc1['df'], params=abc1['pa'], knowledge=False,
                   context=abc1['co'], init_cache=True)

    assert (hcw.data.sample == abc1['df'].sample).any().any()
    assert hcw.data.get_order() == ('A', 'B', 'C')
    assert hcw.params == {'score': 'bic', 'k': 1, 'zero': 0, 'noinc': 10,
                          'tabu': 10, 'prefer': Prefer.NONE}
    assert hcw.knowledge is False
    assert tuple(HCWorker.score_cache.keys()) == \
        (('A', ()), ('B', ()), ('C', ()), ('B', ('A',)), ('C', ('A',)),
         ('A', ('B',)), ('C', ('B',)), ('A', ('C',)), ('B', ('C',)))
    assert hcw.tabulist.tabu == \
        [{'A': set(), 'B': set(), 'C': set()}, None, None, None, None, None,
         None, None, None, None]
    assert list(hcw.deltas.keys()) == \
        [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'),
         ('C', 'B')]

    assert hcw.trace.context['id'] == abc1['co']['id']
    assert hcw.trace.context['in'] == abc1['co']['in']
    assert hcw.trace.context['algorithm'] == 'HC'
    assert hcw.trace.context['N'] == 1000
    assert hcw.trace.context['params'] == hcw.params
    assert hcw.trace.trace['activity'] == ['init']
    assert hcw.paused is None


def test_hc_worker_abc_2_ok(abc1):  # using ABC BN and Tabu params
    HCWorker.score_cache = {'initial': 'shoud be deleted'}
    hcw = HCWorker(data=abc1['bn'], params=abc1['pa'], knowledge=False,
                   context=abc1['co'], init_cache=True)

    assert hcw.data.bn == abc1['bn'].bn
    assert hcw.data.get_order() == ('A', 'B', 'C')
    assert {k: v for k, v in hcw.params.items() if k != 'zero'} == \
        {'score': 'bic', 'k': 1, 'noinc': 10, 'tabu': 10,
         'prefer': Prefer.NONE}
    assert values_same(hcw.params['zero'], 1E-3, sf=10)
    assert hcw.knowledge is False
    assert tuple(HCWorker.score_cache.keys()) == \
        (('A', ()), ('B', ()), ('C', ()), ('B', ('A',)), ('C', ('A',)),
         ('A', ('B',)), ('C', ('B',)), ('A', ('C',)), ('B', ('C',)))
    assert hcw.tabulist.tabu == \
        [{'A': set(), 'B': set(), 'C': set()}, None, None, None, None, None,
         None, None, None, None]
    assert list(hcw.deltas.keys()) == \
        [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'),
         ('C', 'B')]

    assert hcw.trace.context['id'] == abc1['co']['id']
    assert hcw.trace.context['in'] == abc1['co']['in']
    assert hcw.trace.context['algorithm'] == 'HC'
    assert hcw.trace.context['N'] == 1000
    assert hcw.trace.context['params'] == hcw.params
    assert hcw.trace.trace['activity'] == ['init']
    assert hcw.paused is None


def test_hc_worker_asia_1_ok(asia1):  # using Asia Dataframe and Tabu params
    HCWorker.score_cache = {'initial': 'shoud be deleted'}
    hcw = HCWorker(data=asia1['df'], params=asia1['pa'], knowledge=False,
                   context=asia1['co'], init_cache=True)

    assert (hcw.data.sample == asia1['df'].sample).any().any()
    assert hcw.data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert hcw.params == {'score': 'bic', 'k': 1, 'zero': 0, 'noinc': 10,
                          'tabu': 10, 'prefer': Prefer.NONE}
    assert hcw.knowledge is False
    assert len(HCWorker.score_cache.keys()) == 64  # 8 + 8 x 7
    assert hcw.tabulist.tabu == \
        [{'asia': set(), 'bronc': set(), 'dysp': set(), 'either': set(),
          'lung': set(), 'smoke': set(), 'tub': set(), 'xray': set()}, None,
         None, None, None, None, None, None, None, None]
    assert len(hcw.deltas.keys()) == 56

    assert hcw.trace.context['id'] == asia1['co']['id']
    assert hcw.trace.context['in'] == asia1['co']['in']
    assert hcw.trace.context['algorithm'] == 'HC'
    assert hcw.trace.context['N'] == 1000
    assert hcw.trace.context['params'] == hcw.params
    assert hcw.trace.trace['activity'] == ['init']
    assert hcw.paused is None


def test_hc_worker_asia_2_ok(asia1):  # using Asia BN and Tabu params
    HCWorker.score_cache = {'initial': 'shoud be deleted'}
    hcw = HCWorker(data=asia1['bn'], params=asia1['pa'], knowledge=False,
                   context=asia1['co'], init_cache=True)

    assert hcw.data.bn == asia1['bn'].bn
    assert hcw.data.get_order() == \
        ('asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray')
    assert {k: v for k, v in hcw.params.items() if k != 'zero'} == \
        {'score': 'bic', 'k': 1, 'noinc': 10, 'tabu': 10,
         'prefer': Prefer.NONE}
    assert values_same(hcw.params['zero'], 1E-3, sf=10)
    assert hcw.knowledge is False
    assert len(HCWorker.score_cache.keys()) == 64  # 8 + 8 x 7
    assert hcw.tabulist.tabu == \
        [{'asia': set(), 'bronc': set(), 'dysp': set(), 'either': set(),
          'lung': set(), 'smoke': set(), 'tub': set(), 'xray': set()}, None,
         None, None, None, None, None, None, None, None]
    assert len(hcw.deltas.keys()) == 56

    assert hcw.trace.context['id'] == asia1['co']['id']
    assert hcw.trace.context['in'] == asia1['co']['in']
    assert hcw.trace.context['algorithm'] == 'HC'
    assert hcw.trace.context['N'] == 1000
    assert hcw.trace.context['params'] == hcw.params
    assert hcw.trace.trace['activity'] == ['init']
    assert hcw.paused is None


# Test score cache

def test_cache_ab_1_ok(ab1):  # using AB Dataframe and plain HC params
    HCWorker.score_cache = {'initial': 'shoud be deleted'}
    hcw = HCWorker(data=ab1['df'], params=ab1['pa'], knowledge=False,
                   context=ab1['co'], init_cache=True)

    # check score cache has correct values

    cache = hcw.score_cache
    assert set(cache.keys()) == \
        {('B', ('A',)), ('A', ('B',)), ('B', ()), ('A', ())}

    assert values_same(cache[('B', ('A',))][0], -30.32904566, sf=10)
    assert cache[('B', ('A',))][1] == \
        {'mean': 25.0, 'max': 39, 'min': 6, 'lt5': 0.0, 'fpa': 2}

    assert values_same(cache[('A', ('B',))][0], -25.82099817, sf=10)
    assert cache[('A', ('B',))][1] == \
        {'mean': 25.0, 'max': 39, 'min': 6, 'lt5': 0.0, 'fpa': 2}

    assert values_same(cache[('B', ())][0], -30.39559319, sf=10)
    assert cache[('B', ())][1] == \
        {'mean': 50.0, 'max': 59, 'min': 41, 'lt5': 0.0, 'fpa': 1}

    assert values_same(cache[('A', ())][0], -25.88754569, sf=10)
    assert cache[('A', ())][1] == \
        {'mean': 50.0, 'max': 74, 'min': 26, 'lt5': 0.0, 'fpa': 1}

    # Check cache initialisation clears cache

    HCWorker.init_score_cache()

    assert HCWorker.score_cache == {}


# Test clone() method

def test_clone_type_error_1(ab1):  # bad sequence type
    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (True, ), 'pause': True})
    hcw = HCWorker(data=ab1['df'], params=ab1['pa'], knowledge=know,
                   context=ab1['co'], init_cache=True)
    with pytest.raises(TypeError):
        hcw.clone(sequence=True)
    with pytest.raises(TypeError):
        hcw.clone(sequence='bad type', pause=True)
    with pytest.raises(TypeError):
        hcw.clone(sequence=(True, False, 1), pause=False)
    with pytest.raises(TypeError):
        hcw.clone(sequence=tuple(), pause=False)


def test_clone_type_error_2(ab1):  # bad pause type
    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (True, ), 'pause': True})
    hcw = HCWorker(data=ab1['df'], params=ab1['pa'], knowledge=know,
                   context=ab1['co'], init_cache=True)
    with pytest.raises(TypeError):
        hcw.clone(sequence=(True, ), pause=2)
    with pytest.raises(TypeError):
        hcw.clone(sequence=(True, ), pause=[True])


def test_clone_value_error_1(ab1):  # specify args with no knowledge
    hcw = HCWorker(data=ab1['df'], params=ab1['pa'], knowledge=False,
                   context=ab1['co'], init_cache=True)
    with pytest.raises(ValueError):
        hcw.clone(sequence=(True, ))
    with pytest.raises(ValueError):
        hcw.clone(pause=False)


def test_clone_value_error_2(ab1):  # specify args non equiv_seq knowledge
    know = Knowledge(rules=RuleSet.STOP_ARC,
                     params={'stop': {('B', 'C'): True}})
    hcw = HCWorker(data=ab1['df'], params=ab1['pa'], knowledge=know,
                   context=ab1['co'], init_cache=True)
    with pytest.raises(ValueError):
        hcw.clone(sequence=(True, ))
    with pytest.raises(TypeError):
        hcw.clone(pause=False)


def test_clone_ab_1_ok(ab1):  # check which variables have separate copies
    hcw = HCWorker(data=ab1['df'], params=ab1['pa'], knowledge=False,
                   context=ab1['co'], init_cache=True)
    clone = hcw.clone()

    assert id(hcw) != id(clone)  # clone is a different object

    assert id(hcw.data) == id(clone.data)  # using same copy of data
    assert id(hcw.params) == id(clone.params)  # using same copy of params

    assert id(hcw.score) == id(clone.score)  # simple type, same ref
    assert id(hcw.iter) == id(clone.iter)  # simple type, same ref
    assert id(hcw.num_noinc) == id(clone.num_noinc)  # simple type, same ref
    assert id(hcw.best_parents_score) == id(clone.best_parents_score)  # ditto
    assert id(hcw.paused) == id(clone.paused)  # ditto

    assert id(hcw.knowledge) == id(clone.knowledge)  # both reference False
    assert id(hcw.deltas) != id(clone.deltas)  # different copies
    assert id(hcw.parents) != id(clone.parents)  # different copies
    assert id(hcw.tabulist) == id(clone.tabulist)  # both reference None
    assert id(hcw.trace) != id(clone.trace)  # different copies
    assert id(hcw.best) == id(clone.best)  # both reference None
    assert id(hcw.best_parents) != id(clone.best_parents)  # different copies


def test_clone_ab_2_ok(ab1):  # check run on clone gives same result
    hcw = HCWorker(data=ab1['df'], params=ab1['pa'], knowledge=False,
                   context=ab1['co'], init_cache=True)
    clone = hcw.clone()

    hcw.run()
    clone.run()

    assert hcw.score == clone.score
    assert id(hcw.parents) != id(clone.parents)
    assert hcw.parents == clone.parents
    assert id(hcw.trace) != id(clone.trace)
    assert hcw.trace == clone.trace
    assert hcw.paused is False
    assert clone.paused is False


def test_clone_abc_1_ok(abc1):  # check which variables have separate copies
    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (True, ), 'pause': True})
    hcw = HCWorker(data=abc1['df'], params=abc1['pa'], knowledge=know,
                   context=abc1['co'], init_cache=True)
    clone = hcw.clone()

    assert id(hcw) != id(clone)  # clone is a different object

    assert id(hcw.data) == id(clone.data)  # using same copy of data
    assert id(hcw.params) == id(clone.params)  # using same copy of params

    assert id(hcw.score) == id(clone.score)  # simple type, same ref
    assert id(hcw.iter) == id(clone.iter)  # simple type, same ref
    assert id(hcw.num_noinc) == id(clone.num_noinc)  # simple type, same ref
    assert id(hcw.best_parents_score) == id(clone.best_parents_score)  # ditto
    assert id(hcw.paused) == id(clone.paused)  # ditto

    assert id(hcw.knowledge) != id(clone.knowledge)  # different copies
    assert id(hcw.deltas) != id(clone.deltas)  # different copies
    assert id(hcw.parents) != id(clone.parents)  # different copies
    assert id(hcw.tabulist) != id(clone.tabulist)  # different copies
    assert id(hcw.trace) != id(clone.trace)  # different copies
    assert id(hcw.best) == id(clone.best)  # both reference None
    assert id(hcw.best_parents) != id(clone.best_parents)  # different copies


def test_clone_abc_2_ok(abc1):  # check run on clone gives same result
    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (True, ), 'pause': True})
    hcw = HCWorker(data=abc1['df'], params=abc1['pa'], knowledge=know,
                   context=abc1['co'], init_cache=True)
    clone = hcw.clone()

    hcw.run()
    clone.run()

    assert hcw.score == clone.score
    assert id(hcw.parents) != id(clone.parents)
    assert hcw.parents == clone.parents
    assert id(hcw.trace) != id(clone.trace)
    assert hcw.trace == clone.trace
    assert hcw.paused is False
    assert clone.paused is False


def test_clone_asia_1_ok(asia1):  # check which variables have separate copies
    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (True, False)})
    hcw = HCWorker(data=asia1['df'], params=asia1['pa'], knowledge=know,
                   context=asia1['co'], init_cache=True)
    clone = hcw.clone()

    assert id(hcw) != id(clone)  # clone is a different object

    assert id(hcw.data) == id(clone.data)  # using same copy of data
    assert id(hcw.params) == id(clone.params)  # using same copy of params

    assert id(hcw.score) == id(clone.score)  # simple type, same ref
    assert id(hcw.iter) == id(clone.iter)  # simple type, same ref
    assert id(hcw.num_noinc) == id(clone.num_noinc)  # simple type, same ref
    assert id(hcw.best_parents_score) == id(clone.best_parents_score)  # ditto
    assert id(hcw.paused) == id(clone.paused)  # ditto

    assert id(hcw.knowledge) != id(clone.knowledge)  # different copies
    assert id(hcw.deltas) != id(clone.deltas)  # different copies
    assert id(hcw.parents) != id(clone.parents)  # different copies
    assert id(hcw.tabulist) != id(clone.tabulist)  # different copies
    assert id(hcw.trace) != id(clone.trace)  # different copies
    assert id(hcw.best) == id(clone.best)  # both reference None
    assert id(hcw.best_parents) != id(clone.best_parents)  # different copies


def test_clone_asia_2_ok(asia1):  # check run on clone gives same result
    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (True, False)})
    hcw = HCWorker(data=asia1['df'], params=asia1['pa'], knowledge=know,
                   context=asia1['co'], init_cache=True)
    clone = hcw.clone()

    hcw.run()
    clone.run()

    assert hcw.score == clone.score
    assert id(hcw.parents) != id(clone.parents)
    assert hcw.parents == clone.parents
    assert id(hcw.trace) != id(clone.trace)
    assert hcw.trace == clone.trace
    assert hcw.paused is False
    assert clone.paused is False


# Check pause and restart

def test_pause_asia_1_ok(asia1):
    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (False,), 'pause': True})
    hcw = HCWorker(data=asia1['df'], params=asia1['pa'], knowledge=know,
                   context=asia1['co'], init_cache=True)

    # run() over one iteration with sequence (False,) then pause

    hcw.run()

    # Add first arc bronc-->dysp which is in var order as seq[0] is False

    assert hcw.trace.trace['activity'] == ['init', 'add']
    assert hcw.trace.trace['arc'] == [None, ('bronc', 'dysp')]
    assert hcw.trace.trace['knowledge'] == \
        [None, ('equiv_seq', True, 'no_op', ('bronc', 'dysp'))]
    assert hcw.parents == \
        {'asia': set(), 'bronc': set(), 'dysp': {'bronc'}, 'either': set(),
         'lung': set(), 'smoke': set(), 'tub': set(), 'xray': set()}
    assert hcw.iter == 1
    assert hcw.paused is True
    assert hcw.knowledge.count == 1
    print('\n\nTrace after 1st run():\n{}'.format(hcw.trace))

    # extend the sequence by one more decision point & restart

    hcw.knowledge.set_sequence((False, True), True)
    hcw.run()

    # Second add is equiv_add against var order as seq[1] is True,
    # followed then by 2 non-equiv adds

    assert hcw.trace.trace['activity'] == \
        ['init', 'add', 'add', 'add', 'add']
    assert hcw.trace.trace['arc'] == \
        [None, ('bronc', 'dysp'), ('lung', 'either'), ('either', 'xray'),
         ('tub', 'either')]
    assert hcw.trace.trace['knowledge'] == \
        [None,
         ('equiv_seq', True, 'no_op', ('bronc', 'dysp')),
         ('equiv_seq', True, 'swap_best', ('either', 'lung')),
         None, None]
    assert hcw.parents == \
        {'asia': set(), 'bronc': set(), 'dysp': {'bronc'},
         'either': {'lung', 'tub'}, 'lung': set(), 'smoke': set(),
         'tub': set(), 'xray': {'either'}}
    assert hcw.iter == 4
    assert hcw.paused is True
    assert hcw.knowledge.count == 2
    print('\n\nTrace after 2nd run():\n{}'.format(hcw.trace))

    # Clone hcw, extending sequence to FTF, check cloning OK

    hcw2 = hcw.clone(((False, True, False)))
    assert hcw2.trace.trace['activity'] == \
        ['init', 'add', 'add', 'add', 'add']
    assert hcw2.trace.trace['arc'] == \
        [None, ('bronc', 'dysp'), ('lung', 'either'), ('either', 'xray'),
         ('tub', 'either')]
    assert hcw2.trace.trace['knowledge'] == \
        [None,
         ('equiv_seq', True, 'no_op', ('bronc', 'dysp')),
         ('equiv_seq', True, 'swap_best', ('either', 'lung')),
         None, None]
    assert hcw2.parents == \
        {'asia': set(), 'bronc': set(), 'dysp': {'bronc'},
         'either': {'lung', 'tub'}, 'lung': set(), 'smoke': set(),
         'tub': set(), 'xray': {'either'}}
    assert hcw2.iter == 4
    assert hcw2.paused is True
    assert hcw2.knowledge.count == 2
    assert hcw2.knowledge.sequence == (False, True, False)
    assert hcw2.knowledge.pause is True

    # run the clone with extended sequence

    hcw2.run()

    # Third and final equiv add is against var order as seq[2] is False. hcw2
    # now runs to completion.

    assert hcw2.trace.trace['activity'] == \
        ['init', 'add', 'add', 'add', 'add', 'add', 'add', 'add', 'reverse',
         'reverse', 'add', 'reverse', 'reverse', 'add', 'delete', 'reverse',
         'reverse', 'add', 'reverse', 'stop']
    assert hcw2.trace.trace['arc'] == \
        [None, ('bronc', 'dysp'), ('lung', 'either'), ('either', 'xray'),
         ('tub', 'either'), ('bronc', 'smoke'), ('smoke', 'lung'),
         ('either', 'dysp'), ('bronc', 'smoke'), ('smoke', 'lung'),
         ('either', 'asia'), ('lung', 'smoke'), ('smoke', 'bronc'),
         ('smoke', 'tub'), ('either', 'asia'), ('bronc', 'smoke'),
         ('smoke', 'lung'), ('either', 'asia'), ('lung', 'smoke'), None]
    assert hcw2.trace.trace['knowledge'] == \
        [None, ('equiv_seq', True, 'no_op', ('bronc', 'dysp')),
         ('equiv_seq', True, 'swap_best', ('either', 'lung')),
         None, None, ('equiv_seq', True, 'no_op', ('bronc', 'smoke')),
         None, None, None, None, None, None, None, None, None, None, None,
         None, None, None]
    assert hcw2.parents == \
        {'asia': set(),
         'bronc': set(),
         'dysp': {'bronc', 'either'},
         'either': {'tub', 'lung'},
         'lung': {'smoke'},
         'smoke': {'bronc'},
         'tub': set(),
         'xray': {'either'}}
    assert hcw2.iter == 19
    assert hcw2.paused is False
    assert hcw2.knowledge.count == 3

    print('\n\nTrace after 1st hc2w run():\n{}'.format(hcw2.trace))


def test_pause_asia_2_ok(asia1):  # check paused and non-paused -> same result
    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (False,), 'pause': True})

    # hcw1 runs without knowledge - uses variable order

    hcw1 = HCWorker(data=asia1['df'], params=asia1['pa'], knowledge=False,
                    context=asia1['co'], init_cache=True)
    hcw1.run()

    # hcw2 runs with sequence of FF, then pause

    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': (False, False), 'pause': True})
    hcw2 = HCWorker(data=asia1['df'], params=asia1['pa'], knowledge=know,
                    context=asia1['co'], init_cache=False)
    hcw2.run()

    # clone hcw2 to hwc3 and extend sequence to FFFFFF....
    # clone hcw2 to hwc4 but extend sequence to FFT
    # clone hcw2 to hwc5 extending sequence to FFTTTTTT...
    # modify hcw2 to extend sequence to FFFFFF

    hcw3 = hcw2.clone(sequence=tuple([False]*30))
    hcw4 = hcw2.clone(sequence=(False, False, True), pause=False)
    hcw2.knowledge.set_sequence(tuple([False] * 6), False)

    # run hcw2, hcw3, hcw4 and hcw5 to completion

    hcw2.run()
    hcw3.run()
    hcw4.run()

    # hcw1, 2 and 3 should have same trace eve though 2 and 3 paused

    assert hcw1.trace == hcw2.trace
    assert hcw1.trace == hcw3.trace

    # hcw4 has different trace since we put T at seq[2], though same learnt DAG

    assert hcw1.trace != hcw4.trace
    assert hcw1.parents == hcw4.parents

    # First difference appears at iteration 3

    assert hcw1.trace.trace['activity'][3] == 'add'
    assert hcw1.trace.trace['arc'][3] == ('either', 'xray')
    assert hcw1.trace.trace['knowledge'][3] is None

    assert hcw4.trace.trace['activity'][3] == 'add'
    assert hcw4.trace.trace['arc'][3] == ('xray', 'either')
    assert hcw4.trace.trace['knowledge'][3] == \
        ('equiv_seq', True, 'swap_best', ('either', 'xray'))

    # hcw5 has decision sequence TTTT.... - does result in different DAG

    know = Knowledge(rules=RuleSet.EQUIV_SEQ,
                     params={'sequence': tuple([True] * 30)})
    hcw5 = HCWorker(data=asia1['df'], params=asia1['pa'], knowledge=know,
                    context=asia1['co'], init_cache=False)
    hcw5.run()

    # hcw1 and hcw5 do learnt different DAGs

    assert hcw1.parents == \
        {'asia': set(),
         'bronc': {'smoke'},
         'dysp': {'bronc', 'either'},
         'either': {'tub', 'lung'},
         'lung': set(),
         'smoke': {'lung'},
         'tub': set(),
         'xray': {'either'}}
    assert hcw5.parents == \
        {'asia': set(),
         'bronc': {'dysp', 'either'},
         'dysp': set(),
         'either': {'tub', 'lung'},
         'lung': {'dysp'},
         'smoke': {'bronc', 'lung'},
         'tub': set(),
         'xray': {'either'}}
