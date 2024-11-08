
# Test the TabuList class

import pytest

from learn.tabulist import TabuList
from learn.trace import Activity, Trace, Detail
from learn.dagchange import DAGChange


@pytest.fixture  # an initialised Trace object for testing
def _trace():
    _trace = Trace({'id': 'TabuList module testing'})
    _trace.add(Activity.INIT, {Detail.DELTA: 31.2})
    return _trace


# Testing the constructtor


def test_tabulist_type_error_1():  # no constructor arguments
    with pytest.raises(TypeError):
        TabuList()


def test_tabulist_type_error_2():  # wrong constructor argument type
    with pytest.raises(TypeError):
        TabuList(None)
    with pytest.raises(TypeError):
        TabuList(False)


def test_tabulist_value_error_1():  # Tabu length too small
    with pytest.raises(ValueError):
        TabuList(0)
    with pytest.raises(ValueError):
        TabuList(-1)


def test_tabulist_value_error_2():  # Tabu length too big
    with pytest.raises(ValueError):
        TabuList(101)
    with pytest.raises(ValueError):
        TabuList(1000000000)


def test_tabulist_ok_1():  # Initialise TabuList with 10 entries
    tabulist = TabuList(10)
    assert tabulist.tabu == [None] * 10
    assert tabulist.ptr == 0


def test_tabulist_ok_2():  # Initialise TabuList with 1 entry
    tabulist = TabuList(1)
    assert tabulist.tabu == [None]
    assert tabulist.ptr == 0


def test_tabulist_ok_3():  # Initialise TabuList with 100 entries
    tabulist = TabuList(100)
    assert tabulist.tabu == [None] * 100
    assert tabulist.ptr == 0

# Testing the add function


def test_tabulist_add_type_error_1():  # no arguments
    tabulist = TabuList(10)
    with pytest.raises(TypeError):
        tabulist.add()


def test_tabulist_add_type_error_2():  # arg not a dict
    tabulist = TabuList(10)
    with pytest.raises(TypeError):
        tabulist.add(12)
    with pytest.raises(TypeError):
        tabulist.add('A')
    with pytest.raises(TypeError):
        tabulist.add(True)


def test_tabulist_add_type_error_3():  # arg dict values not all sets
    tabulist = TabuList(10)
    with pytest.raises(TypeError):
        tabulist.add({'A': 1})
    with pytest.raises(TypeError):
        tabulist.add({'A': 'B'})
    with pytest.raises(TypeError):
        tabulist.add({'A': ['B']})
    with pytest.raises(TypeError):
        tabulist.add({'A': set(), 'B': None})


def test_tabulist_add_ok_1():  # check valid adds into length 1 list
    tabulist = TabuList(1)
    assert tabulist.tabu == [None]
    assert tabulist.ptr == 0
    tabulist.add({'A': set(), 'B': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set()}]
    assert tabulist.ptr == 0
    tabulist.add({'A': set('B'), 'B': set()})
    assert tabulist.tabu == [{'A': set('B'), 'B': set()}]
    assert tabulist.ptr == 0
    tabulist.add({'A': set(), 'B': set('A')})
    assert tabulist.tabu == [{'A': set(), 'B': set('A')}]
    assert tabulist.ptr == 0
    tabulist.add({'A': set(), 'B': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set()}]
    assert tabulist.ptr == 0


def test_tabulist_add_ok_2():  # check valid adds into length 2 list
    tabulist = TabuList(2)
    assert tabulist.tabu == [None,
                             None]
    assert tabulist.ptr == 0

    tabulist.add({'A': set(), 'B': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set()},
                             None]
    assert tabulist.ptr == 1

    tabulist.add({'A': set('B'), 'B': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set()},
                             {'A': set('B'), 'B': set()}]
    assert tabulist.ptr == 0

    tabulist.add({'A': set(), 'B': set('A')})
    assert tabulist.tabu == [{'A': set(), 'B': set('A')},
                             {'A': set('B'), 'B': set()}]
    assert tabulist.ptr == 1

    tabulist.add({'A': set(), 'B': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set('A')},
                             {'A': set(), 'B': set()}]


def test_tabulist_add_ok_3():  # check valid adds into length 3 list
    tabulist = TabuList(3)
    assert tabulist.tabu == [None,
                             None,
                             None]
    assert tabulist.ptr == 0

    tabulist.add({'A': set(), 'B': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set()},
                             None,
                             None]
    assert tabulist.ptr == 1

    tabulist.add({'A': set('B'), 'B': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set()},
                             {'A': set('B'), 'B': set()},
                             None]
    assert tabulist.ptr == 2

    tabulist.add({'A': set(), 'B': set('A')})
    assert tabulist.tabu == [{'A': set(), 'B': set()},
                             {'A': set('B'), 'B': set()},
                             {'A': set(), 'B': set('A')}]
    assert tabulist.ptr == 0

    tabulist.add({'A': set(), 'B': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set()},
                             {'A': set('B'), 'B': set()},
                             {'A': set(), 'B': set('A')}]
    assert tabulist.ptr == 1

# Testing the hit function


def test_tabulist_hit_type_error_1():  # no arguments
    tabulist = TabuList(10)
    with pytest.raises(TypeError):
        tabulist.hit()


def test_tabulist_hit_type_error_2():  # missing parents arg
    tabulist = TabuList(10)
    proposed = DAGChange(Activity.ADD, ('A', 'B'), 1.0, {})
    with pytest.raises(TypeError):
        tabulist.hit(proposed=proposed)


def test_tabulist_hit_type_error_3():  # missing proposed arg
    tabulist = TabuList(10)
    parents = {'A': set(), 'B': set()}
    with pytest.raises(TypeError):
        tabulist.hit(parents)


def test_tabulist_hit_type_error_4():  # parents not a dict
    tabulist = TabuList(10)
    proposed = DAGChange(Activity.ADD, ('A', 'B'), 1.0, {})
    with pytest.raises(TypeError):
        tabulist.hit(12, proposed)
    with pytest.raises(TypeError):
        tabulist.hit('A', proposed)
    with pytest.raises(TypeError):
        tabulist.hit(True, proposed)


def test_tabulist_hit_type_error_5():  # arg dict values not all sets
    tabulist = TabuList(10)
    proposed = DAGChange(Activity.ADD, ('A', 'B'), 1.0, {})
    with pytest.raises(TypeError):
        tabulist.hit({'A': 1}, proposed)
    with pytest.raises(TypeError):
        tabulist.hit({'A': 'B'}, proposed)
    with pytest.raises(TypeError):
        tabulist.hit({'A': ['B']}, proposed)
    with pytest.raises(TypeError):
        tabulist.hit({'A': set(), 'B': None}, proposed)


def test_tabulist_hit_type_error_6():  # proposed not a DAGChange
    tabulist = TabuList(10)
    parents = {'A': set(), 'B': set()}
    with pytest.raises(TypeError):
        tabulist.hit(parents, True)
    with pytest.raises(TypeError):
        tabulist.hit(parents, 17)
    with pytest.raises(TypeError):
        tabulist.hit(parents, ['A', 1])


# Check successful hit calls, including checking input arguments not modified


def test_tabulist_hit_ok_1():  # hits on empty length 1 list always false
    tabulist = TabuList(1)
    assert tabulist.tabu == [None]
    assert tabulist.ptr == 0

    parents = {'A': set(), 'B': set()}
    proposed = DAGChange(Activity.ADD, ('A', 'B'), 1.0, {})
    assert tabulist.hit(parents, proposed) is None
    assert parents == {'A': set(), 'B': set()}
    assert proposed.activity == Activity.ADD
    assert proposed.arc == ('A', 'B')
    assert proposed.delta == 1.0
    assert proposed.counts == {}

    parents = {'A': set(), 'B': set('A')}
    proposed = DAGChange(Activity.DEL, ('A', 'B'), 1.0, {})
    assert tabulist.hit(parents, proposed) is None
    assert parents == {'A': set(), 'B': set('A')}
    assert proposed.activity == Activity.DEL
    assert proposed.arc == ('A', 'B')
    assert proposed.delta == 1.0
    assert proposed.counts == {}

    parents = {'A': set(), 'B': set('A')}
    proposed = DAGChange(Activity.REV, ('A', 'B'), 1.0, {})
    assert tabulist.hit(parents, proposed) is None
    assert parents == {'A': set(), 'B': set('A')}
    assert proposed.activity == Activity.REV
    assert proposed.arc == ('A', 'B')
    assert proposed.delta == 1.0
    assert proposed.counts == {}

    assert tabulist.blocked() == []


def test_tabulist_hit_ok_2():  # hits on empty length 2 list always false
    tabulist = TabuList(2)
    assert tabulist.tabu == [None, None]
    assert tabulist.ptr == 0

    parents = {'A': set(), 'B': set(), 'C': set()}
    proposed = DAGChange(Activity.ADD, ('A', 'B'), 1.0, {})
    assert tabulist.hit(parents, proposed) is None
    assert parents == {'A': set(), 'B': set(), 'C': set()}
    assert proposed.activity == Activity.ADD
    assert proposed.arc == ('A', 'B')
    assert proposed.delta == 1.0
    assert proposed.counts == {}

    parents = {'A': set(), 'B': set('A'), 'C': set()}
    proposed = DAGChange(Activity.DEL, ('A', 'B'), 1.0, {})
    assert tabulist.hit(parents, proposed) is None
    assert parents == {'A': set(), 'B': set('A'), 'C': set()}
    assert proposed.activity == Activity.DEL
    assert proposed.arc == ('A', 'B')
    assert proposed.delta == 1.0
    assert proposed.counts == {}

    parents = {'A': set(), 'B': set('C'), 'C': set()}
    proposed = DAGChange(Activity.REV, ('B', 'C'), 1.0, {})
    assert tabulist.hit(parents, proposed) is None
    assert parents == {'A': set(), 'B': set('C'), 'C': set()}
    assert proposed.activity == Activity.REV
    assert proposed.arc == ('B', 'C')
    assert proposed.delta == 1.0
    assert proposed.counts == {}

    assert tabulist.blocked() == []


def test_tabulist_hit_ok_3():  # length 1 list contains empty DAG
    tabulist = TabuList(1)
    assert tabulist.tabu == [None]
    assert tabulist.ptr == 0
    tabulist.add({'A': set(), 'B': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set()}]
    assert tabulist.ptr == 0

    parents = {'A': set(), 'B': set('')}  # A  B
    proposed = DAGChange(Activity.ADD, ('A', 'B'), 1.0, {})  # add A -- > B
    assert tabulist.hit(parents, proposed) is None  # misses empty DAG

    parents = {'A': set(), 'B': set('A')}  # A --> B
    proposed = DAGChange(Activity.DEL, ('A', 'B'), 1.0, {})  # delete A -- > B
    assert tabulist.hit(parents, proposed) == 1  # hits element 1

    parents = {'A': set(), 'B': set('A')}  # A --> B
    proposed = DAGChange(Activity.REV, ('A', 'B'), 1.0, {})  # reverse A --> B
    assert tabulist.hit(parents, proposed) is None  # misses empty DAG

    assert tabulist.blocked() == [(Activity.DEL.value, ('A', 'B'), 1.0,
                                  {'elem': 1})]


def test_tabulist_hit_ok_4():  # length 1 list contains A <-- B
    tabulist = TabuList(1)
    assert tabulist.tabu == [None]
    assert tabulist.ptr == 0
    tabulist.add({'A': set('B'), 'B': set()})
    assert tabulist.tabu == [{'A': set('B'), 'B': set()}]
    assert tabulist.ptr == 0

    parents = {'A': set(), 'B': set('')}  # A  B
    proposed = DAGChange(Activity.ADD, ('A', 'B'), 1.0, {})  # add A --> B
    assert tabulist.hit(parents, proposed) is None  # misses A <-- B

    parents = {'A': set(), 'B': set('')}  # A  B
    proposed = DAGChange(Activity.ADD, ('B', 'A'), 1.0, {})  # add A <-- B
    assert tabulist.hit(parents, proposed) == 1  # hits element 1

    parents = {'A': set(), 'B': set('A')}  # A --> B
    proposed = DAGChange(Activity.DEL, ('A', 'B'), 1.0, {})  # delete A -- > B
    assert tabulist.hit(parents, proposed) is None  # misses A <-- B

    parents = {'A': set(), 'B': set('A')}  # A --> B
    proposed = DAGChange(Activity.REV, ('A', 'B'), 1.0, {})  # reverse A --> B
    assert tabulist.hit(parents, proposed) == 1  # hits element 1

    assert tabulist.blocked() == [(Activity.ADD.value, ('B', 'A'), 1.0,
                                   {'elem': 1}),
                                  (Activity.REV.value, ('A', 'B'), 1.0,
                                   {'elem': 1})]


def test_tabulist_hit_ok_5():  # length 2 list contain empty DAG only
    tabulist = TabuList(2)
    assert tabulist.tabu == [None, None]
    assert tabulist.ptr == 0
    tabulist.add({'A': set(), 'B': set(), 'C': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set(), 'C': set()}, None]
    assert tabulist.ptr == 1

    parents = {'A': set(), 'B': set(), 'C': set()}  # A  B  C
    proposed = DAGChange(Activity.ADD, ('B', 'C'), 1.0, {})  # add B --> C
    assert tabulist.hit(parents, proposed) is None  # misses A  B  C

    parents = {'A': set('C'), 'B': set(), 'C': set()}  # C --> A  B
    proposed = DAGChange(Activity.DEL, ('C', 'A'), 1.0, {})  # delete C --> A
    assert tabulist.hit(parents, proposed) == 1  # hits element 1

    parents = {'A': set(), 'B': set('C'), 'C': set()}  # A  B <-- C
    proposed = DAGChange(Activity.REV, ('C', 'B'), 1.0, {})  # reverse C --> B
    assert tabulist.hit(parents, proposed) is None  # misses A  B  C

    assert tabulist.blocked() == [(Activity.DEL.value, ('C', 'A'), 1.0,
                                  {'elem': 1})]


def test_tabulist_hit_ok_6():  # length 2 list contains empty & B --> C
    tabulist = TabuList(2)
    assert tabulist.tabu == [None, None]
    assert tabulist.ptr == 0
    tabulist.add({'A': set(), 'B': set(), 'C': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set(), 'C': set()}, None]
    assert tabulist.ptr == 1
    tabulist.add({'A': set(), 'B': set(), 'C': set('B')})
    assert tabulist.tabu == [{'A': set(), 'B': set(), 'C': set()},
                             {'A': set(), 'B': set(), 'C': set('B')}]
    assert tabulist.ptr == 0

    parents = {'A': set(), 'B': set(), 'C': set()}  # A  B  C
    proposed = DAGChange(Activity.ADD, ('C', 'B'), 1.0, {})  # add C --> B
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {'A': set(), 'B': set(), 'C': set()}  # A  B  C
    proposed = DAGChange(Activity.ADD, ('B', 'C'), 1.0, {})  # add C --> B
    assert tabulist.hit(parents, proposed) == 2  # hit element 2

    parents = {'A': set(), 'B': set(), 'C': set()}  # A  B  C
    proposed = DAGChange(Activity.ADD, ('A', 'B'), 1.0, {})  # add A --> B
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {'A': set(), 'B': set('A'), 'C': set('B')}  # A -> B -> C
    proposed = DAGChange(Activity.DEL, ('A', 'B'), 1.0, {})  # delete A --> B
    assert tabulist.hit(parents, proposed) == 2  # hit element 2

    parents = {'A': set(), 'B': set('A'), 'C': set('B')}  # A -> B -> C
    proposed = DAGChange(Activity.DEL, ('B', 'C'), 1.0, {})  # delete B --> C
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {'A': set(), 'B': set(), 'C': set('B')}  # A  B -> C
    proposed = DAGChange(Activity.REV, ('B', 'C'), 1.0, {})  # reverse B --> C
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {'A': set(), 'B': set('C'), 'C': set()}  # A  B <- C
    proposed = DAGChange(Activity.REV, ('C', 'B'), 1.0, {})  # reverse B <- C
    assert tabulist.hit(parents, proposed) == 2  # hits element 2

    parents = {'A': set(), 'B': set(), 'C': set('B')}  # A  B -> C
    proposed = DAGChange(Activity.DEL, ('B', 'C'), 1.0, {})  # reverse B -> C
    assert tabulist.hit(parents, proposed) == 1  # hits element 1

    assert tabulist.blocked() == [(Activity.ADD.value, ('B', 'C'), 1.0,
                                   {'elem': 2}),
                                  (Activity.DEL.value, ('A', 'B'), 1.0,
                                   {'elem': 2}),
                                  (Activity.REV.value, ('C', 'B'), 1.0,
                                   {'elem': 2}),
                                  (Activity.DEL.value, ('B', 'C'), 1.0,
                                   {'elem': 1})]


def test_tabulist_hit_ok_7():  # length 2 list A -> B <- C and A  B <- C
    tabulist = TabuList(2)
    assert tabulist.tabu == [None, None]
    assert tabulist.ptr == 0
    tabulist.add({'A': set(), 'B': set(), 'C': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set(), 'C': set()}, None]
    assert tabulist.ptr == 1
    tabulist.add({'A': set(), 'B': set('C'), 'C': set()})
    assert tabulist.tabu == [{'A': set(), 'B': set(), 'C': set()},
                             {'A': set(), 'B': set('C'), 'C': set()}]
    assert tabulist.ptr == 0
    tabulist.add({'A': set(), 'B': {'A', 'C'}, 'C': set()})
    assert tabulist.tabu == [{'A': set(), 'B': {'A', 'C'}, 'C': set()},
                             {'A': set(), 'B': set('C'), 'C': set()}]
    assert tabulist.ptr == 1

    parents = {'A': set(), 'B': set(), 'C': set()}  # A  B  C
    proposed = DAGChange(Activity.ADD, ('C', 'B'), 1.0, {})  # add C --> B
    assert tabulist.hit(parents, proposed) == 2  # hit element 2

    parents = {'A': set(), 'B': {'A'}, 'C': set()}  # A -> B  C
    proposed = DAGChange(Activity.ADD, ('C', 'B'), 1.0, {})  # add C -> B
    assert tabulist.hit(parents, proposed) == 1  # hit element 1

    parents = {'A': set(), 'B': {'C'}, 'C': set()}  # A  B <- C
    proposed = DAGChange(Activity.ADD, ('A', 'B'), 1.0, {})  # add C -> B
    assert tabulist.hit(parents, proposed) == 1  # hit element 1

    parents = {'A': set(), 'B': {'A', 'C'}, 'C': set()}  # A -> B <- C
    proposed = DAGChange(Activity.ADD, ('A', 'C'), 1.0, {})  # add A -> C
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {'A': set(), 'B': {'A', 'C'}, 'C': set()}  # A -> B <- C
    proposed = DAGChange(Activity.DEL, ('A', 'B'), 1.0, {})  # delete A -> B
    assert tabulist.hit(parents, proposed) == 2  # hit element 2

    parents = {'A': set(), 'B': {'A', 'C'}, 'C': set()}  # A -> B <- C
    proposed = DAGChange(Activity.DEL, ('C', 'B'), 1.0, {})  # delete B <- C
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {'A': {'C'}, 'B': {'A', 'C'}, 'C': set()}  # C -> A -> B <- C
    proposed = DAGChange(Activity.DEL, ('C', 'A'), 1.0, {})  # delete A <- C
    assert tabulist.hit(parents, proposed) == 1  # hit element 1

    parents = {'A': set(), 'B': {'A', 'C'}, 'C': set()}  # A -> B <- C
    proposed = DAGChange(Activity.REV, ('A', 'B'), 1.0, {})  # reverse A -> B
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {'A': set(), 'B': {'A', 'C'}, 'C': set()}  # A -> B <- C
    proposed = DAGChange(Activity.REV, ('C', 'B'), 1.0, {})  # reverse B -> C
    assert tabulist.hit(parents, proposed) is None  # miss

    parents = {'A': set(), 'B': {'A'}, 'C': {'B'}}  # A -> B -> C
    proposed = DAGChange(Activity.REV, ('B', 'C'), 1.0, {})  # reverse B -> C
    assert tabulist.hit(parents, proposed) == 1  # hit element 1

    parents = {'A': {'B'}, 'B': {'C'}, 'C': set()}  # A <- B <- C
    proposed = DAGChange(Activity.REV, ('B', 'A'), 1.0, {})  # reverse A -> B
    assert tabulist.hit(parents, proposed) == 1  # hit element 1

    parents = {'A': set(), 'B': set(), 'C': {'B'}}  # A  B -> C
    proposed = DAGChange(Activity.REV, ('B', 'C'), 1.0, {})  # reverse B -> C
    assert tabulist.hit(parents, proposed) == 2  # hit element 2

    assert ([(Activity.ADD.value, ('C', 'B'), 1.0, {'elem': 2}),
             (Activity.ADD.value, ('C', 'B'), 1.0, {'elem': 1}),
             (Activity.ADD.value, ('A', 'B'), 1.0, {'elem': 1}),
             (Activity.DEL.value, ('A', 'B'), 1.0, {'elem': 2}),
             (Activity.DEL.value, ('C', 'A'), 1.0, {'elem': 1}),
             (Activity.REV.value, ('B', 'C'), 1.0, {'elem': 1}),
             (Activity.REV.value, ('B', 'A'), 1.0, {'elem': 1}),
             (Activity.REV.value, ('B', 'C'), 1.0, {'elem': 2})]
            == tabulist.blocked())
