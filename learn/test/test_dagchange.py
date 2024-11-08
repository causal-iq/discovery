
# Test the DAGChange class

import pytest

from learn.dagchange import DAGChange, BestDAGChanges
from learn.trace import Activity


# Check DAGChange initialiser

def test_dag_change_type_error_1():  # initialiser with no arg
    with pytest.raises(TypeError):
        DAGChange()


def test_dag_change_type_error_2():  # initialiser bad activity type
    with pytest.raises(TypeError):
        DAGChange(2)
    with pytest.raises(TypeError):
        DAGChange('add')
    with pytest.raises(TypeError):
        DAGChange({'activity': 'stop'})
    with pytest.raises(TypeError):
        DAGChange(None)
    with pytest.raises(TypeError):
        DAGChange(23.2)


def test_dag_change_init_ok_1():  # initialiser with single arg
    change = DAGChange(Activity.STOP)
    assert change.activity == Activity.STOP
    assert change.arc is None
    assert change.delta is None
    assert change.counts is None
    print('\nChange is {}'.format(change))


def test_dag_change_init_ok_2():  # initialiser with full args
    change = DAGChange(Activity.ADD, ('A', 'B'), 1.1, {'lt5': 0.0})
    assert change.activity == Activity.ADD
    assert change.arc == ('A', 'B')
    assert change.delta == 1.1
    assert change.counts == {'lt5': 0.0}
    print('\nChange is {}'.format(change))


# Check DAGChange equality

def test_dag_change_eq_ok_1():  # equality with same instance
    change = DAGChange(Activity.REV, ('A', 'B'), -0.2, {'lt5': 0.0})
    assert change == change


def test_dag_change_eq_ok_2():  # equality with same instance
    change = DAGChange(Activity.STOP, None, None, None)
    assert change == change


def test_dag_change_eq_ok_3():  # equality with diff instance
    change1 = DAGChange(Activity.REV, ('A', 'B'), -0.2, {'lt5': 0.0})
    change2 = DAGChange(Activity.REV, ('A', 'B'), -0.2, {'lt5': 0.0})
    assert change1 == change2


def test_dag_change_eq_ok_4():  # equality with diff instance
    change1 = DAGChange(Activity.STOP)
    change2 = DAGChange(Activity.STOP)
    assert change1 == change2


# Check DAGChange inequality

def test_dag_change_ne_ok_1():  # equality with diff instance
    change1 = DAGChange(Activity.REV, ('A', 'B'), -0.2, {'lt5': 0.0})
    change2 = DAGChange(Activity.STOP)
    assert change1 != change2


# Check BestDAGChanges init errors

def test_best_changes_type_error_1():  # initialiser bad top type
    with pytest.raises(TypeError):
        BestDAGChanges(17)
    with pytest.raises(TypeError):
        BestDAGChanges(True)
    with pytest.raises(TypeError):
        BestDAGChanges(17.3)
    with pytest.raises(TypeError):
        BestDAGChanges('bad type')


def test_best_changes_type_error_2():  # initialiser bad top type
    change = DAGChange(Activity.REV, ('A', 'B'), -0.2, {'lt5': 0.0})
    with pytest.raises(TypeError):
        BestDAGChanges(17, change)
    with pytest.raises(TypeError):
        BestDAGChanges(True, change)
    with pytest.raises(TypeError):
        BestDAGChanges(17.3, change)
    with pytest.raises(TypeError):
        BestDAGChanges('bad type', change)


def test_best_changes_type_error_3():  # initialiser bad second type
    change = DAGChange(Activity.REV, ('A', 'B'), -0.2, {'lt5': 0.0})
    with pytest.raises(TypeError):
        BestDAGChanges(change, 17)
    with pytest.raises(TypeError):
        BestDAGChanges(change, True)
    with pytest.raises(TypeError):
        BestDAGChanges(change, 17.3)
    with pytest.raises(TypeError):
        BestDAGChanges(change, 'bad type')


def test_best_changes_type_error_4():  # initialiser bad second type
    with pytest.raises(TypeError):
        BestDAGChanges(None, 17)
    with pytest.raises(TypeError):
        BestDAGChanges(None, True)
    with pytest.raises(TypeError):
        BestDAGChanges(None, 17.3)
    with pytest.raises(TypeError):
        BestDAGChanges(None, 'bad type')


# Check BestDAGChanges init OK

def test_best_changes_init_ok_1():  # init OK with two Nones
    best = BestDAGChanges(None, None)
    assert best.top is None
    assert best.second is None


def test_best_changes_init_ok_2():  # init with defaults
    best = BestDAGChanges()
    assert best.top == DAGChange(Activity.STOP)
    assert best.second is None


def test_best_changes_init_ok_3():  # init with top default
    change = DAGChange(Activity.REV, ('A', 'B'), -0.2, {'lt5': 0.0})
    best = BestDAGChanges(second=change)
    assert best.top == DAGChange(Activity.STOP)
    assert best.second == change


def test_best_changes_init_ok_4():  # init with second default
    change = DAGChange(Activity.REV, ('A', 'B'), -0.2, {'lt5': 0.0})
    best = BestDAGChanges(change)
    assert best.top == change
    assert best.second is None


def test_best_changes_init_ok_5():  # init with specific args
    change1 = DAGChange(Activity.ADD, ('A', 'B'), 1.1, {'lt5': 0.0})
    change2 = DAGChange(Activity.REV, ('A', 'B'), -0.2, {'lt5': 0.0})
    best = BestDAGChanges(change1, change2)
    assert best.top == change1
    assert best.second == change2


# Check BestDAGChanges eq OK

def test_best_changes_eq_ok_1():  # eq with same specific
    change = DAGChange(Activity.ADD, ('A', 'B'), 1.1, {'lt5': 0.0})
    assert BestDAGChanges(change) == BestDAGChanges(change)


def test_best_changes_eq_ok_2():  # eq with defaults
    assert BestDAGChanges() == BestDAGChanges()


# Check BestDAGChanges ne OK

def test_best_changes_ne_ok_1():  # eq with same specific
    change = DAGChange(Activity.ADD, ('A', 'B'), 1.1, {'lt5': 0.0})
    assert BestDAGChanges(change) != BestDAGChanges()


def test_best_changes_ne_ok_2():  # ne with defaults
    change1 = DAGChange(Activity.ADD, ('A', 'B'), 1.1, {'lt5': 0.0})
    change2 = DAGChange(Activity.REV, ('A', 'B'), -0.2, {'lt5': 0.0})
    assert BestDAGChanges(change1) != BestDAGChanges(change1, change2)


def test_best_changes_ne_ok_3():  # ne with defaults
    change1 = DAGChange(Activity.ADD, ('A', 'B'), 1.1, {'lt5': 0.0})
    change2 = DAGChange(Activity.REV, ('A', 'B'), -0.2, {'lt5': 0.0})
    assert BestDAGChanges(second=change1) != BestDAGChanges(change1, change2)
