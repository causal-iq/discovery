
# Test the Timing class

import pytest
from time import time, sleep

from core.timing import Timing

TOO_LONG = ''.join(['*'] * (Timing.MAX_ACTION_LEN + 1))


@pytest.fixture(scope="module")  # Ensure timing enabled with module fixture
def on():
    filter = {'ok_1', 'ok_2', 'ok_3a', 'ok_3b'}
    Timing.on(True, filter)

    # Have to directly set filter to include illegal action type/values
    # so that can test for the relevant exceptions

    Timing.filter = filter | {'error', None, False, 2, '', TOO_LONG}
    return Timing.filter


# Test on() function

def test_timing_on_type_error_1_(on):  # no arguments
    with pytest.raises(TypeError):
        Timing.on()


def test_timing_on_type_error_2_(on):  # bad type for active
    with pytest.raises(TypeError):
        Timing.on(1)
    with pytest.raises(TypeError):
        Timing.on([False])
    with pytest.raises(TypeError):
        Timing.on('False')
    with pytest.raises(TypeError):
        Timing.on(None, set())


def test_timing_on_type_error_3_(on):  # bad type for filter
    with pytest.raises(TypeError):
        Timing.on(True, filter=False)
    with pytest.raises(TypeError):
        Timing.on(True, filter='invalid')
    with pytest.raises(TypeError):
        Timing.on(True, filter=['invalid'])


def test_timing_on_ok_1_(on):  # fixture should ensure timing on
    assert Timing.active is True
    assert Timing.filter == on


# Test now() function

def test_timing_now_type_error_1_(on):  # no arguments expected
    with pytest.raises(TypeError):
        Timing.now(False)


def test_timing_now_ok_1_(on):  # check return is as expected
    before = time()
    sleep(1.0)
    now = Timing.now()
    assert isinstance(now, float)
    assert now >= before + 1.0


# Test record() function

def test_timing_record_type_error_1_(on):  # no arguments specified
    with pytest.raises(TypeError):
        Timing.record()


def test_timing_record_type_error_2_(on):  # insufficient arguments
    with pytest.raises(TypeError):
        Timing.record('error', 0)
    with pytest.raises(TypeError):
        Timing.record('error', 0)


def test_timing_record_type_error_3_(on):  # bad action type
    with pytest.raises(TypeError):
        Timing.record(None, 0, time())
    with pytest.raises(TypeError):
        Timing.record(False, 0, time())
    with pytest.raises(TypeError):
        Timing.record(2, 0, time())


def test_timing_record_type_error_4_(on):  # bad scale type
    with pytest.raises(TypeError):
        Timing.record('error', None, time())
    with pytest.raises(TypeError):
        Timing.record('error', False, time())
    with pytest.raises(TypeError):
        Timing.record('error', 'invalid', time())
    with pytest.raises(TypeError):
        Timing.record('error', 2.3, time())


def test_timing_record_type_error_5_(on):  # bad start type
    with pytest.raises(TypeError):
        Timing.record('error', 1, None)
    with pytest.raises(TypeError):
        Timing.record('error', 1, 1)
    with pytest.raises(TypeError):
        Timing.record('error', 1, False)
    with pytest.raises(TypeError):
        Timing.record('error', 1, 'invalid')
    with pytest.raises(TypeError):
        Timing.record('error', 1, [32.5])


def test_timing_record_value_error_1_(on):  # bad action length
    with pytest.raises(ValueError):
        Timing.record('', 1, time())
    with pytest.raises(ValueError):
        Timing.record(TOO_LONG, 1, time())


def test_timing_record_ok_1_(on):  # record single action of one class
    start = time()
    sleep(0.1)
    Timing.record('ok_1', 0, start)

    print('\n\nok_1 timings are:\n{}'.format(Timing.to_string({'ok_1'})))

    assert 'ok_1' in Timing.times
    assert 0 in Timing.times['ok_1']
    assert Timing.times['ok_1'][0]['count'] == 1
    assert Timing.times['ok_1'][0]['total'] == Timing.times['ok_1'][0]['max']


def test_timing_record_ok_2_(on):  # record single action with different scales
    start = time()
    sleep(0.05)
    Timing.record('ok_2', 1, start)
    sleep(0.03)
    Timing.record('ok_2', 0, start)
    sleep(0.01)
    Timing.record('ok_2', 1, start)

    print('\n\nok_2 timings are:\n{}'.format(Timing.to_string({'ok_2'})))

    assert 'ok_2' in Timing.times
    assert 0 in Timing.times['ok_2']
    assert Timing.times['ok_2'][0]['count'] == 1
    assert Timing.times['ok_2'][0]['total'] == Timing.times['ok_2'][0]['max']
    assert Timing.times['ok_2'][1]['count'] == 2


def test_timing_record_ok_3_(on):  # record different actions
    start = time()
    sleep(0.05)
    Timing.record('ok_3b', 1, start)
    sleep(0.03)
    Timing.record('ok_3b', 0, start)
    sleep(0.01)
    Timing.record('ok_3a', 1, start)
    sleep(0.03)
    Timing.record('ok_3b', 0, start)

    print('\n\nok_3 timings are:\n{}'
          .format(Timing.to_string({'ok_3a', 'ok_3b'})))

    assert 'ok_3a' in Timing.times
    assert 'ok_3b' in Timing.times
    assert set(Timing.times['ok_3a']) == {1}
    assert set(Timing.times['ok_3b']) == {0, 1}
    assert Timing.times['ok_3a'][1]['count'] == 1
    assert Timing.times['ok_3a'][1]['total'] == Timing.times['ok_3a'][1]['max']
    assert Timing.times['ok_3b'][0]['count'] == 2
    assert Timing.times['ok_3b'][1]['count'] == 1
    assert Timing.times['ok_3b'][1]['total'] == Timing.times['ok_3b'][1]['max']
