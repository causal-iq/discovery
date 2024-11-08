
#   Test the strucure learning Trace class

import pytest
from time import sleep
from os.path import exists
from os import remove, rmdir

from learn.trace import Trace, Activity, Detail
from core.common import SOFTWARE_VERSION, Randomise
from fileio.common import TESTDATA_DIR
import testdata.example_dags as ex_dag


def test_trace_constructor_type_error1():   # Constructor bad arg types
    with pytest.raises(TypeError):
        Trace(True)
    with pytest.raises(TypeError):
        Trace(3.17)


def test_trace_constructor_type_error2():   # Bad context value types
    with pytest.raises(TypeError):
        Trace({'N': 'should be int'})


def test_trace_constructor_type_error3():   # Bad context value types
    with pytest.raises(TypeError):
        Trace({'id': 32})
    with pytest.raises(TypeError):
        Trace({'id': 'trace/hc', 'in': {}})
    with pytest.raises(TypeError):
        Trace({'algorithm': 'HC', 'id': True})
    with pytest.raises(TypeError):
        Trace({'algorithm': True})
    with pytest.raises(TypeError):
        Trace({'in': 40.3})


def test_trace_constructor_type_error4():   # Bad context value types
    with pytest.raises(TypeError):
        Trace({'params': 32})


def test_trace_constructor_type_error5():   # Bad context score types
    with pytest.raises(TypeError):
        Trace({'score': 10})
    with pytest.raises(TypeError):
        Trace({'score': 'invalid'})
    with pytest.raises(TypeError):
        Trace({'score': [-10.1]})


def test_trace_constructor_type_error6():   # Bad context var_order type
    with pytest.raises(TypeError):
        Trace({'var_order': {'A', 'B'}})
    with pytest.raises(TypeError):
        Trace({'score': 'invalid'})
    with pytest.raises(TypeError):
        Trace({'score': [-10.1]})


def test_trace_constructor_type_error7():   # Bad context randomise type
    with pytest.raises(TypeError):
        Trace({'randomise': True})
    with pytest.raises(TypeError):
        Trace({'randomise': 'order'})
    with pytest.raises(TypeError):
        Trace({'randomise': {-10.1}})


def test_trace_constructor_type_error8():   # Bad randomise elements type
    with pytest.raises(TypeError):
        Trace({'randomise': [2]})
    with pytest.raises(TypeError):
        Trace({'randomise': [True, False]})
    with pytest.raises(TypeError):
        Trace({'randomise': ['a', 'b']})
    with pytest.raises(TypeError):
        Trace({'randomise': [Randomise.ORDER, 'bad']})


def test_trace_constructor_value_error1():   # Bad context keys
    with pytest.raises(ValueError):
        Trace({'invalid': 33})
    with pytest.raises(ValueError):
        Trace({'N': 100, 'invalid': 33})


def test_trace_constructor_value_error2():   # Bad id values
    with pytest.raises(ValueError):
        Trace({'id': ''})
    with pytest.raises(ValueError):
        Trace({'id': 'hi/ee/ww?'})


def test_trace_constructor_value_error3():   # More bad id values
    with pytest.raises(ValueError):
        Trace({'id': 'a//b'})
    with pytest.raises(ValueError):
        Trace({'id': '/.'})
    with pytest.raises(ValueError):
        Trace({'id': 'aa/./bb'})
    with pytest.raises(ValueError):
        Trace({'id': 'aa/../bb'})
    with pytest.raises(ValueError):
        Trace({'id': 'aa/bb  cc.test1'})
    with pytest.raises(ValueError):
        Trace({'id': 'aa/bb--1'})
    with pytest.raises(ValueError):
        Trace({'id': '__'})
    with pytest.raises(ValueError):
        Trace({'id': '  '})


def test_trace_constructor_value_error4():   # More bad id values
    with pytest.raises(ValueError):
        Trace({'id': ' a'})
    with pytest.raises(ValueError):
        Trace({'id': ' '})
    with pytest.raises(ValueError):
        Trace({'id': 'a '})
    with pytest.raises(ValueError):
        Trace({'id': '.a'})
    with pytest.raises(ValueError):
        Trace({'id': '_'})
    with pytest.raises(ValueError):
        Trace({'id': '- '})


def test_trace_constructor_1_ok():  # Constructor called with no arg
    trace = Trace()
    assert trace.context['software_version'] == SOFTWARE_VERSION


def test_trace_constructor_2_ok():  # Constructor called with empty dict
    trace = Trace({})
    assert trace.context['software_version'] == SOFTWARE_VERSION
    assert set(trace.context.keys()) == {'cpu', 'os', 'python', 'ram',
                                         'software_version'}


def test_trace_constructor_3_ok():  # Constructor called with dict
    trace = Trace({'N': 250, 'id': 'my expt'})
    assert trace.context['software_version'] == SOFTWARE_VERSION
    assert trace.context['N'] == 250
    assert trace.context['id'] == 'my expt'
    assert set(trace.context.keys()) == {'cpu', 'os', 'python', 'ram',
                                         'software_version', 'N', 'id'}


def test_trace_constructor_4_ok():  # Constructor called with dict
    trace = Trace({'N': 10, 'id': 'another expt', 'algorithm': 'PC',
                   'in': 'discrete/small/asia.dsc',
                   'params': {'alpha': 0.02}})
    assert trace.context['software_version'] == SOFTWARE_VERSION
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'another expt'
    assert trace.context['algorithm'] == 'PC'
    assert trace.context['in'] == 'discrete/small/asia.dsc'
    assert trace.context['params'] == {'alpha': 0.02}
    assert set(trace.context.keys()) == {'cpu', 'os', 'python', 'ram', 'in',
                                         'software_version', 'N', 'id',
                                         'params', 'algorithm'}


def test_trace_constructor_5_ok():  # Constructor called with dict incl. know
    trace = Trace({'N': 10, 'id': 'another expt', 'algorithm': 'PC',
                   'in': 'discrete/small/asia.dsc',
                   'params': {'alpha': 0.02},
                   'knowledge': 'Ruleset "Swap equivalent add" with limit 5'})
    assert set(trace.context.keys()) == {'cpu', 'os', 'python', 'ram', 'in',
                                         'software_version', 'N', 'id',
                                         'params', 'algorithm', 'knowledge'}
    assert trace.context['software_version'] == SOFTWARE_VERSION
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'another expt'
    assert trace.context['algorithm'] == 'PC'
    assert trace.context['in'] == 'discrete/small/asia.dsc'
    assert trace.context['params'] == {'alpha': 0.02}
    assert trace.context['knowledge'] == ('Ruleset "Swap equivalent add"'
                                          + ' with limit 5')


def test_trace_constructor_6_ok():  # Order randomisation with list
    trace = Trace({'N': 10, 'id': 'another expt', 'algorithm': 'PC',
                   'in': 'discrete/small/asia.dsc',
                   'params': {'alpha': 0.02},
                   'randomise': [Randomise.ORDER],
                   'var_order': ['xray', 'tub', 'dysp', 'either',
                                 'asia', 'lung', 'smoke', 'bronc']})
    assert set(trace.context.keys()) == {'cpu', 'os', 'python', 'ram', 'in',
                                         'software_version', 'N', 'id',
                                         'params', 'algorithm', 'var_order',
                                         'randomise'}
    assert trace.context['software_version'] == SOFTWARE_VERSION
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'another expt'
    assert trace.context['algorithm'] == 'PC'
    assert trace.context['in'] == 'discrete/small/asia.dsc'
    assert trace.context['params'] == {'alpha': 0.02}
    assert trace.context['var_order'] == ['xray', 'tub', 'dysp', 'either',
                                          'asia', 'lung', 'smoke', 'bronc']
    assert trace.context['randomise'] == [Randomise.ORDER]


def test_trace_constructor_7_ok():  # Order randomisation with single
    trace = Trace({'N': 10, 'id': 'another expt', 'algorithm': 'PC',
                   'in': 'discrete/small/asia.dsc',
                   'params': {'alpha': 0.02},
                   'randomise': Randomise.ORDER,
                   'var_order': ['xray', 'tub', 'dysp', 'either',
                                 'asia', 'lung', 'smoke', 'bronc']})
    assert set(trace.context.keys()) == {'cpu', 'os', 'python', 'ram', 'in',
                                         'software_version', 'N', 'id',
                                         'params', 'algorithm', 'var_order',
                                         'randomise'}
    assert trace.context['software_version'] == SOFTWARE_VERSION
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'another expt'
    assert trace.context['algorithm'] == 'PC'
    assert trace.context['in'] == 'discrete/small/asia.dsc'
    assert trace.context['params'] == {'alpha': 0.02}
    assert trace.context['var_order'] == ['xray', 'tub', 'dysp', 'either',
                                          'asia', 'lung', 'smoke', 'bronc']
    assert trace.context['randomise'] == Randomise.ORDER


def test_trace_constructor_8_ok():  # Order randomisation with list
    trace = Trace({'N': 10, 'id': 'another expt', 'algorithm': 'PC',
                   'in': 'discrete/small/asia.dsc',
                   'params': {'alpha': 0.02},
                   'randomise': [Randomise.ORDER, Randomise.KNOWLEDGE],
                   'var_order': ['xray', 'tub', 'dysp', 'either',
                                 'asia', 'lung', 'smoke', 'bronc']})
    assert set(trace.context.keys()) == {'cpu', 'os', 'python', 'ram', 'in',
                                         'software_version', 'N', 'id',
                                         'params', 'algorithm', 'var_order',
                                         'randomise'}
    assert trace.context['software_version'] == SOFTWARE_VERSION
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'another expt'
    assert trace.context['algorithm'] == 'PC'
    assert trace.context['in'] == 'discrete/small/asia.dsc'
    assert trace.context['params'] == {'alpha': 0.02}
    assert trace.context['var_order'] == ['xray', 'tub', 'dysp', 'either',
                                          'asia', 'lung', 'smoke', 'bronc']
    assert trace.context['randomise'] == [Randomise.ORDER, Randomise.KNOWLEDGE]


def test_trace_constructor_9_ok():  # Score
    trace = Trace({'N': 10, 'id': 'another expt', 'algorithm': 'HC',
                   'in': 'discrete/small/asia.dsc',
                   'params': {'score': 'bic', 'k': 1},
                   'score': -99.99})
    assert set(trace.context.keys()) == {'cpu', 'os', 'python', 'ram', 'in',
                                         'software_version', 'N', 'id',
                                         'params', 'algorithm', 'score'}
    assert trace.context['software_version'] == SOFTWARE_VERSION
    assert trace.context['N'] == 10
    assert trace.context['id'] == 'another expt'
    assert trace.context['algorithm'] == 'HC'
    assert trace.context['in'] == 'discrete/small/asia.dsc'
    assert trace.context['params'] == {'score': 'bic', 'k': 1}
    assert trace.context['score'] == -99.99


def test_trace_add_type_error1():   # add bad Activity Type
    trace = Trace()
    with pytest.raises(TypeError):
        trace.add()
    with pytest.raises(TypeError):
        trace.add(True)
    with pytest.raises(TypeError):
        trace.add('bad arg type')


def test_trace_add_type_error2():   # add bad details type
    trace = Trace()
    with pytest.raises(TypeError):
        trace.add(Activity.INIT, 37)


def test_trace_add_type_error3():   # add empty details dict
    trace = Trace()
    with pytest.raises(TypeError):
        trace.add(Activity.INIT, {})


def test_trace_add_type_error4():   # bad details key type
    trace = Trace()
    with pytest.raises(TypeError):
        trace.add(Activity.INIT, {'Score': 23.1})


def test_trace_add_type_error5():   # wrong type for Detail item
    trace = Trace()
    with pytest.raises(TypeError):
        trace.add(Activity.INIT, {Detail.DELTA: 'wrong type'})


def test_trace_add_attribute_error1():   # unknown Attribute type
    trace = Trace()
    with pytest.raises(AttributeError):
        trace.add(Activity.INIT, {Detail.SCORE: +00.09})


def test_trace_value_error_1():   # mandatory details not provided
    trace = Trace()
    with pytest.raises(ValueError):
        trace.add(Activity.INIT, {Detail.ARC: ('A', 'B')})
    with pytest.raises(ValueError):
        trace.add(Activity.ADD, {Detail.DELTA: 21.0})


def test_trace_ok_1():   # correct trace calls
    trace = Trace()
    sleep(0.02)
    trace.add(Activity.INIT, {Detail.DELTA: 31.2})
    print("\n\n{}".format(trace.get()))
    trace = trace.get().drop(labels='time', axis=1).to_dict('records')
    assert trace[0] == {'activity': 'init', 'arc': None, 'delta/score': 31.2,
                        'activity_2': None, 'arc_2': None, 'delta_2': None,
                        'min_N': None, 'mean_N': None, 'max_N': None,
                        'free_params': None, 'lt5': None, 'knowledge': None,
                        'blocked': None}


def test_trace_ok_2():   # correct trace calls usng chaining
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -200.8}) \
        .add(Activity.ADD, {Detail.ARC: ('A', 'B'), Detail.DELTA: 41.4}) \
        .add(Activity.DEL, {Detail.ARC: ('B', 'C'), Detail.DELTA: 22.7}) \
        .add(Activity.REV, {Detail.ARC: ('A', 'B'), Detail.DELTA: 39.9}) \
        .add(Activity.STOP, {Detail.DELTA: -100.8})
    print("\n\n{}".format(trace.get()))
    trace = trace.get().drop(labels='time', axis=1).to_dict('records')
    assert trace[0] == {'activity': 'init', 'arc': None, 'delta/score': -200.8,
                        'activity_2': None, 'arc_2': None, 'delta_2': None,
                        'min_N': None, 'mean_N': None, 'max_N': None,
                        'free_params': None, 'lt5': None, 'knowledge': None,
                        'blocked': None}
    assert trace[1] == {'activity': 'add', 'arc': ('A', 'B'),
                        'delta/score': 41.4, 'activity_2': None, 'arc_2': None,
                        'delta_2': None, 'min_N': None, 'mean_N': None,
                        'max_N': None, 'free_params': None, 'lt5': None,
                        'knowledge': None, 'blocked': None}
    assert trace[2] == {'activity': 'delete', 'arc': ('B', 'C'),
                        'delta/score': 22.7, 'activity_2': None, 'arc_2': None,
                        'delta_2': None, 'min_N': None, 'mean_N': None,
                        'max_N': None, 'free_params': None, 'lt5': None,
                        'knowledge': None, 'blocked': None}
    assert trace[3] == {'activity': 'reverse', 'arc': ('A', 'B'),
                        'delta/score': 39.9, 'activity_2': None, 'arc_2': None,
                        'delta_2': None, 'min_N': None, 'mean_N': None,
                        'max_N': None, 'free_params': None, 'lt5': None,
                        'knowledge': None, 'blocked': None}
    assert trace[4] == {'activity': 'stop', 'arc': None, 'delta/score': -100.8,
                        'activity_2': None, 'arc_2': None, 'delta_2': None,
                        'min_N': None, 'mean_N': None, 'max_N': None,
                        'free_params': None, 'lt5': None, 'knowledge': None,
                        'blocked': None}


def test_trace_ok_3():   # correct trace calls usng chaining including blocked
    trace = Trace().add(Activity.INIT, {Detail.DELTA: -200.8}) \
        .add(Activity.ADD, {Detail.ARC: ('A', 'B'), Detail.DELTA: 41.4,
                            Detail.BLOCKED: []}) \
        .add(Activity.DEL, {Detail.ARC: ('B', 'C'), Detail.DELTA: 22.7,
                            Detail.BLOCKED: []}) \
        .add(Activity.REV, {Detail.ARC: ('A', 'B'), Detail.DELTA: 39.9,
                            Detail.BLOCKED: [(Activity.ADD, ('B', 'A'),
                                              3.0, {})]}) \
        .add(Activity.STOP, {Detail.DELTA: -100.8})
    print("\n\n{}".format(trace.get()))
    trace = trace.get().drop(labels='time', axis=1).to_dict('records')
    assert trace[0] == {'activity': 'init', 'arc': None, 'delta/score': -200.8,
                        'activity_2': None, 'arc_2': None, 'delta_2': None,
                        'min_N': None, 'mean_N': None, 'max_N': None,
                        'free_params': None, 'lt5': None, 'knowledge': None,
                        'blocked': None}
    assert trace[1] == {'activity': 'add', 'arc': ('A', 'B'),
                        'delta/score': 41.4, 'activity_2': None, 'arc_2': None,
                        'delta_2': None, 'min_N': None, 'mean_N': None,
                        'max_N': None, 'free_params': None, 'lt5': None,
                        'knowledge': None, 'blocked': []}
    assert trace[2] == {'activity': 'delete', 'arc': ('B', 'C'),
                        'delta/score': 22.7, 'activity_2': None, 'arc_2': None,
                        'delta_2': None, 'min_N': None, 'mean_N': None,
                        'max_N': None, 'free_params': None, 'lt5': None,
                        'knowledge': None, 'blocked': []}
    assert trace[3] == {'activity': 'reverse', 'arc': ('A', 'B'),
                        'delta/score': 39.9, 'activity_2': None, 'arc_2': None,
                        'delta_2': None, 'min_N': None, 'mean_N': None,
                        'max_N': None, 'free_params': None, 'lt5': None,
                        'knowledge': None, 'blocked': [(Activity.ADD,
                                                        ('B', 'A'),
                                                        3.0, {})]}
    assert trace[4] == {'activity': 'stop', 'arc': None, 'delta/score': -100.8,
                        'activity_2': None, 'arc_2': None, 'delta_2': None,
                        'min_N': None, 'mean_N': None, 'max_N': None,
                        'free_params': None, 'lt5': None, 'knowledge': None,
                        'blocked': None}


def test_trace_set_result_type_error():  # bad result type
    with pytest.raises(TypeError):
        Trace().set_result('bad type')


def test_trace_set_result_ok():  # set Trace result
    trace = Trace().add(Activity.INIT, {Detail.DELTA: 0.0}) \
                   .add(Activity.ADD, {Detail.ARC: ('A', 'B'),
                                       Detail.DELTA: 31.3,
                                       Detail.ACTIVITY_2: 'add',
                                       Detail.ARC_2: ('B', 'C'),
                                       Detail.DELTA_2: 10.77}) \
                   .set_result(ex_dag.ab())
    ex_dag.ab(trace.result)

#   Tests on save and read functions


def test_trace_read_type_error_1():  # no argument type
    with pytest.raises(TypeError):
        Trace.read()


def test_trace_read_type_error_2():  # bad id argument
    with pytest.raises(TypeError):
        Trace.read(True)
    with pytest.raises(TypeError):
        Trace.read(39)
    with pytest.raises(TypeError):
        Trace.read(-11.2)
    with pytest.raises(TypeError):
        Trace.read([-11.2])
    with pytest.raises(TypeError):
        Trace.read('misc/trace', 32)
    with pytest.raises(TypeError):
        Trace.read('misc/trace', {'name': 'what'})


def test_trace_read_filenotfound_error():  # non-existent root directory
    with pytest.raises(FileNotFoundError):
        Trace.read('test/test1', 'nonexistent')


def test_trace_read_value_error_1():  # partial id is zero length
    with pytest.raises(ValueError):
        Trace.read('', TESTDATA_DIR + '/experiments')


def test_trace_read_value_error_2():  # binary file
    with pytest.raises(ValueError):
        Trace.read('misc/null.sys', TESTDATA_DIR)


def test_trace_read_value_error_3():  # textual file
    with pytest.raises(ValueError):
        Trace.read('misc/a_1.csv', TESTDATA_DIR)


def test_trace_read_value_error_4():  # textual file
    with pytest.raises(ValueError):
        Trace.read('misc/ab_cb.dsc', TESTDATA_DIR)


def test_trace_read_value_error_5():  # serialised file, but not Trace
    with pytest.raises(ValueError):
        Trace.read('misc/list', TESTDATA_DIR)


def test_trace_read_ok_1():  # non-existent trace file
    traces = Trace.read('HC_N_1/nonexistent',
                        TESTDATA_DIR + '/experiments')
    assert traces is None


def test_trace_read_ok_2():  # read OK from single entry file
    traces = Trace.read('HC_N_1/single', TESTDATA_DIR + '/experiments')
    assert isinstance(traces, dict)
    assert 'entry1' in traces
    assert isinstance(traces['entry1'], Trace)
    assert traces['entry1'].context['id'] == 'HC_N_1/single/entry1'


def test_trace_read_ok_3():  # non-existent entry
    traces = Trace.read('HC_N_1/single/nonexistent',
                        TESTDATA_DIR + '/experiments')
    assert traces is None


def test_trace_read_ok_4():  # read OK from double entry file
    traces = Trace.read('HC_N_1/double', TESTDATA_DIR + '/experiments')
    assert isinstance(traces, dict)
    assert 'entry1' in traces
    assert isinstance(traces['entry1'], Trace)
    assert traces['entry1'].context['id'] == 'HC_N_1/double/entry1'
    assert 'entry2' in traces
    assert isinstance(traces['entry2'], Trace)
    assert traces['entry2'].context['id'] == 'HC_N_1/double/entry2'


def test_trace_read_ok_5():  # non-existent entry
    trace = Trace.read('HC_N_1/double/nonexistent',
                       TESTDATA_DIR + '/experiments')
    assert trace is None


def test_trace_read_ok_6():  # empty trace file treated as non-existent
    trace = Trace.read('misc/empty/test', TESTDATA_DIR + '/experiments')
    assert trace is None


def test_trace_read_ok_7():  # check KS_R16_E100 trace containing knowledge
    traces = Trace.read('HC/KS_R16_E100/asia', TESTDATA_DIR + '/experiments')
    trace = traces['N10'].trace
    assert trace['activity'] == ['init', 'add', 'add', 'stop']
    assert trace['arc'] == \
        [None, ('xray', 'bronc'), ('smoke', 'tub'), None]
    assert trace['knowledge'] == \
        [None,
         ('reqd_arc', True, 'stop_del', ('either', 'dysp')),
         ('reqd_arc', True, 'stop_del', ('either', 'dysp')),
         ('reqd_arc', True, 'stop_del', ('either', 'dysp'))]


def test_trace_read_ok_8():  # check KW_L16 trace containing knowledge
    traces = Trace.read('HC/KS_L16/sachs', TESTDATA_DIR + '/experiments')
    trace = traces['N10'].trace
    assert trace['activity'] == \
        ['init', 'add', 'add', 'add', 'add', 'add', 'add', 'stop']
    assert trace['arc'] == \
        [None, ('PKA', 'PKC'), ('PKA', 'Raf'), ('PKA', 'P38'), ('PKA', 'Jnk'),
         ('Mek', 'Erk'), ('PKA', 'Mek'), None]
    assert trace['knowledge'] == \
        [None,
         None,
         ('equiv_add', True, 'no_op', ('PKA', 'Raf')),
         ('equiv_add', True, 'swap_best', ('P38', 'PKA')),
         ('equiv_add', True, 'swap_best', ('Jnk', 'PKA')),
         ('equiv_add', True, 'swap_best', ('Erk', 'Mek')),
         ('equiv_add', True, 'swap_best', ('Mek', 'PKA')),
         ('act_cache', True, 'stop_rev', ('PKA', 'Jnk'))]


def test_trace_read_ok_9():  # check KS_L16 trace containing knowledge
    traces = Trace.read('HC/KS_L16/asia', TESTDATA_DIR + '/experiments')
    trace = traces['N10'].trace

    assert trace['activity'] == \
        ['init', 'add', 'add', 'add', 'add', 'none', 'add', 'add', 'reverse',
         'stop']
    assert trace['arc'] == \
        [None, ('bronc', 'dysp'), ('either', 'xray'), ('either', 'lung'),
         ('tub', 'either'), ('smoke', 'tub'), ('xray', 'smoke'),
         ('dysp', 'smoke'), ('either', 'lung'), None]
    assert trace['knowledge'] == \
        [None,
         ('equiv_add', True, 'no_op', ('bronc', 'dysp')),
         ('equiv_add', True, 'no_op', ('either', 'xray')),
         None,
         ('equiv_add', True, 'swap_best', ('either', 'tub')),
         ('equiv_add', True, 'ext_add', ('smoke', 'tub')),
         ('act_cache', True, 'stop_add', ('smoke', 'tub')),
         None,
         None,
         ('act_cache', True, 'stop_rev', ('bronc', 'dysp'))]


def test_trace_save_type_error_1():  # no argument specified
    with pytest.raises(TypeError):
        Trace({'id': 'test'}).save(None)


def test_trace_save_type_error_2():  # bad argument type
    with pytest.raises(TypeError):
        Trace({'id': 'test'}).save(37)
    with pytest.raises(TypeError):
        Trace({'id': 'test'}).save(True)
    with pytest.raises(TypeError):
        Trace({'id': 'test'}).save(-11.2)
    with pytest.raises(TypeError):
        Trace({'id': 'test'}).save([-11.2])
    with pytest.raises(TypeError):
        Trace({'id': 'test'}).save('misc/trace', 32)
    with pytest.raises(TypeError):
        Trace({'id': 'test'}).save('misc/trace', {'name': 'what'})


def test_trace_save_filenotfound_error():  # non-existent root_dir
    with pytest.raises(FileNotFoundError):
        Trace({'id': 'test'}).save('nonexistent')


def test_trace_save_value_error_1():  # id not defined
    with pytest.raises(ValueError):
        Trace().save()


def test_trace_save_value_error_2():  # invalid id
    with pytest.raises(ValueError):
        Trace({'id': 'invalid'}).save()


def test_trace_save_value_error_3():  # binary file
    with pytest.raises(ValueError):
        Trace({'id': 'misc/null.sys/test'}).save(TESTDATA_DIR)


def test_trace_save_value_error_4():  # textual file
    with pytest.raises(ValueError):
        Trace({'id': 'misc/a_1.csv/test'}).save(TESTDATA_DIR)


def test_trace_save_value_error_5():  # textual file
    with pytest.raises(ValueError):
        Trace({'id': 'misc/ab_cb.dsc/test'}).save(TESTDATA_DIR)


def test_trace_save_value_error_6():  # serialised file, but not Trace
    with pytest.raises(ValueError):
        Trace({'id': 'misc/list/test'}).save(TESTDATA_DIR)


def test_trace_save_ok_1():  # save to file at root_dir, 1 entry
    Trace({'id': 'single/entry1'}).save(TESTDATA_DIR + '/tmp')
    assert exists(TESTDATA_DIR + '/tmp/single.pkl.gz')
    traces = Trace.read('single', TESTDATA_DIR + '/tmp')
    assert isinstance(traces, dict)
    assert set(traces.keys()) == {'entry1'}
    assert isinstance(traces['entry1'], Trace)
    assert traces['entry1'].context['id'] == 'single/entry1'
    remove(TESTDATA_DIR + '/tmp/single.pkl.gz')


def test_trace_save_ok_2():  # save to file at root_dir, 2 entries
    Trace({'id': 'double/entry1'}).save(TESTDATA_DIR + '/tmp')
    assert exists(TESTDATA_DIR + '/tmp/double.pkl.gz')
    traces = Trace.read('double', TESTDATA_DIR + '/tmp')
    assert isinstance(traces, dict)
    assert set(traces.keys()) == {'entry1'}
    assert isinstance(traces['entry1'], Trace)
    assert traces['entry1'].context['id'] == 'double/entry1'
    Trace({'id': 'double/entry2'}).save(TESTDATA_DIR + '/tmp')
    assert exists(TESTDATA_DIR + '/tmp/double.pkl.gz')
    traces = Trace.read('double', TESTDATA_DIR + '/tmp')
    assert isinstance(traces, dict)
    assert set(traces.keys()) == {'entry1', 'entry2'}
    assert isinstance(traces['entry1'], Trace)
    assert traces['entry1'].context['id'] == 'double/entry1'
    assert isinstance(traces['entry2'], Trace)
    assert traces['entry2'].context['id'] == 'double/entry2'
    remove(TESTDATA_DIR + '/tmp/double.pkl.gz')


def test_trace_save_ok_3():  # save to file below root_dir, 2 entries
    Trace({'id': 'sub/double/entry1'}).save(TESTDATA_DIR + '/tmp')
    assert exists(TESTDATA_DIR + '/tmp/sub/double.pkl.gz')
    traces = Trace.read('sub/double', TESTDATA_DIR + '/tmp')
    assert isinstance(traces, dict)
    assert set(traces.keys()) == {'entry1'}
    assert isinstance(traces['entry1'], Trace)
    assert traces['entry1'].context['id'] == 'sub/double/entry1'
    Trace({'id': 'sub/double/entry2'}).save(TESTDATA_DIR + '/tmp')
    assert exists(TESTDATA_DIR + '/tmp/sub/double.pkl.gz')
    traces = Trace.read('sub/double', TESTDATA_DIR + '/tmp')
    assert isinstance(traces, dict)
    assert set(traces.keys()) == {'entry1', 'entry2'}
    assert isinstance(traces['entry1'], Trace)
    assert traces['entry1'].context['id'] == 'sub/double/entry1'
    assert isinstance(traces['entry2'], Trace)
    assert traces['entry2'].context['id'] == 'sub/double/entry2'
    remove(TESTDATA_DIR + '/tmp/sub/double.pkl.gz')
    rmdir(TESTDATA_DIR + '/tmp/sub')
