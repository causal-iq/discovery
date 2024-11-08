
# Module test the graphviz module

import pytest
from random import random
from os import remove

from fileio.common import TESTDATA_DIR
from learn.trace import Trace
from core.bn import BN
from analysis.graphviz import traceviz
from analysis.trace import TraceAnalysis
import testdata.example_dags as ex_dag


@pytest.fixture(scope='module')  # directory where tests write files
def dir():
    return TESTDATA_DIR + '/tmp'


@pytest.fixture(scope='module')  # directory where reference files are
def refdir():
    return TESTDATA_DIR + '/experiments/analysis/graphviz/'


@pytest.fixture(scope="function")  # temp file, automatically removed
def filename():
    _filename = '{}'.format(int(random() * 10000000))
    yield _filename
    remove(TESTDATA_DIR + '/tmp/' + _filename + '.png')
    remove(TESTDATA_DIR + '/tmp/' + _filename + '.gv')


@pytest.fixture(scope='module')  # return TraceAnalysis for Cancer N10
def cancer10():
    trace = Trace.read('HC_N_1/cancer', TESTDATA_DIR + '/experiments')['N10']
    ref = ex_dag.cancer()
    return TraceAnalysis(trace, ref)


def cmp(fn1, fn2):  # compare file, different ordering, comments allowed
    with open(fn1, 'r') as f1, open(fn2, 'r') as f2:
        _f1 = {li for li in f1.readlines() if not li.startswith('//')}
        _f2 = {li for li in f2.readlines() if not li.startswith('//')}
        return _f1 == _f2


# traceviz exceptions

def test_traceviz_type_error_1():  # no arguments
    with pytest.raises(TypeError):
        traceviz()


def test_traceviz_type_error_2(dir):  # bad or missing analysis arg
    with pytest.raises(TypeError):
        traceviz(dir=dir)
    with pytest.raises(TypeError):
        traceviz(analysis=None, dir=dir)
    with pytest.raises(TypeError):
        traceviz(analysis=7, dir=dir)


def test_traceviz_type_error_3(cancer10):  # bad or missing dir arg
    with pytest.raises(TypeError):
        traceviz(analysis=cancer10)
    with pytest.raises(TypeError):
        traceviz(cancer10)
    with pytest.raises(TypeError):
        traceviz(cancer10, None)
    with pytest.raises(TypeError):
        traceviz(analysis=cancer10, dir=7)
    with pytest.raises(TypeError):
        traceviz(analysis=cancer10, dir=[''])


def test_traceviz_type_error_4(cancer10, dir):  # bad or missing filename arg
    with pytest.raises(TypeError):
        traceviz(analysis=cancer10, dir=dir, filename=7)
    with pytest.raises(TypeError):
        traceviz(analysis=cancer10, dir=dir, filename=True)
    with pytest.raises(TypeError):
        traceviz(analysis=cancer10, dir=dir, filename=['bad type'])


# successful tracevizs

def test_traceviz_cancer_ok_1(cancer10, dir, filename, refdir):
    traceviz(cancer10, dir, filename)
    assert cmp(refdir + 'HC_N_1_cancer_N10.gv', dir + '/' + filename + '.gv')


def test_traceviz_asia_ok_1(dir, filename, refdir):
    trace = Trace.read('HC_N_1/asia', TESTDATA_DIR + '/experiments')['N40']
    ref = ex_dag.asia()
    analysis = TraceAnalysis(trace, ref)
    traceviz(analysis, dir, filename)
    assert cmp(refdir + 'HC_N_1_asia_N40.gv', dir + '/' + filename + '.gv')


def test_traceviz_sachs_ok_1(dir, filename, refdir):
    trace = Trace.read('HC/STD/sachs', TESTDATA_DIR + '/experiments')['N100']
    ref = BN.read(TESTDATA_DIR + '/discrete/small/sachs.dsc').dag
    analysis = TraceAnalysis(trace, ref)
    traceviz(analysis, dir, filename)
    assert cmp(refdir + 'HC_STD_sachs_N100.gv', dir + '/' + filename + '.gv')


def test_traceviz_sports_ok_1(dir, filename, refdir):
    trace = Trace.read('HC/STD/sports', TESTDATA_DIR + '/experiments')['N1000']
    ref = BN.read(TESTDATA_DIR + '/discrete/small/sports.dsc').dag
    analysis = TraceAnalysis(trace, ref)
    traceviz(analysis, dir, filename)
    assert cmp(refdir + 'HC_STD_sports_N1000.gv', dir + '/' + filename + '.gv')


def test_traceviz_hepar2_ok_1(dir, filename, refdir):
    trace = Trace.read('HC/STD/hepar2', TESTDATA_DIR + '/experiments')['N1000']
    ref = BN.read(TESTDATA_DIR + '/discrete/large/hepar2.dsc').dag
    analysis = TraceAnalysis(trace, ref)
    traceviz(analysis, dir, filename)
    assert cmp(refdir + 'HC_STD_hepar2_N1000.gv', dir + '/' + filename + '.gv')
