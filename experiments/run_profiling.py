
#   Rund code profiling

from pprofile import Profile

from fileio.common import EXPTS_DIR, TESTDATA_DIR
from core.bn import BN
from learn.hc import hc


def run_profile_asia():
    dsc = EXPTS_DIR + '/bn/asia.dsc'
    context = {'in': dsc, 'id': 'tabu/asia/profiling'}
    bn = BN.read(dsc)
    data = bn.generate_cases(100)
    profiler = Profile()
    with profiler:
        dag, trace = hc(data, params={'tabu': 10}, context=context)
    profiler.dump_stats(TESTDATA_DIR + '/profiling/asia.data')
    print('\n{} learning gives:\n{}'.format(trace, dag))


def run_profile_child():
    dsc = EXPTS_DIR + '/bn/child.dsc'
    context = {'in': dsc, 'id': 'hc/child/profiling'}
    bn = BN.read(dsc)
    data = bn.generate_cases(1000)
    # profiler = Profile()
    # with profiler:
    dag, trace = hc(data, context=context)
    # profiler.dump_stats(TESTDATA_DIR + '/profiling/child.data')
    print('\n{} learning gives:\n{}'.format(trace, dag))


def run_profile_diarrhoea():
    dsc = EXPTS_DIR + '/bn/diarrhoea.dsc'
    context = {'in': dsc, 'id': 'tabu/diarrhoea/profiling'}
    bn = BN.read(dsc)
    data = bn.generate_cases(1000000)
    # profiler = Profile()
    # with profiler:
    dag, trace = hc(data, context=context)
    # profiler.dump_stats(TESTDATA_DIR + '/profiling/diarrhoea.data')
    print('\n{} learning gives:\n{}'.format(trace, dag))
