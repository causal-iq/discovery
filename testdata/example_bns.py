#
#   Example DAGs for testing and demonstration
#
#   Functions follow a common signature of no arguments to generate a graph
#   and a graph argument to validate that graph e.g. ab() generates the A-->B
#   graph, and ab(graph) validates graph as being A-->B
#

from core.bn import BN
from core.cpt import CPT
from core.lingauss import LinGauss
import testdata.example_dags as dag


# Some simple discrete BNs

def a(check=None):  # single categoical A node
    if check is None:
        return BN(dag.a(), {'A': (CPT, {'0': 0.25, '1': 0.75})})

    assert isinstance(check, BN)
    dag.a(check.dag)
    assert check.cnds == {'A': CPT({'0': 0.25, '1': 0.75})}
    assert check.free_params == 1
    assert check.estimated_pmfs == {}


def ab(check=None):  # A --> B categorical network
    if check is None:
        return BN(dag.ab(), {'A': (CPT, {'0': 0.25, '1': 0.75}),
                             'B': (CPT, [({'A': '0'}, {'0': 0.2, '1': 0.8}),
                                         ({'A': '1'}, {'0': 0.7, '1': 0.3})])})

    assert isinstance(check, BN)
    dag.ab(check.dag)
    assert check.cnds == {'A': CPT({'0': 0.25, '1': 0.75}),
                          'B': CPT([({'A': '0'}, {'0': 0.2, '1': 0.8}),
                                    ({'A': '1'}, {'0': 0.7, '1': 0.3})])}
    assert check.free_params == 3
    assert check.estimated_pmfs == {}


# Some simple continuous BNs

def x(check=None):  # single Gaussian X node
    if check is None:
        return BN(dag.x(),
                  {'X': (LinGauss, {'mean': 0.0,
                                    'sd': 1.0,
                                    'coeffs': {}})})

    assert isinstance(check, BN)
    dag.x(check.dag)
    assert check.cnds == {'X': LinGauss({'mean': 0.0,
                                         'sd': 1.0,
                                         'coeffs': {}})}
    assert check.free_params == 2
    assert check.estimated_pmfs == {}


def xy(check=None):  # X --> Y  Gaussian network
    if check is None:
        return BN(dag.xy(),
                  {'X': (LinGauss, {'mean': 2.0,
                                    'sd': 1.0,
                                    'coeffs': {}}),
                   'Y': (LinGauss, {'mean': 0.5,
                                    'sd': 0.5,
                                    'coeffs': {'X': 1.5}})})

    assert isinstance(check, BN)
    dag.xy(check.dag)
    assert check.cnds == {'X': LinGauss({'mean': 2.0,
                                         'sd': 1.0,
                                         'coeffs': {}}),
                          'Y': LinGauss({'mean': 0.5,
                                         'sd': 0.5,
                                         'coeffs': {'X': 1.5}})}
    assert check.free_params == 5
    assert check.estimated_pmfs == {}


def x_y(check=None):  # X  Y Gaussian network
    if check is None:
        return BN(dag.x_y(),
                  {'X': (LinGauss, {'mean': 0.2,
                                    'sd': 0.1,
                                    'coeffs': {}}),
                   'Y': (LinGauss, {'mean': -4.0,
                                    'sd': 2.0,
                                    'coeffs': {}})})

    assert isinstance(check, BN)
    dag.x_y(check.dag)
    assert check.cnds == {'X': LinGauss({'mean': 0.2,
                                         'sd': 0.1,
                                         'coeffs': {}}),
                          'Y': LinGauss({'mean': -4.0,
                                         'sd': 2.0,
                                         'coeffs': {}})}
    assert check.free_params == 4
    assert check.estimated_pmfs == {}


def xyz(check=None):  # X --> Y  --> Z Gaussian network
    if check is None:
        return BN(dag.xyz(),
                  {'X': (LinGauss, {'mean': 0.0,
                                    'sd': 5.0,
                                    'coeffs': {}}),
                   'Y': (LinGauss, {'mean': -0.7,
                                    'sd': 0.5,
                                    'coeffs': {'X': -1.2}}),
                   'Z': (LinGauss, {'mean': 0.03,
                                    'sd': 0.05,
                                    'coeffs': {'Y': 0.3}})})

    assert isinstance(check, BN)
    dag.xyz(check.dag)
    assert check.cnds == {'X': LinGauss({'mean': 0.0,
                                         'sd': 5.0,
                                         'coeffs': {}}),
                          'Y': LinGauss({'mean': -0.7,
                                         'sd': 0.5,
                                         'coeffs': {'X': -1.2}}),
                          'Z': LinGauss({'mean': 0.03,
                                         'sd': 0.05,
                                         'coeffs': {'Y': 0.3}})}
    assert check.free_params == 8
    assert check.estimated_pmfs == {}


def xy_zy(check=None):  # X --> Y  <-- Z Gaussian network
    if check is None:
        return BN(dag.xy_zy(),
                  {'X': (LinGauss, {'mean': -3.07,
                                    'sd': 0.45,
                                    'coeffs': {}}),
                   'Y': (LinGauss, {'mean': -2.7,
                                    'sd': 1.5,
                                    'coeffs': {'X': 1.2, 'Z': -0.4}}),
                   'Z': (LinGauss, {'mean': 6.2,
                                    'sd': 1.4,
                                    'coeffs': {}})})

    assert isinstance(check, BN)
    dag.xy_zy(check.dag)
    assert check.cnds == {'X': LinGauss({'mean': -3.07,
                                         'sd': 0.45,
                                         'coeffs': {}}),
                          'Y': LinGauss({'mean': -2.7,
                                         'sd': 1.5,
                                         'coeffs': {'X': 1.2, 'Z': -0.4}}),
                          'Z': LinGauss({'mean': 6.2,
                                         'sd': 1.4,
                                         'coeffs': {}})}
    assert check.free_params == 8
    assert check.estimated_pmfs == {}
