
import pytest
from pandas import DataFrame

from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas
from core.graph import DAG
from core.bn import BN
from core.metrics import values_same, dicts_same
import testdata.example_dags as dag


@pytest.fixture(scope="function")  # temp file, automatically removed
def bn():
    # Creates DAG N1->N2 with 3 (1,1), 2 (1,2), 3 (2, 1) and 4 (2,2) cases
    dag = DAG(['N1', 'N2'], [('N1', '->', 'N2')])
    data = Pandas.read(TESTDATA_DIR + '/simple/heckerman.csv',
                       dstype='categorical')
    return BN.fit(dag, data)


def test_bn_prob_lnprob_case_type_error(bn):  # bad arg types
    with pytest.raises(TypeError):
        bn.lnprob_case()
    with pytest.raises(TypeError):
        bn.lnprob_case(77)
    with pytest.raises(TypeError):
        bn.lnprob_case([])
    with pytest.raises(TypeError):
        bn.lnprob_case(True)
    with pytest.raises(TypeError):
        bn.lnprob_case({}, 76.1)


def test_bn_prob_lnprob_case_value_error1(bn):  # bad arg values
    with pytest.raises(ValueError):
        bn.lnprob_case({})
    with pytest.raises(ValueError):
        bn.lnprob_case({'N1': '1', 'N2': '2'}, 6)


def test_bn_prob_lnprob_case_value_error2(bn):  # bad node name in case
    with pytest.raises(ValueError):
        bn.lnprob_case({'N3': '1', 'N2': '1'})


def test_bn_prob_lnprob_case_value_error3(bn):  # bad value in case
    with pytest.raises(ValueError):
        bn.lnprob_case({'N1': 'bad value', 'N2': '1'})


def test_bn_prob_lnprob_case_heckerman(bn):  # correct values for N=12 dataset
    assert values_same(10 ** (bn.lnprob_case({'N1': '1', 'N2': '1'})),
                       0.25, sf=10) is True
    assert values_same(10 ** (bn.lnprob_case({'N1': '1', 'N2': '2'})),
                       0.1666666667, sf=10) is True
    assert values_same(10 ** (bn.lnprob_case({'N1': '2', 'N2': '1'})),
                       0.25, sf=10) is True
    assert values_same(10 ** (bn.lnprob_case({'N1': '2', 'N2': '2'})),
                       0.3333333333, sf=10) is True


def test_bn_prob_marginal_distribution_type_error1(bn):  # bad arg types
    with pytest.raises(TypeError):
        bn.marginal_distribution()
    with pytest.raises(TypeError):
        bn.marginal_distribution(12)
    with pytest.raises(TypeError):
        bn.marginal_distribution(3.8)
    with pytest.raises(TypeError):
        bn.marginal_distribution('N1', 'N2')
    with pytest.raises(TypeError):
        bn.marginal_distribution('N1', [3])
    with pytest.raises(TypeError):
        bn.marginal_distribution('N1', [3, 'N2'])


def test_bn_prob_marginal_distribution_value_error1(bn):  # bad node value
    with pytest.raises(ValueError):
        bn.marginal_distribution('N3')


def test_bn_prob_marginal_distribution_value_error2(bn):  # bad parents value
    with pytest.raises(ValueError):
        bn.marginal_distribution('N1', ['N1', 'N2'])
    with pytest.raises(ValueError):
        bn.marginal_distribution('N1', ['N2', 'N3'])
    with pytest.raises(ValueError):
        bn.marginal_distribution('N1', ['N2', 'N2'])


# TODO: check shape and values of marginal distributions, comparing with
# global distribution sometimes

def test_bn_prob_marginal_distribution_heckerman1(bn):

    p = bn.marginal_distribution('N1')

    # check distribution has correct shape and values

    assert len(p.columns) == 1  # 1 column of N1 values
    assert len(p.index) == 2  # for 2 values of N1
    assert values_same(p.loc['1'][''], 5/12, sf=10)  # N1=1
    assert values_same(p.loc['2'][''], 7/12, sf=10)  # N1=2
    print(p)


def test_bn_prob_marginal_distribution_heckerman2(bn):

    p = bn.marginal_distribution('N2')

    # check distribution has correct shape and values

    assert len(p.columns) == 1  # 1 column of N2 values
    assert len(p.index) == 2  # 2 values of N2
    assert values_same(p.loc['1'][''], 6/12, sf=10)  # N2=1
    assert values_same(p.loc['2'][''], 6/12, sf=10)  # N2=2


def test_bn_prob_marginal_distribution_heckerman3(bn):

    p = bn.marginal_distribution('N2', ['N1'])

    # check distribution has correct shape and values

    assert len(p.columns) == 2  # 2 columns for N1 values
    assert len(p.index) == 2  # 2 rows for N2 values
    assert values_same(float(p.loc['1', '1'].iloc[0]), 3/12, sf=10)  # 1, 1
    assert values_same(float(p.loc['1', '2'].iloc[0]), 3/12, sf=10)  # 2, 1
    assert values_same(float(p.loc['2', '1'].iloc[0]), 2/12, sf=10)  # 1, 2
    assert values_same(float(p.loc['2', '2'].iloc[0]), 4/12, sf=10)  # 2, 2


def test_bn_prob_marginal_distribution_heckerman4(bn):

    p = bn.marginal_distribution('N1', ['N2'])

    # check distribution has correct shape and values

    assert len(p.columns) == 2  # 2 columns for N2 values: 1, 2
    assert len(p.index) == 2  # 2 rows for N1 values
    assert values_same(float(p.loc['1', '1'].iloc[0]), 3/12, sf=10)  # 1, 1
    assert values_same(float(p.loc['1', '2'].iloc[0]), 2/12, sf=10)  # 1, 2
    assert values_same(float(p.loc['2', '1'].iloc[0]), 3/12, sf=10)  # 2, 1
    assert values_same(float(p.loc['2', '2'].iloc[0]), 4/12, sf=10)  # 2, 2


def test_bn_prob_global_distribution_heckerman_demo(bn):

    global_dist = bn.global_distribution()

    # check distribution has correct shape

    assert len(global_dist.columns) == len(bn.dag.nodes) + 1  # # nodes + prob
    assert len(global_dist.index) == 4  # 4 combinations of values

    # check individual entries are correct

    assert dict(global_dist.iloc[0, 0:2]) == {'N1': '2', 'N2': '2'}
    assert values_same(global_dist.iloc[0][''], 4/12, sf=10)
    assert dict(global_dist.iloc[1, 0:2]) == {'N1': '1', 'N2': '1'}
    assert values_same(global_dist.iloc[1][''], 3/12, sf=10)
    assert dict(global_dist.iloc[2, 0:2]) == {'N1': '2', 'N2': '1'}
    assert values_same(global_dist.iloc[2][''], 3/12, sf=10)
    assert dict(global_dist.iloc[3, 0:2]) == {'N1': '1', 'N2': '2'}
    assert values_same(global_dist.iloc[3][''], 2/12, sf=10)

    # check probabilities sum to 1

    assert values_same(global_dist[''].sum(), 1.0, sf=10)

    # print out distribution nicely

    print('\nGlobal prob. distribution for Heckerman 12 instance dataset:')
    print(global_dist.to_string(index=False))
    print('Local distributions:')
    for n in bn.dag.nodes:
        print('{}: {}'.format(n, {v: '{:.6f}'.format(p) for v, p in
                                  global_dist.groupby(n)[''].sum().items()}))


def test_bn_prob_marginal_distribution_ac_bc():
    data = {'A': ['0', '0', '1', '1', '2', '2', '2', '3', '3', '3'],
            'B': ['0', '1', '1', '2', '0', '1', '1', '3', '3', '3'],
            'C': ['0', '1', '1', '1', '0', '1', '1', '2', '2', '1']}
    data = Pandas(df=DataFrame(data, dtype='category'))
    bn = BN.fit(dag.ac_bc(), data)

    # Only 7 combinations of the 16 possible parental combinations for C
    # are present so 9 pmfs for C should have been estimated

    assert dicts_same(bn.estimated_pmfs, {'C': 9})

    # Marginal distributions for A and B just match the data as these are
    # parentless nodes

    assert dicts_same(dict(bn.marginal_distribution('A')['']),
                      {'0': 0.2, '1': 0.2, '2': 0.3, '3': 0.3})
    assert dicts_same(dict(bn.marginal_distribution('B')['']),
                      {'0': 0.2, '1': 0.4, '2': 0.1, '3': 0.3})

    # Marginal distribution for C doesn't match data as it involves summing
    # over parental combinations, some of which had estimated pmfs. Instead
    # need to compare it with distribution obtained directly from global
    # distribution

    gd = bn.global_distribution()
    assert dicts_same(dict(bn.marginal_distribution('C')['']),
                      dict(gd.groupby('C')[''].sum()))

    # Compare output from marginal_distribution for node C with that
    # obtained directly from global distribution

    p = bn.marginal_distribution('C', ['A', 'B'])
    for pvs in p:

        # obtain part of global distribution that pertains to this combination
        # of parental values, and convert to a pmf over C values

        from_gd = gd.loc[(gd['A'] == pvs[0]) & (gd['B'] == pvs[1]),
                         ['C', '']].set_index('C')[''].to_dict()
        assert dicts_same(dict(p[pvs[0]][pvs[1]]), from_gd)


def test_bn_prob_global_distribution_cancer_demo():
    bn = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')

    global_dist = bn.global_distribution()

    # check distribution has correct shape

    assert len(global_dist.columns) == len(bn.dag.nodes) + 1
    assert len(global_dist.index) == 32

    # most probable case

    assert dict(global_dist.iloc[0, 0:-1]) == \
        {'Cancer': 'False', 'Dyspnoea': 'False', 'Pollution': 'low',
         'Smoker': 'False', 'Xray': 'negative'}
    assert values_same(global_dist.iloc[0][''], 0.3524472, sf=10)

    # least probable case

    assert dict(global_dist.iloc[-1, 0:-1]) == \
        {'Cancer': 'True', 'Dyspnoea': 'False', 'Pollution': 'low',
         'Smoker': 'False', 'Xray': 'negative'}
    assert values_same(global_dist.iloc[-1][''], 2.205e-05, sf=10)

    # probabilities should sum to 1

    assert values_same(global_dist[''].sum(), 1.0, sf=10)

    print('\nSorted global log probability distribution for Cancer BN:')
    # print(global_dist.to_string(index=False))
    print('Local distributions:')
    for n in bn.dag.nodes:
        print('{}: {}'.format(n, {v: '{:.6f}'.format(p) for v, p in
                                  global_dist.groupby(n)[''].sum().items()}))


def test_bn_prob_marginal_distribution_common_effect():
    dag = DAG(['Cause1', 'Cause2', 'CommonEffect'],
              [('Cause1', '->', 'CommonEffect'),
               ('Cause2', '->', 'CommonEffect')])
    data = Pandas.read(TESTDATA_DIR + '/simple/common_effect_1k.csv',
                       dstype='categorical')
    bn = BN.fit(dag, data)

    # Check CPTs have correct values (the ones used to construct data)

    assert dicts_same(bn.cnds['Cause1'].cdist(), {'no': 0.8, 'yes': 0.2},
                      sf=10)
    assert dicts_same(bn.cnds['Cause2'].cdist(), {'no': 0.6, 'yes': 0.4},
                      sf=10)
    cpt = bn.cnds['CommonEffect']
    assert dicts_same(cpt.cdist({'Cause1': 'no', 'Cause2': 'no'}),
                      {'no': 0.7, 'yes': 0.3}, sf=10)
    assert dicts_same(cpt.cdist({'Cause1': 'no', 'Cause2': 'yes'}),
                      {'no': 0.5, 'yes': 0.5}, sf=10)
    assert dicts_same(cpt.cdist({'Cause1': 'yes', 'Cause2': 'no'}),
                      {'no': 0.4, 'yes': 0.6}, sf=10)
    assert dicts_same(cpt.cdist({'Cause1': 'yes', 'Cause2': 'yes'}),
                      {'no': 0.6, 'yes': 0.4}, sf=10)

    # these marginals should just be same as the parentless CPT

    p = bn.marginal_distribution('Cause1')
    assert dicts_same(dict(p['']), {'no': 0.8, 'yes': 0.2}, sf=10)
    p = bn.marginal_distribution('Cause2')
    assert dicts_same(dict(p['']), {'no': 0.6, 'yes': 0.4}, sf=10)

    # this marginal should show independence

    p = bn.marginal_distribution('Cause1', ['Cause2'])
    assert dicts_same(dict(p[tuple(['no'])]), {'no': 0.48, 'yes': 0.12})
    assert dicts_same(dict(p[tuple(['yes'])]), {'no': 0.32, 'yes': 0.08})
    assert values_same(p.sum().sum(), 1, sf=10)

    p = bn.marginal_distribution('CommonEffect', ['Cause1', 'Cause2'])
    assert dicts_same(dict(p[('no', 'no')]), {'no': 0.336, 'yes': 0.144})
    assert dicts_same(dict(p[('no', 'yes')]), {'no': 0.16, 'yes': 0.16}, sf=10)
    assert dicts_same(dict(p[('yes', 'no')]), {'no': 0.048, 'yes': 0.072})
    assert dicts_same(dict(p[('yes', 'yes')]), {'no': 0.048, 'yes': 0.032})


def test_bn_prob_marginal_distribution_cancer():
    bn = BN.read(TESTDATA_DIR + '/cancer/cancer.dsc')

    # generate 20,000 cases for cancer BN, then fit bn to this data,
    # and generate global probability distribution for this data

    data = Pandas(df=bn.generate_cases(20000))
    bn = BN.fit(bn.dag, data)
    gd = bn.global_distribution()
    print('\nGlobal distribution for Cancer BN from 20,000 cases:\n{}'
          .format(gd))

    # Loop over each node - we are going to get marginals at that node
    # for all combinations of all the other nodes. We will check that
    # the PMF returned from the marginal_distribution() function matches
    # that obtained directly from the global distribution

    for node in bn.dag.nodes:
        others = list(bn.dag.nodes)
        others.remove(node)
        md = bn.marginal_distribution(node, others)
        for combo in md:

            # construct query to extract marginals from global distribution

            query = ' & '.join(['{} == "{}"'.format(p[0], p[1])
                               for p in zip(others, combo)])
            gd_margs = dict(gd.query(query)[[node, '']].set_index(node)[''])
            assert dicts_same(dict(md[combo]), gd_margs)


def test_bn_prob_bn_score_heckerman(bn):
    bn.score(12, 'bic')
