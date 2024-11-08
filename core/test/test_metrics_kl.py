
from pandas import Series
from numpy import NaN
import pytest

from core.bn import BN
from core.metrics import kl, values_same
from fileio.common import TESTDATA_DIR
from fileio.pandas import Pandas


def test_kl_type_error():  # bad argument types
    with pytest.raises(TypeError):
        kl()
    with pytest.raises(TypeError):
        kl('bad arg')
    with pytest.raises(TypeError):
        kl('bad arg', 'bad arg')
    dist = Series(data={'a': 0.2, 'b': 0.8})
    with pytest.raises(TypeError):
        kl(3.2, dist)
    with pytest.raises(TypeError):
        kl(dist, True)


def test_kl_value_error1():  # inconsistent indices
    dist1 = Series(data={'a': 0.2, 'b': 0.8})
    dist2 = Series(data={'a': 0.2, 'c': 0.8})
    with pytest.raises(ValueError):
        kl(dist1, dist2)
    dist3 = Series(data={'a': 0.2, 'b': 0.7, 'c': 0.1})
    with pytest.raises(ValueError):
        kl(dist1, dist3)
    with pytest.raises(ValueError):
        kl(dist3, dist1)


def test_kl_value_error2():  # NaN numbers
    dist1 = Series(data={'a': 0.2, 'b': NaN})
    dist2 = Series(data={'a': 0.2, 'b': 0.8})
    with pytest.raises(ValueError):
        kl(dist1, dist2)
    with pytest.raises(ValueError):
        kl(dist2, dist1)
    dist3 = Series(data={'a': NaN, 'b': NaN})
    with pytest.raises(ValueError):
        kl(dist1, dist3)
    with pytest.raises(ValueError):
        kl(dist3, dist1)


def test_kl_value_error3():  # bad values
    dist1 = Series(data={'a': 0.2, 'b': 1.1})
    dist2 = Series(data={'a': 0.2, 'b': 0.8})
    dist3 = Series(data={'a': -0.1, 'b': 0.8})
    with pytest.raises(ValueError):
        kl(dist1, dist2)
    with pytest.raises(ValueError):
        kl(dist2, dist3)


def test_kl_value_ok1():  # check can cope with one dist have 0 prob
    dist1 = Series(data={'a': 0.2, 'b': 0.8})
    dist2 = Series(data={'a': 0.0, 'b': 1.0})
    assert values_same(kl(dist1, dist2), 6.867869874)


def test_kl_value_ok2():  # check can cope with both dists have 0 prob
    dist1 = Series(data={'a': 1.0, 'b': 0.0})
    dist2 = Series(data={'a': 0.0, 'b': 1.0})
    assert values_same(kl(dist1, dist2), 36.84136149)


def test_metrics_kl_wiki_ok():  # Kullback-Leibler_divergence Wikipedia
    p = Series(data={'0': 0.36, '1': 0.48, '2': 0.16})
    q = Series(data={'0': 1 / 3, '1': 1 / 3, '2': 1 / 3})

    # Check get KL values reported in Wikipedia article

    assert values_same(kl(p, q), 0.0852996, sf=6)
    assert values_same(kl(q, p), 0.097455, sf=5)

    # Check KL of distribution with itself is zero

    assert values_same(kl(p, p), 0)
    assert values_same(kl(q, q), 0)


def test_metrics_kl_ab_ok():  # KL of sample from true in A --> B
    ab = BN.read(TESTDATA_DIR + '/dsc/ab.dsc')  # get A-->B BN

    limit = ab.global_distribution()  # theoretical distribution
    limit = limit.set_index(ab.dag.nodes).squeeze()  # convert to Series

    print('\nKL divergence of sample from true distribution: A --> B')
    for N in [50, 100, 200, 500, 1000, 2000, 5000]:
        data = Pandas(df=ab.generate_cases(N))
        dist = data.sample.value_counts().divide(N)  # df to count series
        print("KL is {:.3E} at sample size {}".format(kl(dist, limit), N))

        fit = BN.fit(ab.dag, data)  # re-fit BN so CPTs match data
        dist2 = fit.global_distribution()
        dist2 = dist2.set_index(ab.dag.nodes).squeeze()
        assert kl(dist, dist2) < 1E-10


def test_metrics_kl_abc_ok():  # KL of sample from true in A --> B --> C
    abc = BN.read(TESTDATA_DIR + '/dsc/abc.dsc')  # get A-->B-->C BN

    limit = abc.global_distribution()  # theoretical distribution
    limit = limit.set_index(abc.dag.nodes).squeeze()  # convert to Series

    print('\nKL divergence of sample from true distribution: A --> B --> C')
    for N in [50, 100, 200, 500, 1000, 2000, 5000]:
        # for N in [10000]:
        data = abc.generate_cases(N)  # generate data for N cases
        dist = data.value_counts().divide(N)  # dataframe to count series
        print("KL is {:.3E} at sample size {}".format(kl(dist, limit), N))


def test_metrics_kl_ab_cb_ok():  # KL of sample from true in A --> B <-- C
    ab_cb = BN.read(TESTDATA_DIR + '/dsc/ab_cb.dsc')  # get A-->B<--C BN

    limit = ab_cb.global_distribution()  # theoretical distribution
    limit = limit.set_index(ab_cb.dag.nodes).squeeze()  # convert to Series

    print('\nKL divergence of sample from true distribution: A --> B <-- C')
    for N in [200, 500, 1000, 2000, 5000]:
        data = ab_cb.generate_cases(N)  # generate data for N cases
        dist = data.value_counts().divide(N)  # dataframe to count series
        print("KL is {:.3E} at sample size {}".format(kl(dist, limit), N))
