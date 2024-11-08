
# Linear Guassian implementation of Conditional Node Distribution

from sklearn.linear_model import LinearRegression
from math import sqrt
from numpy import sum as npsum

from core.common import random_generator, rndsf
from core.metrics import values_same, dicts_same
from fileio.pandas import Pandas


class LinGauss():

    MAX_SF = 10  # max no of significant

    _model = None

    """
        Conditional Linear Gaussian Distribution

        :param dict lg: specification of Linear Guassian in following form:
                        {'coeffs': {node: coeff}, 'mean': mean, 'sd': sd}

        :ivar dict coeffs: linear coeffeicient of parents {parent: coeff}
        :ivar float mean: mean of Gaussian noise (aka intercept, mu)
        :ivar float sd: S.D. of Gaussian noise (aka sigma)

        :raises TypeError: if called with bad arg types
        :raises ValueError: if called with bad arg values
    """
    def __init__(self, lg):

        if (not isinstance(lg, dict)
            or set(lg) != {'coeffs', 'mean', 'sd'}
            or not isinstance(lg['coeffs'], dict)
            or not all([isinstance(c, str) and isinstance(v, float)
                        for c, v in lg['coeffs'].items()])
            or not isinstance(lg['mean'], float)
                or not isinstance(lg['sd'], float)):
            raise TypeError('LinGauss() bad arg types')

        if lg['sd'] < 0.0:
            raise ValueError('LinGauss() bad arg value')

        self.coeffs = lg['coeffs']
        self.mean = lg['mean']
        self.sd = lg['sd']
        self.has_parents = True if len(self.coeffs) > 0 else False
        self.free_params = 2 + len(self.coeffs)

    @classmethod
    def fit(self, node, parents, data, autocomplete=True):
        """
            Fit a Linear Gaussian to data.

            :param str node: node that Linear Gaussian applies to
            :param tuple/None parents: parents of node
            :param Data data: data to fit Linear Gaussian to
            :param bool autocomplete: not used for Linear Gaussian

            :raises TypeError: with bad arg types
            :raises ValueError: with bad arg values

            :returns tuple: (lg_spec, None) where
                             lg is (LinGauss class, lg_spec)
        """
        if (not isinstance(node, str)
            or (parents is not None
                and (not isinstance(parents, tuple) or len(parents) == 0
                     or not all([isinstance(p, str) for p in parents])))
            or not isinstance(data, Pandas)
                or autocomplete is not True):
            raise TypeError('LinGauss.fit() bad arg type')

        if parents is None:

            # Just need to determine mean and sd for univariate Gaussian

            values = data.values((node, ))
            lg = {'mean': values.mean().item(), 'sd': values.std().item(),
                  'coeffs': {}}

        else:

            # Get values for child and its parents and fit a linear regression
            # model for the parents predicting the child value

            values = data.values(tuple([node] + list(parents)))
            if LinGauss._model is None:
                LinGauss._model = LinearRegression()
            LinGauss._model.fit(values[:, 1:], values[:, 0])

            # Parent coefficientsare the linear regression coefficents and
            # the regression intercept is the mean of the child Gaussian

            coeffs = {p: LinGauss._model.coef_[i].item()
                      for i, p in enumerate(parents)}
            mean = LinGauss._model.intercept_.item()

            # Use model to predict child values, calculate residuals and
            # hence noise S.D.

            residuals = values[:, 0] - LinGauss._model.predict(values[:, 1:])
            sd = sqrt(npsum(residuals ** 2) / len(residuals))

            lg = {'mean': mean, 'sd': sd, 'coeffs': coeffs}

        return ((LinGauss, lg), None)

    def cdist(self, parental_values=None):
        """
            Return conditional distribution for specified parental values.

            :param dict/None parental_values: parental values for which dist.
                                              required for non-orphans

            :raises TypeError: if args are of wrong type
            :raises ValueError: if args have invalid or conflicting values

            :return tuple: (mean, sd) of child Gaussian distribution
        """
        if ((self.coeffs == {} and parental_values is not None)
            or (len(self.coeffs) > 0 and parental_values is None)
            or (len(self.coeffs) > 0 and
                set(self.coeffs) != set(parental_values))):
            raise TypeError('lingauss.cpt() coeffs/parent values mismatch')

        mean = self.mean + sum([parental_values[p] * self.coeffs[p]
                                for p in self.coeffs])
        return mean, self.sd

    def random_value(self, pvs):
        """
            Generate a random value for a node given the value of its parents.

            :param dict/None pvs: parental values, {parent1: value1, ...}

            :return str/float: random value for node.
        """
        mean, sd = self.cdist(pvs)
        return mean + random_generator().normal() * sd

    def parents(self):
        """
            Return parents of node CND relates to

            :returns list: parent node names in alphabetical order
        """
        pass

    def to_spec(self, name_map):
        """
            Returns external specification format of LinGauss, renaming nodes
            according to a name map.

            :param dict name_map: map of node names {old: new}

            :raise TypeError: if bad arg type
            :raise ValueError: if bad arg value, e.g. coeff keys not in map

            :returns dict: LinGauss specification with renamed nodes
        """
        if (not isinstance(name_map, dict)
                or not all([isinstance(k, str) for k in name_map])
                or not all([isinstance(v, str) for v in name_map.values()])):
            raise TypeError('LinGauss.to_spec() bad arg type')

        if len(set(self.coeffs) - set(name_map)) != 0:
            raise ValueError('LinGauss.to_spec() bad arg value')

        coeffs = {name_map[n]: v for n, v in self.coeffs.items()}
        return {'coeffs': coeffs, 'mean': self.mean, 'sd': self.sd}

    def __str__(self):
        """
            Human-friendly formula description of the Linear Guassian
        """
        def _term(node, coeff):
            # val = _val(coeff)
            val = rndsf(coeff, self.MAX_SF)
            return ('' if val == '0.0' else ('{}*{}'.format(val, node)
                    if coeff < 0 else '+{}*{}'.format(val, node)))

        terms = ''.join([_term(n, self.coeffs[n])
                         for n in sorted(self.coeffs)])
        terms = terms[1:] if len(terms) > 0 and terms[0] == '+' else terms
        normal = 'Normal({},{})'.format(rndsf(self.mean, self.MAX_SF),
                                        rndsf(self.sd, self.MAX_SF))
        return '{}{}'.format(terms + '+' if len(terms) else '', normal)

    def __eq__(self, other):
        """
            Return whether two CNDs are the same allowing for probability
            rounding errors

            :param CND other: CND to compared to self

            :returns bool: whether LinGauss objects are the same up to 10 sf
        """
        return (isinstance(other, LinGauss)
                and values_same(self.mean, other.mean, sf=10)
                and values_same(self.sd, other.sd, sf=10)
                and set(self.coeffs) == set(other.coeffs)
                and dicts_same(self.coeffs, other.coeffs, sf=10))

    def validate_parents(self, node, parents, node_values):
        """
            Check LinGauss coeff keys consistent with parents in DAG.

            :param str node: name of node
            :param dict parents: parents of all nodes defined in DAG
            :param dict node_values: values of each cat. node [UNUSED]
        """
        if ((node not in parents and len(self.coeffs) > 0)
                or (node in parents
                    and set(parents[node]) != set(self.coeffs))):
            raise ValueError('LinGauss.validate_parents() parent mismatch')
