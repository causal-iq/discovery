
# Data concrete implementation with data held in NumPy arrays

from numpy import array, ndarray, bincount, unique as npunique, zeros, \
                  prod as npprod, nonzero, empty
from numpy.random import default_rng
from pandas import read_csv, factorize, DataFrame, Categorical
from pandas.errors import EmptyDataError
from gzip import BadGzipFile

from core.timing import Timing
from fileio.common import DatasetType, is_valid_path, FileFormatError
from fileio.pandas import Pandas
from fileio.data import Data

MAX_CATEGORY = 100  # maximum number of different values in category


class NumPy(Data):
    """
        Concrete Data subclass which holds data in NumPy arrays

        :param dict data: data provided as a 2-D NumPy array
        :param DatasetType dstype: type of variables in dataset
        :param dict col_values: column names, and its categorical values
                                {node: (val1, val2, ...), ....}

        :ivar ndarray data: the original data values
        :ivar ndarray sample: sample values of size N, rows possibly reordered

        :ivar tuple nodes: internal (i.e. original) node names
        :ivar categories: categories for each categorical node:
                          (ndarray['c1', 'c2', ...], ...)
        :ivar tuple order: order in which nodes should be processed
        :ivar dict ext_to_orig: map from external to original names
        :ivar dict orig_to_ext: map from original to external names
        :ivar int N: current sample size being used by the algorithm
        :ivar dict node_types: node types {n1: t1, n2: ....}
        :ivar DatasetType dstype: type of dataset (categorical/numeric/mixed)
        :ivar dict node_values: values and their counts for categorical nodes
                                in sample {n1: {v1: c1, v2: ...}, n2 ...}

        :raises TypeError: if bad arg type
        :raises ValueError: if bad arg value
    """

    MAX_BINCOUNT = 1000000

    def __init__(self, data, dstype, col_values):

        if (not isinstance(data, ndarray) or len(data.shape) != 2
            or not isinstance(dstype, (DatasetType, str))
            or dstype not in {v for v in DatasetType}
            or not isinstance(col_values, dict)
            or not all([isinstance(k, str) for k in col_values])
            or (dstype == 'categorical' and
                (not all([isinstance(t, tuple)
                          for t in col_values.values()]) or
                 not all([isinstance(s, str)
                          for t in col_values.values() for s in t]))
            or (dstype == 'continuous' and
                not all([v is None for v in col_values.values()])))):
            raise TypeError('NumPy() bad arg type')

        if (data.shape[0] < 2 or data.shape[1] < 2
                or data.shape[1] != len(col_values)
                or dstype == 'categorical' and data.dtype != 'uint8'
                or dstype == 'continuous' and data.dtype != 'float32'):
            raise ValueError('NumPy bad arg values')

        self.data = data
        self.nodes = tuple(col_values)

        node_type = 'category' if dstype == 'categorical' else 'float32'
        self.node_types = {n: node_type for n in self.nodes}

        self.categories = (array([col_values[n] for n in self.nodes]) if
                           dstype == 'categorical' else None)
        self.order = tuple(i for i in range(len(self.nodes)))
        self.ext_to_orig = {n: n for n in self.nodes}
        self.orig_to_ext = {n: n for n in self.nodes}
        self.dstype = dstype if isinstance(dstype, str) else dstype.value

        # set N, sample and categorical node_values and counts for that N

        self.set_N(N=data.shape[0])

    @classmethod
    def read(self, filename, dstype, N=None):
        """
            Read a file into a NumPy object.

            :param str filename: full path of data file
            :param DatasetType/str dstype: type of dataset
            :param int/None N: number of rows to read

            :raises TypeError: if argument types incorrect
            :raises ValueError: if illegal values in args or file
            :raises FileNotFoundError: if file does not exist
            :raises FileFormatError: if format of file incorrect

            :returns NumPy: data contained in file
        """
        if (not isinstance(filename, str)
            or (N is not None
                and (not isinstance(N, int) or isinstance(N, bool)))
            or (not isinstance(dstype, (DatasetType, str))
                or dstype not in {v for v in DatasetType})):
            raise TypeError('Bad argument types for data.read')
        if (N is not None and N < 2):
            raise ValueError('Bad argument values for data.read')

        is_valid_path(filename)

        if dstype == 'mixed':
            raise ValueError('Mixed datasets not supported')

        try:

            # Read from file treating as floats/strings according to dstype

            nrows = {} if N is None else {'nrows': N}
            dtype = 'float32' if dstype == 'continuous' else 'category'
            df = read_csv(filename, sep=',', header=0, encoding='utf-8',
                          keep_default_na=False, na_values='<NA>',
                          dtype=dtype, **nrows)

            if N is not None and N > len(df):
                raise ValueError('Bad argument values for Pandas.read')

        except (UnicodeDecodeError, PermissionError, EmptyDataError,
                BadGzipFile) as e:
            raise FileFormatError('File format error: {}'.format(e))

        print('\n\n{}\n'.format(df.tail()))

        return NumPy.from_df(df, dstype, keep_df=False)

    @classmethod
    def from_df(self, df, dstype, keep_df):
        """
            Create a NumPy object from a Pandas dataframe - used externally
            just for unit testing.

            :param str filename: full path of data file
            :param DatasetType/str dstype: type of dataset
            :param bool keep_df: whether df is retained or overwritten -
                                 the latter is more memory efficient

            :raises TypeError: if argument types incorrect
            :raises ValueError: if illegal values in args or file
            :raises FileNotFoundError: if file does not exist
            :raises FileFormatError: if format of file incorrect

            :returns NumPy: data contained in file
        """
        if (not isinstance(df, DataFrame) or not isinstance(keep_df, bool)
            or (not isinstance(dstype, (DatasetType, str))
                or dstype not in {v for v in DatasetType})):
            raise TypeError('NumPy.from_df() bad arg type')

        dtypes = {df[c].dtype.__str__() for c in df.columns}
        if (len(df.columns) == 1 or len(df) == 1
                or (dstype == 'categorical' and dtypes != {'category'})
                or (dstype == 'continuous' and dtypes != {'float32'})):
            raise ValueError('NumPy.from_df() bad arg value')

        # if keep_df is True:
        #     df = df.copy()
        df2 = df.copy(deep=True) if keep_df is True else df

        if dstype == 'categorical':

            # convert categorical values to integer codes, and capture code to
            # value mapping as a tuple for each node, i.e. ('yes', 'no')
            # implies integer code 0 maps to 'yes', 1 to 'no'

            col_values = {}
            for col in df2.columns:
                df2[col], uniques = factorize(df2[col])
                if len(uniques) > MAX_CATEGORY:
                    raise ValueError('data.read() too many categories')
                col_values[col] = tuple(uniques.categories[uniques.codes]
                                        .unique())
        else:

            # col_values just holds node names for continuous data

            col_values = {col: None for col in df.columns}

        # convert data frame to numpy array of appropriate dtype

        dtype = 'uint8' if dstype == 'categorical' else 'float32'
        data = df2.to_numpy(dtype=dtype)

        return NumPy(data, dstype, col_values)

    def set_N(self, N, seed=None):
        """
            Set current working sample size, and optionally randomise the row
            order.

            :param int N: current working sample size
            :param int/None seed: seed for row order randomisation if reqd.,
                                  0 and None both imply original order.

            :raises TypeError: if bad argument type
            :raises ValueError: if bad argument value
        """
        if (not isinstance(N, int) or isinstance(N, bool) or
            seed is not None and (not isinstance(seed, int)
                                  or isinstance(seed, bool))):
            raise TypeError('NumPy.set_N() bad arg type')

        if (N < 1 or N > self.data.shape[0] or
                (seed is not None and (seed < 0 or seed > 100))):
            raise ValueError('Pandas.set_N() bad arg value')

        self.N = N

        # if sample contains all rows in data then it is just a reference to
        # self.data, otherwise it is a copy of the required subset of rows.

        self.sample = (self.data if N == self.data.shape[0]
                       else self.data[:N, :])

        # compute the node values and counts for categorical variables for
        # the sample

        self.node_values = {}
        if self.dstype == 'categorical':
            for j in range(self.sample.shape[1]):
                counts = {self.categories[j][v]: c for v, c
                          in enumerate(bincount(self.sample[:, j]))}
                counts = {v: counts[v] for v in sorted(counts)}
                self.node_values[self.nodes[j]] = counts

        # Randomise the row order in self.sample if required

        if seed is not None and seed != 0:
            order = default_rng(seed).permutation(N)
            self.sample = self.sample[order]

    def randomise_names(self, seed=None):
        """
            Randomises the node names that the learning algorithm uses
            (so sensitivity to these names can be assessed).

            :param int/None seed: randomisation seed (if None, names revert
                                  back to original names)

            :raises TypeError: for bad argument types
            :raises ValueError: for bad argument values
        """
        if seed is not None and not isinstance(seed, int):
            raise TypeError('Data.randomise_names() bad arg type')

        # Generate new column names

        old_orig_to_ext = {orig: ext for orig, ext in self.orig_to_ext.items()}
        self._generate_random_names(seed)

        # Update keys in node_types and node_values

        map = {old_orig_to_ext[orig]: self.orig_to_ext[orig]
               for orig in self.orig_to_ext}
        self.node_values = {map[n]: vc for n, vc in self.node_values.items()}
        self.node_types = {map[n]: t for n, t in self.node_types.items()}

    def unique(self, j_reqd, num_vals):
        """
            Counts unique combinations of categorical variables in specified
            set of columns

            :param tuple j_reqd: indices of columns required
            :param ndarray num_vals: number of values in each of those columns

            :returns tuple: (ndarray: array of unique combinations,
                             ndarray: vector of corresponding counts)
        """
        minlength = npprod(num_vals).item()

        if minlength <= self.MAX_BINCOUNT:

            # If maximum number of possible combinations below threshold then
            # pack combinations into integers, and count those for speed.
            # First, generate the packed integers

            multipliers = array([npprod(num_vals[:i])
                                 for i in range(len(j_reqd))])
            packed = self.sample[:, j_reqd] @ multipliers

            # Count the frquency of unique packed integers, removing all
            # entries with zero counts

            counts = bincount(packed, minlength=minlength)
            packed = nonzero(counts)[0]
            counts = counts[packed]

            # Unpack integers back into their combinations of values using
            # the same multipliers used to pack them into one integer

            combos = empty((len(packed), len(multipliers)), dtype=int)
            for jj, r in enumerate(reversed(multipliers)):
                combos[:, len(multipliers) - jj - 1] = packed // r
                packed = packed % r

        else:

            # If maximum number of possible combinations above threshold then
            # using the much slower numpy unique function.

            combos, counts = npunique(self.sample[:, j_reqd], axis=0,
                                      return_counts=True)

        return combos, counts

    def marginals(self, node, parents, values_reqd=False):
        """
            Return marginal counts for a node and its parents.

            :param str node: node for which marginals required.
            :param dict parents: {node: parents} parents of non-orphan nodes
            :param bool values_reqd: whether parent and child values required

            :raises TypeError: for bad argument types
            :raises ValueError: for bad argument values

            :returns tuple: of counts, and optionally, values:
                            - ndarray counts: 2D, rows=child, cols=parents
                            - int maxcol: maximum number of parental values
                            - tuple rowval: child values for each row
                            - tuple colval: parent combo (dict) for each col
        """
        if (not isinstance(node, str) or not isinstance(parents, dict)
                or not all([isinstance(p, list) for p in parents.values()])
                or not isinstance(values_reqd, bool)):
            raise TypeError('NumPy.marginals() bad arg type')

        # determine nodes (external names) for which marginals required

        nodes = tuple([node] + parents[node]) if node in parents else (node,)
        if (len(set(nodes) - set(self.node_values)) != 0
                or len(nodes) != len(set(nodes))):
            raise ValueError('NumPy.marginals() bad arg value')

        maxcol = 1
        rowval = colval = None
        start = Timing.now()

        if len(nodes) == 1:

            # marginals for a single variable - just use node_values

            counts = array([[c] for c in self.node_values[node].values()],
                           dtype=int)
            if values_reqd is True:
                rowval = tuple(self.node_values[node].keys())

        else:

            # Determine required column indices and number of unique values
            # in each column.

            j_reqd = tuple(self.nodes.index(self.ext_to_orig[n])
                           for n in nodes)
            num_vals = array([len(self.node_values[n]) for n in nodes])
            maxcol = npprod(num_vals[1:]).item()

            # identify and count unique combinations of node values

            combos, _counts = self.unique(j_reqd, num_vals)

            # separate the child values and parental combinations

            c_values = array(range(len(self.node_values[node])))
            p_combos = npunique(combos[:, 1:], axis=0)

            # initialise and populate the crosstab-style matrix where rows
            # are child values, and columns are unique parental combinations.

            c_value_to_i = {v: i for i, v in enumerate(c_values)}
            p_combo_to_j = {tuple(c): j for j, c in enumerate(p_combos)}
            counts = zeros((len(c_values), len(p_combos)), dtype='int32')
            for idx, (c_value, *p_combo) in enumerate(combos):
                i = c_value_to_i[c_value]
                j = p_combo_to_j[tuple(p_combo)]
                counts[i, j] = _counts[idx]

            # Generate child category corresponding to each row, and parental
            # category combination to each column if required.

            if values_reqd is True:
                rowval = tuple(self.categories[j_reqd[0]])
                colval = tuple({self.orig_to_ext[self.nodes[j_reqd[j]]]:
                                self.categories[j_reqd[j]][c[j - 1]]
                                for j in range(1, len(j_reqd))}
                               for c in p_combos)

            c_values = p_combos = c_value_to_i = p_combo_to_j = None
            _counts = combos = None

        Timing.record('marginals', len(nodes), start)

        return (counts, maxcol, rowval, colval)

    def values(self, nodes):
        """
            Return the (float) values for the specified set of nodes. Suitable
            for passing into e.g. linearRegression fitting function

            :param tuple nodes: nodes for which data required

            :raises TypeError: if bad arg type
            :raises ValueError: if bad arg value

            :returns ndarray: Numpy array of values, each column for a node
        """
        if (not isinstance(nodes, tuple) or len(nodes) == 0
                or not all([isinstance(n, str) for n in nodes])):
            raise TypeError('Pandas.values() bad arg type')

        numeric = {n for n, t in self.node_types.items() if t != 'category'}
        if len(nodes) != len(set(nodes)) or len(set(nodes) - numeric) != 0:
            raise ValueError('Pandas.values() bad arg values')

        return self.data[:, [self.nodes.index(n) for n in nodes]]

    def as_df(self):
        """
            Return the data as a Pandas dataframe with current sample size,
            column names and column order.

            :returns DataFrame: data as Pandas
        """

        # convert NumPy array to Pandas DataFrame of appropriate type

        dtype = 'uint8' if self.dstype == 'categorical' else 'float32'
        df = DataFrame(data=self.sample, dtype=dtype, columns=self.nodes)

        # Convert integers representing categories back to categories

        if self.dstype == 'categorical':
            for j in range(len(df.columns)):
                df.iloc[:, j] = (Categorical.from_codes(df.iloc[:, j],
                                 categories=self.categories[j]))

        # reorder and rename the columns if required

        if (self.order != tuple(range(self.data.shape[1]))
                or self.orig_to_ext != self.ext_to_orig):
            order = (self.orig_to_ext[self.nodes[j]] for j in self.order)
            print('\n*** Re-ordering/naming to {}\n'.format(self.get_order()))
            df = df.rename(columns=self.orig_to_ext).reindex(columns=order)

        return df

    def write(self, filename, compress=False, sf=10, zero=None, preserve=True):
        """
            Write data into a gzipped CSV data format file

            :param str filename: full path of data file
            :param bool compress: whether to gzip compress the file
            :param int sf: number of s.f. to retain for numeric values
            :param float zero: abs values below this counted as zero
            :param bool preserve: whether self.df is left unchanged by this
                                  function, False conserves memory if writing
                                  out large files.

            :raises TypeError: if argument types incorrect
            :raises ValueError: if data has no columns defined
            :raises FileNotFoundError: if destination folder does not exist
        """
        pandas = Pandas(df=self.as_df())
        pandas.write(filename, compress, sf, zero, preserve)
