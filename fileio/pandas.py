
# Concrete subclass of Data which implements a Pandas data source


from numpy import array
from pandas import DataFrame, crosstab, read_csv, to_numeric
from pandas.errors import EmptyDataError
from gzip import BadGzipFile
from csv import QUOTE_MINIMAL

from fileio.data import Data
from core.common import rndsf
from fileio.common import is_valid_path, FileFormatError, DatasetType
from core.timing import Timing


class Pandas(Data):
    """
        Data subclass which holds data in a Pandas dataframe

        :param DataFrame df: data provided as a Pandas dataframe.

        :ivar DataFrame df: original Pandas dataframe providing data
        :ivar DataFrame sample: sample of self.df with correct column names
                                and sample size which provides sample counts
                                to algorithm
        :ivar DatasetType dstype: type of dataset (categorical/numeric/mixed)
        :ivar dict node_values: values and their counts of categorical nodes
                                in sample {n1: {v1: c1, v2: ...}, n2 ...}
    """
    def __init__(self, df):
        if not isinstance(df, DataFrame):
            raise TypeError('Pandas() bad arg type')

        if len(df) < 2 or len(df.columns) < 2:
            raise ValueError('Pandas() bad dataframe size')

        if df.isna().any().any():
            raise ValueError('Pandas() missing data unsupported')

        self.sample = self.df = df  # all refer to same object initially
        self.nodes = tuple(df.columns)
        self.order = tuple(i for i in range(len(self.nodes)))
        self.ext_to_orig = {n: n for n in self.nodes}
        self.orig_to_ext = {n: n for n in self.nodes}
        self.N = len(df)
        self.node_values = {c: dict(self.sample[c].value_counts())
                            for c in self.sample.columns
                            if self.sample[c].dtype.__str__() == 'category'}

        # Determine node types and overall dataset type

        self.node_types = {n: self.sample[n].dtype.__str__()
                           for n in self.nodes}
        self._set_dstype()

    @classmethod
    def _set_type(self, column):
        """
            Set appropriate variable type for structure learning:
                - integers are set to category if they would fit into
                  int8 otherwise minimum length integer type
                - floats set to minimum length float type
                - everything else set to category

            :param Series column: column to set type for
        """
        type = 'category'
        try:
            type = to_numeric(column).dtype.__str__()
            if type == 'int64':
                type = to_numeric(column, downcast='integer').dtype.__str__()
                type = 'category' if type == 'int8' else type
            else:
                type = to_numeric(column, downcast='float').dtype.__str__()
        except ValueError:
            pass

        print('Variable {} set to type {}'.format(column.name, type))
        return column.astype(str).astype(type)

    @classmethod
    def read(self, filename, dstype=None, N=None):
        """
            Read a file into a Pandas object

            :param str filename: full path of data file
            :param DatasetType/str/None dstype: type of dataset
            :param int/None N: number of rows to read

            :raises TypeError: if argument types incorrect
            :raises ValueError: if illegal value coercion or N < 2
            :raises FileNotFoundError: if file does not exist
            :raises FileFormatError: if format of file incorrect

            :returns Data: data contained in file
        """
        if (not isinstance(filename, str)
            or (N is not None
                and (not isinstance(N, int) or isinstance(N, bool)))
            or (dstype is not None
                and (not isinstance(dstype, (DatasetType, str))
                     or dstype not in {v for v in DatasetType}))):
            raise TypeError('Bad argument types for Pandas.read')
        if (N is not None and N < 2):
            raise ValueError('Bad argument values for Pandas.read')

        is_valid_path(filename)

        try:
            # Read from file treating all values as strings initially

            nrows = {} if N is None else {'nrows': N}
            dtype = ('float32' if dstype == 'continuous' else
                     ('category' if dstype == 'categorical' else 'object'))
            df = read_csv(filename, sep=',', header=0, encoding='utf-8',
                          keep_default_na=False, na_values='<NA>', dtype=dtype,
                          **nrows)
            if N is not None and N > len(df):
                raise ValueError('Bad argument values for Pandas.read')

            # Convert values to appropriate type for structure learning
            # if dstype was unspecified or was mixed

            if dstype not in {'categorical', 'continuous'}:
                print()
                for col in df.columns:
                    df[col] = self._set_type(df[col])

            return Pandas(df=df)

        except (UnicodeDecodeError, PermissionError, EmptyDataError,
                BadGzipFile) as e:
            raise FileFormatError('File format error: {}'.format(e))

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
        if (not isinstance(filename, str) or not isinstance(compress, bool)
                or not isinstance(sf, int) or isinstance(sf, bool)
                or (zero is not None and not isinstance(zero, float))
                or not isinstance(preserve, bool)):
            raise TypeError('Bad argument types for data.write')
        zero = zero if zero is not None else 10 ** (- sf)

        if sf < 2 or sf > 10 or zero < 1e-20 or zero > 0.1:
            raise ValueError('Bad argument types for data.write')

        df = self.df.copy() if preserve is True else self.df
        for col in df.columns:
            if df[col].dtype in ['float32', 'float64']:
                df[col] = df[col].apply(lambda x: rndsf(x, sf, zero))

        df.to_csv(filename, index=False, na_rep='*',
                  quoting=QUOTE_MINIMAL, escapechar='+',
                  compression="gzip" if compress is True else 'infer')

    def _update_sample(self, old_N=None, old_ext_to_orig=None):
        """
            Updates sample dataframe and node_values so they have correct
            sizes, node order and node names.

            :param int/None old_N: old value of sample size if changed
            :param dict/None old_ext_to_orig: old name mapping if changed
        """
        node_order = [self.nodes[i] for i in self.order]

        # This next line uses a possibly inefficient, but the originally used.
        # method to get a subset of rows by converting the DataFrame to a dict
        # and back to a DataFrame. The commented out line below would seem to
        # be more efficient and should produce the same result but seems to
        # result in different scores being produced at low sample sizes.

        self.sample = DataFrame(self.df[:self.N].to_dict())[node_order]
        # self.sample = ((self.df[:self.N])[node_order])

        self.sample.rename(columns=self.orig_to_ext, inplace=True)

        if old_N is not None and old_N != self.N:

            # if N has changed then must recount node values

            self.node_values = {c: dict(self.sample[c].value_counts())
                                for c in self.sample.columns}

        elif old_ext_to_orig is not None:

            # if nodes renamed then must update key values in node_values
            # and node_types

            old_rev_map = {orig: ext for ext, orig in old_ext_to_orig.items()}
            old_to_new = {old_rev_map[orig]: self.orig_to_ext[orig]
                          for orig in self.orig_to_ext}
            self.node_values = {old_to_new[old]: counts
                                for old, counts in self.node_values.items()}
            self.node_types = {old_to_new[old]: _type
                               for old, _type in self.node_types.items()}

    def set_N(self, N, seed=None):
        """
            Set current working sample size, and optionally randomise the row
            order

            :param int N: current working sample size
            :param int/None seed: seed for row order randomisation if reqd.

            :raises TypeError: if bad argument type
            :raises ValueError: if bad argument value
        """
        if (not isinstance(N, int) or isinstance(N, bool) or
            seed is not None and (not isinstance(seed, int)
                                  or isinstance(seed, bool))):
            raise TypeError('Data.set_N() bad arg type')

        if (N < 1 or N > len(self.df) or
                (seed is not None and (seed < 0 or seed > 100))):
            raise ValueError('Pandas.set_N() bad arg value')

        old_N = self.N
        self.N = N
        self._update_sample(old_N=old_N)

        if seed is not None and seed != 0:
            self.sample = self.sample.sample(frac=1.0, random_state=seed) \
                          .reset_index(drop=True)

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

        # Revert node names back to originals

        old_ext_to_orig = self.ext_to_orig
        self.df.rename(columns=self.ext_to_orig, inplace=True)
        self.ext_to_orig = {n: n for n in self.nodes}

        # Generate the new column names

        self._generate_random_names(seed)

        # Update the Pandas sample dataset with the new column names

        self._update_sample(old_ext_to_orig=old_ext_to_orig)

    def marginals(self, node, parents, values_reqd=False):
        """
            Return marginal counts for a node and its parents.

            :param str node: node for which marginals required.
            :param dict parents: {node: [parents]} parents of non-orphan nodes
            :param bool values_reqd: whether parent and child values required

            :raises TypeError: for bad argument types

            :returns tuple: of counts, and optionally, values:
                            - ndarray counts: 2D, rows=child, cols=parents
                            - int maxcol: maximum number of parental values
                            - tuple rowval: child values for each row
                            - tuple colval: parent combo (dict) for each col
        """
        if (not isinstance(node, str) or not isinstance(parents, dict)
                or not all([isinstance(p, list) for p in parents.values()])
                or not isinstance(values_reqd, bool)):
            raise TypeError('Pandas.marginals() bad arg type')

        maxcol = 1
        rowval = None
        colval = None
        start = Timing.now()

        if node in parents:

            # Node has parents so get cross tabulation

            marginals = DataFrame.copy(crosstab(self.sample[node],
                                                [self.sample[p] for p in
                                                 sorted(parents[node])]))

            # max number of parental value combos is geometric product of
            # number of states of each parent

            for p in parents[node]:
                maxcol *= len(self.node_values[p].keys())

            # sort rows and columns by values to get consistent ordering

            marginals.sort_index(axis='index', inplace=True)
            marginals.sort_index(axis='columns', inplace=True)

            # extract row and column values from indices if required

            if values_reqd is True:
                rowval = tuple(marginals.index)
                colval = tuple(dict(zip(marginals.columns.names,
                               (col,) if isinstance(col, str) else col))
                               for col in marginals.columns)

            # Obtain counts as newly instantiated NumPy 2-D array

            counts = marginals.to_numpy(dtype=int, copy=True)

        else:

            # Orphan node so just use node value counts, sorted by value

            marginals = self.sample[node].value_counts().sort_index().to_dict()
            counts = array([[c] for c in marginals.values()], dtype=int)

            if values_reqd is True:
                rowval = tuple(marginals.keys())

        # Free memory and record timing information

        marginals = None
        Timing.record('marginals', (len(parents[node]) + 1
                                    if node in parents else 1), start)

        return (counts, maxcol, rowval, colval)

    def values(self, nodes):
        """
            Return the numeric values for the specified set of nodes. Suitable
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

        values = self.sample[list(nodes)].values

        return values

    def as_df(self):
        """
            Return the data as a Pandas dataframe with current sample size
            and column order.

            :returns DataFrame: data as Pandas
        """
        return self.sample
