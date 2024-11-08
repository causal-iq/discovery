#
# File/directory funcions specific to the Noisy Data Paper experiments
#

from os import walk, sep
from re import compile
import pandas as pd

from fileio.bayesys import read
from fileio.common import FileFormatError, is_valid_path
from core.metrics import dicts_same


results_file_cache = {}

NOISY_PATTERN = compile(r'^(DAG|PAG|\d+\- +PAG|\d+\- +DAG)learned_'  # DAG/PAG
                        r'([0-9A-Za-z\-]+)\_'  # algorithm
                        r'([A-Z]+)\_'  # network
                        r'([0-9A-Za-z]+)\_'  # noise
                        r'([0-9\.]+)'  # size
                        r'(k.csv|k.csv.csv)$')


def _match_filter(filter, case):
    """
        Check whether a set of case parameters matches filter

        :param dict filter: filter which case parameters must match
        :param dict case: parameters for case

        :returns bool: whether case parameters match filter
    """
    if filter is None:
        return True
    if ('algorithm' in filter and case['algorithm']
        not in filter['algorithm']) or \
            ('network' in filter and case['network']
             not in filter['network']) or \
            ('noise' in filter and case['noise'] not in filter['noise']) or \
            ('size' in filter and case['size'] not in filter['size']):
        return False
    else:
        return True


def _get_true_graph(case, true_dir):
    """
        Gets true graph for a specific noisy case. To do so it understands
        the weird naming conventions used in the noisy data experiments.

        :param dict case: algorithm, network, noise and size values for case
        :param str true_dir: root directory where true graphs located

        :raises TypeError: if case argument not valid
        :raises ValueError: if cannot find valid true graph file

        :returns (PDAG, str): true graph for specified case and its file name
    """
    if type(case) is not dict or type(true_dir) is not str or \
            set(case.keys()) != set(['algorithm', 'network', 'noise', 'size']):
        raise TypeError('_get_true_graph called with bad args')

    # Construct name of true graph based on noise and network

    noise = case['noise']
    network = case['network']
    if noise in ['N', 'I5', 'I10', 'M5', 'M10', 'S5', 'S10', 'MI', 'MS', 'IS']:
        file_name = '/DAGs/DAGtrue_{}.csv'.format(network)
    elif network == 'ASIA':
        file_name = '/MAGs/ASIA/ASIA_MAGtrue_L10_cML_cMISL.csv'
    elif network == 'SPORTS':
        file_name = '/MAGs/SPORTS/SPORTS_MAGtrue_L10_cML_cIL_cSL_cMISL.csv'
    elif noise == 'L10':
        file_name = '/MAGs/{0:}/{0:}_MAGtrue_L10.csv'.format(network)
    else:
        file_name = '/MAGs/{0:}/{0:}_MAGtrue_L5_cML_cIL_cSL_cMISL.csv' \
            .format(network)

    return (read(true_dir + file_name, strict=False), file_name)


def validate(name, path, strict=True, filter=None):
    """
        Validate and read in BN from Bayesys format file

        :param str name: file name (excluding directory path)
        :param str path: directory path to file
        :param bool strict: whether strict validation should be applied
        :param dict filter: if not None, filter which processed files match

        :raises FileNotFoundError: if file does not exist
        :raises FileFormatError: if contents of file is not valid

        :returns PDAG: PDAG specified in file
    """

    if not strict:
        # Fix known problems with some file names
        if name == 'runtime.txt':
            return (None, None)
        f_name = name.replace('N_S5', 'S5')
        if path.endswith('SPORTS/SaiyanH') and 'SaiyanH' not in f_name:
            f_name = f_name.replace('DAGlearned_', 'DAGlearned_SPORTS_')
        if path.endswith('GOBN-ILP/N') and \
                ('GOBN-ILP' not in f_name and 'GOBNILP' not in f_name):
            f_name = f_name.replace('ned_', 'ned_GOBNILP_')
    else:
        f_name = name

    matched = NOISY_PATTERN.match(f_name)
    if matched:
        case = {'algorithm': matched.group(2), 'network': matched.group(3),
                'noise': matched.group(4), 'size': matched.group(5)}
        if not _match_filter(filter, case):
            return (None, None)
        file_name = path + sep + name
        try:
            graph = read(file_name, strict=strict)
        except Exception as e:
            raise FileFormatError(e)
    else:
        raise FileNotFoundError(path + sep + name)
    return (graph, case)


def _get_case(name, path, strict=True, filter=None):
    """
        Validate and read in BN from Bayesys format file

        :param str name: file name (excluding directory path)
        :param str path: directory path to file
        :param bool strict: whether strict validation should be applied
        :param dict/None filter: filter which processed files match

        :raises FileNotFoundError: if file name is invalid format

        :returns dict: case this file has data for
    """

    if not strict:
        # Fix known problems with some file names
        if name == 'runtime.txt':
            return None
        f_name = name.replace('N_S5', 'S5')
        if path.endswith('SPORTS/SaiyanH') and 'SaiyanH' not in f_name:
            f_name = f_name.replace('DAGlearned_', 'DAGlearned_SPORTS_')
        if path.endswith('GOBN-ILP/N') and \
                ('GOBN-ILP' not in f_name and 'GOBNILP' not in f_name):
            f_name = f_name.replace('ned_', 'ned_GOBNILP_')
    else:
        f_name = name

    matched = NOISY_PATTERN.match(f_name)
    if matched:
        case = {'algorithm': matched.group(2), 'network': matched.group(3),
                'noise': matched.group(4), 'size': matched.group(5)}
        return case if _match_filter(filter, case) else None
    else:
        raise FileNotFoundError(path + sep + name)


def _get_recorded_metrics(results_dir, case):
    """
        Gets the recorded metrics from the results spreadsheet for a
        specified case.

        :param str results_dir: root directory where results spreadsheets held
        :param dict case: specific case (network, size, noise, algorithm)

        :raise FileNotFoundError: if cannot open results file
        :raise FileFormatError: if problem with results file contents

        :returns dict: recorded metrics for  specified case
    """

    METRICS_REQD = {'# of edges': 'edges', '# of indep graphs': 'fragments',
                    'runtime': 'runtime', 'Precision': 'p-b',
                    'Recall': 'r-b', 'F1 Score': 'f1-b', 'SHD': 'shd-b',
                    'DDM': 'ddm', 'BSF': 'bsf'}

    results_sub_dir = {'N': '1. N'}

    if not isinstance(results_dir, str) or not isinstance(case, dict) or \
            sorted(list(case.keys())) != ['algorithm', 'network',
                                          'noise', 'size'] or \
            case['noise'] not in results_sub_dir:
        raise TypeError('_get_recorded_metrics called with bad args')

    noise = case['noise']
    file_name = '{}/{}/Results_{}_ALL_NEW (with corrected Precision).xls' \
        .format(results_dir, results_sub_dir[noise], noise)

    try:
        is_valid_path(file_name)
    except ValueError:
        raise FileNotFoundError('Results file for noise {} cases not found'
                                .format(case['noise']))

    # Open the spreadsheet, or used cached version if available

    if noise in results_file_cache:
        xls = results_file_cache[noise]
    else:
        xls = pd.read_excel(file_name, sheet_name=None)
        results_file_cache[noise] = xls

    # Access the sheet and row for specific case

    try:
        sheet = xls[case['network']]
        dataset = '{}_{}_{}k'.format(case['network'], noise, case['size'])
        result = sheet.loc[(sheet['Algorithm'] == case['algorithm']) &
                           ((sheet['Dataset'] == dataset) |
                            (sheet['Dataset'] == 'trainingData_' + dataset))]
        if len(result.index) != 1:
            raise KeyError()
    except KeyError:
        raise FileFormatError('Unable to find case in results file')

    return {METRICS_REQD[k]: float(v) for k, v in result.iloc[0].items()
            if k in METRICS_REQD}


def evaluate_noisy(learned_dir, true_dir, results_dir, strict=False,
                   warnings=True, filter=None):
    """
        Validates files created in the noisy data tests

        :param str learned_dir: root directory where learned graphs located
        :param str true_dir: root directory where true graphs located
        :param str results_dir: root directory where results spreadsheets are
        :param bool strict: whether strict validation should be applied
        :param bool warnings: whether detail of problems reported
        :param dict/None filter: only process files matching filter
    """
    num_ok = 0
    num_badname = 0
    num_badcontents = 0
    for root, _, files in walk(learned_dir, topdown=False):
        for name in files:
            try:
                case = _get_case(name, root, strict, filter)
                if case:
                    print(name)
                    num_ok += 1

                    # get true graph for this case

                    true, _ = _get_true_graph(case, true_dir)

                    # get learned graph - ensure contains same nodes as true

                    learned = read(root + sep + name, all_nodes=true.nodes,
                                   strict=strict)
                    metrics = learned.compared_to(true, bayesys='v1.5+')
                    expected = _get_recorded_metrics(results_dir, case)

                    assert dicts_same(metrics, expected, strict=False, sf=3)

                    # print(learned)
            except FileFormatError as e:
                num_badcontents += 1
                if warnings:
                    print('Bad file format: {}'.format(e))
            except FileNotFoundError as e:
                num_badname += 1
                if warnings:
                    print('Bad file name: {}'.format(e))

    print('\nLearned graphs: {} OK, {} bad contents, and {} bad file name'
          .format(num_ok, num_badcontents, num_badname))
    return None
