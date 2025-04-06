
# Analyses results in Noisy Benchmark experiments described in:
#
# Constantinou, A. C., Liu, Y., Chobtham, K., Guo, Z.,
# and Kitson, N. K. (2021). Large-scale empirical validation of Bayesian
# Network structure learning algorithms with noisy data.
# International Journal of Approximate Reasoning, Vol. 131, pp. 151–188
#
# and produces selected results for Ken Kitson PhD Thesis.
#

from pandas import read_excel, DataFrame
import pandas._config.config as config
from numpy import nan, full, tril_indices, triu_indices
from copy import deepcopy

from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_conover_friedman
from matplotlib.pyplot import savefig, figure, xlabel, ylabel, title
from matplotlib.pyplot import get_cmap
from matplotlib.colors import ListedColormap, BoundaryNorm
from seaborn import heatmap

from analysis.statistics import rank_values, normality, equal_variances


PAPER_DIR = 'experiments/papers/ijar_noisy/'
RESULTS_FILE = 'results_noisyPaper.xls'


def get_noise_benchmark_results(metric, method_is_algo=True, rank_reqd=True):
    """
        Read the noise benchmark raw results from an Excel spreadsheet,
        extract F1 values, and rank algorithms for each experiment.

        :param bool method_is_algo: method is algorithm if true otherwise
                                    it is noise type.
        :param bool rank_reqd: whether to return F1 ranks or values

        :returns tuple: DataFrame(quiet ranks), DataFrame(noisy_ranks)
    """

    def results_matrix(block_results, rank_reqd=True, ignore_methods=set()):
        """
            Return values for each sample

            :param dict f1_data: {block: {method: result}}
            :param bool rank_reqd: whether ranks or values required
            :param set ignore_methoods: methods to ignore

            :returns dict: {method: [results]}
        """
        def method_to_result(block_result):  # {method: result} for a block
            return (rank_values(block_result) if rank_reqd is True else
                    {a: 0.0 if isinstance(v, str) else v
                     for a, v in block_result.items()})

        results_matrix = {}
        for block_result in block_results.values():
            for method, result in method_to_result(block_result).items():
                if method in ignore_methods:
                    continue
                if method not in results_matrix:
                    results_matrix[method] = []
                results_matrix[method].append(result)

        return DataFrame.from_dict(results_matrix, orient='columns')

    # Read results from Excel file into DataFrame with dataset, algo & F1 cols

    sheets = read_excel(io=PAPER_DIR + RESULTS_FILE, sheet_name=None)
    datasets = []
    algos = []
    results = []
    for sheet in sheets.values():  # Each sheet has results for one network
        sheet = sheet.to_dict(orient='list')
        datasets += sheet['Dataset']
        algos += sheet['Algorithm']
        results += sheet[metric]

    # Restructure data into dicts so it is keyed by block and then method.
    # Do this separately for quiet and noisy data if the method is algorithm

    quiet = {}
    noisy = {}
    if method_is_algo is True:
        for i, dataset in enumerate(datasets):
            if '_N_' in dataset:
                if dataset not in quiet:
                    quiet[dataset] = {}
                quiet[dataset][algos[i]] = results[i]
            else:
                if dataset not in noisy:
                    noisy[dataset] = {}
                noisy[dataset][algos[i]] = results[i]
    else:
        for i, dataset in enumerate(datasets):
            dataset = dataset.split('_')
            block = dataset[0] + '_' + algos[i] + '_' + dataset[2]
            if block not in noisy:
                noisy[block] = {}
            noisy[block][dataset[1]] = results[i]
        noisy = {b: vs for b, vs in noisy.items() if len(vs) == 16}
        quiet = deepcopy(noisy)

    # Convert results into results_matrix DataFrame format - columns are
    # methods and rows are blocks

    quiet = results_matrix(quiet, rank_reqd, ignore_methods={'NOTEARS'})
    noisy = results_matrix(noisy, rank_reqd, ignore_methods={'NOTEARS'})

    # Order the columns in each dataset by the algorithm's overall rank in
    # the quiet data

    avg_ranks = quiet.rank(axis=1).mean(axis=0)
    sorted_algorithms = avg_ranks.sort_values().index
    quiet = quiet[sorted_algorithms]
    noisy = noisy[sorted_algorithms]

    return (quiet, noisy)


def paired_significance(group1, group2):
    """
        Perform paired value significance tests

        :param DataFrame group1: first group to compare, columns are comparison
                                 attribute, rows are experiment results
        :param DataFrame group2: second group

        :returns tuple: (compare - list of things being compared,
                         p_values - array of p_values)
    """

    compare = list(group1.columns)
    group1 = group1.to_numpy().copy()
    group2 = group2.to_numpy().copy()

    stat, p = friedmanchisquare(*group1.T)
    print("Friedman Test for group 1 data: statistic={:.6f}, p-value={:.6e}"
          .format(stat, p))
    stat, p = friedmanchisquare(*group2.T)
    print("Friedman Test for group 2 data: statistic={:.6f}, p-value={:.6e}"
          .format(stat, p))

    # Perform pairwise paired value tests with Conover correction

    group1_p = posthoc_conover_friedman(group1, p_adjust='fdr_bh').to_numpy()
    group2_p = posthoc_conover_friedman(group2, p_adjust='fdr_bh').to_numpy()

    # Place group 1 p_values below diag and group 2 above

    dim = len(compare)
    p_values = full((dim, dim), nan)
    below_diag = tril_indices(dim, k=-1)
    p_values[below_diag] = group1_p[below_diag]
    above_diag = triu_indices(dim, k=1)
    p_values[above_diag] = group2_p[above_diag]

    return compare, p_values


def plot_heatmap(labels, values, plot_file):
    """
        Write a heatmap plot to file

        :param list labels: list of labels for axes
        :param ndarray values: 2D array of values to plot
        :param str plot_file: name of plot file
    """
    bins = [0, 0.001, 0.01, 0.05, 1]
    colors = ['#00441b', '#1b7837', '#a6dba0', '#fdae61']
    cmap = ListedColormap(colors)
    cmap = get_cmap(cmap, len(colors))
    cmap.set_bad(color='gray')
    norm = BoundaryNorm(bins, len(colors))

    figure(figsize=(14, 10))
    heatmap(values, cmap=cmap, norm=norm, annot=True, fmt=".1e",
            cbar_kws={'label': 'p-value'}, annot_kws={'size': 9},
            xticklabels=labels, yticklabels=labels)

    # Add titles and labels
    title('Pairwise P-Value Heatmap')
    xlabel('Algorithms')
    ylabel('Algorithms')

    savefig(plot_file, dpi=600)


def values_ijar_noisy_f1_rank_algo_signif():

    quiet, noisy = get_noise_benchmark_results(metric='F1 Score',
                                               rank_reqd=True)
    print("\n\nAlgorithm ranks for quiet data:\n{}".format(quiet))
    print("\nAlgorithm ranks for noisy data:\n{}".format(noisy))

    print('\nNormality of quiet data:\n{}'.format(normality(quiet)))
    print('\nNormality of noisy data:\n{}'.format(normality(noisy)))

    print('\nQuiet data variances equal:\n{}'.format(equal_variances(quiet)))
    print('\nNoisy data variances equal:\n{}'.format(equal_variances(noisy)))

    algos, p_values = paired_significance(quiet, noisy)
    plot_heatmap(algos, p_values, PAPER_DIR + 'noise_f1_rank_algo_signif.png')


def values_ijar_noisy_f1_algo_signif():
    config.option_context("display.float_format", "{:.2e}".format)

    quiet, noisy = get_noise_benchmark_results(metric='F1 Score',
                                               rank_reqd=False)
    print("\n\nAlgorithm F1 for quiet data:\n{}".format(quiet))
    print("\nAlgorithm F1 for noisy data:\n{}".format(noisy))

    print('\nNormality of quiet data:\n{}'.format(normality(quiet)))
    print('\nNormality of noisy data:\n{}'.format(normality(noisy)))

    print('\nQuiet data variances equal:\n{}'.format(equal_variances(quiet)))
    print('\nNoisy data variances equal:\n{}'.format(equal_variances(noisy)))

    algos, p_values = paired_significance(quiet, noisy)
    plot_heatmap(algos, p_values, PAPER_DIR + 'noise_f1_algo_signif.png')


def values_ijar_noisy_bsf_algo_signif():
    config.option_context("display.float_format", "{:.2e}".format)

    quiet, noisy = get_noise_benchmark_results(metric='BSF', rank_reqd=False)
    print("\n\nAlgorithm F1 for quiet data:\n{}".format(quiet))
    print("\nAlgorithm F1 for noisy data:\n{}".format(noisy))

    print('\nNormality of quiet data:\n{}'.format(normality(quiet)))
    print('\nNormality of noisy data:\n{}'.format(normality(noisy)))

    print('\nQuiet data variances equal:\n{}'.format(equal_variances(quiet)))
    print('\nNoisy data variances equal:\n{}'.format(equal_variances(noisy)))

    algos, p_values = paired_significance(quiet, noisy)
    plot_heatmap(algos, p_values, PAPER_DIR + 'noise_bsf_algo_signif.png')


def values_ijar_noisy_shd_algo_signif():
    config.option_context("display.float_format", "{:.2e}".format)

    quiet, noisy = get_noise_benchmark_results(metric='SHD', rank_reqd=False)
    print("\n\nAlgorithm F1 for quiet data:\n{}".format(quiet))
    print("\nAlgorithm F1 for noisy data:\n{}".format(noisy))

    print('\nNormality of quiet data:\n{}'.format(normality(quiet)))
    print('\nNormality of noisy data:\n{}'.format(normality(noisy)))

    print('\nQuiet data variances equal:\n{}'.format(equal_variances(quiet)))
    print('\nNoisy data variances equal:\n{}'.format(equal_variances(noisy)))

    algos, p_values = paired_significance(quiet, noisy)
    plot_heatmap(algos, p_values, PAPER_DIR + 'noise_shd_algo_signif.png')


def values_ijar_noisy_f1_type_signif():
    config.option_context("display.float_format", "{:.2e}".format)

    quiet, noisy = get_noise_benchmark_results(metric='F1 Score',
                                               method_is_algo=False,
                                               rank_reqd=False)
    print("\n\nAlgorithm F1 for quiet data:\n{}".format(quiet))
    print("\nAlgorithm F1 for noisy data:\n{}".format(noisy))

    print('\nNormality of quiet data:\n{}'.format(normality(quiet)))
    print('\nNormality of noisy data:\n{}'.format(normality(noisy)))

    print('\nQuiet data variances equal:\n{}'.format(equal_variances(quiet)))
    print('\nNoisy data variances equal:\n{}'.format(equal_variances(noisy)))

    algos, p_values = paired_significance(quiet, noisy)
    plot_heatmap(algos, p_values, PAPER_DIR + 'noise_f1_type_signif.png')


def values_ijar_noisy_bsf_type_signif():
    config.option_context("display.float_format", "{:.2e}".format)

    quiet, noisy = get_noise_benchmark_results(metric='BSF',
                                               method_is_algo=False,
                                               rank_reqd=False)
    print("\n\nAlgorithm F1 for quiet data:\n{}".format(quiet))
    print("\nAlgorithm F1 for noisy data:\n{}".format(noisy))

    print('\nNormality of quiet data:\n{}'.format(normality(quiet)))
    print('\nNormality of noisy data:\n{}'.format(normality(noisy)))

    print('\nQuiet data variances equal:\n{}'.format(equal_variances(quiet)))
    print('\nNoisy data variances equal:\n{}'.format(equal_variances(noisy)))

    algos, p_values = paired_significance(quiet, noisy)
    plot_heatmap(algos, p_values, PAPER_DIR + 'noise_bsf_type_signif.png')


def values_ijar_noisy_shd_type_signif():
    config.option_context("display.float_format", "{:.2e}".format)

    quiet, noisy = get_noise_benchmark_results(metric='SHD',
                                               method_is_algo=False,
                                               rank_reqd=False)
    print("\n\nAlgorithm F1 for quiet data:\n{}".format(quiet))
    print("\nAlgorithm F1 for noisy data:\n{}".format(noisy))

    print('\nNormality of quiet data:\n{}'.format(normality(quiet)))
    print('\nNormality of noisy data:\n{}'.format(normality(noisy)))

    print('\nQuiet data variances equal:\n{}'.format(equal_variances(quiet)))
    print('\nNoisy data variances equal:\n{}'.format(equal_variances(noisy)))

    algos, p_values = paired_significance(quiet, noisy)
    plot_heatmap(algos, p_values, PAPER_DIR + 'noise_shd_type_signif.png')
