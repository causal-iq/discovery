
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
from numpy import nan, full, tril_indices, triu_indices

from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_conover_friedman
from matplotlib.pyplot import savefig, figure, xlabel, ylabel, title
from matplotlib.pyplot import get_cmap
from matplotlib.colors import ListedColormap, BoundaryNorm
from seaborn import heatmap

from analysis.statistics import rank_values


PAPER_DIR = 'experiments/papers/ijar_noisy/'
RESULTS_FILE = 'results_noisyPaper.xls'


def get_noise_benchmark_f1_ranks():
    """
        Read the noise benchmark raw results from an Excel spreadsheet,
        extract F1 values, and rank algorithms for each experiment.

        :returns tuple: DataFrame(quiet ranks), DataFrame(noisy_ranks)
    """

    def _to_ranks(f1_data):
        ranks = {}
        for dataset, f1s in f1_data.items():
            for algo, f1 in rank_values(f1s).items():
                if algo == 'NOTEARS':
                    continue
                if algo not in ranks:
                    ranks[algo] = []
                ranks[algo].append(f1)
        return ranks

    # Read results from Excel file into DataFrame with dataset, algo & F1 cols

    sheets = read_excel(io=PAPER_DIR + RESULTS_FILE, sheet_name=None)
    datasets = []
    algorithms = []
    f1 = []
    for df in sheets.values():
        data = df.to_dict(orient='list')
        datasets += data['Dataset']
        algorithms += data['Algorithm']
        f1 += data['F1 Score']

    # Restructure data so it is keyed by dataset and algorithm for the
    # quiet and noisy results separately

    quiet = {}
    noisy = {}
    for i, dataset in enumerate(datasets):
        if '_N_' in dataset:
            if dataset not in quiet:
                quiet[dataset] = {}
            quiet[dataset][algorithms[i]] = f1[i]
        else:
            if dataset not in noisy:
                noisy[dataset] = {}
            noisy[dataset][algorithms[i]] = f1[i]

    # Convert F1 values to F1 ranks for each experiment, and return as
    # DataFrames with columns as algorithms, rows as experiments

    quiet = DataFrame.from_dict(_to_ranks(quiet), orient='columns')
    noisy = DataFrame.from_dict(_to_ranks(noisy), orient='columns')

    # Order the columns in each dataset by the algorithm's overall rank in
    # the quiet data

    avg_ranks = quiet.rank(axis=1).mean(axis=0)
    sorted_algorithms = avg_ranks.sort_values().index
    quiet = quiet[sorted_algorithms]
    noisy = noisy[sorted_algorithms]

    return (quiet, noisy)


def values_ijar_noisy_read_results():

    quiet, noisy = get_noise_benchmark_f1_ranks()
    print("\n\nAlgorithm ranks for quiet data:\n{}".format(quiet))
    print("\nAlgorithm ranks for noisy data:\n{}".format(noisy))

    algos = list(noisy.columns)
    noisy = noisy.to_numpy()
    quiet = quiet.to_numpy()

    # Perform Friedman chi**2 to find if ANY algorithms differ significantly

    stat, p = friedmanchisquare(*quiet.T)
    print("Friedman Test for quiet data: statistic={:.6f}, p-value={:.6e}"
          .format(stat, p))
    stat, p = friedmanchisquare(*noisy.T)
    print("Friedman Test for noisy data: statistic={:.6f}, p-value={:.6e}"
          .format(stat, p))

    # Perform pairwise paired value tests with Conover correction

    quiet_p = posthoc_conover_friedman(quiet).to_numpy()
    noisy_p = posthoc_conover_friedman(noisy).to_numpy()
    print(type(quiet_p))

    dim = len(algos)
    p_values = full((dim, dim), nan)
    below_diag = tril_indices(dim, k=-1)
    p_values[below_diag] = quiet_p[below_diag]
    above_diag = triu_indices(dim, k=1)
    p_values[above_diag] = noisy_p[above_diag]

    # critical_difference_diagram(avg_ranks, p_values)

    bins = [0, 0.001, 0.01, 0.05, 1]
    colors = ['#00441b', '#1b7837', '#a6dba0', '#fdae61']
    cmap = ListedColormap(colors)
    cmap = get_cmap(cmap, len(colors))
    cmap.set_bad(color='gray')
    norm = BoundaryNorm(bins, len(colors))

    figure(figsize=(14, 10))
    heatmap(p_values, cmap=cmap, norm=norm, annot=True, fmt=".2e",
            cbar_kws={'label': 'p-value'}, annot_kws={'size': 7},
            xticklabels=algos, yticklabels=algos)

    # Add titles and labels
    title('Pairwise P-Value Heatmap')
    xlabel('Algorithms')
    ylabel('Algorithms')

    savefig(PAPER_DIR + 'noise_p_values.png', dpi=600)
