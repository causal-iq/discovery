
# Analyses results in Knowledge Impact experiments described in:
#
# Constantinou, A.C., Guo, Z. and Kitson, N.K., 2023.
# The impact of prior knowledge on causal structure learning.
# Knowledge and Information Systems, pp.1-50.
#
# and produces selected results for Ken Kitson PhD Thesis.
#

from numpy import isnan, sum
from scipy.stats import normaltest, wilcoxon, binomtest
from pandas import DataFrame, read_excel
from statsmodels.stats.multitest import multipletests

PAPER_DIR = 'experiments/papers/kis_know/'
RESULTS_FILE = 'kis_knowledge.xls'
ALGORITHMS = {'SaiyanH', 'HC', 'TABU', 'MAHC', 'GES'}


def robust_wilcoxon(x, y, force_signtest=False):
    """
        Robust two-sample Wilcoxon Signed-Rank test that uses the appropriate
        strategy depending upon the number of zero differences

        Non-parametric paired test. Note that results must be ordered so that
        pairs of results have same position in the two lists.

        :param list x: results for one factor-level
        :param list y: results for other factor-level
        :param bool force_signtest: use the Sign test regardless of # zeroes
    """
    diffs = x - y
    num_zeros = sum(diffs == 0)
    total_pairs = len(diffs)
    percent_zeros = (num_zeros / total_pairs) * 100

    # print(f"Total pairs: {total_pairs}")
    # print(f"Zero differences: {num_zeros} ({percent_zeros:.2f}%)")

    if percent_zeros < 10 and not force_signtest:

        # Remove zero differences and use exact method

        x_nonzero = x[diffs != 0]
        y_nonzero = y[diffs != 0]
        stat, p = wilcoxon(x_nonzero, y_nonzero, method='exact')
        method_used = "Wilcoxon (Exact, No Zeros)"

    elif percent_zeros < 30 and not force_signtest:

        # Use approximate method - suitable for between 10 and 30% zeroes

        stat, p = wilcoxon(x, y, method='approx')
        method_used = "Wilcoxon (Approx)"

    else:
        # Use a paired sign test
        n_pos = sum(diffs > 0)
        n_total = sum(diffs != 0)  # Ignore zero differences
        p = binomtest(n_pos, n_total, p=0.5, alternative='two-sided').pvalue
        stat = None  # Sign test doesn't return a test statistic
        method_used = "Sign Test ({:.1f}% Zeros)".format(percent_zeros)

    # print(f"Test Used: {method_used}")
    # print(f"P-value: {p}\n")
    return stat, p, method_used


def read_kis_data(metric):
    """
        Read results from Excel file into DataFrame with dataset, algo &
        metrics cols.

        :param str metric: metric required: 'F1', 'BSF', 'BIC (log2)'

        :returns DataFrame: of results, columns are knowledge methods
    """

    results = read_excel(io=PAPER_DIR + RESULTS_FILE, sheet_name='all results')
    results = results[results.columns[:14]].to_dict(orient='list')

    print("\n\nColumns in results sheet are:\n{}".format(list(results.keys())))

    n_expts = 0
    methods = {}
    for i, algo in enumerate(results['Algorithm']):
        if algo not in ALGORITHMS:
            continue
        n_expts += 1
        method = results['constraint'][i]
        if method not in methods:
            methods[method] = {}
        expt = (algo + '_' + results['Case study'][i] + '_' +
                results['sample size'][i])
        if expt not in methods[method]:
            methods[method][expt] = results[metric][i]
        else:
            raise ValueError('Duplicate {}:{}'.format(method, expt))
    methods = DataFrame(methods)
    print("\n{} experiments found:\n{}".format(n_expts, methods.head()))

    return methods


def kis_signif_test(metric):
    """
        Read the noise benchmark raw results from an Excel spreadsheet,
        extract F1 values, and rank algorithms for each experiment.

        :param str metric: metric required: 'F1', 'BSF', 'BIC (log2)'
    """

    # Test each set of results for normality and whether significantly
    # different from baseline ignoring tests with missing value result.

    methods = read_kis_data('F1')

    # separate out baseline where no knowledge used

    baseline = methods['none'].to_numpy()
    methods = methods.drop(columns='none')

    max_norm_p = None
    num_normal = 0
    p_raw = []
    for method in methods.columns:  # loop over different methods

        # get difference between knowledge and baseline disregarding cases
        # where no result was produced

        values = methods[method].to_numpy()
        valid = ~isnan(values)
        diff = methods[method].to_numpy() - baseline
        diff = diff[valid]

        # Ignore methods with less than 10 cases, check normality and perform
        # robust non-parametric paired significance test.

        if len(diff) > 10:
            _, norm_p = normaltest(diff)
            if norm_p >= 0.05:
                num_normal += 1
            if max_norm_p is None or norm_p > max_norm_p:
                max_norm_p = norm_p
            _, p, test = robust_wilcoxon(values[valid], baseline[valid],
                                         force_signtest=True)
            p_raw.append(p)
            print(('{:20s} has {:3d} results, normality {:.2e}, ' +
                   '{} significance {:.2e}')
                  .format(method, len(diff), norm_p, test, p))
        else:
            print('*** {} has less than 10 experiments'.format(method))
    print('\nAll experiments have p-value of {:.2e} or below for normality'
          .format(max_norm_p))
    print('{}/{} experiment results have normal distribution at p < 0.05'
          .format(num_normal, len(methods.columns)))

    # perform correction to avoid false positives since doing multiple tests.

    corrected = multipletests(p_raw, method='bonferroni')
    print('\nKnowledge methods significantly different to no knowledge:')
    for i, method in enumerate(methods.columns):
        print("{}: {} ({:.2e})"
              .format((' TRUE' if corrected[1][i] < 0.05 else 'FALSE'),
                      method, corrected[1][i]))


def values_kis_f1_signif():
    kis_signif_test('F1')


def values_kis_arcs_signif():
    kis_signif_test('Arcs learnt')


def values_kis_bic_signif():
    kis_signif_test('BIC (log2)')
