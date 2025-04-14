
# Functions to perform statistical analysis

from numpy import exp, number
from pandas import DataFrame
from scipy.stats import shapiro, normaltest, kstest, anderson, levene, \
    bartlett, fligner, wilcoxon, binomtest
from statsmodels.stats.multitest import multipletests


def rank_values(scores):
    """
        Ranks "scores" by repacing absolute score with the ranking

        :param dict scores: dict of absolute score values

        :returns dict: has same keys as "scores" but with values replaced by
                       a rank
    """
    if (not isinstance(scores, dict) or not len(scores)
            or not all([isinstance(k, str) for k in scores])
            or not all([isinstance(v, (str, int, float))
                        for v in scores.values()])
            or any([isinstance(v, bool) for v in scores.values()])):
        raise TypeError("rank_values() called with bad args")

    # Separate valid scores from failures which are indicated by str typevalues

    valid = sorted([(s, a) for a, s in scores.items()
                    if not isinstance(s, str)], reverse=True)
    previous_score = None
    rank = 0
    tie_count = 1
    ranked = {}
    for score, algo in valid:
        if score != previous_score:
            rank += tie_count
            tie_count = 1
        else:
            tie_count += 1

        # print(algo, rank)
        ranked[algo] = rank
        previous_score = score

    if len(scores) > len(valid):
        ranked.update({a: len(valid) + 1 for a, s in scores.items()
                       if isinstance(s, str)})

    return ranked


def anderson_p(A2):
    """
        Approximate, emprical p-value from Anderson-Darling test based on:
        Marsaglia, G., & Marsaglia, J. (2004).
        "Evaluating the Anderson-Darling Distribution".
        Journal of Statistical Software, 9(2), 1-5.

        :param float A2: statistic returned by Anderson-Darling Test

        :returns float: EMPIRICAL, APPROXIMATE, EQUIVALENT p-value
    """
    if A2 <= 0.2:
        p = 1 - exp(-13.436 + 101.14 * A2 - 223.73 * A2**2)
    elif A2 <= 0.34:
        p = 1 - exp(-8.318 + 42.796 * A2 - 59.938 * A2**2)
    elif A2 <= 0.6:
        p = exp(0.9177 - 4.279 * A2 - 1.38 * A2**2)
    elif A2 <= 10:
        p = exp(1.2937 - 5.709 * A2 + 0.0186 * A2**2)
    else:
        p = 0  # Very strong rejection of normality
    return p


def normality(samples):
    """
        Test normality of samples using four different tests. Null hypothesis
        is that they are normal.

        :param DataFrame samples: samples to test normality of. Each column is
                                  one sample to test

        :raises TypeError: if argument not a DataFrame
        :raises ValueError: if sample size < 3

        :returns DataFrame: p-value that we would see the sample data given
                            the null hypothesis.
    """
    if (not isinstance(samples, DataFrame)
        or not (samples.select_dtypes(include=[number]).shape[1] ==
                samples.shape[1])):
        raise TypeError('normality() bad arg type')
    if len(samples) < 3:
        raise ValueError('normality() bad arg value')

    results = {}
    for sample in samples.columns:
        data = samples[sample].dropna()  # Remove NaNs if any

        # Shapiro-Wilk test - good for n < 50
        _, shapiro_p = shapiro(data) if len(data) <= 5000 else (None, None)

        # D'Agostino & Pearson test - more power than Shapiro-Wilk
        _, dagostino_p = normaltest(data) if len(data) > 20 else (None, None)

        # Kolmogorov-Smirnov test - tests for different distributions
        _, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))

        # Anderson-Darling test - good for large sample sizes
        and_stat = anderson(data, dist='norm')

        results[sample] = {
            'Shapiro-Wilk': shapiro_p,
            'D\'Agostino': dagostino_p,
            'Kolmogorov-Smirnov': ks_p,
            'Anderson-Darling': anderson_p(and_stat.statistic),
        }

    return DataFrame.from_dict(results, orient='index')


def equal_variances(samples):
    """
        Test whether samples have same variance

        :param DataFrame samples: samples to test variance of. Each column is
                                  one sample to test

        :returns Series: p_value that all samples have same variance from
                         the three tests
    """
    if not isinstance(samples, DataFrame):
        raise TypeError('equal_variances() bad arg type')

    data = [samples[s] for s in samples.columns]
    return DataFrame({'levene': [levene(*data)[1]],
                      'bartlett': [bartlett(*data)[1]],
                      'fligner': [fligner(*data)[1]]})


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


def correct_p_values(p_values: dict, correction: str = 'bonferroni',
                     significance_level: float = 0.05) -> dict:
    """
        Corrects p-values using the specified correction method and filters
        significant methods.

        :param dict p_values: Dictionary of methods and their raw p-values.
        :param str correction: Correction method ('bonferroni', 'fdr_bh', etc.)
                               or None to skip correction. Default is
                               'bonferroni'.
        :param float significance_level: Significance level to filter corrected
                                         p-values. Default is 0.05.
        :returns dict: Dictionary of methods with corrected p-values that meet
                       the significance level.
    """
    if (not isinstance(p_values, dict)
        or not all(isinstance(v, float) for v in p_values.values())
            or (correction is not None and not isinstance(correction, str))):
        raise TypeError("correct_p_values bad arg types")

    if (not len(p_values)
        or not (0 < significance_level < 1)
        or correction not in {None, 'bonferroni', 'fdr_bh', 'holm',
                              'sidak'}):
        raise ValueError("correct_p_values bad arg values")

    methods = list(p_values.keys())
    raw_p_values = list(p_values.values())

    # Skip correction if correction is None
    if correction is None:
        corrected_p_values = raw_p_values
    else:
        # Apply correction
        corrected = multipletests(raw_p_values, method=correction)
        corrected_p_values = corrected[1]

    # Filter significant methods
    significant_methods = {
        method: corrected_p
        for method, corrected_p in zip(methods, corrected_p_values)
        if corrected_p < significance_level
    }

    return significant_methods
