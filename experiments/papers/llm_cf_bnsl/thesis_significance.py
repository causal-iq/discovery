
# Statistical Significance Analysis of structure learning guided by ChatGPT-4
# generated constraints

from pandas import DataFrame, read_excel, melt
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

from analysis.statistics import normality, robust_wilcoxon, correct_p_values


PAPER_DIR = 'experiments/papers/llm_cf_bnsl/'
RESULTS_FILE = 'resultsMultiInput_v3_for_R1.xls'


def read_gpt4_results(metric):
    """
        Read results from GPT4 results spreadsheet and convert to percentage
        change values plotted in original paper.

        :param str metric: required metric: F1, BSF, SHD or BIC

        :returns tuple: absolute, and percentage relative results as
                        DataFranes where rows, columns are blocks, methods
    """
    sheet = read_excel(io=PAPER_DIR + RESULTS_FILE,
                       sheet_name='results by network size for R1')

    # Loop through rows in sheet, extracting block (algorithm + network).
    # method (constraints) and result (F1/BSF/SHD/BIC) in dict of dict
    # {block: {method: metric}}

    results = {}
    algorithm = list(sheet['Algorithm'])
    dag_true = list(sheet['DAGtrue'])
    directed = list(sheet['directedConstraints'])
    initial = list(sheet['initialGraphConstraints'])
    temporal = list(sheet['temporalConstraints'])
    result = list(sheet[metric if metric == 'BIC' else metric + '-CPDAG'])
    for i, algo in enumerate(algorithm):
        if not isinstance(algo, str):
            continue
        block = dag_true[i].split('_')[1].lower() + '_' + algo.lower()
        if block not in results:
            results[block] = {}
        if isinstance(directed[i], str):
            method = 'directed_' + directed[i].split('_')[2]
            results[block][method] = result[i]
        elif isinstance(initial[i], str):
            method = 'initial_' + initial[i].split('_')[2]
            results[block][method] = result[i]
        elif isinstance(temporal[i], str):
            method = 'temporal_' + temporal[i].split('_')[2]
            results[block][method] = result[i]
        else:
            results[block]['_none'] = result[i]

    # Transform to DataFrame

    df = DataFrame.from_dict(results, orient='index').sort_index()
    df = df[sorted(df.columns)]

    # Convert absolute values to percentage difference from no-constraint
    # value. BIC percentage uses constraint value in denominators, all others
    # no-constraint in denominator to agree with bar chart in original paper

    df_pct = df.copy()
    for col in df.columns:
        if col != '_none':
            if metric != 'BIC':
                df_pct[col] = (df[col] / df['_none'] - 1) * 100
            else:
                df_pct[col] = (df['_none'] / df[col] - 1) * 100
    df_pct.drop('_none', axis=1, inplace=True)
    print('\n\nMeans and counts for {}:\n{}'
          .format(metric, DataFrame([df_pct.mean()]).round(3)))
    print(DataFrame([df_pct.count()]))
    return df, df_pct


def stat_analysis(metric, abs, rel):
    """
        Perform the statistical analysis on GPT4 results

        :param str metric: metric being analysed
        :param DataFrame abs: absolute metrics
        :param DataFrane rel: relative percentage metrics
    """

    # Do normality tests

    print('\nNormality tests for absolute {} values:\n{}'
          .format(metric, normality(abs)))
    print('\nNormality tests for relative {} values:\n{}'
          .format(metric, normality(rel)))

    # Perform non-parametric paired significance tests between each knowledge
    # method and the no-knowledge result ("_none")

    p_values = {}
    for method in [c for c in abs.columns if c != '_none']:
        compare = abs[['_none', method]].dropna()
        _, p_values[method], test = robust_wilcoxon(compare['_none'],
                                                    compare[method], False)
        print('{}: {:.2e} ({})'.format(method, p_values[method], test))

    # List the p < 0.05 significant methods with different corrections

    for correction in (None, 'bonferroni', 'fdr_bh'):
        signif = correct_p_values(p_values=p_values, correction=correction)
        print('{} significant at p < 0.05 with {} correction: {}'
              .format(len(signif), correction, list(signif)))

    _, kruskal_p = kruskal(*[rel[m].dropna() for m in rel.columns])
    print(f"\nKruskal-Wallis Test p-value={kruskal_p:.6e}")

    rel_long = melt(rel.reset_index(), id_vars='index', var_name='method',
                    value_name='value').dropna()
    pairwise_p_values = posthoc_dunn(rel_long, group_col='method',
                                     val_col='value', p_adjust='fdr_bh')
    print("\nPairwise p-values (Dunn test):")
    print(pairwise_p_values)

    # List significant pairs at p < 0.05
    significant_pairs = []
    for method1 in pairwise_p_values.index:
        for method2 in pairwise_p_values.columns:
            # Ensure each pair is listed only once (method1 < method2)
            if (method1 < method2
                    and pairwise_p_values.loc[method1, method2] < 0.05):
                significant_pairs.append((method1, method2))

    print("\n{}/{} significant pairs at p < 0.05:"
          .format(len(significant_pairs),
                  len(rel.columns) * (len(rel.columns) - 1) / 2))
    for pair in significant_pairs:
        print(f"{pair[0]} vs {pair[1]}")


def values_gpt4_signif_f1():
    abs, rel = read_gpt4_results('F1')
    stat_analysis('F1', abs, rel)


def values_gpt4_signif_bsf():
    abs, rel = read_gpt4_results('BSF')
    stat_analysis('BSF', abs, rel)


def values_gpt4_signif_shd():
    abs, rel = read_gpt4_results('SHD')
    stat_analysis('SHD', abs, rel)


def values_gpt4_signif_bic():
    abs, rel = read_gpt4_results('BIC')
    stat_analysis('BIC', abs, rel)
