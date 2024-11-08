
# Generate the results and analysis used in Version 2 (1st resubmission) of the
# the Variable Ordering paper to Data Mining and Knowledge Discovery (DMKD)

from experiments.run_analysis import run_analysis


NETWORKS = ('asia,sports,sachs,child,' +
            'insurance,property,diarrhoea,water,' +
            'mildew,alarm,barley,hailfinder,' +
            'hepar2,win95pts,formed,pathfinder')


def figure_ord_hc_arbitrary_small():
    """
        Arbitrary changes for smaller networks
    """
    args = {'action': 'trace',
            'series': 'HC/STD',
            'networks': ('asia,sports,sachs,child,' +
                         'insurance,property,diarrhoea,water'),
            'N': '10k',
            'file': '/papers/var_ordering/ord_hc_arbitrary_small.png',
            'params': ('fig:ord_hc_arbitrary_small;' +
                       'arb:60')}
    run_analysis(args)


def figure_ord_hc_arbitrary_large():
    """
        Arbitrary changes for larger networks
    """
    args = {'action': 'trace',
            'series': 'HC/STD',
            'networks': ('mildew,alarm,barley,hailfinder,' +
                         'hepar2,win95pts,formed,pathfinder'),
            'N': '10k',
            'file': '/papers/var_ordering/ord_hc_arbitrary_large.png',
            'params': ('fig:ord_hc_arbitrary_large;' +
                       'arb:160')}
    run_analysis(args)


def figure_ord_hc_f1():
    """
        CPDAG F1 vs. sample size for different orderings
    """
    args = {'action': 'series',
            'series': ('BNLEARN/HC_OPT,' +
                       'BNLEARN/HC_BAD,' +
                       'BNLEARN/HC_STD'),
            'metrics': 'f1-e',
            'networks': NETWORKS,
            'N': '10-10m',
            'params': ('fig:ord_hc_f1;' +
                       'figure.title;' +
                       'subplot.aspect:1.05;'
                       'figure.subplots_left:0.04;' +
                       'figure.subplots_right:0.86;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.22')}
    run_analysis(args)


def figure_ord_hc_opt_edges():
    """
        Normalised edges vs. sample size for optimal ordering
    """
    args = {'action': 'metrics',
            'series': 'BNLEARN/HC_OPT',
            'metrics': 'a-ext,a-mis,e-ori,f1-e',
            'networks': NETWORKS,
            'N': '10-10m',
            'params': ('fig:ord_hc_opt_edges;' +
                       'figure.title;' +
                       'yaxis.label:Scaled edges or F1;' +
                       'legend.labels:{a-ext,extra\nedges,' +
                       'a-mis,missing\nedges,' +
                       'e-ori,misorientated\nedges,f1-e,F1};'
                       'subplot.aspect:1.05;' +
                       'figure.subplots_left:0.05;' +
                       'figure.subplots_right:0.84;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.22')}
    run_analysis(args)


def figure_ord_hc_worst_edges():
    """
        Normalised edges vs. sample size for worst ordering
    """
    args = {'action': 'metrics',
            'series': 'BNLEARN/HC_BAD',
            'metrics': 'a-ext,a-mis,e-ori,f1-e',
            'networks': NETWORKS,
            'N': '10-10m',
            'params': ('fig:ord_hc_worst_edges;' +
                       'figure.title;' +
                       'yaxis.label:Scaled edges or F1;' +
                       'legend.labels:{a-ext,extra\nedges,' +
                       'a-mis,missing\nedges,' +
                       'e-ori,misorientated\nedges,f1-e,F1};'
                       'subplot.aspect:1.05;' +
                       'figure.subplots_left:0.05;' +
                       'figure.subplots_right:0.84;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.22')}
    run_analysis(args)


def figure_ord_hc_reversed():
    """
        Normalised reversed arcs vs. sample size for different orderings
    """
    args = {'action': 'series',
            'series': ('BNLEARN/HC_OPT,' +
                       'BNLEARN/HC_BAD,' +
                       'BNLEARN/HC_STD'),
            'metrics': 'a-rev',
            'networks': NETWORKS,
            'N': '10-10m',
            'params': ('fig:ord_hc_reversed;' +
                       'figure.title;' +
                       'yaxis.label:Normalised reversed arcs;' +
                       'subplot.aspect:1.05;' +
                       'figure.subplots_left:0.04;' +
                       'figure.subplots_right:0.87;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.22')}
    run_analysis(args)


def figure_ord_hc_extra():
    """
        Normalised extra arcs vs. sample size for different orderings
    """
    args = {'action': 'series',
            'series': ('BNLEARN/HC_OPT,' +
                       'BNLEARN/HC_BAD,' +
                       'BNLEARN/HC_STD'),
            'metrics': 'a-ext',
            'networks': NETWORKS,
            'N': '10-10m',
            'params': ('fig:ord_hc_extra;' +
                       'figure.title;' +
                       'yaxis.label:Normalised extra arcs;' +
                       'subplot.aspect:1.05;' +
                       'figure.subplots_left:0.04;' +
                       'figure.subplots_right:0.87;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.22')}
    run_analysis(args)


def figure_ord_hc_missing():
    """
        Normalised extra arcs vs. sample size for different orderings
    """
    args = {'action': 'series',
            'series': ('BNLEARN/HC_OPT,' +
                       'BNLEARN/HC_BAD,' +
                       'BNLEARN/HC_STD'),
            'metrics': 'a-mis',
            'networks': NETWORKS,
            'N': '10-10m',
            'params': ('fig:ord_hc_missing;' +
                       'figure.title;' +
                       'yaxis.label:Normalised missing arcs;' +
                       'subplot.aspect:1.05;' +
                       'figure.subplots_left:0.04;' +
                       'figure.subplots_right:0.87;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.22')}
    run_analysis(args)


def figure_ord_hc_shd():
    """
        Normalised extra arcs vs. sample size for different orderings
    """
    args = {'action': 'series',
            'series': ('BNLEARN/HC_OPT,' +
                       'BNLEARN/HC_BAD,' +
                       'BNLEARN/HC_STD'),
            'metrics': 'shd-e',
            'networks': NETWORKS,
            'N': '10-10m',
            'params': ('fig:ord_hc_shd;' +
                       'figure.title;' +
                       'yaxis.label:CPDAG SHD;' +
                       'yaxis.range:¬;' +
                       'yaxis.shared:False;' +
                       'subplot.aspect:1.05;' +
                       'figure.subplots_left:0.05;' +
                       'figure.subplots_right:0.87;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_wspace:0.40;' +
                       'figure.subplots_hspace:0.22')}
    run_analysis(args)


def figure_ord_hc_score():
    """
        Score vs. sample size for different orderings
    """
    args = {'action': 'series',
            'series': ('BNLEARN/HC_OPT,' +
                       'BNLEARN/HC_BAD,' +
                       'BNLEARN/HC_STD'),
            'metrics': 'score',
            'networks': NETWORKS,
            'N': '10-10m',
            'params': ('fig:ord_hc_score;' +
                       'figure.title;' +
                       'subplot.aspect:1.05;'
                       'figure.subplots_left:0.06;' +
                       'figure.subplots_right:0.87;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.22;' +
                       'figure.subplots_wspace:0.40;' +
                       'yaxis.label:Normalised BIC Score;' +
                       'yaxis.range:¬')}
    run_analysis(args)


def figure_ord_hc_loglik_bic():
    """
        Log. Likelihood & BIC vs. sample size for STD orderng
    """
    args = {'action': 'metrics',
            'series': 'BNLEARN/HC_STD',
            'metrics': 'loglik,score',
            'networks': NETWORKS,
            'N': '10-10M',
            'params': ('fig:ord_hc_loglik_bic;' +
                       'figure.title;' +
                       'subplot.aspect:1.05;'
                       'figure.subplots_left:0.07;' +
                       'figure.subplots_right:0.87;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.22;' +
                       'figure.subplots_wspace:0.40;' +
                       'yaxis.label:Normalised score;' +
                       'legend.labels:{loglik,Log. Likelihood,score,BIC};' +
                       'yaxis.shared:False;' +
                       'yaxis.range:¬')}
    run_analysis(args)


def figure_ord_hc_loglik_score():
    """
        Log. Likelihood vs. sample size for different orderings
    """
    args = {'action': 'series',
            'series': ('BNLEARN/HC_OPT,' +
                       'BNLEARN/HC_BAD,' +
                       'BNLEARN/HC_STD'),
            'metrics': 'loglik',
            'networks': NETWORKS,
            'N': '10-10M',
            'params': ('fig:ord_hc_loglik;' +
                       'figure.title;' +
                       'subplot.aspect:1.20;'
                       'figure.subplots_left:0.07;' +
                       'figure.subplots_right:0.87;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_hspace:0.22;' +
                       'figure.subplots_wspace:0.45;' +
                       'yaxis.label:Normalised Log. Likelihood;' +
                       'yaxis.shared:False;' +
                       'yaxis.range:¬')}
    run_analysis(args)


def figure_ord_hc_impact():
    """
        Impact of ordering compared to sample size and hyperparameters
    """
    args = {'action': 'impact',
            'series': 'HC_IMPACT',
            'metrics': 'f1-e',
            'networks': NETWORKS,
            'N': '1k-1m',
            'params': ('fig:ord_hc_impact;' +
                       'subplot.aspect:2.0;'
                       'figure.subplots_top:0.98;' +
                       'figure.title')}
    run_analysis(args)


def figure_ord_hc_lowd_impact():
    """
        Impact of ordering compared to sample size and hyperparameters
        for low-dimensional cases
    """
    args = {'action': 'impact',
            'series': 'HC_IMPACT',
            'metrics': 'f1-e',
            'networks': NETWORKS,
            'N': '10-1k',
            'params': ('fig:ord_hc_impact_lowd;' +
                       'subplot.aspect:2.0;'
                       'figure.subplots_top:0.98;' +
                       'figure.title')}
    run_analysis(args)


def figure_ord_algo_impact():
    """
        Impact of ordering compared to sample size and hyperparameters for
        different algorithms
    """
    args = {'action': 'impact',
            'series': 'ALGO_IMPACT',
            'metrics': 'f1-e',
            'networks': NETWORKS,
            'N': '1k-1m',
            'params': ('fig:ord_algo_impact;' +
                       'subplot.aspect:1.7;'
                       'figure.subplots_left:0.08;' +
                       'figure.subplots_top:0.98;' +
                       'figure.subplots_right:0.82;' +
                       'figure.title')}
    run_analysis(args)


# These generate some of the values quoted in the text

def values_ord_density():
    """
        Sensitivity of graph density to ISS and K complexity hyperparameter
    """
    args = {'action': 'tree',
            'series': ('BNLEARN/HC_STD,' +
                       'BNLEARN/HC_K_5,' +
                       'BNLEARN/HC_ISS_5,' +
                       'BNLEARN/HC_BDS,' +
                       'BNLEARN/HC_BDE'),
            'networks': NETWORKS,
            'N': '1k-1m',
            'params': ('stats:dens,|E|,n')}
    run_analysis(args)
