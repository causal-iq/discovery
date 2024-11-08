
# Definitions which configure experiments, analysis and charts used in
# research output

from enum import Enum
from core.common import Randomise
from learn.knowledge import RuleSet


NETWORKS = ['cancer', 'asia', 'sports', 'sachs', 'covid',  # networks
            'child', 'insurance', 'property', 'diarrhoea', 'water',
            'mildew', 'alarm', 'barley', 'hailfinder', 'hailfinder2',
            'hepar2', 'win95pts', 'win95pts2', 'formed', 'pathfinder']


class Package(Enum):  # supported learning packages
    BNBENCH = {'arg': 'bnbench', 'name': 'BNBENCH'}
    BNLEARN = {'arg': 'bnlearn', 'name': 'BNLEARN'}
    TETRAD = {'arg': 'tetrad', 'name': 'TETRAD'}


class Algorithm(Enum):  # supported algorithms
    HC = {'method': 'hc', 'name': 'HC', 'label': 'HC', 'colour': '#c44e52'}
    TABU = {'method': 'tabu', 'name': 'TABU', 'label': 'TABU',
            'colour': '#dd8452'}
    H2PC = {'method': 'h2pc', 'name': 'H2PC', 'label': 'H2PC',
            'colour': '#ccb974'}
    MMHC = {'method': 'mmhc', 'name': 'MMHC', 'label': 'MMHC',
            'colour': '#937860'}
    PC = {'method': 'pc.stable', 'name': 'PC', 'label': 'PC-Stable',
          'colour': '#4c72b0'}
    GS = {'method': 'gs', 'name': 'GS', 'label': 'GS', 'colour': '#64b5cd'}
    IIAMB = {'method': 'inter.iamb', 'name': 'IIAMB', 'label': 'Inter-IAMB',
             'colour': '#da8bc3'}
    FGES = {'method': 'fges', 'name': 'FGES', 'label': 'FGES',
            'colour': '#aaaa00'}


class Ordering(Enum):  # types of node ordering supported
    STANDARD = {'name': 'alphabetic', 'colour': '#000000'}
    OPTIMAL = {'name': 'optimal', 'colour': '#66bd63'}
    WORST = {'name': 'worst', 'colour': '#d73027'}


# Series used in papers and for system testing

SERIES_P = {
    'BNLEARN/HC_STD': {},
    'BNLEARN/HC_OPT': {'ordering': Ordering.OPTIMAL},
    'BNLEARN/HC_BAD': {'ordering': Ordering.WORST},

    # BELOW ARE SERIES USED ONLY IN SYSTEM TESTS

    # These are used to check for name insensitivity in system test

    'TABU/NAM': {'package': Package.BNBENCH,
                 'params': {'tabu': 10, 'bnlearn': False},
                 'randomise': ([Randomise.NAMES], 2)},
    'BNLEARN/TABU_NAM': {'algorithm': Algorithm.TABU,
                         'randomise': ([Randomise.NAMES], 2)},
    'TETRAD/FGES_NAM': {'package': Package.TETRAD,
                        'algorithm': Algorithm.FGES,
                        'randomise': ([Randomise.NAMES], 2),
                        'params': {'score': 'bic', 'k': 1}},

    # These are used for integer limit knowledge system tests

    'HC/KS_L16': {'package': Package.BNBENCH, 'knowledge': RuleSet.EQUIV_ADD,
                  'kparams': {'limit': 16, 'ignore': 0, 'expertise': 1.0}},
    'HC/KS_R16_E100': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.REQD_ARC,
                       'kparams': {'reqd': 16, 'expertise': 1.0}},
    'HC/KW_L1': {'package': Package.BNBENCH,
                 'ordering': Ordering.WORST,
                 'knowledge': RuleSet.EQUIV_ADD,
                 'kparams': {'limit': 1, 'ignore': 0, 'expertise': 1.0}},
}


# Series Groups used in papers

SERIES_GROUPS_P = {

    # Used in Variable ordering paper

    'HC_BNLEARN': ['BNLEARN/HC_OPT',  # used for series plots in Figs 2
                   'BNLEARN/HC_BAD',
                   'BNLEARN/HC_STD'],

    'HC_IMPACT': ['BNLEARN/HC_OPT',
                  'BNLEARN/HC_BAD',
                  'BNLEARN/HC_STD',
                  'BNLEARN/HC_BDE',
                  'BNLEARN/HC_BDS',
                  'BNLEARN/HC_K_5',
                  'BNLEARN/HC_ISS_5'],

    'ALGO_IMPACT': ['BNLEARN/HC_OPT', 'BNLEARN/HC_BAD', 'BNLEARN/HC_STD',
                    'BNLEARN/HC_BDE', 'BNLEARN/HC_K_5',
                    'BNLEARN/TABU_OPT', 'BNLEARN/TABU_BAD', 'BNLEARN/TABU_STD',
                    'BNLEARN/TABU_BDE', 'BNLEARN/TABU_K_5',
                    'TETRAD/FGES_OPT', 'TETRAD/FGES_BAD', 'TETRAD/FGES_STD',
                    'BNLEARN/H2PC_OPT', 'BNLEARN/H2PC_BAD', 'BNLEARN/H2PC_STD',
                    'BNLEARN/H2PC_BDE', 'BNLEARN/H2PC_K_5',
                    'BNLEARN/MMHC_OPT', 'BNLEARN/MMHC_BAD', 'BNLEARN/MMHC_STD',
                    'BNLEARN/MMHC_BDE', 'BNLEARN/MMHC_K_5',
                    'BNLEARN/PC_OPT', 'BNLEARN/PC_BAD', 'BNLEARN/PC_STD',
                    'BNLEARN/PC_X2', 'BNLEARN/PC_ALPHA_01',
                    'BNLEARN/GS_OPT', 'BNLEARN/GS_BAD', 'BNLEARN/GS_STD',
                    'BNLEARN/GS_X2', 'BNLEARN/GS_ALPHA_01',
                    'BNLEARN/IIAMB_OPT', 'BNLEARN/IIAMB_BAD',
                    'BNLEARN/IIAMB_STD', 'BNLEARN/IIAMB_X2',
                    'BNLEARN/IIAMB_ALPHA_01'],
}


# Parameters for figures used in papers

FIGURE_PARAMS_P = {

    # Figures in Ordering paper

    'ord_hc_arbitrary_small': {  # Arbitrray changes for small networks
                               'figure.title': '',
                               'figure.subplots_left': 0.125,
                               'figure.subplots_right': 0.995,
                               'figure.subplots_top': 0.98,
                               'figure.subplots_bottom': 0.12,
                               'legend.fontsize': 15,
                               'line.sizes': [3, 3, 3, 3, 3, 3, 3, 3],
                               'legend.title_fontsize': 15,
                               'subplot.axes_fontsize': 15,
                               'xaxis.ticks_fontsize': 15,
                               'yaxis.ticks_fontsize': 15,
                               'legend.outside': False,
                               'subplot.aspect': 1.17},

    'ord_hc_arbitrary_large': {  # Arbitrray changes for large networks
                               'figure.title': '',
                               'figure.subplots_left': 0.125,
                               'figure.subplots_right': 0.995,
                               'figure.subplots_top': 0.98,
                               'figure.subplots_bottom': 0.12,
                               'legend.fontsize': 15,
                               'line.sizes': [3, 3, 3, 3, 3, 3, 3, 3],
                               'legend.title_fontsize': 15,
                               'subplot.axes_fontsize': 15,
                               'xaxis.ticks_fontsize': 15,
                               'yaxis.ticks_fontsize': 15,
                               'legend.outside': False,
                               'subplot.aspect': 1.60},

    'ord_hc_f1': {},  # Effect of ordering on CPDAG F1

    'tabu_stab_order_f1': {  # Effect of ordering on CPDAG F1
                           'line.dashes': {'TABU/BASE3': (1, 0),
                                           'TABU/STABLE3/DEC_SCORE': (2, 2),
                                           'TABU/STABLE3/INC_SCORE': (2, 2),
                                           'TABU/STABLE3/SCORE_PLUS': (1, 0)}
                          },

    'tabu_stab_order_score': {},  # Effect of ordering on score

    'ord_hc_reversed': {},  # Effect of ordering on reversed arcs

    'ord_hc_extra': {},  # Effect of ordering on extra arcs

    'ord_hc_missing': {},  # Effect of ordering on missing arcs

    'ord_hc_score': {  # Effect of ordering on normalised BIC score
                     'figure.subplots_right': 0.87,
                     'figure.subplots_left': 0.06,
                     'figure.subplots_wspace': 0.40,
                     'yaxis.range': None,
                     'yaxis.label': 'Normalised BIC Score',
                     'subplot.aspect': 1.1}

}
