
#   Common code which supports some common operations required by several
#   experiments but which is nevertheless not general enough to warrant
#   inclusion in the main code base.

from pandas import DataFrame
from enum import Enum
from itertools import chain
from numpy import nan
from math import floor, ceil
from re import compile

from conftest import EXPT_ARGS
from experiments.config import Ordering, SERIES_GROUPS_P, SERIES_P, \
    FIGURE_PARAMS_P, Package, Algorithm
from core.common import Randomise, ln
from core.bn import BN
from fileio.common import EXPTS_DIR
from learn.knowledge import RuleSet

INT = compile(r'^\-*\d+$')
FLOAT = compile(r'^\-*\d+\.\d+$')

# Types of experimental analysis supported

ANALYSIS = {'trace': 'Analyse and print traces in full',
            'check': 'Check consistency of traces between series',
            'metrics': 'Compare learnt graph metrics',
            'series': 'Compare learnt graphs from different series',
            'score': 'Plot scores of specified nodes',
            'score2': 'Comparing learnt and reference scores',
            'score3': 'CPS scores against parameter values',
            'impact': 'Impact of experimental factors e.g. node ordering',
            'bn': 'Detailed analysis of test networks',
            'network': 'Impact of network properties on accuracy',
            'error': 'Conditions with cause learning errors',
            'summary': 'Summary analysis'}

METRICS = {'f1': {'label': 'F1', 'size': 5, 'colour': '#000000',
                  'dashes': (1, 0)},
           'f1-e': {'label': 'F1 (CPDAG)', 'size': 3, 'colour': '#000000',
                    'dashes': (6, 3)},
           'a-ok': {'label': 'correct arcs', 'size': 3, 'colour': '#66bd63',
                    'dashes': (1, 0)},
           'p': {'label': 'precision', 'size': 3, 'colour': '#f46d43',
                 'dashes': (4, 1)},
           'r': {'label': 'recall', 'size': 3, 'colour': '#74add1',
                 'dashes': (4, 1)},
           'a-rev': {'label': 'reversed arcs', 'size': 3, 'colour': '#d73027',
                     'dashes': (1, 0)},
           'a-non': {'label': 'reversed (non-equiv)', 'size': 2,
                     'colour': '#d73027', 'dashes': (2, 1)},
           'a-eqv': {'label': 'reversed (equiv)', 'size': 2,
                     'colour': '#d73027', 'dashes': (1, 2)},
           'a-ext': {'label': 'extra arcs', 'size': 3, 'colour': '#f46d43',
                     'dashes': (1, 0)},
           'e-ori': {'label': "misoriented edges", 'size': 3,
                     'colour': '#d73027', 'dashes': (1, 0)},
           'a-mis': {'label': 'missing arcs', 'size': 3, 'colour': '#74add1',
                     'dashes': (1, 0)},
           'shd': {'label': 'SHD', 'size': 3, 'colour': '#74add1',
                   'dashes': (1, 0)},
           'shd-s': {'label': 'SHD (normalised)', 'size': 3,
                     'colour': '#d73027', 'dashes': (1, 0)},
           'shd-b': {'label': 'SHD (norm., bayesys)', 'size': 3,
                     'colour': '#d73027', 'dashes': (4, 1)},
           'shd-e': {'label': 'SHD (norm., CPDAG)', 'size': 3,
                     'colour': '#d73027', 'dashes': (4, 1)},
           'shd-es': {'label': 'SHD (norm., CPDAG)', 'size': 3,
                      'colour': '#d73027', 'dashes': (4, 1)},
           'f1-b': {'label': 'F1 (bayesys)', 'size': 3,
                    'colour': '#000000', 'dashes': (4, 1)},
           'bsf': {'label': 'BSF', 'size': 3,
                   'colour': '#21908d', 'dashes': (4, 1)},
           'bsf-e': {'label': 'BSF', 'size': 3,
                     'colour': '#21908d', 'dashes': (4, 1)},
           'score': {'label': 'score', 'size': 3, 'colour': '#FF0000',
                     'dashes': (1, 0)},
           'time': {'label': 'time', 'size': 3, 'colour': '#000000'},
           'n': {'label': '# variables', 'size': 3},
           '|A|': {'label': '# true edges'},
           '|E|': {'label': '# edges'},
           'loglik': {'label': 'loglik', 'size': 3, 'colour': '#0000FF',
                      'dashes': (1, 0)},
           'f1-e-std': {'label': 'S.D. of f1-e'},
           'score-std': {'label': 'S.D. of score'},
           'loglik-std': {'label': 'S.D. of loglik'},
           'p-e': {'label': 'precision'},
           'r-e': {'label': 'recall'},
           'expts': {'label': 'Number of experiments'},
           'dens': {'label': 'Learnt graph density'},
           'dens-std': {'label': 'Learnt graph density SD'},
           'nonex': {'label': 'Non-extendable PDAG'},
           'pretime': {'label': 'Preprocessing time'}}

METRIC_GROUPS = {'arcs': ['a-ok', 'a-ext', 'a-rev', 'a-non', 'a-eqv',
                          'a-mis', 'f1', 'f1-e'],
                 'prf1': ['p', 'r', 'f1'],
                 'bayesys': ['f1-b', 'shd-b', 'bsf', 'f1'],
                 'shdf1': ['shd-s', 'f1']}

R = '#c44e52'
B = '#4c72b0'
LB = '#64b5cd'
CB = '#046DDC'
G = '#55a868'
Y = '#ccb974'
AL = 'Active learning\n0.{} x n requests'
ALE = 'Active learning\nexpertise = {}'
AO = 'Active learning (orientation)\n0.{} x n requests'
AOE = 'Active learning (orientation)\nexpertise = {}'
MX = 'Predefined knowledge\n0.{} x n mixed arcs'
MXE = 'Mixed arcs\nexpertise = {}'
RQ = 'Predefined knowledge\n0.{} x n required arcs'
RQE = 'Required arcs\nexpertise = {}'
PH = 'Predefined knowledge\n0.{} x n prohibited arcs'
TR = 'Predefined knowledge\n0.{} x n nodes in tiers'
TRE = 'Predefined tiers\nexpertise = {}'
SS = 'Increase sample\nsize by {} times'
FIGURE_PARAMS = \
    {'al_limit': {'figure.subplots_right': 0.8,
                  'figure.subplots_wspace': 0.20},
     'ALEXF': {'subplot.kind': 'violin',
               'palette': [R, LB, B, R, LB, B, R, LB, B, R, LB, B, R, LB, B],
               'xaxis.tick_labels': (AO.format('0.50'), TR.format('0.50'),
                                     RQE.format('0.50'), ALE.format('0.67'),
                                     MXE.format('0.67'), RQE.format('0.67'),
                                     ALE.format('0.80'), MXE.format('0.80'),
                                     RQE.format('0.80'), ALE.format('0.90'),
                                     MXE.format('0.90'), RQE.format('0.90'),
                                     ALE.format('1.00'), MXE.format('1.00'),
                                     RQE.format('1.00')),
               'legend.key': {'Active learning': '#c44e52',
                              'Predefined mixed arcs': '#64b5cd',
                              'Predefined required arcs': '#4c72b0'},
               'legend.ncol': 3,
               'legend.loc': 'upper left',
               'legend.fontsize': 8,
               'xaxis.ticks_fontsize': 8,
               'xaxis.ticks_rotation': -50,
               'yaxis.label': 'F1 increase',
               'figure.dpi': 600,
               'violin.width': 0.9,
               'violin.scale': 'width'},
     'ALEXP': {'subplot.kind': 'violin',
               'palette': [R, B, R, B, R, B, R, B, R, B],
               'xaxis.tick_labels': (AOE.format('0.50'), TRE.format('0.50'),
                                     AOE.format('0.67'), TRE.format('0.67'),
                                     AOE.format('0.80'), TRE.format('0.80'),
                                     AOE.format('0.90'), TRE.format('0.90'),
                                     AOE.format('1.00'), TRE.format('1.00')),
               'legend.key': {'Active learning (orientation)': '#c44e52',
                              'Predefined tiers': '#4c72b0'},
               'legend.ncol': 2,
               'legend.loc': 'upper left',
               'legend.fontsize': 8,
               'xaxis.ticks_fontsize': 8,
               'xaxis.ticks_rotation': -50,
               'yaxis.label': 'F1 increase',
               'figure.dpi': 600,
               'violin.width': 0.9,
               'violin.scale': 'width'},
     'XAIP': {'subplot.kind': 'violin',
              'palette': [R, B, LB, G],
              'xaxis.tick_labels': (AL.format('0.50'), TR.format('0.50'),
                                    PH.format('0.50'), SS.format('100')),
              'legend.key': {'Active learning': R,
                             'Predefined tiers': B,
                             'Predefined prohibited arcs': LB,
                             'Increase sample size by 100x': G},
              'legend.ncol': 2,
              'legend.loc': 'upper right',
              'legend.fontsize': 12,
              'xaxis.ticks_fontsize': 8,
              'xaxis.ticks_rotation': -50,
              'yaxis.label': 'F1 increase',
              'figure.dpi': 600,
              'violin.width': 0.9,
              'violin.scale': 'width',
              'violin.fontsize': 12},
     'XAIE': {'subplot.kind': 'violin',
              'palette': [R, B, R, B, R, B],
              'xaxis.tick_labels': ('67%\ncorrect', ' 67%\ncorrect ',
                                    '80%\ncorrect', ' 80%\ncorrect ',
                                    '100%\ncorrect', ' 100%\ncorrect '),
              'legend.key': {'Active learning': R,
                             'Predefined tiers': B},
              'legend.ncol': 2,
              'legend.loc': 'upper right',
              'legend.fontsize': 12,
              'xaxis.ticks_fontsize': 12,
              'xaxis.ticks_rotation': 0,
              'xaxis.ticks_halign': 'center',
              'yaxis.label': 'F1 increase',
              'figure.dpi': 600,
              'violin.width': 0.9,
              'violin.scale': 'width',
              'violin.fontsize': 12},
     'XAIL': {'subplot.kind': 'violin',
              'palette': [R, R, R, R],
              'xaxis.tick_labels': ('0.125 x n', '0.250 x n',
                                    '0.500 x n', 'Unlimited'),
              'legend.key': {'Active learning': R},
              'legend.ncol': 1,
              'legend.loc': 'upper left',
              'legend.fontsize': 12,
              'xaxis.ticks_fontsize': 12,
              'xaxis.ticks_rotation': 0,
              'xaxis.ticks_halign': 'center',
              'yaxis.label': 'F1 increase',
              'figure.dpi': 600,
              'violin.width': 0.9,
              'violin.scale': 'width',
              'violin.fontsize': 12},
     'act_algo': {'subplot.kind': 'violin',
                  'xaxis.label': '',
                  'figure.subplots_bottom': 0.18,
                  'figure.subplots_left': 0.10,
                  'figure.subplots_right': 0.90,
                  'violin.scale': 'width',
                  'violin.width': 1.0,
                  'xaxis.tick_labels': ['Tabu', 'FGES', 'PC-Stable',
                                        'Tabu AL - 0.8 expertise',
                                        'Tabu AL - 1.0 expertise'],
                  'xaxis.ticks_rotation': -60,
                  'xaxis.ticks_halign': 'left'}}
FIGURE_PARAMS.update(FIGURE_PARAMS_P)


SERIES_DEFAULTS = {'package': Package.BNLEARN,
                   'algorithm': Algorithm.HC,
                   'datagen': 'v1',
                   'ordering': Ordering.STANDARD,
                   'params': {'score': 'bic', 'base': 'e', 'k': 1},
                   'knowledge': False,
                   'kparams': None,
                   'randomise': False,
                   }

SERIES_COMPARATORS = ('package', 'algorithm', 'ordering', 'score', 'test', 'k',
                      'iss', 'alpha', 'knowledge', 'limit', 'ignore',
                      'expertise', 'reqd', 'threshold', 'earlyok', 'partial',
                      'stop', 'nodes')

SERIES_GROUPS = {'HC_BNBENCH': ['HC/OPT', 'HC/BAD', 'HC/STD'],
                 'HC_TABU': ['BNLEARN/TABU_STD', 'BNLEARN/HC_STD'],
                 'CHECK_TABU': ['BNLEARN/TABU_STD', 'HC/STD', 'TABU/STD'],
                 'GROW': ['HC/GROW_OPT', 'HC/GROW', 'HC/GROW_BAD'],
                 'GROW_STD': ['HC/OPT', 'HC/STD', 'HC/BAD', 'HC/GROW'],
                 'GROW_BAD': ['HC/OPT', 'HC/STD', 'HC/BAD', 'HC/GROW_BAD'],
                 'GROW_OPT': ['HC/OPT', 'HC/STD', 'HC/BAD', 'HC/GROW_OPT'],
                 'ORDER': ['HC/ORDER', 'HC/ORDER_KL4', 'HC/ORDER_KL16'],
                 'KS_IMPACT': ['BNLEARN/HC_STD', 'BNLEARN/HC_BDE',
                               'BNLEARN/HC_BDS', 'BNLEARN/HC_K_5',
                               'BNLEARN/HC_ISS_5'],
                 'TABU_IMPACT': ['BNLEARN/TABU_OPT', 'BNLEARN/TABU_BAD',
                                 'BNLEARN/TABU_STD', 'BNLEARN/TABU_BDE',
                                 'BNLEARN/TABU_BDS', 'BNLEARN/TABU_K_5',
                                 'BNLEARN/TABU_ISS_5'],
                 'H2PC_IMPACT': ['BNLEARN/H2PC_OPT', 'BNLEARN/H2PC_BAD',
                                 'BNLEARN/H2PC_STD', 'BNLEARN/H2PC_BDE',
                                 'BNLEARN/H2PC_BDS', 'BNLEARN/H2PC_K_5',
                                 'BNLEARN/H2PC_ISS_5'],
                 'MMHC_IMPACT': ['BNLEARN/MMHC_OPT', 'BNLEARN/MMHC_BAD',
                                 'BNLEARN/MMHC_STD', 'BNLEARN/MMHC_BDE',
                                 'BNLEARN/MMHC_BDS', 'BNLEARN/MMHC_K_5',
                                 'BNLEARN/MMHC_ISS_5'],
                 'PC_IMPACT': ['BNLEARN/PC_OPT', 'BNLEARN/PC_BAD',
                               'BNLEARN/PC_STD', 'BNLEARN/PC_ALPHA_01',
                               'BNLEARN/PC_X2'],
                 'GS_IMPACT': ['BNLEARN/GS_OPT', 'BNLEARN/GS_BAD',
                               'BNLEARN/GS_STD', 'BNLEARN/GS_ALPHA_01',
                               'BNLEARN/GS_X2'],
                 'IIAMB_IMPACT': ['BNLEARN/IIAMB_OPT', 'BNLEARN/IIAMB_BAD',
                                  'BNLEARN/IIAMB_STD',
                                  'BNLEARN/IIAMB_ALPHA_01',
                                  'BNLEARN/IIAMB_X2'],
                 'HC_CHECK': ['BNLEARN/HC_OPT', 'HC/OPT', 'BNLEARN/HC_BAD',
                              'HC/BAD', 'BNLEARN/HC_STD', 'HC/STD'],
                 'ALGO_STD': ['BNLEARN/HC_STD',  'BNLEARN/TABU_STD',
                              'TETRAD/FGES_STD',
                              'BNLEARN/H2PC_STD', 'BNLEARN/MMHC_STD',
                              'BNLEARN/PC_STD', 'BNLEARN/GS_STD',
                              'BNLEARN/IIAMB_STD'],
                 'ALGO_OPT': ['BNLEARN/HC_OPT', 'BNLEARN/TABU_OPT',
                              'TETRAD/FGES_OPT',
                              'BNLEARN/H2PC_OPT', 'BNLEARN/MMHC_OPT',
                              'BNLEARN/PC_OPT', 'BNLEARN/GS_OPT',
                              'BNLEARN/IIAMB_OPT'],
                 'ALGO_BAD': ['BNLEARN/HC_BAD', 'BNLEARN/TABU_BAD',
                              'TETRAD/FGES_BAD',
                              'BNLEARN/H2PC_BAD', 'BNLEARN/MMHC_BAD',
                              'BNLEARN/PC_BAD', 'BNLEARN/GS_BAD',
                              'BNLEARN/IIAMB_BAD'],
                 'ALGO_ORD': ['BNLEARN/HC_OPT', 'BNLEARN/TABU_OPT',
                              'BNLEARN/H2PC_OPT', 'BNLEARN/MMHC_OPT',
                              'BNLEARN/PC_OPT', 'BNLEARN/GS_OPT',
                              'BNLEARN/IIAMB_OPT',
                              'BNLEARN/HC_BAD', 'BNLEARN/TABU_BAD',
                              'BNLEARN/H2PC_BAD', 'BNLEARN/MMHC_BAD',
                              'BNLEARN/PC_BAD', 'BNLEARN/GS_BAD',
                              'BNLEARN/IIAMB_BAD'],
                 'SCHY': ['BNLEARN/HC_STD', 'BNLEARN/TABU_STD',
                          'BNLEARN/H2PC_STD', 'BNLEARN/MMHC_STD',
                          'BNLEARN/HC_OPT', 'BNLEARN/TABU_OPT',
                          'BNLEARN/H2PC_OPT', 'BNLEARN/MMHC_OPT',
                          'BNLEARN/HC_BAD', 'BNLEARN/TABU_BAD',
                          'BNLEARN/H2PC_BAD', 'BNLEARN/MMHC_BAD'],
                 'FGES': ['TETRAD/FGES_OPT', 'TETRAD/FGES_STD',
                          'TETRAD/FGES_BAD'],
                 'FGES_HC': ['HC/STD', 'BNLEARN/TABU_STD', 'TETRAD/FGES_STD'],
                 'IIAMB': ['BNLEARN/IIAMB_OPT', 'BNLEARN/IIAMB_BAD',
                           'BNLEARN/IIAMB_STD'],
                 'GS': ['BNLEARN/GS_OPT', 'BNLEARN/GS_BAD', 'BNLEARN/GS_STD'],
                 'HC': ['BNLEARN/HC_OPT', 'BNLEARN/HC_BAD', 'BNLEARN/HC_STD'],
                 'PC': ['BNLEARN/PC_OPT', 'BNLEARN/PC_BAD', 'BNLEARN/PC_STD'],
                 'TABU': ['BNLEARN/TABU_OPT', 'BNLEARN/TABU_BAD',
                          'BNLEARN/TABU_STD'],
                 'MMHC': ['BNLEARN/MMHC_OPT', 'BNLEARN/MMHC_BAD',
                          'BNLEARN/MMHC_STD'],
                 'H2PC': ['BNLEARN/H2PC_OPT', 'BNLEARN/H2PC_BAD',
                          'BNLEARN/H2PC_STD'],
                 'ALGO_KNW': ['TABU/BASE', 'TETRAD/FGES_STD',
                              'BNLEARN/PC/BASE', 'TABU/EQVP/L050_E80',
                              'TABU/EQVP/L050'],
                 'TEXP': ['TABU/BASE', 'TABU/EQV/L050', 'TABU/EQV/L050_E67'],
                 'IGN': ['HC/ORDER/KL16', 'HC/ORDER/KLI16', 'HC/ORDER/BASE'],
                 'TRIG': ['TABU/BASE', 'TABU/LODL/L050', 'TABU/LT5/L050',
                          'TABU/STAB/L050', 'TABU/EQV/L050',
                          'TABU/EQVLT5/L050', 'TABU/POS/L050'],
                 'LIM_OLD': ['HC/ORDER/BASE', 'HC/ORDER/KL4', 'HC/ORDER/KL16',
                             'HC/ORDER/KL0'],
                 'LIM': ['TABU/BASE', 'TABU/EQV/L012', 'TABU/EQV/L025',
                         'TABU/EQV/L050', 'TABU/EQV/E100'],
                 'LIMP': ['TABU/BASE', 'TABU/EQVP/L012', 'TABU/EQVP/L025',
                          'TABU/EQVP/L050', 'TABU/EQVP/E100'],
                 'LIMR': ['TABU/BASE', 'TABU/EQV/L012', 'TABU/REQD/L012',
                          'TABU/EQV/L025', 'TABU/REQD/L025', 'TABU/EQV/L050',
                          'TABU/REQD/L050', 'TABU/EQV/E100'],
                 'LIMF': ['TABU/BASE',
                          'TABU/EQV/L012', 'TABU/MIX9/L012', 'TABU/REQD/L012',
                          'TABU/EQV/L025', 'TABU/MIX9/L025', 'TABU/REQD/L025',
                          'TABU/EQV/L050', 'TABU/MIX9/L050', 'TABU/REQD/L050',
                          'TABU/EQV/E100'],
                 'LIMH': ['HC/BASE', 'HC/EQV/L025', 'HC/EQV/L050',
                          'HC/REQD/L025', 'HC/REQD/L050'],
                 'EXP_OLD': ['HC/ORDER/BASE', 'HC/ORDER/KE50', 'HC/ORDER/KE67',
                             'HC/ORDER/KE80', 'HC/ORDER/KL0'],
                 # 'EXP': ['TABU/BASE','TABU/EQV/L050_E50','TABU/EQV/L050_E67',
                 #         'TABU/EQV/L050_E80', 'TABU/EQV/L050_E90',
                 #         'TABU/EQV/L050'],
                 'EXPR': ['TABU/BASE',
                          'TABU/EQV/L050_E50', 'TABU/REQD/L050_E50',
                          'TABU/EQV/L050_E67', 'TABU/REQD/L050_E67',
                          'TABU/EQV/L050_E80', 'TABU/REQD/L050_E80',
                          'TABU/EQV/L050_E90', 'TABU/REQD/L050_E90',
                          'TABU/EQV/L050', 'TABU/REQD/L050'],
                 'EXP25': ['TABU/BASE', 'TABU/EQV/L025_E67', 'TABU/EQV/L025',
                           'TABU/REQD/L025_E67', 'TABU/REQD/L025'],
                 'EXPH': ['HC/BASE', 'HC/EQV/L050_E50', 'HC/EQV/L050_E67',
                          'HC/EQV/L050_E80', 'HC/EQV/L050'],
                 'COMBO': ['HC/BASE', 'HC/EQV/L050', 'HC/LT5/L050',
                           'HC/EQVLT5/L050'],
                 'STAB': ['TABU/BASE', 'TABU/STAB/L050_T20', 'TABU/STAB/L050',
                          'TABU/STAB/L050_T01', 'TABU/STAB/L050_T001',
                          'TABU/EQV/L050'],
                 'STAB12': ['TABU/BASE', 'TABU/STAB/L012_T20',
                            'TABU/STAB/L012', 'TABU/STAB/L012_T01',
                            'TABU/STAB/L012_T001', 'TABU/EQV/L050'],
                 'LODL': ['TABU/BASE', 'TABU/LODL/L050_T20', 'TABU/LODL/L050',
                          'TABU/LODL/L050_T01', 'TABU/LODL/L050_T001',
                          'TABU/EQV/L050'],
                 'LT5': ['TABU/BASE', 'TABU/LT5/L050_T20', 'TABU/LT5/L050',
                         'TABU/LT5/L050_T01', 'TABU/LT5/L050_T001',
                         'TABU/EQV/L050'],
                 'PARTIAL': ['TABU/BASE', 'TABU/EQV/L050',
                             'TABU/EQV/L050_PAR'],
                 'AL': ['TABU/BASE', 'TABU/STOP/L050', 'TABU/MIX/R05S45',
                        'TABU/MIX/R25S25', 'TABU/REQD/L050', 'TABU/EQV/L050',
                        'TABU/TIERS/L050'],
                 'ALI': ['TABU/BASE', 'TABU/STOP/L050', 'TABU/MIX/R05S45',
                         'TABU/MIX/R25S25', 'TABU/REQD/L050', 'TABU/EQV/L050',
                         'TABU/EQV/L050_PAR', 'TABU/TIERS/L050'],
                 'R1': ['TABU/TIERS/L050_E50', 'TABU/TIERS/L050_E67'],
                 'R2': ['TABU/TIERS/L050_E80', 'TABU/TIERS/L050_E90'],

                 'TB': ['TABU/BASE'],
                 'L50': ['TABU/REQD/L050', 'TABU/MIX9/L050',
                         'TABU/EQV/L050', 'TABU/TIERS/L050',
                         'TABU/STOP/L050', 'TABU/EQVP/L050'],
                 'EQVL': ['TABU/EQV/L012', 'TABU/EQV/L025',
                          'TABU/EQV/L050', 'TABU/EQV/E100'],
                 'EQVPL':  ['TABU/EQVP/L012', 'TABU/EQVP/L025',
                            'TABU/EQVP/L050', 'TABU/EQVP/E100'],
                 'MIX9L': ['TABU/MIX9/L012', 'TABU/MIX9/L025',
                           'TABU/MIX9/L050'],
                 'REQDL': ['TABU/REQD/L012', 'TABU/REQD/L025',
                           'TABU/REQD/L050'],
                 'STOPL': ['TABU/STOP/L012', 'TABU/STOP/L025',
                           'TABU/STOP/L050'],
                 'TIERSL': ['TABU/TIERS/L012', 'TABU/TIERS/L025',
                            'TABU/TIERS/L050'],
                 'EQVE': ['TABU/EQV/L050_E50', 'TABU/EQV/L050_E67',
                          'TABU/EQV/L050_E80', 'TABU/EQV/L050_E90',
                          'TABU/EQV/L050'],
                 'EQVPE': ['TABU/EQVP/L050_E50', 'TABU/EQVP/L050_E67',
                           'TABU/EQVP/L050_E80', 'TABU/EQVP/L050_E90',
                           'TABU/EQVP/L050'],
                 'REQDE': ['TABU/REQD/L050_E50', 'TABU/REQD/L050_E67',
                           'TABU/REQD/L050_E80', 'TABU/REQD/L050_E90',
                           'TABU/REQD/L050'],
                 'TIERSE': ['TABU/TIERS/L050_E50', 'TABU/TIERS/L050_E67',
                            'TABU/TIERS/L050_E80', 'TABU/TIERS/L050_E90',
                            'TABU/TIERS/L050'],
                 'MIX9E': ['TABU/MIX9/L050_E50', 'TABU/MIX9/L050_E67',
                           'TABU/MIX9/L050_E80', 'TABU/MIX9/L050_E90'],

                 'ALEXF': ['TABU/BASE',  # Active Learning/predefined comp
                           'TABU/EQV/L050_E50', 'TABU/MIX9/L050_E50',
                           'TABU/REQD/L050_E50', 'TABU/EQV/L050_E67',
                           'TABU/MIX9/L050_E67', 'TABU/REQD/L050_E67',
                           'TABU/EQV/L050_E80', 'TABU/MIX9/L050_E80',
                           'TABU/REQD/L050_E80', 'TABU/EQV/L050_E90',
                           'TABU/MIX9/L050_E90', 'TABU/REQD/L050_E90',
                           'TABU/EQV/L050', 'TABU/MIX9/L050',
                           'TABU/REQD/L050'],

                 'ALEXP': ['TABU/BASE',  # Active Learning/predefined comp
                           'TABU/EQVP/L050_E50', 'TABU/TIERS/L050_E50',
                           'TABU/EQVP/L050_E67', 'TABU/TIERS/L050_E67',
                           'TABU/EQVP/L050_E80', 'TABU/TIERS/L050_E80',
                           'TABU/EQVP/L050_E90', 'TABU/TIERS/L050_E90',
                           'TABU/EQVP/L050', 'TABU/TIERS/L050'],

                 'XAIP': ['TABU/BASE', 'TABU/EQVP/L050',  # for XAI poster
                          'TABU/TIERS/L050', 'TABU/STOP/L050'],

                 'XAIE': ['TABU/BASE',  # for XAI poster expertise plot
                          'TABU/EQVP/L050_E67', 'TABU/TIERS/L050_E67',
                          'TABU/EQVP/L050_E80', 'TABU/TIERS/L050_E80',
                          'TABU/EQVP/L050', 'TABU/TIERS/L050'],

                 'XAIL': ['TABU/BASE',  # for XAI poster limit plot
                          'TABU/EQVP/L012', 'TABU/EQVP/L025',
                          'TABU/EQVP/L050', 'TABU/EQVP/E100']}
SERIES_GROUPS.update(SERIES_GROUPS_P)


SERIES = {

    # Series for Knowledge Paper

    'HC/BASE': {'package': Package.BNBENCH,
                'randomise': ([Randomise.ORDER], 10)},
    'HC/BASE2': {'package': Package.BNBENCH,
                 'randomise': ([Randomise.ORDER, Randomise.NAMES], 25)},
    'HC/BASE3': {'package': Package.BNBENCH,
                 'randomise': ([Randomise.ORDER, Randomise.NAMES,
                                Randomise.ROWS], 25)},
    'HC/SCORE/EMPTY': {'package': Package.BNBENCH},
    'HC/SCORE/REF': {'package': Package.BNBENCH},
    'HC/SCORE/HCREF': {'package': Package.BNBENCH},
    'HC/GREEDY_SP': {'package': Package.BNBENCH,
                     'params': {'stable': 'score+', 'tree': (1, 1, 100)},
                     'randomise': ([Randomise.ORDER, Randomise.NAMES], 3)},
    'HC/STABLE/DEC_SCORE': {'package': Package.BNBENCH,
                            'params': {'stable': 'dec_score'},
                            'randomise': ([Randomise.ORDER,
                                           Randomise.NAMES], 3)},
    'HC/STABLE/INC_SCORE': {'package': Package.BNBENCH,
                            'params': {'stable': 'inc_score'},
                            'randomise': ([Randomise.ORDER,
                                           Randomise.NAMES], 3)},
    'HC/STABLE/SCORE': {'package': Package.BNBENCH,
                        'params': {'stable': 'score'},
                        'randomise': ([Randomise.ORDER,
                                       Randomise.NAMES], 3)},
    'HC/STABLE/SCORE_PLUS': {'package': Package.BNBENCH,
                             'params': {'stable': 'score+'},
                             'randomise': ([Randomise.ORDER,
                                            Randomise.NAMES], 3)},
    'HC/STABLE3/SCORE_PLUS': {'package': Package.BNBENCH,
                              'params': {'stable': 'score+'},
                              'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                             Randomise.NAMES], 25)},
    'HC/STABLE3/BDEU_BASE': {'package': Package.BNBENCH,
                             'params': {'score': 'bde', 'iss': 1},
                             'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                            Randomise.NAMES], 25)},
    'HC/STABLE3/BDEU_PLUS': {'package': Package.BNBENCH,
                             'params': {'score': 'bde', 'iss': 1,
                                        'stable': 'score+'},
                             'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                            Randomise.NAMES], 25)},
    'HC/SAMPLE/STD': {'package': Package.BNBENCH,
                      'randomise': ([Randomise.SAMPLE], 25)},
    'HC/SAMPLE/BASE': {'package': Package.BNBENCH,
                       'randomise': ([Randomise.SAMPLE, Randomise.ROWS,
                                      Randomise.ORDER, Randomise.NAMES],
                                     25)},
    'HC/SAMPLE/STABLE': {'package': Package.BNBENCH,
                         'params': {'stable': 'score+'},
                         'randomise': ([Randomise.SAMPLE, Randomise.ROWS,
                                        Randomise.ORDER, Randomise.NAMES],
                                       25)},
    'HC/SAMPLE/SC4': {'package': Package.BNBENCH,
                      'params': {'stable': 'sc4+'},
                      'randomise': ([Randomise.SAMPLE, Randomise.ROWS,
                                     Randomise.ORDER, Randomise.NAMES],
                                    25)},
    'HC/STABLE/ORDER': {'package': Package.BNBENCH,
                        'params': {'stable': 'order'},
                        'randomise': ([Randomise.ORDER,
                                       Randomise.NAMES], 3)},
    'HC/REQD/L025': {'package': Package.BNBENCH, 'knowledge': RuleSet.REQD_ARC,
                     'kparams': {'reqd': 0.25, 'expertise': 1.0},
                     'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                   10)},
    'HC/REQD/L025_E50': {'package': Package.BNBENCH,
                         'knowledge': RuleSet.REQD_ARC,
                         'kparams': {'reqd': 0.25, 'expertise': 0.5},
                         'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                       3)},

    'HC/REQD/L050': {'package': Package.BNBENCH, 'knowledge': RuleSet.REQD_ARC,
                     'kparams': {'reqd': 0.50, 'expertise': 1.0},
                     'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                   10)},
    'HC/REQD/L050_E80': {'package': Package.BNBENCH,
                         'knowledge': RuleSet.REQD_ARC,
                         'kparams': {'reqd': 0.50, 'expertise': 0.8},
                         'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                       10)},
    'HC/REQD/L050_E67': {'package': Package.BNBENCH,
                         'knowledge': RuleSet.REQD_ARC,
                         'kparams': {'reqd': 0.50, 'expertise': 0.67},
                         'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                       10)},
    'HC/REQD/L050_E50': {'package': Package.BNBENCH,
                         'knowledge': RuleSet.REQD_ARC,
                         'kparams': {'reqd': 0.50, 'expertise': 0.50},
                         'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                       10)},
    'HC/REQD/L4': {'package': Package.BNBENCH, 'knowledge': RuleSet.REQD_ARC,
                   'kparams': {'reqd': 4, 'expertise': 1.0},
                   'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE], 10)},
    'HC/REQD/L16': {'package': Package.BNBENCH, 'knowledge': RuleSet.REQD_ARC,
                    'kparams': {'reqd': 16, 'expertise': 1.0},
                    'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE], 10)},

    'HC/EQV/L025': {'package': Package.BNBENCH, 'knowledge': RuleSet.EQUIV_ADD,
                    'kparams': {'limit': 0.25, 'ignore': 0, 'expertise': 1.0,
                                'partial': True},
                    'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE], 10)},
    'HC/EQV/L025_E50': {'package': Package.BNBENCH,
                        'knowledge': RuleSet.EQUIV_ADD,
                        'kparams': {'limit': 0.25, 'ignore': 0,
                                    'expertise': 0.5, 'partial': True},
                        'randomise': ([Randomise.ORDER,
                                       Randomise.KNOWLEDGE], 3)},
    'HC/EQV/L050': {'package': Package.BNBENCH, 'knowledge': RuleSet.EQUIV_ADD,
                    'kparams': {'limit': 0.50, 'ignore': 0, 'expertise': 1.0,
                                'partial': True},
                    'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE], 10)},
    'HC/EQV/L050_E80': {'package': Package.BNBENCH,
                        'knowledge': RuleSet.EQUIV_ADD,
                        'kparams': {'limit': 0.50, 'ignore': 0,
                                    'expertise': 0.8, 'partial': True},
                        'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                      10)},
    'HC/EQV/L050_E67': {'package': Package.BNBENCH,
                        'knowledge': RuleSet.EQUIV_ADD,
                        'kparams': {'limit': 0.50, 'ignore': 0,
                                    'expertise': 0.67, 'partial': True},
                        'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                      10)},
    'HC/EQV/L050_E50': {'package': Package.BNBENCH,
                        'knowledge': RuleSet.EQUIV_ADD,
                        'kparams': {'limit': 0.50, 'ignore': 0,
                                    'expertise': 0.50, 'partial': True},
                        'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                      10)},

    'HC/POS/L050': {'package': Package.BNBENCH, 'knowledge': RuleSet.POS_DELTA,
                    'kparams': {'limit': 0.50, 'ignore': 0, 'expertise': 1.0},
                    'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE], 3)},
    'HC/LT5/L050': {'package': Package.BNBENCH, 'knowledge': RuleSet.HI_LT5,
                    'kparams': {'limit': 0.50, 'ignore': 0, 'expertise': 1.0},
                    'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE], 3)},
    'HC/STAB/L050': {'package': Package.BNBENCH,
                     'knowledge': RuleSet.BIC_UNSTABLE,
                     'kparams': {'limit': 0.50, 'ignore': 0, 'expertise': 1.0},
                     'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE], 3)},

    'HC/EQVLT5/L050': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.EQUIV_LT5,
                       'kparams': {'limit': 0.50, 'ignore': 0,
                                   'expertise': 1.0, 'partial': True},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     3)},

    'TABU/BASE': {'package': Package.BNBENCH,
                  'params': {'tabu': 10, 'bnlearn': False},
                  'randomise': ([Randomise.ORDER], 10)},
    'TABU/BASE2': {'package': Package.BNBENCH, 'params': {'tabu': 10},
                   'randomise': ([Randomise.ORDER, Randomise.NAMES], 25)},
    'TABU/BASE3': {'package': Package.BNBENCH, 'params': {'tabu': 10},
                   'randomise': ([Randomise.ORDER, Randomise.NAMES,
                                  Randomise.ROWS], 25)},
    'TABU/OPT': {'package': Package.BNBENCH, 'ordering': Ordering.OPTIMAL,
                 'params': {'tabu': 10}},
    'TABU/BAD': {'package': Package.BNBENCH, 'ordering': Ordering.WORST,
                 'params': {'tabu': 10}},
    'TABU/SAMPLE/STD': {'package': Package.BNBENCH, 'params': {'tabu': 10},
                        'randomise': ([Randomise.SAMPLE], 25)},
    'TABU/SAMPLE/BASE': {'package': Package.BNBENCH, 'params': {'tabu': 10},
                         'randomise': ([Randomise.SAMPLE, Randomise.ROWS,
                                        Randomise.ORDER, Randomise.NAMES],
                                       25)},
    'TABU/SAMPLE/STABLE': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'stable': 'score+'},
                           'randomise': ([Randomise.SAMPLE, Randomise.ROWS,
                                          Randomise.ORDER, Randomise.NAMES],
                                         25)},
    'TABU/SAMPLE/SC4': {'package': Package.BNBENCH,
                        'params': {'tabu': 10, 'stable': 'sc4+'},
                        'randomise': ([Randomise.SAMPLE, Randomise.ROWS,
                                       Randomise.ORDER, Randomise.NAMES],
                                      25)},
    'TABU/STABLE': {'package': Package.BNBENCH,
                    'params': {'tabu': 10, 'bnlearn': False, 'stable': True},
                    'randomise': ([Randomise.ORDER, Randomise.NAMES], 3)},
    'TABU/STABLE/DEC_SCORE': {'package': Package.BNBENCH,
                              'params': {'tabu': 10, 'stable': 'dec_score'},
                              'randomise': ([Randomise.ORDER,
                                             Randomise.NAMES], 25)},
    'TABU/STABLE/INC_SCORE': {'package': Package.BNBENCH,
                              'params': {'tabu': 10, 'stable': 'inc_score'},
                              'randomise': ([Randomise.ORDER,
                                             Randomise.NAMES], 25)},
    'TABU/STABLE/SCORE': {'package': Package.BNBENCH,
                          'params': {'tabu': 10, 'stable': 'score'},
                          'randomise': ([Randomise.ORDER,
                                         Randomise.NAMES], 25)},
    'TABU/STABLE/SCORE_PLUS': {'package': Package.BNBENCH,
                               'params': {'tabu': 10, 'stable': 'score+'},
                               'randomise': ([Randomise.ORDER,
                                              Randomise.NAMES], 25)},
    'TABU/STABLE3/DEC_SCORE': {'package': Package.BNBENCH,
                               'params': {'tabu': 10, 'stable': 'dec_score'},
                               'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                              Randomise.NAMES], 25)},
    'TABU/STABLE3/DEC_1': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'stable': 'dec_1'},
                           'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                          Randomise.NAMES], 25)},
    'TABU/STABLE3/DEC_2': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'stable': 'dec_2'},
                           'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                          Randomise.NAMES], 25)},
    'TABU/STABLE3/INC_SCORE': {'package': Package.BNBENCH,
                               'params': {'tabu': 10, 'stable': 'inc_score'},
                               'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                              Randomise.NAMES], 25)},
    'TABU/STABLE3/SCORE': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'stable': 'score'},
                           'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                          Randomise.NAMES], 25)},
    'TABU/STABLE3/SCORE_PLUS': {'package': Package.BNBENCH,
                                'params': {'tabu': 10, 'stable': 'score+'},
                                'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                               Randomise.NAMES], 25)},
    'TABU/STABLE3/SC4_PLUS': {'package': Package.BNBENCH,
                              'params': {'tabu': 10, 'stable': 'sc4+'},
                              'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                             Randomise.NAMES], 25)},
    'TABU/STABLE3/SP_GREEDY': {'package': Package.BNBENCH,
                               'params': {'tabu': 10, 'stable': 'score+',
                                          'tree': (1, 1, 100)},
                               'randomise': ([Randomise.ORDER, Randomise.NAMES,
                                              Randomise.ROWS], 25)},
    'TABU/STABLE3/BDEU_BASE': {'package': Package.BNBENCH,
                               'params': {'tabu': 10, 'score': 'bde',
                                          'iss': 1},
                               'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                              Randomise.NAMES], 5)},
    'TABU/STABLE3/BDEU_DEC': {'package': Package.BNBENCH,
                              'params': {'tabu': 10, 'stable': 'dec_score',
                                         'score': 'bde', 'iss': 1},
                              'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                             Randomise.NAMES], 5)},
    'TABU/STABLE3/BDEU_INC': {'package': Package.BNBENCH,
                              'params': {'tabu': 10, 'stable': 'inc_score',
                                         'score': 'bde', 'iss': 1},
                              'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                             Randomise.NAMES], 5)},
    'TABU/STABLE3/BDEU_PLUS': {'package': Package.BNBENCH,
                               'params': {'tabu': 10, 'stable': 'score+',
                                          'score': 'bde', 'iss': 1},
                               'randomise': ([Randomise.ORDER, Randomise.ROWS,
                                              Randomise.NAMES], 5)},
    'TABU/T1_1_20': {'package': Package.BNBENCH,
                     'params': {'tabu': 10, 'bnlearn': False, 'stable': True,
                                'tree': (1, 1, 20)},
                     'randomise': ([Randomise.ORDER, Randomise.NAMES], 3)},
    'TABU/T1_1_20S': {'package': Package.BNBENCH,
                      'params': {'tabu': 10, 'bnlearn': False,
                                 'stable': 'score', 'tree': (1, 1, 20)},
                      'randomise': ([Randomise.ORDER, Randomise.NAMES], 3)},
    'TABU/T1_1_20SP': {'package': Package.BNBENCH,
                       'params': {'tabu': 10, 'bnlearn': False,
                                  'stable': 'score+', 'tree': (1, 1, 20)},
                       'randomise': ([Randomise.ORDER, Randomise.NAMES], 3)},
    'HC/T1_1_20SP': {'package': Package.BNBENCH,
                     'params': {'stable': 'score+', 'tree': (1, 1, 20)},
                     'randomise': ([Randomise.ORDER, Randomise.NAMES], 3)},
    'TABU/T4_1_100': {'package': Package.BNBENCH,
                      'params': {'tabu': 10, 'bnlearn': False, 'stable': True,
                                 'tree': (4, 1, 100)},
                      'randomise': ([Randomise.ORDER, Randomise.NAMES], 3)},
    'TABU/EQV/L012': {'package': Package.BNBENCH,
                      'knowledge': RuleSet.EQUIV_ADD,
                      'params': {'tabu': 10, 'bnlearn': False},
                      'kparams': {'limit': 0.125, 'expertise': 1.0,
                                  'ignore': 0},
                      'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                    10)},
    'TABU/EQV/L025': {'package': Package.BNBENCH,
                      'knowledge': RuleSet.EQUIV_ADD,
                      'params': {'tabu': 10, 'bnlearn': False},
                      'kparams': {'limit': 0.25, 'expertise': 1.0,
                                  'ignore': 0},
                      'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                    10)},
    'TABU/EQV/L025_E67': {'package': Package.BNBENCH,
                          'params': {'tabu': 10, 'bnlearn': False},
                          'knowledge': RuleSet.EQUIV_ADD,
                          'kparams': {'limit': 0.25, 'expertise': 0.67,
                                      'ignore': 0},
                          'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                        3)},
    'TABU/EQV/L050': {'package': Package.BNBENCH,
                      'knowledge': RuleSet.EQUIV_ADD,
                      'params': {'tabu': 10, 'bnlearn': False},
                      'kparams': {'limit': 0.50, 'expertise': 1.0,
                                  'ignore': 0, 'partial': False},
                      'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                    10)},
    'TABU/EQV/L050_E90': {'package': Package.BNBENCH,
                          'params': {'tabu': 10, 'bnlearn': False},
                          'knowledge': RuleSet.EQUIV_ADD,
                          'kparams': {'limit': 0.50, 'expertise': 0.90,
                                      'ignore': 0},
                          'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                        10)},
    'TABU/EQV/L050_E80': {'package': Package.BNBENCH,
                          'params': {'tabu': 10, 'bnlearn': False},
                          'knowledge': RuleSet.EQUIV_ADD,
                          'kparams': {'limit': 0.50, 'expertise': 0.80,
                                      'ignore': 0, 'earlyok': False},
                          'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                        10)},
    'TABU/EQV/L050_E67': {'package': Package.BNBENCH,
                          'params': {'tabu': 10, 'bnlearn': False},
                          'knowledge': RuleSet.EQUIV_ADD,
                          'kparams': {'limit': 0.50, 'expertise': 0.67,
                                      'ignore': 0},
                          'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                        10)},
    'TABU/EQV/L050_E50': {'package': Package.BNBENCH,
                          'params': {'tabu': 10, 'bnlearn': False},
                          'knowledge': RuleSet.EQUIV_ADD,
                          'kparams': {'limit': 0.50, 'expertise': 0.5,
                                      'ignore': 0},
                          'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                        10)},
    'TABU/EQV/E67': {'package': Package.BNBENCH,
                     'params': {'tabu': 10, 'bnlearn': False},
                     'knowledge': RuleSet.EQUIV_ADD,
                     'kparams': {'limit': False, 'expertise': 0.67,
                                 'ignore': 0},
                     'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE], 3)},
    'TABU/EQV/E80': {'package': Package.BNBENCH,
                     'params': {'tabu': 10, 'bnlearn': False},
                     'knowledge': RuleSet.EQUIV_ADD,
                     'kparams': {'limit': False, 'expertise': 0.80,
                                 'ignore': 0},
                     'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE], 3)},
    'TABU/EQV/E100': {'package': Package.BNBENCH,
                      'params': {'tabu': 10, 'bnlearn': False},
                      'knowledge': RuleSet.EQUIV_ADD,
                      'kparams': {'limit': False, 'expertise': 1.0,
                                  'ignore': 0},
                      'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                    10)},

    'TABU/EQVP/L012': {'package': Package.BNBENCH,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'knowledge': RuleSet.EQUIV_ADD,
                       'kparams': {'limit': 0.125, 'expertise': 1.0,
                                   'ignore': 0, 'partial': True},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/EQVP/L025': {'package': Package.BNBENCH,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'knowledge': RuleSet.EQUIV_ADD,
                       'kparams': {'limit': 0.25, 'expertise': 1.0,
                                   'ignore': 0, 'partial': True},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/EQVP/L050': {'package': Package.BNBENCH,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'knowledge': RuleSet.EQUIV_ADD,
                       'kparams': {'limit': 0.50, 'expertise': 1.0,
                                   'ignore': 0, 'partial': True},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/EQVP/E100': {'package': Package.BNBENCH,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'knowledge': RuleSet.EQUIV_ADD,
                       'kparams': {'limit': False, 'expertise': 1.0,
                                   'ignore': 0, 'partial': True},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/EQVP/L050_E50': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.EQUIV_ADD,
                           'kparams': {'limit': 0.50, 'expertise': 0.5,
                                       'ignore': 0, 'partial': True},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/EQVP/L050_E67': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.EQUIV_ADD,
                           'kparams': {'limit': 0.50, 'expertise': 0.67,
                                       'ignore': 0, 'partial': True},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/EQVP/L050_E80': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.EQUIV_ADD,
                           'kparams': {'limit': 0.50, 'expertise': 0.8,
                                       'ignore': 0, 'partial': True},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/EQVP/L050_E90': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.EQUIV_ADD,
                           'kparams': {'limit': 0.50, 'expertise': 0.9,
                                       'ignore': 0, 'partial': True},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/EQV/L050_PAR': {'package': Package.BNBENCH,
                          'params': {'tabu': 10, 'bnlearn': False},
                          'knowledge': RuleSet.EQUIV_ADD,
                          'kparams': {'limit': 0.50, 'expertise': 1.0,
                                      'ignore': 0, 'partial': True},
                          'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                        10)},

    'TABU/REQD/L012': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.REQD_ARC,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'kparams': {'reqd': 0.125, 'expertise': 1.0,
                                   'ignore': 0},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/REQD/L025': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.REQD_ARC,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'kparams': {'reqd': 0.25, 'expertise': 1.0,
                                   'ignore': 0},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/REQD/L025_E67': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.REQD_ARC,
                           'kparams': {'reqd': 0.25, 'expertise': 0.67},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 3)},
    'TABU/REQD/L050': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.REQD_ARC,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'kparams': {'reqd': 0.50, 'expertise': 1.0},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/REQD/L050_E90': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.REQD_ARC,
                           'kparams': {'reqd': 0.50, 'expertise': 0.90},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/REQD/L050_E80': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.REQD_ARC,
                           'kparams': {'reqd': 0.50, 'expertise': 0.80},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/REQD/L050_E67': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.REQD_ARC,
                           'kparams': {'reqd': 0.50, 'expertise': 0.67},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/REQD/L050_E50': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.REQD_ARC,
                           'kparams': {'reqd': 0.50, 'expertise': 0.5},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/STOP/L012': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.STOP_ARC,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'kparams': {'stop': 0.125, 'expertise': 1.0},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/STOP/L025': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.STOP_ARC,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'kparams': {'stop': 0.25, 'expertise': 1.0},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/STOP/L050': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.STOP_ARC,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'kparams': {'stop': 0.50, 'expertise': 1.0},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/TIERS/L012': {'package': Package.BNBENCH,
                        'knowledge': RuleSet.TIERS,
                        'params': {'tabu': 10, 'bnlearn': False},
                        'kparams': {'nodes': 0.125, 'expertise': 1.0},
                        'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                      10)},
    'TABU/TIERS/L025': {'package': Package.BNBENCH,
                        'knowledge': RuleSet.TIERS,
                        'params': {'tabu': 10, 'bnlearn': False},
                        'kparams': {'nodes': 0.25, 'expertise': 1.0},
                        'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                      10)},
    'TABU/TIERS/L050': {'package': Package.BNBENCH,
                        'knowledge': RuleSet.TIERS,
                        'params': {'tabu': 10, 'bnlearn': False},
                        'kparams': {'nodes': 0.50, 'expertise': 1.0},
                        'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                      10)},
    'TABU/TIERS/L050_E50': {'package': Package.BNBENCH,
                            'knowledge': RuleSet.TIERS,
                            'params': {'tabu': 10, 'bnlearn': False},
                            'kparams': {'nodes': 0.50, 'expertise': 0.5},
                            'randomise': ([Randomise.ORDER,
                                           Randomise.KNOWLEDGE], 10)},
    'TABU/TIERS/L050_E67': {'package': Package.BNBENCH,
                            'knowledge': RuleSet.TIERS,
                            'params': {'tabu': 10, 'bnlearn': False},
                            'kparams': {'nodes': 0.50, 'expertise': 0.67},
                            'randomise': ([Randomise.ORDER,
                                           Randomise.KNOWLEDGE], 10)},
    'TABU/TIERS/L050_E80': {'package': Package.BNBENCH,
                            'knowledge': RuleSet.TIERS,
                            'params': {'tabu': 10, 'bnlearn': False},
                            'kparams': {'nodes': 0.50, 'expertise': 0.8},
                            'randomise': ([Randomise.ORDER,
                                           Randomise.KNOWLEDGE], 10)},
    'TABU/TIERS/L050_E90': {'package': Package.BNBENCH,
                            'knowledge': RuleSet.TIERS,
                            'params': {'tabu': 10, 'bnlearn': False},
                            'kparams': {'nodes': 0.50, 'expertise': 0.9},
                            'randomise': ([Randomise.ORDER,
                                           Randomise.KNOWLEDGE], 10)},
    'TABU/MIX/R25S25': {'package': Package.BNBENCH,
                        'knowledge': RuleSet.MIX_ARC,
                        'params': {'tabu': 10, 'bnlearn': False},
                        'kparams': {'stop': 0.25, 'reqd': 0.25,
                                    'expertise': 1.0},
                        'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                      3)},
    'TABU/MIX/R05S45': {'package': Package.BNBENCH,
                        'knowledge': RuleSet.MIX_ARC,
                        'params': {'tabu': 10, 'bnlearn': False},
                        'kparams': {'stop': 0.45, 'reqd': 0.05,
                                    'expertise': 1.0},
                        'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                      10)},
    'TABU/MIX9/L050': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.MIX_ARC,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'kparams': {'stop': 0.45, 'reqd': 0.05,
                                   'expertise': 1.0},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/MIX9/L025': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.MIX_ARC,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'kparams': {'stop': 0.225, 'reqd': 0.025,
                                   'expertise': 1.0},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/MIX9/L012': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.MIX_ARC,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'kparams': {'stop': 0.1125, 'reqd': 0.0125,
                                   'expertise': 1.0},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/MIX9/L050_E50': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.MIX_ARC,
                           'kparams': {'stop': 0.45, 'reqd': 0.05,
                                       'expertise': 0.5},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/MIX9/L050_E67': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.MIX_ARC,
                           'kparams': {'stop': 0.45, 'reqd': 0.05,
                                       'expertise': 0.67},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/MIX9/L050_E80': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.MIX_ARC,
                           'kparams': {'stop': 0.45, 'reqd': 0.05,
                                       'expertise': 0.8},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/MIX9/L050_E90': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.MIX_ARC,
                           'kparams': {'stop': 0.45, 'reqd': 0.05,
                                       'expertise': 0.9},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/MI/L050': {'package': Package.BNBENCH, 'knowledge': RuleSet.MI_CHECK,
                     'kparams': {'limit': 0.50, 'ignore': 0, 'expertise': 1.0},
                     'params': {'tabu': 10, 'bnlearn': False},
                     'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                   3)},
    'TABU/POS/L050': {'package': Package.BNBENCH,
                      'params': {'tabu': 10, 'bnlearn': False},
                      'knowledge': RuleSet.POS_DELTA,
                      'kparams': {'limit': 0.50, 'ignore': 0,
                                  'expertise': 1.0},
                      'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                    3)},
    'TABU/LT5/L050': {'package': Package.BNBENCH, 'knowledge': RuleSet.HI_LT5,
                      'params': {'tabu': 10, 'bnlearn': False},
                      'kparams': {'limit': 0.50, 'ignore': 0,
                                  'threshold': 0.05, 'expertise': 1.0},
                      'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                    10)},
    'TABU/LT5/L050_T01': {'package': Package.BNBENCH,
                          'knowledge': RuleSet.HI_LT5,
                          'params': {'tabu': 10, 'bnlearn': False},
                          'kparams': {'limit': 0.50, 'ignore': 0,
                                      'threshold': 0.01, 'expertise': 1.0},
                          'randomise': ([Randomise.ORDER,
                                         Randomise.KNOWLEDGE], 10)},
    'TABU/LT5/L050_T001': {'package': Package.BNBENCH,
                           'knowledge': RuleSet.HI_LT5,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'kparams': {'limit': 0.50, 'ignore': 0,
                                       'threshold': 0.001, 'expertise': 1.0},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/LT5/L050_T20': {'package': Package.BNBENCH,
                          'knowledge': RuleSet.HI_LT5,
                          'params': {'tabu': 10, 'bnlearn': False},
                          'kparams': {'limit': 0.50, 'ignore': 0,
                                      'threshold': 0.20, 'expertise': 1.0},
                          'randomise': ([Randomise.ORDER,
                                         Randomise.KNOWLEDGE], 10)},
    'TABU/STAB/L050': {'package': Package.BNBENCH,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'knowledge': RuleSet.BIC_UNSTABLE,
                       'kparams': {'limit': 0.50, 'ignore': 0,
                                   'expertise': 1.0, 'threshold': 0.05},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/STAB/L050_T20': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.BIC_UNSTABLE,
                           'kparams': {'limit': 0.50, 'ignore': 0,
                                       'expertise': 1.0, 'threshold': 0.20},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/STAB/L050_T01': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.BIC_UNSTABLE,
                           'kparams': {'limit': 0.50, 'ignore': 0,
                                       'expertise': 1.0, 'threshold': 0.01},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/STAB/L050_T001': {'package': Package.BNBENCH,
                            'params': {'tabu': 10, 'bnlearn': False},
                            'knowledge': RuleSet.BIC_UNSTABLE,
                            'kparams': {'limit': 0.50, 'ignore': 0,
                                        'expertise': 1.0, 'threshold': 0.001},
                            'randomise': ([Randomise.ORDER,
                                           Randomise.KNOWLEDGE], 10)},
    'TABU/STAB/L050_T0001': {'package': Package.BNBENCH,
                             'params': {'tabu': 10, 'bnlearn': False},
                             'knowledge': RuleSet.BIC_UNSTABLE,
                             'kparams': {'limit': 0.50, 'threshold': 0.0001,
                                         'expertise': 1.0, 'ignore': 0},
                             'randomise': ([Randomise.ORDER,
                                            Randomise.KNOWLEDGE], 3)},
    'TABU/STAB/L012': {'package': Package.BNBENCH,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'knowledge': RuleSet.BIC_UNSTABLE,
                       'kparams': {'limit': 0.125, 'ignore': 0,
                                   'expertise': 1.0, 'threshold': 0.05},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     3)},
    'TABU/STAB/L012_T20': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.BIC_UNSTABLE,
                           'kparams': {'limit': 0.125, 'ignore': 0,
                                       'expertise': 1.0, 'threshold': 0.20},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 3)},
    'TABU/STAB/L012_T01': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.BIC_UNSTABLE,
                           'kparams': {'limit': 0.125, 'ignore': 0,
                                       'expertise': 1.0, 'threshold': 0.01},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 3)},
    'TABU/STAB/L012_T001': {'package': Package.BNBENCH,
                            'params': {'tabu': 10, 'bnlearn': False},
                            'knowledge': RuleSet.BIC_UNSTABLE,
                            'kparams': {'limit': 0.125, 'ignore': 0,
                                        'expertise': 1.0, 'threshold': 0.001},
                            'randomise': ([Randomise.ORDER,
                                           Randomise.KNOWLEDGE], 3)},
    'TABU/LODL/L050': {'package': Package.BNBENCH,
                       'params': {'tabu': 10, 'bnlearn': False},
                       'knowledge': RuleSet.LO_DELTA,
                       'kparams': {'limit': 0.50, 'ignore': 0,
                                   'expertise': 1.0, 'threshold': 0.05},
                       'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                     10)},
    'TABU/LODL/L050_T20': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.LO_DELTA,
                           'kparams': {'limit': 0.50, 'ignore': 0,
                                       'expertise': 1.0, 'threshold': 0.20},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/LODL/L050_T01': {'package': Package.BNBENCH,
                           'params': {'tabu': 10, 'bnlearn': False},
                           'knowledge': RuleSet.LO_DELTA,
                           'kparams': {'limit': 0.50, 'ignore': 0,
                                       'expertise': 1.0, 'threshold': 0.01},
                           'randomise': ([Randomise.ORDER,
                                          Randomise.KNOWLEDGE], 10)},
    'TABU/LODL/L050_T001': {'package': Package.BNBENCH,
                            'params': {'tabu': 10, 'bnlearn': False},
                            'knowledge': RuleSet.LO_DELTA,
                            'kparams': {'limit': 0.50, 'ignore': 0,
                                        'expertise': 1.0, 'threshold': 0.001},
                            'randomise': ([Randomise.ORDER,
                                           Randomise.KNOWLEDGE], 10)},

    'TABU/EQVLT5/L050': {'package': Package.BNBENCH,
                         'params': {'tabu': 10, 'bnlearn': False},
                         'knowledge': RuleSet.EQUIV_LT5,
                         'kparams': {'limit': 0.50, 'ignore': 0,
                                     'expertise': 1.0},
                         'randomise': ([Randomise.ORDER, Randomise.KNOWLEDGE],
                                       3)},

    'HC/STD': {'package': Package.BNBENCH},
    'HC/STD_LLH': {'package': Package.BNBENCH, 'params': {'score': 'loglik'}},
    'HC/OPT_LLH': {'package': Package.BNBENCH, 'params': {'score': 'loglik'},
                   'ordering': Ordering.OPTIMAL},
    'HC/OPT_ORA': {'package': Package.BNBENCH, 'datagen': 'none',
                   'ordering': Ordering.OPTIMAL},
    'TABU/STD': {'package': Package.BNBENCH,
                 'params': {'tabu': 10, 'bnlearn': False}},
    'TABU/STD_NI20': {'package': Package.BNBENCH,
                      'params': {'tabu': 10, 'bnlearn': False, 'noinc': 20}},
    'HC/ORDER/BASE': {'package': Package.BNBENCH,
                      'randomise': ([Randomise.ORDER], 10)},
    'HC/ORDER/KL4': {'package': Package.BNBENCH,
                     'knowledge': RuleSet.EQUIV_ADD,
                     'kparams': {'limit': 4, 'ignore': 0, 'expertise': 1.0},
                     'randomise': ([Randomise.ORDER], 10)},
    'HC/ORDER/KL16': {'package': Package.BNBENCH,
                      'knowledge': RuleSet.EQUIV_ADD,
                      'kparams': {'limit': 16, 'ignore': 0, 'expertise': 1.0},
                      'randomise': ([Randomise.ORDER], 10)},
    'HC/ORDER/KL0': {'package': Package.BNBENCH,
                     'knowledge': RuleSet.EQUIV_ADD,
                     'kparams': {'limit': False, 'ignore': 0,
                                 'expertise': 1.0},
                     'randomise': ([Randomise.ORDER], 10)},
    'HC/ORDER/KE50': {'package': Package.BNBENCH,
                      'knowledge': RuleSet.EQUIV_ADD,
                      'kparams': {'limit': False, 'ignore': 0,
                                  'expertise': 0.50},
                      'randomise': ([Randomise.ORDER], 10)},
    'HC/ORDER/KE67': {'package': Package.BNBENCH,
                      'knowledge': RuleSet.EQUIV_ADD,
                      'kparams': {'limit': False, 'ignore': 0,
                                  'expertise': 0.67},
                      'randomise': ([Randomise.ORDER], 10)},
    'HC/ORDER/KE80': {'package': Package.BNBENCH,
                      'knowledge': RuleSet.EQUIV_ADD,
                      'kparams': {'limit': False, 'ignore': 0,
                                  'expertise': 0.80},
                      'randomise': ([Randomise.ORDER], 10)},
    'HC/ORDER/KLI16': {'package': Package.BNBENCH,
                       'knowledge': RuleSet.EQUIV_ADD,
                       'kparams': {'limit': 16, 'ignore': 16,
                                   'expertise': 1.0},
                       'randomise': ([Randomise.ORDER], 10)},
    'HC/GROW': {'package': Package.BNBENCH,
                'params': {'base': 'e', 'score': 'bic', 'k': 1,
                           'maxiter': 10000}},  # use as grow surrogate
    'HC/GROW_BAD': {'package': Package.BNBENCH, 'ordering': Ordering.WORST,
                    'params': {'base': 'e', 'score': 'bic', 'k': 1,
                               'maxiter': 10000}},  # use as grow surrogate
    'HC/GROW_OPT': {'package': Package.BNBENCH, 'ordering': Ordering.OPTIMAL,
                    'params': {'base': 'e', 'score': 'bic', 'k': 1,
                               'maxiter': 10000}},  # use as grow surrogate
    'HC/OPT': {'package': Package.BNBENCH, 'ordering': Ordering.OPTIMAL},
    'HC/BAD': {'package': Package.BNBENCH, 'ordering': Ordering.WORST},
    'TETRAD/FGES_STD': {'package': Package.TETRAD,
                        'algorithm': Algorithm.FGES,
                        'params': {'score': 'bic', 'k': 1}},
    'TETRAD/FGES_BASE2': {'package': Package.TETRAD,
                          'algorithm': Algorithm.FGES,
                          'randomise': ([Randomise.NAMES, Randomise.ORDER],
                                        25),
                          'params': {'score': 'bic', 'k': 1}},
    'TETRAD/FGES_BASE3': {'package': Package.TETRAD,
                          'algorithm': Algorithm.FGES,
                          'randomise': ([Randomise.NAMES, Randomise.ORDER,
                                         Randomise.ROWS], 25),
                          'params': {'score': 'bic', 'k': 1}},
    'TETRAD/FGES_BASE4': {'package': Package.TETRAD,
                          'algorithm': Algorithm.FGES,
                          'randomise': ([Randomise.NAMES, Randomise.ORDER,
                                         Randomise.ROWS, Randomise.SAMPLE],
                                        25),
                          'params': {'score': 'bic', 'k': 1}},
    'TETRAD/FGES_BDEU': {'package': Package.TETRAD,
                         'algorithm': Algorithm.FGES,
                         'randomise': ([Randomise.NAMES, Randomise.ORDER,
                                        Randomise.ROWS], 5),
                         'params': {'score': 'bde', 'iss': 1}},
    'TETRAD/FGES_BAD': {'package': Package.TETRAD,
                        'ordering': Ordering.WORST,
                        'algorithm': Algorithm.FGES,
                        'params': {'score': 'bic', 'k': 1}},
    'TETRAD/FGES_OPT': {'package': Package.TETRAD,
                        'ordering': Ordering.OPTIMAL,
                        'algorithm': Algorithm.FGES,
                        'params': {'score': 'bic', 'k': 1}},
    'BNLEARN/HC_BDE': {'params': {'score': 'bde', 'iss': 1}},
    'BNLEARN/HC_BDS': {'params': {'score': 'bds', 'iss': 1}},
    'BNLEARN/HC_K_5': {'params': {'score': 'bic', 'k': 5}},
    'BNLEARN/HC_OPT_K_02': {'ordering': Ordering.OPTIMAL,
                            'params': {'score': 'bic', 'k': 0.2}},
    'BNLEARN/HC_ISS_5': {'params': {'score': 'bde', 'iss': 5}},
    'BNLEARN/HC_BASE2': {'algorithm': Algorithm.HC,
                         'randomise': ([Randomise.NAMES,
                                        Randomise.ORDER], 25)},
    'BNLEARN/TABU_STD': {'algorithm': Algorithm.TABU},
    'BNLEARN/TABU_BASE2': {'algorithm': Algorithm.TABU,
                           'randomise': ([Randomise.NAMES,
                                          Randomise.ORDER], 25)},
    'BNLEARN/TABU_OPT': {'algorithm': Algorithm.TABU,
                         'ordering': Ordering.OPTIMAL},
    'BNLEARN/TABU_BAD': {'algorithm': Algorithm.TABU,
                         'ordering': Ordering.WORST},
    'BNLEARN/TABU_BDE': {'algorithm': Algorithm.TABU,
                         'params': {'score': 'bde', 'iss': 1}},
    'BNLEARN/TABU_BDS': {'algorithm': Algorithm.TABU,
                         'params': {'score': 'bds', 'iss': 1}},
    'BNLEARN/TABU_K_5': {'algorithm': Algorithm.TABU,
                         'params': {'score': 'bic', 'k': 5}},
    'BNLEARN/TABU_ISS_5': {'algorithm': Algorithm.TABU,
                           'params': {'score': 'bde', 'iss': 5}},
    'BNLEARN/H2PC_STD': {'algorithm': Algorithm.H2PC},
    'BNLEARN/H2PC_BASE2': {'algorithm': Algorithm.H2PC,
                           'randomise': ([Randomise.NAMES,
                                          Randomise.ORDER], 25)},
    'BNLEARN/H2PC_BASE3': {'algorithm': Algorithm.H2PC,
                           'params': {'test': 'mi', 'score': 'bic'},
                           'randomise': ([Randomise.NAMES, Randomise.ROWS,
                                          Randomise.ORDER], 25)},
    'BNLEARN/H2PC_BASE4': {'algorithm': Algorithm.H2PC,
                           'params': {'test': 'mi', 'score': 'bic'},
                           'randomise': ([Randomise.NAMES, Randomise.ROWS,
                                          Randomise.ORDER, Randomise.SAMPLE],
                                         25)},
    'BNLEARN/H2PC_OPT': {'algorithm': Algorithm.H2PC,
                         'ordering': Ordering.OPTIMAL},
    'BNLEARN/H2PC_BAD': {'algorithm': Algorithm.H2PC,
                         'ordering': Ordering.WORST},
    'BNLEARN/H2PC_BDE': {'algorithm': Algorithm.H2PC,
                         'params': {'score': 'bde', 'iss': 1}},
    'BNLEARN/H2PC_BDS': {'algorithm': Algorithm.H2PC,
                         'params': {'score': 'bds', 'iss': 1}},
    'BNLEARN/H2PC_K_5': {'algorithm': Algorithm.H2PC,
                         'params': {'score': 'bic', 'k': 5}},
    'BNLEARN/H2PC_ISS_5': {'algorithm': Algorithm.H2PC,
                           'params': {'score': 'bde', 'iss': 5}},
    'BNLEARN/MMHC_STD': {'algorithm': Algorithm.MMHC},
    'BNLEARN/MMHC_BASE2': {'algorithm': Algorithm.MMHC,
                           'randomise': ([Randomise.NAMES,
                                          Randomise.ORDER], 25)},
    'BNLEARN/MMHC_BASE3': {'algorithm': Algorithm.MMHC,
                           'params': {'test': 'mi', 'score': 'bic'},
                           'randomise': ([Randomise.NAMES, Randomise.ROWS,
                                          Randomise.ORDER], 25)},
    'BNLEARN/MMHC_BASE4': {'algorithm': Algorithm.MMHC,
                           'params': {'test': 'mi', 'score': 'bic'},
                           'randomise': ([Randomise.NAMES, Randomise.ROWS,
                                          Randomise.ORDER, Randomise.SAMPLE],
                                         25)},
    'BNLEARN/MMHC_OPT': {'algorithm': Algorithm.MMHC,
                         'ordering': Ordering.OPTIMAL},
    'BNLEARN/MMHC_BAD': {'algorithm': Algorithm.MMHC,
                         'ordering': Ordering.WORST},
    'BNLEARN/MMHC_BDE': {'algorithm': Algorithm.MMHC,
                         'params': {'score': 'bde', 'iss': 1}},
    'BNLEARN/MMHC_BDS': {'algorithm': Algorithm.MMHC,
                         'params': {'score': 'bds', 'iss': 1}},
    'BNLEARN/MMHC_K_5': {'algorithm': Algorithm.MMHC,
                         'params': {'score': 'bic', 'k': 5}},
    'BNLEARN/MMHC_ISS_5': {'algorithm': Algorithm.MMHC,
                           'params': {'score': 'bde', 'iss': 5}},
    'BNLEARN/PC/BASE': {'algorithm': Algorithm.PC,
                        'params': {'test': 'mi', 'alpha': 0.05},
                        'randomise': ([Randomise.ORDER], 3)},
    'BNLEARN/PC_BASE2': {'algorithm': Algorithm.PC,
                         'randomise': ([Randomise.NAMES, Randomise.ORDER], 25),
                         'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/PC_BASE3': {'algorithm': Algorithm.PC,
                         'randomise': ([Randomise.NAMES, Randomise.ROWS,
                                        Randomise.ORDER], 25),
                         'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/PC_BASE4': {'algorithm': Algorithm.PC,
                         'randomise': ([Randomise.NAMES, Randomise.ROWS,
                                        Randomise.ORDER, Randomise.SAMPLE],
                                       25),
                         'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/PC_STD': {'algorithm': Algorithm.PC,
                       'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/PC_OPT': {'algorithm': Algorithm.PC,
                       'params': {'test': 'mi', 'alpha': 0.05},
                       'ordering': Ordering.OPTIMAL},
    'BNLEARN/PC_BAD': {'algorithm': Algorithm.PC,
                       'params': {'test': 'mi', 'alpha': 0.05},
                       'ordering': Ordering.WORST},
    'BNLEARN/PC_ALPHA_01': {'algorithm': Algorithm.PC,
                            'params': {'test': 'mi', 'alpha': 0.01}},
    'BNLEARN/PC_X2': {'algorithm': Algorithm.PC,
                      'params': {'test': 'x2', 'alpha': 0.05}},
    'BNLEARN/GS_STD': {'algorithm': Algorithm.GS,
                       'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/GS_BASE2': {'algorithm': Algorithm.GS,
                         'randomise': ([Randomise.NAMES, Randomise.ORDER], 3),
                         'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/GS_BASE3': {'algorithm': Algorithm.GS,
                         'randomise': ([Randomise.NAMES, Randomise.ORDER,
                                        Randomise.ROWS], 25),
                         'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/GS_BASE4': {'algorithm': Algorithm.GS,
                         'randomise': ([Randomise.NAMES, Randomise.ORDER,
                                        Randomise.ROWS, Randomise.SAMPLE],
                                       25),
                         'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/GS_OPT': {'algorithm': Algorithm.GS,
                       'ordering': Ordering.OPTIMAL,
                       'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/GS_BAD': {'algorithm': Algorithm.GS,
                       'ordering': Ordering.WORST,
                       'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/GS_ALPHA_01': {'algorithm': Algorithm.GS,
                            'params': {'test': 'mi', 'alpha': 0.01}},
    'BNLEARN/GS_X2': {'algorithm': Algorithm.GS,
                      'params': {'test': 'x2', 'alpha': 0.05}},
    'BNLEARN/IIAMB_STD': {'algorithm': Algorithm.IIAMB,
                          'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/IIAMB_BASE2': {'algorithm': Algorithm.IIAMB,
                            'randomise': ([Randomise.NAMES, Randomise.ORDER],
                                          25),
                            'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/IIAMB_BASE3': {'algorithm': Algorithm.IIAMB,
                            'randomise': ([Randomise.NAMES, Randomise.ORDER,
                                           Randomise.ROWS], 25),
                            'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/IIAMB_BASE4': {'algorithm': Algorithm.IIAMB,
                            'randomise': ([Randomise.NAMES, Randomise.ORDER,
                                           Randomise.ROWS, Randomise.SAMPLE],
                                          25),
                            'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/IIAMB_OPT': {'algorithm': Algorithm.IIAMB,
                          'ordering': Ordering.OPTIMAL,
                          'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/IIAMB_BAD': {'algorithm': Algorithm.IIAMB,
                          'ordering': Ordering.WORST,
                          'params': {'test': 'mi', 'alpha': 0.05}},
    'BNLEARN/IIAMB_ALPHA_01': {'algorithm': Algorithm.IIAMB,
                               'params': {'test': 'mi', 'alpha': 0.01}},
    'BNLEARN/IIAMB_X2': {'algorithm': Algorithm.IIAMB,
                         'params': {'test': 'x2', 'alpha': 0.05}}}
SERIES.update(SERIES_P)

NETWORKS_GRID_DESIGN = {'figure.subplots_top': 0.9,
                        'figure.subplots_left': 0.1,
                        'figure.subplots_right': 0.9,
                        'figure.subplots_hspace': 0.20,
                        'figure.subplots_wspace': 0.08,
                        'figure.per_row': 4,
                        'subplot.grid': True,
                        'subplot.grid_colour': 'lightgray',
                        'subplot.background': 'white',
                        'subplot.kind': 'line',
                        'subplot.axes_fontsize': 12,
                        'subplot.title_fontsize': 16,
                        'xaxis.scale': 'log',
                        'xaxis.ticks_fontsize': 12,
                        'yaxis.ticks_fontsize': 12,
                        'legend.title': 'Metrics',
                        'legend.title_fontsize': 16,
                        'legend.outside': True,
                        'legend.fontsize': 12}


def convert_str(string):
    """
        Convert string parameter value to its correct type

        :param str string: value expressed as a string

        :returns <type>: string converted to appropriate type
    """
    if string is None:
        value = ''
    elif string == '¬':
        value = None
    elif string == 'False':
        value = False
    elif string == 'True':
        value = True
    elif string.startswith('(') and string.endswith(')'):
        value = tuple(convert_str(v) for v in string[1:-1].split(','))
    elif string.startswith('{') and string.endswith('}'):
        value = string[1:-1].split(',')
        value = {value[2 * i]: value[2 * i + 1]
                 for i in range(round(len(value) / 2))}
    elif FLOAT.match(string):
        value = float(string)
    elif INT.match(string):
        value = int(string)
    else:
        value = string
    return value


def to_num(string):
    """
        Converts a string to numeric allowing syntax like "50K".

        :param string str: string to be converted.

        :raises TypeError: is string is not str type

        :returns int/float/None: integer/float value if valid string else None.
    """
    INT_PATTERN = compile(r'^(\d+)([KMG]*)$')
    FLOAT_PATTERN = compile(r'^\d+\.\d+$')

    if not isinstance(string, str):
        raise TypeError('to_num() bad arg type')

    mult = {'': 1, 'K': 10**3, 'M': 10**6, 'G': 10**9}
    i = INT_PATTERN.match(string.upper())
    f = FLOAT_PATTERN.match(string)
    return (int(i.group(1)) * mult[i.group(2)] if i is not None else
            (float(f.group(0)) if f is not None else f))


def run_help(analysis):
    """
        Returns run help text in human readable format

        :param bool analysis: whether help required for analysis or experiments
    """
    txt = ('\n\n RUN ANALYSIS HELP\n===================\n' if analysis
           else '\n\n RUN EXPERIMENTS HELP\n======================\n')

    txt += ('\nMandatory argument --series= should be one or more' +
            ' (comma separated) series or series groups\n')

    txt += '\nMandatory argument --networks= should be one or more networks\n'

    txt += ('\nOptional argument --N= should be single or range of integers ' +
            'e.g. --N=10-1K, optionally followed by mantissa values and ' +
            'subsample range e.g. --N=10K-1M;1,5;0-3\n')

    if analysis:
        txt += '\nOptional argument --action= should be one of:\n'
        for name, desc in ANALYSIS.items():
            txt += '{:>15}: {}\n'.format(name, desc)

        txt += '\nOptional argument --metrics= should be one or more of:\n'
        for name, metrics in METRIC_GROUPS.items():
            txt += '{:>15}: {}\n'.format(name, metrics)
        for name, value in METRICS.items():
            txt += '{:>15}: {}\n'.format(name, value['label'])

        txt += ('\nOptional argument --file=path/filename specified output' +
                ' file name\n')

        txt += ('\nOptional argument --maxtime=nn specifies maximum' +
                ' execution time in minutes\n')

        txt += ('\nArgument --nodes= specifies nodes to compute scores of' +
                ', and must be specified when --action=score')
    else:
        txt += ('\nOptional argument --action= should be one of: ' +
                '"skip", "compare" or "replace"')

    return txt


def process_args(args, analyse):
    """
        Check and interpret the command line arguments for analyse or learn

        :param dict args: command line arguments {name: value}
        :param bool analyse: whether analysing experiments, otherwise learning

        :raises TypeError: if bad arg types
        :raises ValueError: if bad arg values

        :returns tuple: of execution parameters
    """
    if not isinstance(args, dict) or not isinstance(analyse, bool):
        raise TypeError('process_args() bad arg type')

    # Ensure all arguments in present in args

    args = {a: (args[a] if a in args else None) for a in EXPT_ARGS}

    # Check action argument

    error = ''  # info on incorrect args specified
    if analyse is True:
        action = 'trace' if args['action'] is None else args['action']
        if action not in ANALYSIS:
            error += ' - invalid --action argument\n'
    else:
        action = 'skip' if args['action'] is None else args['action']
        if action not in ['skip', 'compare', 'replace']:
            error += ' - invalid --action argument\n'

    # check each item in --series is a series or series group, and expand
    # groups into individual series

    series = args['series'].split(',') if args['series'] else None
    allowed = set(list(SERIES.keys()) + list(SERIES_GROUPS.keys()))
    if (action not in ['bn', 'network', 'score2', 'score3']
            and (series is None or (set(series) - allowed))):
        error += ' - missing or invalid --series argument\n'
    series = list(chain(*[SERIES_GROUPS[s] if s in SERIES_GROUPS else [s]
                          for s in series])) if series is not None else series

    # check each item in --metrics is a metric or metric group, and expand
    # groups into individual metrics

    metrics = (args['metrics'].split(',') if 'metrics' in args
               and args['metrics'] else ['f1'])
    if (set(metrics) - set(list(METRICS.keys()) +
                           list(METRIC_GROUPS.keys()))):
        error += ' - missing or invalid --metrics argument\n'
    metrics = list(chain(*[METRIC_GROUPS[m] if m in METRIC_GROUPS else [m]
                           for m in metrics]))

    if args['networks'] is None:
        if action != 'bn':
            error += ' no --networks specified\n'
        networks = None
    else:
        networks = args['networks'].split(',')

    # Obtain set of sample sizes from --N argument

    try:
        Ns, Ss = sample_sizes(args['N'] if args['N'] is not None else '10-1M')
        print('\n\nSample sizes are {}'.format(Ns))
    except (ValueError, TypeError):
        Ns = Ss = None
        error += ' - invalid --N specified\n'

    nodes = args['nodes'].split(',') \
        if 'nodes' in args and args['nodes'] is not None else None
    if action == 'score' and nodes is None:
        error += ' - --nodes not specified for score analysis\n'

    # Check any maximum time specified

    if args['maxtime'] is not None:
        maxtime = convert_str(args['maxtime'])
        if not isinstance(maxtime, int) or maxtime < 1 or maxtime > 7200:
            error += ' - invalid maxtime, must be between 1 and 7200\n'
    else:
        maxtime = None

    # Check any file name specified

    if args['file'] is not None:
        file = convert_str(args['file'])
        if not isinstance(file, str):
            error += ' - invalid file, must be a string\n'
    else:
        file = None

    # Extract action-specific parameters as a dictionary

    params = args['params'] if 'params' in args else None
    if params is not None:
        params = [p.split(':') + [None] for p in params.split(';')]
        params = {p[0]: convert_str(p[1]) for p in params}
    else:
        params = {}

    if len(error):
        print(run_help(analysis=analyse))
        print('\n\nERRORS IN COMMAND LINE ARGS:\n{}'.format(error))
        action = None

    return (action, series, metrics, networks, nodes, Ns, Ss, maxtime, file,
            params)


def series_props(series):
    """
        Obtains properties for series

        :param str series: series properties required for. e.g. TABU/BASE

        :raises ValueError: if unknown series specified

        :return dict: properties for series {name: value}
    """
    if series not in SERIES:
        raise ValueError('series_props(): unknown series')

    return {k: SERIES[series][k] if k in SERIES[series] else v
            for k, v in SERIES_DEFAULTS.items()}


def series_comparator(series, as_dict=False):
    """
    """
    def _flatten(key, props):
        _dict = (props[key] if key in props and props[key] is not None else {})
        props.update(_dict)
        props.pop(key, None)

    # obtain series, flatten it, and replace Enums with their name attribute

    props = series_props(series)
    _flatten('params', props)
    _flatten('kparams', props)
    props = {k: ((v.value['name'] if isinstance(v.value, dict) else v.value)
                 if isinstance(v, Enum) else v)
             for k, v in props.items()}
    props = {c: props[c] if c in props and props[c] is not None else None
             for c in SERIES_COMPARATORS}

    return props if as_dict is True else tuple(props.values())


def compare_series_properties(series):
    """
        Compare properties of a list of series and determine which properties
        they have in common and which are different.

        :param list series: names of series to compare

        :returns tuple: of common and different properties:
                        (dict of common properties: {prop: value},
                         dict of different properties {series: {prop: value}})
    """

    # Loop through series extracting the key properties

    rows = []
    for s in series:
        row = series_comparator(s, as_dict=True)
        print('{} has comparator properties: {}'.format(s, row))
        row.update({'series': s})
        rows.append(row)

    # Analyse which properties the series have in common and which different

    rows = DataFrame(rows).set_index('series').replace({nan: None})
    common = {}
    for column in rows.columns:
        values = rows[column].value_counts().to_dict()
        if len(values) < 2:
            common.update({column: (list(values)[0] if len(values) else None)})
            if len(rows.columns) == 1:  # all properties in are in common
                return (common, {s: {} for s in series})
            rows.drop(labels=column, axis=1, inplace=True)

    return ((common if len(common) else {}), rows.to_dict(orient='index'))


def comma_to_and(txt):
    """
        Replaces last comma in a comma-separated list with "and".

        :param str txt: comma-separated textual list

        :returns str: text with last comma replaced with "and"
    """
    return ' and'.join(txt.rsplit(',', 1))


def reference_bn(network, root_dir=EXPTS_DIR):
    """
        Obtains reference BN for specified network.

        :param str network: BN network e.g. alarm, pathfinder etc.
        :param str root_dir: root location of files

        :returns tuple: (BN, BN file name)
    """
    bn_file = ('{}/bn/xdsl/{}.xdsl'.format(EXPTS_DIR, network)
               if network.endswith('_c') else
               '{}/bn/{}.dsc'.format(EXPTS_DIR, network))
    return (BN.read(bn_file), bn_file)


def sample_sizes(N_arg='10-1m'):
    """
        Return list of sample sizes and sub-samples to use in experiments as
        determined by command line argument.

        :param str N_arg: sample sizes required from command line

        raises TypeError: if bad arg type
        raises ValueError: if bad arg values

        :returns tuple: list: of numeric sample sizes to use.
                        tuple/None: range of subsamples
    """
    if not isinstance(N_arg, str):
        raise TypeError('sample_sizes() bad arg type')

    # Split into range, multipliers and subsample parts

    parts = N_arg.split(';')

    # Parse the range specification

    N_range = parts[0].split('-')
    N_range = N_range + N_range if len(N_range) == 1 else N_range
    N_range = tuple([to_num(N) for N in N_range])
    if (len(N_range) != 2
        or (not all([isinstance(N, int) for N in N_range])
            and not all([isinstance(N, float) for N in N_range]))):
        raise TypeError('sample_sizes() bad arg type')
    if ((isinstance(N_range[0], int) and N_range[0] < 2)
            or N_range[1] < N_range[0]):
        raise ValueError('sample_sizes() bad arg value')

    # obtain the mantissa specification

    mantissa = ([to_num(m) for m in parts[1].split(',')]
                if len(parts) > 1 and len(parts[1])
                else [1, 2, 4, 5, 8])
    if not all([isinstance(m, int) for m in mantissa]):
        raise TypeError('sample_sizes() bad arg type')
    if (any([m < 1 or m > 9 for m in mantissa])
        or any([(mantissa[i - 1] >= mantissa[i])
                for i in range(1, len(mantissa))])):
        raise ValueError('sample_sizes() bad arg value')

    # process sub-sample range part if specified

    if len(parts) == 3:
        s_range = parts[2].split('-')
        s_min = convert_str(s_range[0])
        s_max = None if len(s_range) == 1 else convert_str(s_range[1])
        if (not isinstance(s_min, int) and
                (s_max is not None and not isinstance(s_max, int))):
            raise TypeError('sample_sizes() bad arg type')
        if (s_max is not None and s_max <= s_min) or s_min < 0:
            raise ValueError('sample_sizes() bad arg value')
        s_range = (s_min, s_max if s_max is not None else s_min)
    else:
        s_range = None

    # Generate the individual sample numbers

    emin = floor(ln(N_range[0], 10))
    emax = ceil(ln(N_range[1], 10))
    print(N_range, mantissa, emin, emax)
    Ns = [int(m * 10**c) if isinstance(N_range[0], int) else float(m * 10**c)
          for c in range(emin, emax + 1) for m in mantissa
          if N_range[0] <= m * 10**c and m * 10**c <= N_range[1]]
    if not len(Ns):
        raise ValueError('sample_sizes() bad arg value')

    return (Ns, s_range)
