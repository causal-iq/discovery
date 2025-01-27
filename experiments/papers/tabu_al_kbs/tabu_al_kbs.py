
# Generate the raw results, tables and charts for Tabu-AL (Dynamically
# Requested Knowledge) paper Knowledge Based Systems Revision 1

from experiments.run_learn import run_learn
from experiments.run_analysis import run_analysis
from fileio.common import EXPTS_DIR

NETWORKS = ('asia,sports,sachs,child,' +  # Note covid not included
            'insurance,property,diarrhoea,water,' +
            'mildew,alarm,barley,hailfinder,' +
            'hepar2,win95pts,formed,pathfinder')
NETW_SUBPLOTS = ','.join([n + ',' + n for n in NETWORKS.split(',')])

FIX_NS = '1K-100K;1,5;0-9'  # Fixed sample sizes
REL_NS = '0.1-10.0;1,2,5;0-9'  # Relative sample sizes (to # parameters)

PAPER_DIR = EXPTS_DIR + '/papers/tabu_al_kbs/'

# Standard colours for charts

R = '#c44e52'  # red
LB = '#64b5cd'  # light blue
B = '#4c72b0'  # blue
G = '#55a868'  # green

# Templates for x-axis labels for charts

AL = 'Active learning\n0.{} x n requests'
ALU = 'Active learning\nunlimited requests'
ALO = 'Active learning (orientation)\nunlimited requests'
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

IMP_SS = ';N.impact:True'  # param to include sample size impact

# Series and chart properties for f1 vs limit chart

LIM_F1 = 'TABU/BASE,TABU/EQV/L012,TABU/EQV/L025,TABU/EQV/L050,TABU/EQV/E100'

# Series and chart properties for request limit charts

LIM_S = ('TABU/BASE,' +
         'TABU/EQV/L012,TABU/MIX9/L012,TABU/REQD/L012,' +
         'TABU/EQV/L025,TABU/MIX9/L025,TABU/REQD/L025,' +
         'TABU/EQV/L050,TABU/MIX9/L050,TABU/REQD/L050,' +
         'TABU/EQV/E100')
LIM_X = ','.join([AL.format(125), MX.format(125), RQ.format(125),
                  AL.format(250), MX.format(250), RQ.format(250),
                  AL.format(500), MX.format(500), RQ.format(500),
                  ALU, SS.format(10), SS.format(100)])
LIM_L = ('{Active learning,' + R + ',' +
         'Predefined mixed arcs,' + LB + ',' +
         'Predefined required arcs,' + B + ',' +
         'Increased sample size,' + G + '}')
LIM_P = ','.join([R, LB, B, R, LB, B, R, LB, B, R, G, G])

# Series and chart properties for orientation-only request limit charts

LIO_S = ('TABU/BASE,' +
         'TABU/EQVP/L012,TABU/STOP/L012,TABU/TIERS/L012,' +
         'TABU/EQVP/L025,TABU/STOP/L025,TABU/TIERS/L025,' +
         'TABU/EQVP/L050,TABU/STOP/L050,TABU/TIERS/L050,' +
         'TABU/EQVP/E100')
LIO_X = ','.join([AO.format(125), PH.format(125), TR.format(125),
                  AO.format(250), PH.format(250), TR.format(250),
                  AO.format(500), PH.format(500), TR.format(500),
                  ALO, SS.format(10), SS.format(100)])
LIO_L = ('{Active learning (orientation),' + R + ',' +
         'Predefined prohibited arcs,' + LB + ',' +
         'Predefined tiers,' + B + ',' +
         'Increased sample size,' + G + '}')
LIO_P = ','.join([R, LB, B, R, LB, B, R, LB, B, R, G, G])

# Series and chart properties for expertise charts

EXP_S = ('TABU/BASE,' +
         'TABU/EQV/L050_E50,TABU/MIX9/L050_E50,TABU/REQD/L050_E50,' +
         'TABU/EQV/L050_E67,TABU/MIX9/L050_E67,TABU/REQD/L050_E67,' +
         'TABU/EQV/L050_E80,TABU/MIX9/L050_E80,TABU/REQD/L050_E80,' +
         'TABU/EQV/L050_E90,TABU/MIX9/L050_E90,TABU/REQD/L050_E90,' +
         'TABU/EQV/L050,TABU/MIX9/L050,TABU/REQD/L050')
EXP_X = ','.join([ALE.format('0.50'), MXE.format('0.50'), RQE.format('0.50'),
                  ALE.format('0.67'), MXE.format('0.67'), RQE.format('0.67'),
                  ALE.format('0.80'), MXE.format('0.80'), RQE.format('0.80'),
                  ALE.format('0.90'), MXE.format('0.90'), RQE.format('0.90'),
                  ALE.format('1.00'), MXE.format('1.00'), RQE.format('1.00')])
EXP_L = ('{Active learning,' + R + ',' +
         'Predefined mixed arcs,' + LB + ',' +
         'Predefined required arcs,' + B + '}')
EXP_P = ','.join([R, LB, B, R, LB, B, R, LB, B, R, LB, B, R, LB, B])

# Series and chart properties for orientation-only expertise charts

EXO_S = ('TABU/BASE,' +
         'TABU/EQVP/L050_E50,TABU/TIERS/L050_E50,' +
         'TABU/EQVP/L050_E67,TABU/TIERS/L050_E67,' +
         'TABU/EQVP/L050_E80,TABU/TIERS/L050_E80,' +
         'TABU/EQVP/L050_E90,TABU/TIERS/L050_E90,' +
         'TABU/EQVP/L050,TABU/TIERS/L050')
EXO_X = ','.join([AOE.format('0.50'), TRE.format('0.50'),
                  AOE.format('0.67'), TRE.format('0.67'),
                  AOE.format('0.80'), TRE.format('0.80'),
                  AOE.format('0.90'), TRE.format('0.90'),
                  AOE.format('1.00'), TRE.format('1.00')])
EXO_L = ('{Active learning (orientation),' + R + ',' +
         'Predefined tiers,' + B + '}')
EXO_P = ','.join([R, B, R, B, R, B, R, B, R, B])

# Series and chart properties for algorithm comparison

ALG_S = ('TABU/BASE,TETRAD/FGES_STD,BNLEARN/PC/BASE,' +
         'TABU/EQVP/L050_E80,TABU/EQVP/L050')
# ALG_S = ('TABU/BASE,TABU/EQVP/L050_E80,TABU/EQVP/L050')

# Properties of violin impact charts

VIOLIN = ('subplot.kind:violin;' +
          'violin.width:0.9;' +
          'violin.scale:width;' +
          'xaxis.tick_labels:({});' +
          'xaxis.ticks_fontsize:8;' +
          'xaxis.ticks_rotation:-50;' +
          'yaxis.label:F1 increase;' +
          'palette:({});' +
          'legend.key:{};' +
          'legend.ncol:2;' +
          'legend.fontsize:8;' +
          'legend.loc:upper left;' +
          'figure.dpi:600' +
          '{}')

# Properties of algorithm comparison chart

ALGO = ('subplot.kind:violin;' +
        'xaxis.label:;' +
        'figure.subplots_bottom:0.18;' +
        'figure.subplots_left:0.10;' +
        'figure.subplots_right:0.90;' +
        'subplot.title:{' + NETW_SUBPLOTS + '};'
        'violin.scale:width;' +
        'violin.width:1.0;' +
        'xaxis.tick_labels:(Tabu,FGES,PC-Stable,Tabu AL - 0.8 expertise,' +
        'Tabu AL - 1.0 expertise);' +
        'xaxis.ticks_rotation:-60;' +
        'xaxis.ticks_halign:left')


#  STRUCTURE LEARNING RUNS
# =========================

# Base no knowledge runs

def learn_tabu_al_base():  # TABU/BASE no knowledge runs
    run_learn({'action': 'skip', 'series': 'TABU/BASE',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/BASE',
               'networks': NETWORKS, 'N': REL_NS}) is True


# Active learning at different request limits

def learn_tabu_al_eqv_l012():  # TABU/EQV/L012, equiv add, limit 0.125
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L012',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L012',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqv_l025():  # TABU/EQV/L025, equiv add, limited 0.250
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L025',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L025',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqv_l050():  # TABU/EQV/L050, equiv add, limit 0.500
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L050',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L050',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqv_e100():  # TABU/EQV/E100, equiv add, unlimited
    run_learn({'action': 'skip', 'series': 'TABU/EQV/E100',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/EQV/E100',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqvp_l012():  # TABU/EQVP/L012, equiv add (ori), limit 0.125
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L012',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L012',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqvp_l025():  # TABU/EQVP/L025, equiv add (ori), limit 0.250
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L025',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L025',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqvp_l050():  # TABU/EQVP/L050, equiv add (ori), limit 0.500
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L050',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L050',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqvp_e100():  # TABU/EQVP/E100, equiv add, unlimited
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/E100',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/E100',
               'networks': NETWORKS, 'N': REL_NS}) is True


# Predefined required arcs at different request limits

def learn_tabu_al_reqd_l012():  # TABU/REQD/L012, reqd arc, limit 0.125
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L012',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L012',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_reqd_l025():  # TABU/REQD/L025, reqd arc, limit 0.250
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L025',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L025',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_reqd_l050():  # TABU/REQD/L050, reqd arc, limit 0.500
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L050',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L050',
               'networks': NETWORKS, 'N': REL_NS}) is True


# Predefined prohibited arcs at different request limits

def learn_tabu_al_stop_l012():  # TABU/STOP/L012, stop arc, limit 0.125
    run_learn({'action': 'skip', 'series': 'TABU/STOP/L012',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/STOP/L012',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_stop_l025():  # TABU/STOP/L025, stop arc, limit 0.250
    run_learn({'action': 'skip', 'series': 'TABU/STOP/L025',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/STOP/L025',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_stop_l050():  # TABU/STOP/L050, stop arc, limit 0.500
    run_learn({'action': 'skip', 'series': 'TABU/STOP/L050',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/STOP/L050',
               'networks': NETWORKS, 'N': REL_NS}) is True


# Predefined mixed arcs at different request limits

def learn_tabu_al_mix9_l012():  # TABU/MIX9/L012, mixed arc, limit 0.125
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L012',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L012',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_mix9_l025():  # TABU/MIX9/L025, mixed arc, limit 0.250
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L025',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L025',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_mix9_l050():  # TABU/MIX9/L050, mixed arc, limit 0.500
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L050',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L050',
               'networks': NETWORKS, 'N': REL_NS}) is True


# Predefined tiers at different request limits

def learn_tabu_al_tiers_l012():  # TABU/TIERS/L012, tiers, limit 0.125
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L012',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L012',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_tiers_l025():  # TABU/TIERS/L025, tiers, limit 0.250
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L025',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L025',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_tiers_l050():  # TABU/TIERS/L050, tiers, limit 0.500
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L050',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L050',
               'networks': NETWORKS, 'N': REL_NS}) is True


# Active learning at different expertise levels

def learn_tabu_al_eqv_e50():  # TABU/EQV/L050_E50, AL, exp=0.5
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L050_E50',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L050_E50',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqv_e67():  # TABU/EQV/L050_E67, AL, exp=0.67
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L050_E67',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L050_E67',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqv_e80():  # TABU/EQV/L050_E80, AL, exp=0.8
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L050_E80',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L050_E80',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqv_e90():  # TABU/EQV/L050_E90, AL, exp=0.9
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L050_E90',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/EQV/L050_E90',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqvp_e50():  # TABU/EQVP/L050_E50, orient AL, exp=0.5
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L050_E50',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L050_E50',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqvp_e67():  # TABU/EQVP/L050_E67, orient AL, exp=0.67
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L050_E67',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L050_E67',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqvp_e80():  # TABU/EQVP/L050_E80, orient AL, exp=0.8
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L050_E80',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L050_E80',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_eqvp_e90():  # TABU/EQVP/L050_E90, orient AL, exp=0.9
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L050_E90',
               'networks': NETWORKS, 'N': FIX_NS}) is True
    run_learn({'action': 'skip', 'series': 'TABU/EQVP/L050_E90',
               'networks': NETWORKS, 'N': REL_NS}) is True


# Predefined tiers at different expertise levels

def learn_tabu_al_tiers_e50():  # TABU/TIERS/L050_E50, tiers, exp=0.5
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L050_E50',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L050_E50',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_tiers_e67():  # TABU/TIERS/L050_E67, tiers, exp=0.67
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L050_E67',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L050_E67',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_tiers_e80():  # TABU/TIERS/L050_E80, tiers, exp=0.8
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L050_E80',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L050_E80',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_tiers_e90():  # TABU/TIERS/L050_E90, tiers, exp=0.9
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L050_E90',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/TIERS/L050_E90',
               'networks': NETWORKS, 'N': REL_NS}) is True


# Predefined required arcs at different expertise levels

def learn_tabu_al_reqd_e50():  # TABU/REQD/L050_E50, reqd, exp=0.5
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L050_E50',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L050_E50',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_reqd_e67():  # TABU/REQD/L050_E67, reqd, exp=0.67
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L050_E67',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L050_E67',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_reqd_e80():  # TABU/REQD/L050_E80, reqd, exp=0.8
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L050_E80',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L050_E80',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_reqd_e90():  # TABU/REQD/L050_E90, reqd, exp=0.9
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L050_E90',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/REQD/L050_E90',
               'networks': NETWORKS, 'N': REL_NS}) is True


# Predefined mixed arcs at different expertise levels

def learn_tabu_al_mix9_e50():  # TABU/MIX9/L050_E50, mixed, exp=0.5
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L050_E50',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L050_E50',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_mix9_e67():  # TABU/MIX9/L050_E67, mixed, exp=0.67
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L050_E67',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L050_E67',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_mix9_e80():  # TABU/MIX9/L050_E80, mixed, exp=0.8
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L050_E80',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L050_E80',
               'networks': NETWORKS, 'N': REL_NS}) is True


def learn_tabu_al_mix9_e90():  # TABU/MIX9/L050_E90, mixed, exp=0.9
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L050_E90',
               'networks': NETWORKS, 'N': FIX_NS}) is False
    run_learn({'action': 'skip', 'series': 'TABU/MIX9/L050_E90',
               'networks': NETWORKS, 'N': REL_NS}) is True


# Different trigger criteria

def learn_tabu_al_stab_001():  # TABU/STAB/L050_T001, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/STAB/L050_T001',
               'networks': NETWORKS, 'N': FIX_NS}) is False


def learn_tabu_al_stab_01():  # TABU/STAB/L050_T01, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/STAB/L050_T01',
               'networks': NETWORKS, 'N': FIX_NS}) is False


def learn_tabu_al_stab_05():  # TABU/STAB/L050, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/STAB/L050',
               'networks': NETWORKS, 'N': FIX_NS}) is False


def learn_tabu_al_stab_20():  # TABU/STAB/L050_T20, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/STAB/L050_T20',
               'networks': NETWORKS, 'N': FIX_NS}) is False


def learn_tabu_al_lodl_001():  # TABU/LODL/L050_T001, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/LODL/L050_T001',
               'networks': NETWORKS, 'N': FIX_NS}) is False


def learn_tabu_al_lodl_01():  # TABU/LODL/L050_T01, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/LODL/L050_T01',
               'networks': NETWORKS, 'N': FIX_NS}) is False


def learn_tabu_al_lodl_05():  # TABU/LODL/L050, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/LODL/L050',
               'networks': NETWORKS, 'N': FIX_NS}) is False


def learn_tabu_al_lodl_20():  # TABU/LODL/L050_T20, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/LODL/L050_T20',
               'networks': NETWORKS, 'N': FIX_NS}) is False


def learn_tabu_al_lt5_001():  # TABU/LT5/L050_T001, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/LT5/L050_T001',
               'networks': NETWORKS, 'N': FIX_NS}) is False


def learn_tabu_al_lt5_01():  # TABU/LT5/L050_T01, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/LT5/L050_T01',
               'networks': NETWORKS, 'N': FIX_NS}) is False


def learn_tabu_al_lt5_05():  # TABU/LT5/L050, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/LT5/L050',
               'networks': NETWORKS, 'N': FIX_NS}) is False


def learn_tabu_al_lt5_20():  # TABU/LT5/L050_T20, absolute Ns
    run_learn({'action': 'skip', 'series': 'TABU/LT5/L050_T20',
               'networks': NETWORKS, 'N': FIX_NS}) is False


#  RESULTS ANALYSIS
# ==================

# Analysis which compares the different trigger criteria

def table_tabu_al_criteria_stab():  # Accuracy vs threshold - stable criterion
    args = {'action': 'impact',
            'series': 'STAB',
            'networks': NETWORKS,
            'N': '1K-100K;1,5;0-2'}
    run_analysis(args)


def table_tabu_al_criteria_lodl():  # Accuracy vs threshold - low delta
    args = {'action': 'impact',
            'series': 'LODL',
            'networks': NETWORKS,
            'N': '1K-100K;1,5;0-2'}
    run_analysis(args)


def table_tabu_al_criteria_lt5():  # Accuracy vs threshold - low counts
    args = {'action': 'impact',
            'series': 'LT5',
            'networks': NETWORKS,
            'N': '1K-100K;1,5;0-2'}
    run_analysis(args)


# Line chart of F1 vs sample size for different request limits

def chart_tabu_al_n_limit():
    run_analysis({'action': 'series', 'series': LIM_F1, 'networks': NETWORKS,
                  'N': '1K-100K;1,5', 'file': PAPER_DIR + 'act-n-limit.png',
                  'params': ('figure.subplots_right:0.80;' +
                             'subplot.title:{' + NETW_SUBPLOTS + '}')})


# Analysis comparing AL vs predefined at different request limits

def chart_tabu_al_imp_lim_fix():  # impact vs limit, fix samples
    run_analysis({'action': 'impact', 'series': LIM_S, 'networks': NETWORKS,
                  'N': FIX_NS, 'file': PAPER_DIR + 'imp-lim-fix.png',
                  'params': VIOLIN.format(LIM_X, LIM_P, LIM_L, IMP_SS)})


def chart_tabu_al_imp_lim_rel():  # impact vs limit, rel samples
    run_analysis({'action': 'impact', 'series': LIM_S, 'networks': NETWORKS,
                  'N': REL_NS, 'file': PAPER_DIR + 'imp-lim-rel.png',
                  'params': VIOLIN.format(LIM_X, LIM_P, LIM_L, IMP_SS)})


def chart_tabu_al_imp_lio_fix():  # impact vs limit (orient), fix samples
    run_analysis({'action': 'impact', 'series': LIO_S, 'networks': NETWORKS,
                  'N': FIX_NS, 'file': PAPER_DIR + 'imp-lio-fix.png',
                  'params': VIOLIN.format(LIO_X, LIO_P, LIO_L, IMP_SS)})


def chart_tabu_al_imp_lio_rel():  # impact vs limit (orient), rel samples
    run_analysis({'action': 'impact', 'series': LIO_S, 'networks': NETWORKS,
                  'N': REL_NS, 'file': PAPER_DIR + 'imp-lio-rel.png',
                  'params': VIOLIN.format(LIO_X, LIO_P, LIO_L, IMP_SS)})


# Analysis comparing AL vs predefined at different expertise levels

def chart_tabu_al_imp_exp_fix():  # impact vs expertise, fix samples
    run_analysis({'action': 'impact', 'series': EXP_S, 'networks': NETWORKS,
                  'N': FIX_NS, 'file': PAPER_DIR + 'imp-exp-fix.png',
                  'params': VIOLIN.format(EXP_X, EXP_P, EXP_L, '')})


def chart_tabu_al_imp_exp_rel():  # impact vs expertise, rel samples
    run_analysis({'action': 'impact', 'series': EXP_S, 'networks': NETWORKS,
                  'N': REL_NS, 'file': PAPER_DIR + 'imp-exp-rel.png',
                  'params': VIOLIN.format(EXP_X, EXP_P, EXP_L, '')})


def chart_tabu_al_imp_exo_fix():  # impact vs expertise (orient), fix samples
    run_analysis({'action': 'impact', 'series': EXO_S, 'networks': NETWORKS,
                  'N': FIX_NS, 'file': PAPER_DIR + 'imp-exo-fix.png',
                  'params': VIOLIN.format(EXO_X, EXO_P, EXO_L, '')})


def chart_tabu_al_imp_exo_rel():  # impact vs expertise (orient), rel samples
    run_analysis({'action': 'impact', 'series': EXO_S, 'networks': NETWORKS,
                  'N': REL_NS, 'file': PAPER_DIR + 'imp-exo-rel.png',
                  'params': VIOLIN.format(EXO_X, EXO_P, EXO_L, '')})


# Analysis comparing Tabu-AL with other algorithms without knowledge

def chart_tabu_al_alg_fix():  # algorithm comparisons, rel samples
    run_analysis({'action': 'series', 'series': ALG_S, 'networks': NETWORKS,
                  'N': FIX_NS, 'file': PAPER_DIR + 'algo-fix.png',
                  'params': ALGO})


def chart_tabu_al_algo():  # r0 AL/PDF knowledge expertise comparison
    args = {'action': 'series',
            'series': 'ALGO_KNW',
            'networks': NETWORKS,
            'N': '1K-100K;1,5',
            'params': 'fig:act_algo',
            'file': EXPTS_DIR + '/papers/tabu_al_kbs/act-algo.png'}
    run_analysis(args)
