
#   Experimental runs to generate and analyse BNs

from pandas import DataFrame

from fileio.common import EXPTS_DIR
from experiments.config import NETWORKS
from core.bn import BN
from analysis.bn import BNAnalysis
from experiments.plot import plot_degree_distribution


def analyse_networks():
    """
        Perform detailed structural and parameter analysis on test BNs
    """
    summary = []
    dist = {'network': [], 'metric': [], 'value': []}
    for network in NETWORKS:

        #   Read in network description and anlyse the network

        print('Analysing {} network ...'.format(network))
        bn = BN.read(EXPTS_DIR + '/bn/' + network + '.dsc')
        row = {'network': network}
        analysis = BNAnalysis(bn)

        #   Assemble summary of network's nodes

        nodes = analysis.nodes
        row.update({'n': len(nodes),
                    'in-avg': round(nodes['in'].mean(), 2),
                    'in-max': nodes['in'].max(),
                    'deg-avg': round(nodes['deg'].mean(), 2),
                    'deg-max': nodes['deg'].max(),
                    'mb-avg': round(nodes['mb'].mean(), 2),
                    'mb-max': nodes['mb'].max(),
                    'card.avg': round(nodes['card'].mean(), 2),
                    'card.max': round(nodes['card'].max(), 2),
                    'free.avg': round(nodes['free'].mean(), 2),
                    'free.max': round(nodes['free'].max(), 2),
                    'k-l.avg': round(nodes['k-l'].mean(), 2),
                    'k-l.max': round(nodes['k-l'].max(), 2)
                    })

        #   Assemble summary of network's arcs

        a = analysis.arcs
        ali_rev = len(a.loc[(a['reversible'] == 1) & (a['aligned'] == 1)])
        ali_irr = len(a.loc[(a['reversible'] == 0) & (a['aligned'] == 1)])
        opp_irr = len(a.loc[(a['reversible'] == 0) & (a['aligned'] == 0)])
        opp_rev = len(a.loc[(a['reversible'] == 1) & (a['aligned'] == 0)])
        reversible = sum(analysis.arcs['reversible'] > 0)
        aligned = sum(analysis.arcs['aligned'] > 0)
        row.update({'|A|': len(a),
                    'aligned': round(aligned / len(a), 3),
                    'reversible': round(reversible / len(a), 3),
                    'ali-rev': round(ali_rev / len(a), 3),
                    'ali-irr': round(ali_irr / len(a), 3),
                    'opp-rev': round(opp_rev / len(a), 3),
                    'opp-irr': round(opp_irr / len(a), 3)})
        summary.append(row)

        n = len(nodes)
        if network not in ['cancer', 'pathfinder']:
            dist['value'] += list(nodes['in'].array) + list(nodes['deg'].array)
            dist['network'] += 2 * n * [network]
            dist['metric'] += n * ['in'] + n * ['deg']

    summary = DataFrame(summary).set_index('network')
    summary.to_csv(EXPTS_DIR + '/bn/bn_analysis.csv')
    print(summary)

    plot_degree_distribution(DataFrame(dist),
                             EXPTS_DIR + '/bn/bn_analysis.png')
