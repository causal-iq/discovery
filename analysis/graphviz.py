
# encapsulates visualising a graphs

import graphviz as gv
from pandas import DataFrame
from re import compile

from core.common import ln
from analysis.trace import TraceAnalysis

STATUS_COLOUR = {'ok': 'darkgreen', 'eqv': 'indianred1', 'rev': '#990000',
                 'ext': 'gold3', 'mis': 'lightblue3', 'abs': 'darkorchid4'}

DOT_NODE = compile('^(\\S+)\\s\\[height\\=.*pos\\=\\"(\\S+)\\".*$')


def node_positions(analysis):
    """
        Obtain positions for nodes from the reference network layout.

        :param TraceAnalysis analysis: the Trace analysis including the edge
                                       analysis used to construct the
                                       reference graph.

        :returns dict: {node, 'x,y'} of node positions as x, y strings
    """
    # Use 'dot' layout engine suitable for DAGs and add nodes

    dot = gv.Digraph(engine='dot')
    for node in analysis.result.nodes:
        dot.node(node, node)

    # add reference edges to layout - these will be the ok, missing and
    # reversed edges in the analysis

    for type, edges in analysis.edges['result'].items():
        if type in ['arc_matched', 'arc_missing']:
            for edge in edges:
                dot.edge(edge[0], edge[1], label='[??]  ', fontsize='10')
        elif type == 'arc_reversed':
            for edge in edges:
                dot.edge(edge[1], edge[0], label='[??]  ', fontsize='10')

    # call 'pipe' to do layout in 'dot' format and format so each DOT
    # layout command is an element in  a list

    layout = (dot.pipe('dot').decode()
              .replace('\t', '').replace('\r', '').replace('\n', '')
              .replace('digraph {', '').replace(';}', '').split(';'))

    # Extract the node positions from the DOT reference graph layout

    positions = {}
    for item in layout:
        node_match = DOT_NODE.match(item)
        if node_match is not None:
            pos = node_match.group(2).split(',')
            pos = [float(pos[0]) / 72.0, float(pos[1]) / 72.0]
            node = node_match.group(1).replace('"', '')
            positions[node] = ('{:.3f},{:.3f}!'.format(pos[0], pos[1]))

    return positions


def traceviz(analysis, dir, filename=None):
    """
        Visualises a learning trace analysis

        :param TraceAnalysis analysis: the trace analysis to visualise.
        :param str dir: directory to write output to
        :param str/None filename: filename to write output to (for testing)

        :raises TypeError: if bad arg types
    """
    if (not isinstance(analysis, TraceAnalysis) or not isinstance(dir, str)
            or (filename is not None and not isinstance(filename, str))):
        raise TypeError('graphviz.trace() bad arg type')

    # Obtain node positions from visualisation of reference network

    positions = node_positions(analysis)

    # Now use "neato" engine which respects specified node positions. Set graph
    # attributes so arcs drawn around nodes, and dpi is 300, and add nodes.

    dot = gv.Digraph(comment=analysis.context, engine='neato')
    dot.attr('graph', dpi='300', splines='true')
    for node in analysis.result.nodes:
        dot.node(node, node, pos=positions[node])

    # loop through trace records adding edges coloured to reflect the status of
    # the resultant edge.

    trace = DataFrame(analysis.trace).to_dict(orient='records')
    width_col = ('MI' if 'MI' in trace[0] else
                 ('Oracle MI' if 'Oracle MI' in trace[0] else 'delta/score'))
    print('\nArc with determined by {}\n'.format(width_col))
    total_width = sum(analysis.trace[width_col][1:-1])
    for i, entry in enumerate(trace):
        activity = entry['activity']
        if activity not in ['init', 'stop', 'none']:
            frm = entry['arc'][1] if activity == 'reverse' else entry['arc'][0]
            to = entry['arc'][0] if activity == 'reverse' else entry['arc'][1]
            width = '{:.2f}'.format(max(6 + ln(abs(entry[width_col])
                                               / total_width, 'e'), 1))
            arbitrary = '*' if entry['margin'] == 0 else ''
            colour = STATUS_COLOUR[entry['status']]
            dot.edge(frm, to, color=colour, fontcolor=colour, penwidth=width,
                     fontsize="12", label='[{}{}]  '.format(i, arbitrary),
                     labelfloat="true")

    # Add missing edges

    for arc in analysis.edges['result']['arc_missing']:
        dot.edge(arc[0], arc[1], color=STATUS_COLOUR['mis'])

    filename = (analysis.context['id'].replace('/', '_') + '_' + width_col[:2]
                if filename is None else filename)
    filename = dir + '/' + filename + '.gv'
    dot.render(filename=filename, outfile=filename.replace('.gv', '.png'))

    return True
