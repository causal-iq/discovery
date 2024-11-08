#
#   Functions to read and write DSC format BN specification files
#

import re

from fileio.common import is_valid_path, FileFormatError
from core.metrics import values_same
from core.cpt import CPT

# Patterns matching different types of line in DSC file

NAME = '[a-zA-Z0-9\\-\\_\\.]+'
NETWORK = re.compile('belief network \"(' + NAME + ')\"$')
NODE = re.compile('node (' + NAME + ') {$')
TYPE = re.compile('  type \\: discrete \\[ (\\d+) \\] = { (.+) };$')
PROB = re.compile('probability \\( (' + NAME + ') \\) {$')
PMF = re.compile('   (.+)\\;$')
COND_PROB = re.compile('probability \\( (' + NAME + ') \\| (.+) \\) {$')
COND_PMF = re.compile('  \\((.+)\\) \\: (.+);$')

# Format of output sections

CLOSE_SECT = '}}\n'
NETWORK_SECT = 'belief network "{}"\n'
NODE_SECT = 'node {} {{\n  type : discrete [ {} ] = {{ "{}" }};\n' + CLOSE_SECT
PROB_HDR = 'probability ( {}{} ) {{\n'
CPT_ENTRY = '  ({}) : {};\n'


# Allowed transitions between one type of line and the next in DSC file

TRANSITIONS = {
    'start': ['network'],
    'network': ['node'],
    'node': ['type'],
    'type': ['close'],
    'close': ['node', 'prob', 'cond_prob'],
    'prob': ['pmf'],
    'pmf': ['close'],
    'cond_prob': ['cond_pmf'],
    'cond_pmf': ['cond_pmf', 'close']
}


def _parse_line(line):
    """
        Parse line in DSC format file

        :param str line: line (with line terminator removed)

        :returns dict/None: type of line and type specific attributes
    """
    result = PROB.match(line)  # start of probability section
    if result:
        return {'state': 'prob', 'node': result.group(1)}

    result = COND_PMF.match(line)  # conditional probability entry
    if result:
        try:
            parent_values = [int(s) for s in result.group(1).split(', ')]
            cond_pmf = [float(s) for s in result.group(2).split(', ')]
            if min(parent_values) < 0:
                raise ValueError('Parent value index is < 0')
            if not values_same(sum(cond_pmf), 1, sf=6):
                raise ValueError('Probabilities must sum to 1')
        except ValueError:
            return None
        return {'state': 'cond_pmf', 'parent_values': parent_values,
                'cond_pmf': cond_pmf}

    if line == '}':  # close of node, prob or cond prob section
        return {'state': 'close'}

    result = COND_PROB.match(line)  # start of conditional probability section
    if result:
        return {'state': 'cond_prob', 'node': result.group(1),
                'parents': result.group(2).split(', ')}

    result = NODE.match(line)  # start of node section
    if result:
        return {'state': 'node', 'node': result.group(1)}

    result = TYPE.match(line)  # node type and values line
    if result:
        num_values = int(result.group(1))
        values = []
        for i, value in enumerate(result.group(2).split(', ')):
            if not value.startswith('"') or not value.endswith('"'):
                return None  # if value not double-quoted
            value = value[1:-1]
            if '"' in value or value in values:
                return None  # if value contains " or is a duplicate
            values.append(value)
        if len(values) != num_values:
            return None  # if number of values not correct
        return {'state': 'type', 'values': values}

    result = PMF.match(line)  # non-conditional probabilities line
    if result:
        try:
            pmf = [float(s) for s in result.group(1).split(', ')]
            if not values_same(sum(pmf), 1, sf=6):
                raise ValueError('Probabilities must sum to 1')
        except ValueError:
            return None
        return {'state': 'pmf', 'pmf': pmf}

    result = NETWORK.match(line)  # belief network header line
    if result:
        return {'state': 'network', 'name': result.group(1)}

    return None


def _process_parsed(parsed, nodes, edges, cptdata, node):
    """
        Process parsed line, checking it is consistent with previous lines

        :param dict parsed: parsed input line as returned by _parse_line
        :param list nodes: currently identified node names
        :param list edges: currently identified edges (node1, '->', node2)
        :param dict cptdata: data to construct CPTs {parents, probs}
        :param str node: current node being processed (from previous line)

        :raises FileFormatError: if error in file detected

        :returns tuple: updated nodes, edges and cpts
    """
    state = parsed['state']
    node = parsed['node'] if 'node' in parsed else node

    if state == 'cond_pmf':  # entry in conditional probability table
        if 'cond_pmf' not in cptdata[node]:
            cptdata[node]['cond_pmf'] = {}
        parent_values = parsed['parent_values']
        if len(parent_values) != len(cptdata[node]['parents']):
            raise FileFormatError('Parent value index has wrong length')
        if len(parsed['cond_pmf']) != len(cptdata[node]['values']):
            raise FileFormatError('cond pmf has wrong length')
        if tuple(parent_values) in cptdata[node]['cond_pmf']:
            raise FileFormatError('duplicate cond_pmf parental value keys')
        for idx, parent in enumerate(cptdata[node]['parents']):
            if parent_values[idx] >= len(cptdata[parent]['values']):
                raise FileFormatError('Invalid parental value index')
        cptdata[node]['cond_pmf'][tuple(parent_values)] = parsed['cond_pmf']

    elif state == 'node':  # start of definition of a node
        if node in nodes:
            raise FileFormatError('Duplicate node {}'.format(parsed['node']))
        nodes.append(node)

    elif state == 'prob':  # start of probability entry for parentless node
        if node not in nodes:
            raise FileFormatError('Unknown node {} in prob'.format(node))
        cptdata[node]['parents'] = None

    elif state == 'cond_prob':  # start of prob entry for node with parents
        if node not in nodes:
            raise FileFormatError('Unknown node {} in cond prob'.format(node))
        for parent in parsed['parents']:
            if parent not in nodes:
                raise FileFormatError('Unknown node {} in cond prob'
                                      .format(node))
            edges.append((parent, '->', node))
        cptdata[node]['parents'] = parsed['parents']

    elif state == 'type':  # type of node and allowed values
        cptdata[node] = {'values': parsed['values']}

    elif state == 'close':  # close node, prob or cond_prob section
        if 'parents' in cptdata[node] and cptdata[node]['parents'] is not None:
            num_parental_combos = 1
            for parent in cptdata[node]['parents']:
                num_parental_combos *= len(cptdata[parent]['values'])
            if len(cptdata[node]['cond_pmf']) != num_parental_combos:
                raise FileFormatError('Missing cond_pmf entries')
        node = None

    elif state == 'pmf':  # probabilities for parentless node
        pmf = parsed['pmf']
        if len(pmf) != len(cptdata[node]['values']):
            raise FileFormatError('node {} PMF wrong number of entries ')
        cptdata[node]['pmf'] = pmf

    return (nodes, edges, cptdata, node)


def _reformat_cptdata(cptdata):
    """
        Assemble cptdata in format required by BN constructor

        :param dict cptdata: CPT data from DSC file {node: cptdata}

        :returns dict: CPT data in required format {node: cptdata}
    """
    result = {}
    for node, data in cptdata.items():
        if data['parents'] is None:
            pmf = {v: p for v, p in zip(data['values'], data['pmf'])}
            result.update({node: (CPT, pmf)})
        else:
            cond_pmfs = []
            for pvs_idx, probs in (data['cond_pmf']).items():
                pmf = {v: pr for v, pr in zip(data['values'], probs)}
                pvs = {pa: cptdata[pa]['values'][idx]
                       for pa, idx in zip(data['parents'], pvs_idx)}
                cond_pmfs.append((pvs, pmf))
            result.update({node: (CPT, cond_pmfs)})
    return result


def read(path):
    """
        Reads in a BN from a DSC format BN specification file

        :param str path: full path name of file

        :raises TypeError: if path is not a string
        :raises FileNotFoundError: if file does not exist
        :raises FileFormatError: if file contents not valid

        :returns tuple: nodes, edges, cptdata
    """

    is_valid_path(path)

    with open(path, encoding='utf-8') as f:
        line_num = 0
        prev_state = 'start'
        nodes = []
        edges = []
        cptdata = {}
        node = None

        try:
            for line in f:
                line_num += 1
                line = line.replace('\n', '')
                parsed = _parse_line(line)
                if not parsed or \
                        parsed['state'] not in TRANSITIONS[prev_state]:
                    raise FileFormatError('Line {} invalid: {}'
                                          .format(line_num, line))
                else:
                    nodes, edges, cptdata, node = \
                        _process_parsed(parsed, nodes, edges, cptdata, node)
                prev_state = parsed['state']
        except UnicodeDecodeError:
            raise FileFormatError('File contains unexpected char')

        if prev_state != 'close':
            raise FileFormatError('DSC file empty or didn\'t end with "}"')

        cptdata = _reformat_cptdata(cptdata)

    return (nodes, edges, cptdata)


def write(bn, filename):
    """
        Writes Bayesian Network to disk file in DSC format

        :param BN bn: Bayesian Network to dump to file
        :param str filename: name of file to write

        :raises FileNotFoundError: if file location nonexistent
    """

    def _write_cpt_pmf(f, node, cpt, values, parental_values={}):
        parent = cpt.parents()[len(parental_values)]
        for value in values[parent]:
            parental_values[parent] = value
            if len(cpt.parents()) > len(parental_values):
                _write_cpt_pmf(f, node, cpt, values, parental_values)
            else:
                value_idxs = [str(values[p].index(v))
                              for p, v in parental_values.items()]
                probs = [str(cpt.cdist(parental_values)[v])
                         for v in values[node]]
                f.write(CPT_ENTRY
                        .format(', '.join(value_idxs), ', '.join(probs)))
        del parental_values[parent]

    with open(filename, 'w', encoding='utf-8') as f:

        f.write(NETWORK_SECT.format('unknown'))

        values = {}
        for node in bn.dag.nodes:
            values[node] = (bn.cnds[node]).node_values()
            f.write(NODE_SECT.format(node, len(values[node]),
                                     '", "'.join(values[node])))

        for node in bn.dag.nodes:
            cpt = bn.cnds[node]
            if cpt.has_parents:
                parents = ' | ' + ', '.join(cpt.parents())
                f.write(PROB_HDR.format(node, parents))
                _write_cpt_pmf(f, node, cpt, values)
            else:
                parents = ''
                f.write(PROB_HDR.format(node, parents))
                f.write('   ' + ', '.join([str((cpt.cdist())[v])
                                           for v in values[node]]) + ';\n')
            f.write(CLOSE_SECT.format())
