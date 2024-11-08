#
#   Functions to read Tetrad format graph specification files
#

import re

from fileio.common import is_valid_path, FileFormatError
from core.graph import DAG, PDAG

EDGE = re.compile(r'^\d+\.\s(\w+)\s(\-\-[\>\-])\s(\w+)$')


def read(path):
    """
        Reads in a graph from a Tetrad format graph specification file

        :param str path: full path name of file
        :param list all_nodes: optional specification of nodes
        :param bool strict: whether strict validation should be applied

        :raises TypeError: if argument types incorrect
        :raises ValueError: if file suffix not '.tetrad'
        :raises FileNotFoundError: if specified files does not exist
        :raises FileFormatError: if file contents not valid

        :returns DAG/PDAG: DAG or PDAG specified in file
    """
    if not isinstance(path, str):
        raise TypeError('tetrad.read() bad arg type')

    if path.lower().split('.')[-1] != 'tetrad':
        raise ValueError('tetrad.read() bad file suffix')

    is_valid_path(path)

    pdag = False
    try:
        with open(path, newline='', encoding='utf-8') as f:
            num_line = 0
            error = ''
            edges = []
            for line in f:
                line = line.rstrip('\r\n')
                if not line:  # ignore blank lines
                    continue
                num_line += 1

                # ignore these non-structural elements of the file

                if (line in ['Graph Attributes:', 'Graph Node Attributes:']
                    or line.startswith('Score: ')
                        or line.startswith('BIC: ')):
                    continue

                if (num_line == 1 and line != 'Graph Nodes:') or \
                        (num_line == 3 and line != 'Graph Edges:'):
                    error = ' invalid section header'
                    break

                if num_line == 2:
                    nodes = line.replace(';', ',').split(',')

                if num_line > 3:
                    match = EDGE.match(line)
                    if not match:
                        error = ' invalid edge: ' + line
                        break
                    edges.append((match.group(1),
                                  ('->' if match.group(2) == '-->' else '-'),
                                  match.group(3)))
                    if match.group(2) == '---':
                        pdag = True

        if num_line < 1:
            error = ' is empty'

    except UnicodeDecodeError:
        error = ' not text'

    if error:
        raise FileFormatError('file {}{}'.format(path, error))

    return(PDAG(nodes, edges) if pdag else DAG(nodes, edges))


def write(pdag, path):
    """
        Writes a PDAG to a Tetrad format graph specification file (no scores
        are included).

        :param PDAG pdag: pdag to write to file
        :param str path: full path name of file

        :raises TypeError: if bad arg types
        :raises FileNotFoundError: if path to file does not exist
    """
    if not isinstance(pdag, PDAG) or not isinstance(path, str):
        raise TypeError('tetrad.write() bad arg type')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('Graph Nodes:\n')
        f.write('{}\n'.format(';'.join(pdag.nodes)))
        f.write('\nGraph Edges:\n')
        num_edges = 0
        for edge, type in pdag.edges.items():
            num_edges += 1
            f.write('{}. {} {} {}\n'
                    .format(num_edges, edge[0],
                            ('-->' if type.value[3] == '->' else '---'),
                            edge[1]))
