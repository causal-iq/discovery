#
#   Functions to read Bayesys format graph specification files
#

import csv

from fileio.common import is_valid_path, FileFormatError
from core.graph import PDAG, DAG
from learn.knowledge import Knowledge, RuleSet


def read(path, all_nodes=None, strict=True):
    """
        Reads in a graph from a Bayesys format graph specification file

        :param str path: full path name of file
        :param list all_nodes: optional specification of nodes
        :param bool strict: whether strict validation should be applied

        :raises TypeError: if argument types incorrect
        :raises FileNotFoundError: if specified files does not exist
        :raises FileFormatError: if file contents not valid

        :returns DAG/PDAG: DAG or PDAG specified in file
    """
    is_valid_path(path)

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        num_line = 0
        error = ''
        nodes = list(all_nodes) if all_nodes else []
        edges = []
        try:
            for row in reader:
                num_line += 1
                if len(row) != 4:
                    error = ', line {} has not got 4 values'.format(num_line)
                    break
                if num_line == 1:
                    if not strict:  # fix some known content problems
                        row[0] = row[0].replace('Id', "ID")
                        row[1] = row[1].replace('e.1', 'e 1')
                        row[3] = row[3].replace('e.2', 'e 2')
                    if row != ['ID', 'Variable 1', 'Dependency', 'Variable 2']:
                        error = ' has bad header ({})'.format(row)
                        break
                    continue
                if num_line > 1 and row[0] != '{}'.format(num_line - 1):
                    error = ', line {} has bad id'.format(num_line)
                    break

                if all_nodes and (row[1] not in all_nodes
                                  or row[3] not in all_nodes):
                    raise FileFormatError('file {} contains node not in {}'
                                          .format(path, all_nodes))
                if row[1] not in nodes:
                    nodes.append(row[1])
                if row[3] not in nodes:
                    nodes.append(row[3])
                edges.append((row[1], row[2], row[3]))

        except UnicodeDecodeError:
            raise FileFormatError('file {} not in CSV format'.format(path))

        if num_line < 1:
            error = ' is empty'

        if error:
            raise FileFormatError('file {}{}'.format(path, error))
        else:
            graph = PDAG(nodes, edges)
            return graph if not graph.is_DAG() else DAG(nodes, edges)


def write(pdag, path):
    """
        Writes a PDAG to a Bayesys format graph specification file (only
        has details of edges, not unconnected nodes or any parameters)

        :param PDAG pdag: pdag to write to file
        :param str path: full path name of file

        :raises TypeError: if bad arg types
        :raises FileNotFoundError: if path to file does not exist
    """
    if not isinstance(pdag, PDAG) or not isinstance(path, str):
        raise TypeError('bayesys.write() bad arg type')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('ID,Variable 1,Dependency,Variable 2\n')
        num_edges = 0
        for edge, type in pdag.edges.items():
            num_edges += 1
            f.write('{},{},{},{}\n'
                    .format(num_edges, edge[0], type.value[3], edge[1]))


def read_constraints(path, nodes):
    """
        Reads in constraints from Bayesys format files

        :param str path: full path name of file
        :param set nodes: nodes in dag

        :raises TypeError: if argument types incorrect
        :raises FileNotFoundError: if specified files does not exist
        :raises FileFormatError: if file contents not valid

        :returns Knowledge: encapsulating the constraints
    """
    if not isinstance(path, str) or not isinstance(nodes, set):
        raise TypeError('read_constraints() bad arg type')

    if '/constraintsDirected_' in path:
        knowledge = directed(path, nodes)
    elif '/constraintsTemporal_' in path:
        knowledge = temporal(path, nodes)
    else:
        raise FileFormatError('constraints() unrecognised file name')
    return knowledge


def temporal(path, nodes):
    """
        Reads in temporal constraints from Bayesys format file

        :param str path: full path name of file
        :param set nodes: nodes in dag

        :raises TypeError: if argument types incorrect
        :raises FileNotFoundError: if specified files does not exist
        :raises FileFormatError: if file contents not valid

        :returns Knowledge: encapsulating the constraints
    """
    is_valid_path(path)

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        num_line = 0
        error = ''
        tiers = None
        try:
            for row in reader:
                num_line += 1
                if num_line == 1:
                    if (row[0] != 'ID' or row[-1] != 'END' or len(row) < 4
                        or not all([len(h) == 2 and h[0] == 'Tier'
                                    and str(i + 1) == h[1] for i, h in
                                    enumerate([c.split(' ')
                                               for c in row[1:-1]])])):
                        error = ' has bad header ({})'.format(row)
                        break
                    tiers = [list() for n in range(len(row) - 2)]
                    continue

                if tiers is not None and len(tiers) != len(row) - 2:
                    error = (', line {} has not got {} tiers'
                             .format(num_line, len(tiers)))
                    break
                if row[0] != '{}'.format(num_line - 1):
                    error = ', line {} has bad id'.format(num_line)
                    break

                for i, node in enumerate(row[1:-1]):
                    if node != '':
                        if node not in nodes:
                            error = (', line {} has unknown node')
                            break
                        if node in [n for t in tiers for n in t]:
                            error = (', line {} has node already in tier')
                            break
                        tiers[i] += [node]

        except UnicodeDecodeError:
            raise FileFormatError('file {} not in CSV format'.format(path))

        if num_line < 1:
            error = ' is empty'

        if error:
            raise FileFormatError('file {}{}'.format(path, error))

        print('\nTiers are: {}\n'.format(tiers))
        stop = {}
        for i, tier in enumerate(tiers):
            ancestors = [n for tier in [tiers[j] for j in range(0, i)]
                         for n in tier]
            # print('Ancestors of tier {} are {}'.format(i, ancestors))
            for to in ancestors:
                for frm in tier:
                    stop.update({(frm, to): True})

        return Knowledge(rules=RuleSet.STOP_ARC, params={'stop': stop})


def directed(path, nodes):
    """
        Reads in directed constraints from Bayesys format file

        :param str path: full path name of file
        :param set nodes: nodes in dag

        :raises TypeError: if argument types incorrect
        :raises FileNotFoundError: if specified files does not exist
        :raises FileFormatError: if file contents not valid

        :returns Knowledge: encapsulating the constraints
    """
    is_valid_path(path)

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        num_line = 0
        error = ''
        reqd = {}
        try:
            for row in reader:
                num_line += 1
                if len(row) != 3:
                    error = ', line {} has not got 3 values'.format(num_line)
                    break
                if num_line == 1:
                    if row != ['ID', 'Parent', 'Child']:
                        error = ' has bad header ({})'.format(row)
                        break
                    continue
                if row[0] != '{}'.format(num_line - 1):
                    error = ', line {} has bad id'.format(num_line)
                    break
                if (row[1] == row[2] or row[1] not in nodes
                        or row[2] not in nodes):
                    error = ', line {} has bad nodes'.format(num_line)
                    break
                if (row[2], row[1]) in reqd:
                    error = (', line {} has conflicting constraint'
                             .format(num_line))
                    break

                reqd.update({(row[1], row[2]): True})

        except UnicodeDecodeError:
            raise FileFormatError('file {} not in CSV format'.format(path))

        if num_line < 1:
            error = ' is empty'

        if error:
            raise FileFormatError('file {}{}'.format(path, error))

        initial = DAG(list(nodes), [(e[0], '->', e[1]) for e in reqd])

        return Knowledge(rules=RuleSet.REQD_ARC,
                         params={'reqd': reqd, 'initial': initial})
