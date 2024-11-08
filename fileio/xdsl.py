
#   Read and write XDSL format BN definition files produced by GenIe

from xml.dom import minidom
from math import prod
from re import compile, sub, findall

from fileio.common import is_valid_path, FileFormatError
from core.cpt import CPT, NodeValueCombinations
from core.lingauss import LinGauss

XDSL_HDR = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            + '<!-- This network was created in BNBENCH -->\n'
            + '<smile version="1.0" id="bnbench" numsamples="100000" '
            + 'discsamples="100000">\n    <nodes>\n')
NODES_END = '    </nodes>\n'
XDSL_END = '</smile>\n'

STARTS_WITH_LETTER = compile(r'^[a-zA-Z].*$')

GENIE_HDR = ('    <extensions>\n' +
             '        <genie version="1.0" app="GeNIe 4.0.1922.0 ACADEMIC" ' +
             'name="Network1">\n')
GENIE_NODE = ('            <node id="{0:}">\n' +
              '                <name>{0:}</name>\n' +
              '                <interior color="e5f6f7" />\n' +
              '                <outline color="000080" />\n' +
              '                <font color="000000" name="Arial"' +
              ' size="{1:}" />\n' +
              '                <position>{2:} {3:} {4:} {5:}</position>\n' +
              '            </node>\n')
GENIE_END = '        </genie>\n    </extensions>\n'

# Patterns relating to equation definition

EQN_VALID_CHARS = compile(r'^[a-zA-Z0-9\=\(\)\_\.\,\*\+\-]+$')
EQN_NORMAL = r'[\+\-]?Normal\(.+?\,.+?\)'
EQN_SPLIT_TERMS = compile(r'[\+|\-]')


def _to_float(string):
    """
        Convert string to float

        :param str string: string to convert to a float

        :returns float/None: float value or None if conversion impossible
    """
    try:
        x = float(string)
    except ValueError:
        x = None
    return x


def _parse_equation_normal(string):
    """
        Parse the string defining normal function

        :param str string: Normal definition as textual string

        :returns tuple: of floats: (mean, sd)
    """

    # Determine whether negative or postive, remove leading sign

    if string[0] in {'-', '+'}:
        positive = True if string[0] == '+' else False
        string = string[1:]
    else:
        positive = True

    # Extract & check mean and SD, converting to floats

    mean, sd = tuple(string.replace('Normal(', '').replace(')', '').split(','))
    mean = _to_float(mean)
    sd = _to_float(sd)
    if mean is None or sd is None or sd < 0.0:
        raise FileFormatError('xdsl.read() invalid Normal args')
    mean = mean if positive is True else -1.0 * mean

    return (mean, sd)


def _parse_equation_coeffs(string, parents):
    """
        Parse and validate the parent coefficients in the equation.

        :param str string: text containing coefficients
        :param list parents: parent variables (defined in parents elem)

        :raise FileFormatError: if definition is invalid

        :returns dict: of parent coefficients, {p1: coeff1, ...}
    """
    if len(string) == 0 and len(parents) == 0:
        coeffs = {}

    coeffs = {}
    if len(string) > 0 and len(parents) > 0:

        # split string at "+" and "-" symbols

        terms = EQN_SPLIT_TERMS.split(string)
        terms = terms[1:] if terms[0] == '' else terms

        for term in terms:
            pos = string.find(term)
            sign = 1.0 if pos == 0 or string[pos - 1] == '+' else -1.0
            term = term.split('*')
            if len(term) == 1:
                parent = term[0]
                coeff = 1.0
            elif len(term) == 2:
                coeff = _to_float(term[0])
                if coeff is None:
                    parent = term[0]
                    coeff = _to_float(term[1])
                else:
                    parent = term[1]
            else:
                coeff = None
            if coeff is None or parent in coeffs:
                raise FileFormatError('xdsl.read() bad coeffs')
            coeffs.update({parent: coeff * sign})

    elif len(string) > 0 or len(parents) > 0:
        coeffs = None

    if set(coeffs) != set(parents):
        raise FileFormatError('xdsl.read() bad coeffs')

    return coeffs


def _parse_equation_definition(definition, node, parents):
    """
        Parse and validate the textual equation definition. Only supports
        Linear Gaussian equations.

        :param str definition: textual equation definition
        :param str node: DAG node name
        :param list parents: parents as defined in parents element

        :raises FileFormatError: if definition is invalid

        :returns dict: parsed equation, {'mean': mean, 'sd': sd,
                                         'coeffs': {p1: c1, ...}}
    """
    if not EQN_VALID_CHARS.match(definition):
        raise FileFormatError('xdsl.read() equation invalid chars')

    # remove spaces, check just one equals sign with something either side

    definition = definition.replace(' ', '').split('=')
    if (len(definition) != 2 or len(definition[0]) == 0
            or len(definition[1]) == 0):
        raise FileFormatError('xdsl.read() - missing/misplaced/repeated =')

    # LHS should match node name

    if node != definition[0]:
        raise FileFormatError('xdsl.read() - node / LHS mismatch {}/{}'
                              .format(node, definition[0]))

    # rhs shouldn't start with +, or end with + or -

    rhs = definition[1]
    if rhs[0] == '+' or rhs[-1] in {'-', '+'}:
        raise FileFormatError('xdsl.read() - illegal leading/trailing +/-')

    # Look for, and extract single Normal(...) term in rhs

    normal = findall(EQN_NORMAL, rhs)
    if len(normal) != 1:
        raise FileFormatError('xdsl.read() - no/multiple Normals')
    rhs = rhs.replace(normal[0], '')
    mean, sd = _parse_equation_normal(normal[0])
    coeffs = _parse_equation_coeffs(rhs, parents)

    return {'mean': mean, 'sd': sd, 'coeffs': coeffs}


def _process_equation_element(elem):
    """
        Process a <equation> element from the XDSL file.

        :param Element elem: XDSL element (<equation>) describing a node

        :raises FileFormatError: if not a correctly formed <equation> element

        :returns tuple: (node, coeffs, mean, sd) of node information
    """

    #   Check <cpt> elem has id attribute which is BN node name

    id = elem.attributes.getNamedItem('id')
    if id is None:
        raise FileFormatError('xdsl.read() <equation> has no id attribute')
    node = id.nodeValue

    # <equation> should have one <definition>, and optionally one parents node

    parents = [n for n in elem.childNodes if n.nodeName == 'parents']
    definition = [n for n in elem.childNodes if n.nodeName == 'definition']
    if len(definition) != 1 or len(parents) > 1:
        raise FileFormatError('xdsl.read() <equation> bad children')

    #   Process <parents>

    if len(parents):
        try:
            parents = parents[0].firstChild.nodeValue.split()
        except AttributeError:
            raise FileFormatError('xdsl.read() <parents> has no values')

    #   Process <definition>

    try:
        definition = definition[0].firstChild.nodeValue
    except AttributeError:
        raise FileFormatError('xdsl.read() <definition> is empty')

    # parse the equation definition and return it

    return {node: _parse_equation_definition(definition, node, parents)}


def _process_cpt_element(elem):
    """
        Process a <cpt> element from the XDSL file.

        :param Element elem: XDSL element (<cpt>) describing a node

        :raises FileFormatError: if not a correctly formed <cpt> element

        :returns tuple: (node, values, parents, probs) of node information
    """

    #   Check <cpt> elem has id attribute which is BN node name

    id = elem.attributes.getNamedItem('id')
    if id is None:
        raise FileFormatError('xdsl.read() <cpt> has no id attribute')
    node = id.nodeValue

    # <cpt> must have 2 or more <state> children, optionally a <parents> child,
    # and finally a <probabilities> child

    children = [n for n in elem.childNodes if n.nodeType == 1]
    states = [n for n in elem.childNodes if n.nodeName == 'state']
    parents = [n for n in elem.childNodes if n.nodeName == 'parents']
    probs = [n for n in elem.childNodes if n.nodeName == 'probabilities']
    if len(states) < 2 or len(parents) > 1 or len(probs) != 1 or \
            len(states) + len(parents) + len(probs) != len(children):
        raise FileFormatError('xdsl.read() <cpt> has missing/invalid children')

    #   Process <state>s (i.e. values) node can take

    values = []
    for state in states:
        id = state.attributes.getNamedItem('id')
        if id is None:
            raise FileFormatError('xdsl.read() <state> has no id')
        values.append(id.nodeValue)

    #   Process <parents>

    if len(parents):
        try:
            parents = parents[0].firstChild.nodeValue.split()
        except AttributeError:
            raise FileFormatError('xdsl.read() <parents> has no values')

    #   Process <probabilities>

    if len(probs):
        try:
            probs = [float(v) for v in probs[0].firstChild.nodeValue.split()]
        except AttributeError:
            raise FileFormatError('xdsl.read() <probabilities> has no values')

    return {node: {'values': values, 'parents': parents, 'probs': probs}}


def _bn_data(xdsl_data, correct):
    """
        Return BN (DAG and CNDs) data in format required by BN constructor.

        :param dict xdsl_data: BN data in format extracted from XDSL, that is
                               {node: {parents, values, probs}} for CPTs
                               {node: {mean, sd, coeffs}} for Linear Gaussian
        :param bool correct: whether to correct sets of PMF probabilities that
                             don't sum to 1.

        :raises FileFormatError: if incorrect <probabilities> detected
        :raises ValueError: if PMF probabilities don't sum to 1

        :returns tuple: of BN data in required format: ([nodes],
                        [(n1, '->', n2)], CNDs)
    """
    def _pmf(node, pmf):  # check & optionally adjust pmf
        total = sum(pmf.values())
        if total < 0.999999 or total > 1.000001:
            if correct:
                print('\n*** Correcting probabilities for ' + node)
                pmf = {v: pr / total for v, pr in pmf.items()}
            else:
                raise ValueError('xdsl.read() sum ' + node + ' probs not 1')
        return pmf

    # Collect parent, value and CPT data for all categorical nodes

    parents = {n: d['parents'] for n, d in xdsl_data.items()
               if 'parents' in d}
    values = {n: d['values'] for n, d in xdsl_data.items() if 'values' in d}
    probs = {n: d['probs'] for n, d in xdsl_data.items() if 'probs' in d}

    # Mixed continuous / categorical networks not supported

    if len(probs) != len(xdsl_data) and len(probs) != 0:
        raise FileFormatError('xdsl.read() mixed networks unsupported')

    # Loop over XDSL data for each node extracting CPT/LinGauss info

    cnds = {}
    for node, data in xdsl_data.items():

        if 'coeffs' in data:  # Linear Gaussian node
            cnds.update({node: (LinGauss, data)})
            parents.update({node: list(data['coeffs'])})

        elif len(parents[node]):  # categorical non-orphan node
            num_pvs = prod([len(values[p]) for p in parents[node]])
            if num_pvs * len(values[node]) != len(probs[node]):
                raise FileFormatError('xdsl.read() wrong # of probabilities')
            pvs = {p: values[p] for p in parents[node]}
            num_values = len(values[node])
            offset = 0
            cpt = []
            for pv in NodeValueCombinations(pvs, False):
                pr = probs[node][offset:offset + num_values]
                pmf = {v: pr for v, pr in zip(values[node], pr)}
                cpt.append((pv, _pmf(node, pmf)))
                offset += num_values
            cnds.update({node: (CPT, cpt)})

        else:  # categorical orphan node
            if len(probs[node]) != len(values[node]):
                raise FileFormatError('xdsl.read() wrong # of probabilities')
            pmf = {v: pr for v, pr in zip(values[node], probs[node])}
            cpt = _pmf(node, pmf)
            cnds.update({node: (CPT, cpt)})

    edges = [(p, '->', n) for n in parents.keys() for p in parents[n]]
    return (list(xdsl_data), edges, cnds)


def read(path, correct=False):
    """
        Reads in a BN from a XDSL format BN specification file

        :param str path: full path name of file
        :param bool correct: whether to correct sets of PMF probabilities that
                             don't sum to 1.

        :raises TypeError: if path is not a string
        :raises ValueError: if PMF probabilities don't sum to 1 (adjust=False)
        :raises FileNotFoundError: if file does not exist
        :raises FileFormatError: if file contents not valid

        :returns tuple: nodes, edges, CND_specs
    """
    if not isinstance(path, str) or not isinstance(correct, bool):
        raise TypeError('xdsl.read() bad arg type')

    is_valid_path(path)

    with open(path) as xml_file:
        try:
            xdsl = minidom.parse(xml_file)
        except Exception:
            raise FileFormatError('xdsl.read() invalid XML')

    # Check root element is <smile>

    root = xdsl.documentElement
    if root.tagName != 'smile':
        raise FileFormatError('xdsl.read() invalid root')

    # Check top levels under root are <nodes> and, optionally, <extensions>

    topLevel = [n for n in root.childNodes if n.nodeType == 1]
    if len(topLevel) < 1 or len(topLevel) > 2 \
            or topLevel[0].nodeName != 'nodes' or \
            (len(topLevel) == 2 and topLevel[1].nodeName != 'extensions'):
        raise FileFormatError('xdsl.read() invalid top level')

    # Process all the child elements of <nodes> to get node info

    xdsl_data = {}
    for child in topLevel[0].childNodes:
        if child.nodeType != 1:
            continue

        if child.nodeName == 'cpt':
            xdsl_data.update(_process_cpt_element(child))

        elif child.nodeName == 'equation':
            xdsl_data.update(_process_equation_element(child))

        else:
            raise FileFormatError('xdsl.read() bad elem under <nodes>')

    #   Return BN data in format required by BN constructor

    return _bn_data(xdsl_data, correct)


def genie_str(string, prefix):
    """
        Cleanses string so that it conforms to Genie requirements for a
        node name or value.

        :param str string: node name or value to cleanse
        :param str prefix: prefix to use if doesn't start with letter

        :returns str: cleansed string that meets Genie requirements
    """
    cleansed = string if STARTS_WITH_LETTER.match(string) else prefix + string

    return sub('[^0-9a-zA-Z]+', '_', cleansed)


def write_genie_extension(f, partial_order):
    """
        Writes the XDSL Genie extension XML which defines node placement
        on the visual drawing of the network.

        :param TextIOWrapper f: handle to write text
        :param list partial_order: of (list of) nodes in each partial order
                                   group of nodes
    """
    f.write(GENIE_HDR)
    top = 0
    for group in partial_order:
        top += 100
        left = 20
        for node in group:
            f.write(GENIE_NODE.format(genie_str(node, 'N'), 8, left, top,
                                      left + 80, top + 40))
            left += 100
    f.write(GENIE_END)


def _write_cpt(f, node, cpt, node_values, genie):
    """
        Write a <cpt> element

        :param TextIOWrapper f: handle for writing to file
        :param str node: node that CPT relates to
        :param CPT cpt: CPT to represent in XDSL
        :param dict node_values: node values for categorical nodes
        :param bool genie: whether to be in Genie format
    """

    # <cpt> opening element

    f.write('        <cpt id="{}">\n'.format(node))

    # node values (states)

    for value in node_values[node]:
        f.write('            <state id="{}" />\n'
                .format(genie_str(value, 'S') if genie is True else value))

    # process CPT to get parents name if any, and then probabilities
    # associated with each parent value combo and node value.
    # Take care that values always processed in alphabetical order.

    probs = []
    if cpt.has_parents:
        parents = ' '.join([p for p in cpt.parents()])
        f.write('            <parents>' + parents + '</parents>\n')
        pvs = {p: node_values[p] for p in cpt.parents()}
        for pvc in NodeValueCombinations(pvs):
            probs.extend([cpt.cdist(pvc)[v] for v in node_values[node]])
    else:
        probs.extend([cpt.cdist()[v] for v in node_values[node]])
    f.write('            <probabilities>{}</probabilities>\n'
            .format(' '.join(['{}'.format(p) for p in probs])))

    f.write('        </cpt>\n')


def _write_equation(f, node, lingauss, genie):
    """
        Write <equation> element representing linear gaussian

        :param TextIOWrapper f: handle for writing to file
        :param str node: node that CPT relates to
        :param LinGauss lingauss: Linear Gaussian to represent in XDSL
        :param bool genie: whether to be in Genie format
    """
    f.write('        <equation id="{}">\n'
            .format(genie_str(node, 'N') if genie is True else node))

    if lingauss.coeffs != {}:
        parents = ' '.join([p for p in lingauss.coeffs])
        f.write('            <parents>' + parents + '</parents>\n')

    f.write('            <definition>{}={}</definition>\n'
            .format(node, lingauss))

    f.write('        </equation>\n')


def write(bn, filename, genie=False):
    """
        Writes Bayesian Network to disk file in XDSL format. Ensure
        everything written out with node name and values ordered
        alphabetically.

        :param BN bn: Bayesian Network to dump to file
        :param str filename: name of file to write
        :param bool genie: whether files needs to be read by Genie - if so,
                           will ensure all node and state names start with
                           a letter.

        :raises TypeError: bad argument types
        :raises FileNotFoundError: if file location nonexistent
    """
    if bn.__class__.__name__ != 'BN' or not isinstance(filename, str):
        raise TypeError('xdsl.write() bad arg type')

    # Ensure all node names are Genie-friendly if required

    if genie is True:
        name_map = {n: genie_str(n, 'N') for n in bn.dag.nodes}
        bn.rename(name_map)

    # node_values has each node's allowed values ordered alphabetically.

    node_values = {n: bn.cnds[n].node_values() for n in bn.dag.nodes
                   if isinstance(bn.cnds[n], CPT)}
    parents = {n: bn.dag.parents[n] if n in bn.dag.parents else []
               for n in bn.dag.nodes}
    partial_order = bn.dag.partial_order(parents)
    order = [n for g in partial_order for n in g]

    with open(filename, 'w', encoding='utf-8') as f:

        f.write(XDSL_HDR)

        for node in order:
            if isinstance(bn.cnds[node], CPT):
                _write_cpt(f, node, bn.cnds[node], node_values, genie)
            else:
                _write_equation(f, node, bn.cnds[node], genie)

        f.write(NODES_END)

        if genie is True:
            write_genie_extension(f, partial_order)

        f.write(XDSL_END)
