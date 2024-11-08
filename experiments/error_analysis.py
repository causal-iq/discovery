
# Analyse the conditions associated with learning errors

from pandas import DataFrame, crosstab
from math import ceil

from fileio.common import EXPTS_DIR
from fileio.pandas import Pandas
from core.common import ln
from experiments.common import reference_bn
from learn.trace import Trace
from analysis.trace import TraceAnalysis


def _check_error_args(series, networks, Ns, params, root_dir):
    """
        Check args specified for trace analysis are valid

        :param list series: series to include
        :param list networks: networks to include
        :param list Ns: list of sample sizes to process
        :param dict params: error analysis specific parameters
        :param str root_dir: root directory holding trace files

        :raises TypeError: if any bad argument types
        :raises ValueError: if any bad argument values

        :returns tuple:of criterion, metric
    """
    params = {'criterion': 'eqv', 'metric': 'ok'} if params is None else params

    if (not isinstance(series, list) or not isinstance(networks, list)
            or not isinstance(Ns, list)
            or not isinstance(params, dict)
            or not isinstance(root_dir, str)
            or any([not isinstance(s, str) for s in series])
            or any([not isinstance(n, str) for n in networks])):
        raise TypeError('error_analysis() bad arg types')

    if set(params.keys()) != {'criterion', 'metric'}:
        raise ValueError('error_analysis() bad params keys')

    if (params['criterion'] not in ['none', 'eqv', 'mi', 'lt5'] or
            params['metric'] not in ['ok', 'ok-eqv', 'status', 'al']):
        raise ValueError('error_analysis() bad params values')

    return (params['criterion'], params['metric'])


def active_learning(change):
    """
        Returns active learning status of change trace record

        :param dict change: individual change record from trace

        :returns str: summarising active learning interaction
                        "add", "rev", "del" changes without AL interaction
                        "ok_add", "ok_rev", "ok_del" AL confirms change
                        "eqv_add" AL swaps orientation in add
    """
    activity = change['activity'][:3]
    if (change['knowledge'] is not None and
            change['knowledge'][0] != 'act_cache' and
            change['knowledge'][1] is not None):

        # There is a knowledge entry where correct is not None and it is
        # not serviced from cache - i.e. a request is made to the expert

        op = change['knowledge'][2]

        if op == 'no_op':  # AL confirmed activity
            al = 'ok_' + activity

        elif op == 'swap_best':  # AL swapped orientation
            al = 'eqv_add'

        elif op == 'stop_edge':  # AL stop add/rev of extra arc - obsolete
            al = ('ext_rev' if change['blocked'] is not None and
                  len(change['blocked']) and
                  change['blocked'][0][0] == 'reverse' else 'ext_add')

        else:  # AL stop_add, stop_rev, stop_del, ext_add, ext_rev
            al = op
    else:

        # No request made to expert - just return add/del/rev

        al = activity

    # if al in ['add', 'del', 'rev']:
    #     info = 'none'
    # elif al in ['ext_add', 'ok_del', 'stop_del', 'ext_rev']:
    #     info = 'exist'
    # else:
    #     info = 'orient'

    return al


def _analyse_errors(criterion, metric, analysis):
    """
        Analyse learning errors from a specfic trace.

        :param str criterion: criterion by which you wish to analyse metric:
                                "eqv": is it an equivalent change
                                "mi": is true and actual MI similar
                                "lt5": many low support cells
        :param str metric: metric analysed:
                                "ok": arc resulting from change ok
                                "ok-eqv": arc resulting from change ok or eqv
                                "status": all statuses differentiated
        :param TraceAnalysis analysis: a single trace analysis to be analysed
                                       for learning errors
    """
    # print(analysis)
    changes = DataFrame(analysis.trace).to_dict(orient='records')

    errors = []
    iter = -1
    for change in changes:
        iter += 1
        if change['activity'] in ['init', 'stop']:
            continue

        if criterion == 'none':
            crit_val = 'none'

        if criterion in ['eqv', 'mi']:
            crit_val = ('eqv' if change['margin'] == 0 and
                        change['arc'][0] == change['arc_2'][1] and
                        change['arc'][1] == change['arc_2'][0]
                        else 'non')
            if criterion == 'mi':
                delta = change['delta/score']
                mi = change['MI']
                if delta == 0 and mi == 0:
                    add = 'sim'
                elif mi > 1.5 * delta or delta == 0:
                    add = 'hi'
                elif mi < 0.5 * delta:
                    add = 'lo'
                else:
                    add = 'sim'
                crit_val += '-' + add

        elif criterion == 'lt5':
            lt5 = change['lt5']
            if lt5 == 0:
                crit_val = '0'
            else:
                lt5 = ceil(ln(1.0 / lt5, 2))
                crit_val = '1/{} --> 1/{}'.format(2 ** lt5, 2 ** (lt5 - 1))
                crit_val = crit_val.replace('1/2 --> 1/1', '1/2 --> 1')

        if metric == 'ok-eqv':
            metric_val = 'ok' if change['status'] in ['ok', 'eqv'] else 'err'
        elif metric == 'ok':
            metric_val = 'ok' if change['status'] == 'ok' else 'err'
        elif metric == 'status':
            metric_val = change['status']
        elif metric == 'al':
            metric_val = active_learning(change)

        errors.append(
            {'network': analysis.context['id'].split('/')[-2],
             'N': analysis.context['N'],
             'iter': iter,
             'activity': change['activity'],
             'status': change['status'],
             'metric': metric_val,
             'criterion': crit_val}
        )

    return errors


def error_analysis(series, networks, Ns, params=None, root_dir=EXPTS_DIR):
    """
        Entry point into analysing learning errors

        :param list series: series to include
        :param list networks: networks to include
        :param list Ns: list of sample sizes to process
        :param dict params: error analysis specific parameters
        :param str root_dir: root directory holding trace files

        :raises TypeError: if any bad argument types
        :raises ValueError: if any bad argument values

        :returns Dataframe: contingency table of criterion versus metric values
                            for testing purposes
    """
    criterion, metric = _check_error_args(series, networks, Ns, params,
                                          root_dir)

    print('\nAnalysing {} metric for {} criterion in series {} ...\n'
          .format(metric, criterion, series))

    # Loop over specified networks and series

    errors = []
    for network in networks:
        for _series in series:

            # get series properties, reference BN and learning traces

            ref, _ = reference_bn(network)
            traces = Trace.read(_series + '/' + network, root_dir)
            if traces is None:
                print('\nNo traces available for network {} in series {}'
                      .format(network, _series))
                continue

            # get dataset so MI can be computed

            n_max = Ns[-1] if Ns[-1] <= 10**7 else 10**7
            data = Pandas.read(root_dir + '/datasets/' + network + '.data.gz',
                               dstype='categorical').sample

            # loop over traces by increasing sample size order + sample number

            sorted_keys = sorted([tuple(k.split('_')) for k in traces])
            sorted_keys = {t[0]: [t2[1] for t2 in sorted_keys
                                  if t2[0] == t[0] and len(t2) == 2]
                           for t in sorted_keys}
            for N, samples in sorted_keys.items():
                N = int(N.replace('N', ''))
                if N not in Ns:
                    continue  # ignore N values not in sample size list

                samples = [None] if samples == [] else samples
                for sample in samples:
                    key = 'N{}'.format(N) + ('' if sample is None
                                             else '_{}'.format(sample))
                    trace = traces[key]

                    analysis = TraceAnalysis(trace, ref.dag,
                                             data[:N] if N <= n_max else None)
                    print(analysis)

                    errors += _analyse_errors(criterion, metric, analysis)

    df = DataFrame(errors)
    print(df)
    ct = crosstab(df['metric'], df['criterion'])
    num_in_crit = ct.sum().to_dict()  # number of entries in each criterion
    total_num = sum(num_in_crit.values())
    print(ct)
    print(num_in_crit, total_num)

    print('\n Criterion (fract)          Metric Distribution in criteria')
    print('-------------------------  ---------------------------------------')
    for criterion, dist in ct.to_dict().items():
        frac_crit = num_in_crit[criterion] / total_num
        # dist = ', '.join(['{}: {:1.3f}'.format(k, v / sum(dist.values()))
        #                  for k, v in dist.items()])
        dist = ', '.join(['{}: {:1.3f}'.format(k, v / total_num)
                         for k, v in dist.items()])
        print('{:>16s} ({:1.3f})    {}'.format(criterion, frac_crit, dist))

    return ct.to_dict(orient='index')
