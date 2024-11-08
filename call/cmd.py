
#   Run an OS command

from subprocess import Popen, PIPE


def dispatch_cmd(args):
    """
        Run specified command at OS prompt.

        :param list args: command and arguments e.g. ['ls', '-l']

        :raises TypeError: if bad arg type
        :raises ValueError: if list is empty

        :returns list: lines written to stdout
    """
    if (not isinstance(args, list)
            or any([not isinstance(s, str) for s in args])):
        raise TypeError('dispatch_cmd() bad args type')

    if not len(args):
        raise ValueError('dispatch_cmd() has empty args')

    try:
        r = Popen(args, stdout=PIPE, stderr=PIPE, shell=True)
    except Exception as e:
        raise RuntimeError('dispatch_cmd(): {}'.format(e))

    stdout = [ln.decode('UTF-8')[:-2] for ln in r.stdout if len(ln)]
    stderr = [ln.decode('UTF-8')[:-2] for ln in r.stderr if len(ln)]

    if len(stderr):
        raise RuntimeError('dispatch_cmd() stderr: {}', stderr)

    return stdout
