#
#   Module used to invoke R code using a Rscript subprocess and
#   passing parameters to, and receiving results from, R functions through
#   temporary JSON files
#
from subprocess import Popen, PIPE
from random import random
from json import dump, load
from os import remove
import os
import subprocess
import pytest
from functools import wraps

R_TMP_FILE = 'call/R/tmp/{}.{}'
SUPPORTED = {'test': ['echo', 'error'],
             'bnlearn': ['dagscore', 'citest', 'pdag2cpdag', 'shd', 'learn',
                         'compare', 'import']}

r_script = None  # Rscript path [str], None => uninitialised, or Exception


def dispatch_r(package, method, params=None):
    """
        Dispatch a request to R code

        :param str package: R "package" to use
        :param str method: the R function to call in the package
        :param dict params: parameters for the R function

        :raises TypeError: if arguments not of correct type
        :raises ValueError: if unsupported package/method requested
        :raises RuntimeError: if R code ends abnormally

        :returns tuple: (return from R function, R stdout)
    """
    if not isinstance(package, str) or not isinstance(method, str) \
            or (params is not None and not isinstance(params, dict)):
        raise TypeError('dispatch_r called with bad arg types')

    if package not in SUPPORTED or method not in SUPPORTED[package] \
            or params == {}:
        raise ValueError('dispatch_r called with bad arg values')

    # Serialise package, method and params as a JSON structure and write
    # to temporary file

    id = int(random() * 10000000)
    json_args = {'package': package, 'method': method, 'params': params}
    infile = R_TMP_FILE.format(id, 'in.json')
    with open(infile, 'w', encoding='utf-8') as f:
        dump(json_args, f, ensure_ascii=False, indent=4)

    # run R entry point script as a subprocess and capture stdout and stderr

    try:
        stdout, stderr = run_r_script(id)
    except Exception:
        remove(infile)
        raise

    # stderr should be empty if no errors occurred in R subprocess

    if len(stderr):
        remove(infile)
        raise RuntimeError('R code abend for id {}: {}'.format(id, stderr))

    # R script will have written its return values to a temporary JSON file
    # which we deserialise and return

    outfile = R_TMP_FILE.format(id, 'out.json')
    with open(outfile, 'r', encoding='utf-8') as f:
        result = load(f)

    # print('result: {}'.format(result))

    remove(infile)
    remove(outfile)

    return (result, stdout)


def initialise_r_environment():
    """
        Returns the Rscript executable path
    """
    global r_script

    # Rscript path has already been determined so just return it
    if isinstance(r_script, str):
        print(f"r_script already set as {r_script}")
        return r_script

    # r_script is not initialised so check environment and determine it
    if r_script is None:
        print("Checking environment and determining Rscript")
        try:
            r_script = get_rscript_path()
            check_bnlearn_installed(r_script)
        except RuntimeError as e:
            r_script = e
            raise
        return r_script

    # r_script is an Exception from when first checked, so re-raise
    raise r_script


def get_rscript_path():
    """
    Dynamically find the Rscript executable.
    Returns the path to Rscript if found, otherwise raises a RuntimeError.
    """
    rscript_path = os.getenv("CAUSALIQ_R_SCRIPT")
    # rscript_path = False  # Uncomment this to simulate absence of R
    if not rscript_path:
        raise RuntimeError(
            "Rscript is not available. Please set the CAUSALIQ_R_SCRIPT "
            "environment variable to the path of the Rscript executable. "
            "Recommended version: R 4.5.1."
        )
    return rscript_path


def check_bnlearn_installed(rscript_path):
    """
    Check if the bnlearn package is installed in R.
    Returns True if installed, otherwise raises a RuntimeError.
    """
    result = subprocess.run(
        [
            rscript_path, "-e",
            "if (!requireNamespace('bnlearn', quietly = TRUE)) "
            "quit(status = 1)"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            "The bnlearn package is not installed in R. Please install it "
            "using install.packages('bnlearn'). Recommended version: "
            "5.0.2."
        )


def run_r_script(id):
    """
    Function to run an R script
    """
    r_script = initialise_r_environment()  # obtain RScript path

    # Construct the command to execute the main.R entry point in Rscript
    command = f'"{r_script}" call\\R\\main.R {id}'

    # Run the command in a subprocess
    r = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)

    # Obtain stdout & stderr as lists of UTF-8 lines, stripping <CRLF>
    stdout = [ln.decode('UTF-8')[0:-2] for ln in r.stdout if len(ln)]
    # print('\n\nR method {}.{} ({}) stdout:\n{}'
    #       .format(package, method, id, '\n'.join(stdout)))
    stderr = [ln.decode('UTF-8')[0:-2] for ln in r.stderr if len(ln)]
    # print('\nR stderr: {}\n<ends>\n'.format('\n'.join(stderr)))

    return stdout, stderr


def requires_r_and_bnlearn(func):
    """
    Decorator to skip tests if R or bnlearn is not available.
    """
    from call.r import initialise_r_environment  # Import initialise function

    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Checking environment")
        try:
            initialise_r_environment()
        except RuntimeError as e:
            print("initialise_r_environment threw runtime")
            pytest.skip(f"Skipping test: {e}")
        return func(*args, **kwargs)

    return wrapper


if __name__ == '__main__':
    dispatch_r('test', 'echo', {'alpha': 0.5})
