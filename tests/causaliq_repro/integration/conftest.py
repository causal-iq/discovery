
from pytest import fixture, skip

from causaliq_repro.repro import get_zenodo_token


@fixture  # Determine whether output should be displayed (-s flag) or not
def echo_output(request):
    return request.config.getoption("capture") == "no"


# Fixture to retrieve the Zenodo sandbox authentication token
@fixture(scope="session")
def token():
    token = get_zenodo_token(sandbox=True)
    if token is None:
        skip("User does not have Zenodo upload privilege")
    return token
