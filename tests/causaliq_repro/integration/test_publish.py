
# Integration Tests of CauaslIQ Pipeline deposit publish

import pytest

from causaliq_repro.repro import get_zenodo_token


# Fixture to retrieve the Zenodo sandbox authentication token
@pytest.fixture(scope="session")
def token():
    token = get_zenodo_token(sandbox=True)
    if token is None:
        pytest.skip("User does not have Zenodo upload privilege")
    return token


# -- Actual upload runs - skipped unless authentication token present

# Upload and publish the test root if not already present
def test_root_actual(token):

    print(f"Do some tests here using token : {token}")
    return
