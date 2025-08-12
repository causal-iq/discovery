
# Integration Tests of CauaslIQ Repro Pipeline get_metadata function

import pytest

from tests.common import REPRO_TEST_DATA_DIR
from causaliq_repro.repro import get_zenodo_token
from causaliq_repro.deposit import Deposit

BASE_DIR = REPRO_TEST_DATA_DIR + "integration/root1/"


# Fixture to retrieve the Zenodo sandbox authentication token
@pytest.fixture(scope="session")
def token():
    token = get_zenodo_token(sandbox=True)
    if token is None:
        pytest.skip("User does not have Zenodo upload privilege")
    return token


# -- Dry run calls, so authentication token not needed

# Get metadata from integration test root1
def test_root1_dryrun(capsys, token):

    deposit = Deposit(name="", sandbox=True, base_dir=BASE_DIR)
    recid = deposit.status["recid"]

    info = deposit.get_metadata(dry_run=True, token=token)

    captured = capsys.readouterr()
    print(captured.out)

    assert captured.out == (
        "\n"
        "** Get metadata of '' from sandbox Zenodo\n"
        f"   - get metadata (recid: {recid})\n"
    )

    assert info is None


# Get metadata from integration test root1 hub
def test_hub1_dryrun(capsys, token):

    deposit = Deposit(name="hub1", sandbox=True, base_dir=BASE_DIR)
    recid = deposit.status["recid"]

    info = deposit.get_metadata(dry_run=True, token=token)

    captured = capsys.readouterr()
    print(captured.out)

    assert captured.out == (
        "\n"
        "** Get metadata of 'hub1' from sandbox Zenodo\n"
        f"   - get metadata (recid: {recid})\n"
    )

    assert info is None


# -- Actual get_metadata calls, so need authentication token

# Get metadata from integration test root1
def test_root1_actual(capsys, token):

    deposit = Deposit(name="", sandbox=True, base_dir=BASE_DIR)
    recid = deposit.status["recid"]

    info = deposit.get_metadata(dry_run=False, token=token)

    captured = capsys.readouterr()
    print(captured.out)

    assert captured.out == (
        "\n"
        "** Get metadata of '' from sandbox Zenodo\n"
        f"   - get metadata (recid: {recid})\n"
    )

    assert info["id"] == recid
    assert info["metadata"]["title"] == \
        "CausalIQ Integration Test Root 1 Deposit"
    assert info["files"][0]["filename"] == "readme.md"
    assert info['state'] == "unsubmitted"


# Get metadata from integration test root
def test_hub1_actual(capsys, token):

    deposit = Deposit(name="hub1", sandbox=True, base_dir=BASE_DIR)
    recid = deposit.status["recid"]

    info = deposit.get_metadata(dry_run=False, token=token)

    captured = capsys.readouterr()
    print(captured.out)

    assert captured.out == (
        "\n"
        "** Get metadata of 'hub1' from sandbox Zenodo\n"
        f"   - get metadata (recid: {recid})\n"
    )

    assert info["id"] == recid
    assert info["metadata"]["title"] == \
        "CausalIQ Integration Test Root 1 Hub Deposit"
    assert info["files"][0]["filename"] == "readme.md"
    assert info['state'] == "unsubmitted"
