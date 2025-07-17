
# Integration Tests of CauaslIQ Pipeline repro.py upload

import pytest

from causaliq_repro.repro import get_zenodo_token
from causaliq_repro.deposit import Deposit


# Fixture to retrieve the Zenodo sandbox authentication token
@pytest.fixture(scope="session")
def auth_token():
    token = get_zenodo_token(live=False)
    if token is None:
        pytest.skip("User does not have Zenodo upload privilege")
    return token


# -- Dry upload runs - don't require authentication so always run

# Upload the production root - dry run
def test_root_dryrun(capsys):

    deposit = Deposit(name="", live=False)

    deposit.upload(dry_run=True, auth_token=None)

    captured = capsys.readouterr()
    print(f"\n\nCaptured output: \n{captured.out}")

    assert captured.out == (
        "\n** Uploading '' to sandbox Zenodo\n"
        "   - draft deposit created (id: -1)\n"
    )


# -- Actual upload runs - skipped unless authentication token present

# Upload the production root - dry run
def test_root_actual(capsys, auth_token):

    deposit = Deposit(name="", live=False)

    deposit.upload(dry_run=False, auth_token=auth_token)

    id = deposit.status['id']
    assert deposit.status == {
        "id": id,
        "published": False
    }

    deposit.delete(dry_run=False, auth_token=auth_token)

    assert deposit.status == {"id": None}

    stdout = capsys.readouterr().out
    print(stdout)
    assert stdout == (
        "\n** Uploading '' to sandbox Zenodo\n"
        f"   - draft deposit created (id: {id})\n"
        "\n** Deleting '' from sandbox Zenodo\n"
        f"   - draft deposit deleted (id: {id})\n"
    )

   
