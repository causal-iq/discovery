
# Integration Tests of CauaslIQ Pipeline repro.py upload

import pytest

from causaliq_repro.repro import get_zenodo_token
from causaliq_repro.deposit import Deposit


# Fixture to retrieve the Zenodo sandbox authentication token
@pytest.fixture(scope="session")
def token():
    token = get_zenodo_token(live=False)
    if token is None:
        pytest.skip("User does not have Zenodo upload privilege")
    return token


# -- Dry upload runs - don't require authentication so always run

# Upload the production root - dry run
def test_root_dryrun(capsys):

    deposit = Deposit(name="", live=False)

    deposit.upload(dry_run=True, token=None)

    captured = capsys.readouterr()
    print(f"\n\nCaptured output: \n{captured.out}")

    assert captured.out == (
        "\n** Uploading '' to sandbox Zenodo\n"
        "   - create deposit (recid: -1)\n"
        "   - add file readme.md (1467 bytes)\n"
    )


# -- Actual upload runs - skipped unless authentication token present

# Upload the production root - actual run
def test_root_actual(capsys, token):

    deposit = Deposit(name="", live=False)

    deposit.upload(dry_run=False, token=token)

    assert set(deposit.status) == {
        "recid",
        "conceptid",
        "files",
        "checksum",
        "published",
        "version"
    }
    assert set(deposit.status["files"]) == {"readme.md.j2"}
    assert deposit.status["published"] is False
    assert deposit.status["version"] == 1
    recid = deposit.status["recid"]

    deposit.delete(dry_run=False, token=token)

    assert deposit.status == {}

    stdout = capsys.readouterr().out
    print(stdout)

    assert stdout == (
        "\n** Uploading '' to sandbox Zenodo\n"
        f"   - create deposit (recid: {recid})\n"
        "   - add file readme.md (1467 bytes)\n"
        "\n** Deleting '' from sandbox Zenodo\n"
        f"   - delete deposit (recid: {recid})\n"
    )


# Upload the production datasets list - actual run
def test_datasets_actual(capsys, token):

    deposit = Deposit(name="/data", live=False)

    deposit.upload(dry_run=False, token=token)

    assert set(deposit.status) == {
        "recid",
        "conceptid",
        "files",
        "checksum",
        "published",
        "version"
    }
    assert set(deposit.status["files"]) == {"readme.md.j2"}
    assert deposit.status["published"] is False
    assert deposit.status["version"] == 1
    recid = deposit.status["recid"]

    deposit.delete(dry_run=False, token=token)

    assert deposit.status == {}

    stdout = capsys.readouterr().out
    print(stdout)

    assert stdout == (
        "\n** Uploading '/data' to sandbox Zenodo\n"
        f"   - create deposit (recid: {recid})\n"
        "   - add file readme.md (950 bytes)\n"
        "\n** Deleting '/data' from sandbox Zenodo\n"
        f"   - delete deposit (recid: {recid})\n"
    )
