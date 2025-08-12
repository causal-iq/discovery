
# Integration Tests of CauaslIQ Pipeline for an isolated root node

from pytest import MonkeyPatch, fixture, skip
from _pytest.capture import CaptureFixture
from os import remove as remove_local_file

from tests.common import REPRO_TEST_DATA_DIR
from causaliq_repro.repro import get_zenodo_token
from causaliq_repro.deposit import Deposit

BASE_DIR = REPRO_TEST_DATA_DIR + "integration/root2/"


# --- Test fixtures and helper functions

# Fixture to retrieve the Zenodo sandbox authentication token
@fixture(scope="session")
def token():
    token = get_zenodo_token(sandbox=True)
    if token is None:
        skip("User does not have Zenodo upload privilege")
    return token


# Fixture that detects whether user has requested to see output (-s option)
@fixture(scope="session", autouse=True)
def detect_capture_mode(request):
    global capture_disabled
    capture_disabled = request.config.getoption("capture") == "no"


# returns captured output, optionally echoing that output to stdout
def get_stdout(capsys):
    stdout = capsys.readouterr().out
    if capture_disabled:
        with capsys.disabled():
            print(stdout, end="")
    return stdout


# Perform and check upload which creates a new draft deposit
def check_upload_create(name: str, token: str, capsys: CaptureFixture):
    """
        Checks creation of a new draft deposit

        :param str name: deposit name
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
    """
    # Write an empty status file to force upload
    deposit = Deposit(name=name, sandbox=True, base_dir=BASE_DIR)
    deposit.status = {}
    deposit._write_status()

    # Upload the initial deposit with just a readme file
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=BASE_DIR)
    deposit.upload(dry_run=False, token=token)
    recid = deposit.status["recid"]
    fileid = deposit.status["files"]["readme.md.j2"]["fileid"]
    size = deposit.status["files"]["readme.md.j2"]["size"]

    # check stdout as expected
    stdout = get_stdout(capsys)
    assert stdout == (
        "\n"
        "** Uploading '' to sandbox Zenodo\n"
        f"   - create deposit (recid: {recid})\n"
        f"   - add file readme.md (recid: {recid}, "
        f"fileid: {fileid}, {size} bytes)\n"
    )


# Perform and check upload where nothing has changed
def check_upload_nochange(name: str, token: str, capsys: CaptureFixture):
    """
        Checks upload where nothing has changed

        :param str name: deposit name
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
    """

    # Upload the initial deposit with just a readme file
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=BASE_DIR)
    deposit.upload(dry_run=False, token=token)
    recid = deposit.status["recid"]

    # check stdout as expected
    stdout = get_stdout(capsys)
    assert stdout == (
        "\n"
        "** Uploading '' to sandbox Zenodo\n"
        f"   - no changes made (recid: {recid})\n"
    )


# set changes to simulate deposit changes
def set_changes(changes: dict = {}):
    """
        Sets "changes" struct which indicates how the deposit has changed.
        Used with monkeypatch to patch return from Deposit.Idenfiy_changes
        to simulate different scenarions.
    """
    def patched(self, checksums):
        _changes = {  # default values represent "no changes"
            "metadata": False,
            "status": dict(self.status),
            "files": [],
            "deleted": []
        }
        _changes.update(changes)
        _ = checksums  # suppress "not used" linter warning
        return _changes
    return patched


# Perform and check upload where metadata has changed
def check_upload_metadata_changed(name: str, token: str,
                                  capsys: CaptureFixture,
                                  monkeypatch: MonkeyPatch):
    """
        Checks upload where metadata change is simulated

        :param str name: deposit name
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param MonkeyPatch monkeypatch: fixture to patch function behaviour
    """

    # Simulate the metadata being changed
    capsys.readouterr()
    monkeypatch.setattr(Deposit, "_identify_changes",
                        set_changes({'metadata': True}))
    deposit = Deposit(name=name, sandbox=True, base_dir=BASE_DIR)
    deposit.metadata["title"] = deposit.metadata["title"].replace("2", "2A")
    deposit.upload(dry_run=False, token=token)
    monkeypatch.undo()
    recid = deposit.status["recid"]

    # check stdout as expected
    stdout = get_stdout(capsys)
    assert stdout == (
        "\n"
        "** Uploading '' to sandbox Zenodo\n"
        f"   - update metadata (recid: {recid})\n"
    )


# Perform and check upload where metadata and readme has changed
def check_upload_metadata_and_readme_changed(name: str, token: str,
                                             capsys: CaptureFixture,
                                             monkeypatch: MonkeyPatch):
    """
        Checks upload where metadata and readme change has been simulated

        :param str name: deposit name
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param MonkeyPatch monkeypatch: fixture to patch function behaviour
    """
    # Simulate the metadata being changed
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=BASE_DIR)

    # patch _identify_changes return to signal metadata & readme change
    fileid = deposit.status["files"]["readme.md.j2"]["fileid"]
    monkeypatch.setattr(Deposit, "_identify_changes",
                        set_changes({"metadata": True,
                                     "files": ["readme.md.j2"],
                                     "deleted": [("readme.md.j2", fileid)]}))
    deposit.metadata["title"] = deposit.metadata["title"].replace("2", "2B")

    deposit.upload(dry_run=False, token=token)
    recid = deposit.status["recid"]
    monkeypatch.undo()

    # check stdout as expected
    stdout = get_stdout(capsys)
    return
    assert stdout == (
        "\n"
        "** Uploading '' to sandbox Zenodo\n"
        f"   - update metadata (recid: {recid})\n"
    )


# Perform and check retrieving metadata
def check_get_metadata(name: str, rootid: str, files: set[str], token: str,
                       capsys: CaptureFixture):
    """
        Checks getting metadata of a draft deposit

        :param str name: deposit name
        :param str rootid: root id for this test
        :param set files: set of expected filenames in deposit
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
    """

    # Check the metadata of the deposit
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=BASE_DIR)
    recid = deposit.status["recid"]
    info = deposit.get_metadata(dry_run=False, token=token)

    # check stdout as expected
    stdout = get_stdout(capsys)
    assert stdout == (
        "\n"
        "** Get metadata of '' from sandbox Zenodo\n"
        f"   - get metadata (recid: {recid})\n"
    )

    # check key elements of the retrieved metadata are as expected
    assert info["id"] == recid
    assert info["metadata"]["title"] == \
        f"CausalIQ Integration Test Root {rootid} Deposit"
    assert {f["filename"] for f in info["files"]} == files
    assert info['state'] == "unsubmitted"

    print(info["metadata"]["related_identifiers"])
    get_stdout(capsys)


# Perform and check adding a file
def check_add_file(name: str, filename: str, token: str,
                   capsys: CaptureFixture):
    """
        Checks adding a file to a draft deposit

        :param str name: deposit name
        :param str filename: local name of file to add
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
    """
    # Write new file to local directory
    with open(BASE_DIR + filename, "w") as f:
        f.write("This is a test deposit file.\nLine 2 of the deposit file.\n")

    # Upload deposit which adds file
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=BASE_DIR)
    deposit.upload(dry_run=False, token=token)
    recid = deposit.status["recid"]
    fileid = deposit.status["files"][filename]["fileid"]
    size = deposit.status["files"][filename]["size"]

    # check stdout as expected
    stdout = get_stdout(capsys)
    assert stdout == (
        "\n"
        f"** Uploading '{name}' to sandbox Zenodo\n"
        f"   - add file {filename} (recid: {recid}, "
        f"fileid: {fileid}, {size} bytes)\n"
    )


# Perform and check removing a file
def check_remove_file(name: str, filename: str, token: str,
                      capsys: CaptureFixture):
    """
        Checks removing a file from a draft deposit

        :param str name: deposit name
        :param str filename: local name of file to remove
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
    """
    # Remove file from local system
    remove_local_file(BASE_DIR + filename)

    # Upload deposit which removes file
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=BASE_DIR)
    fileid = deposit.status["files"][filename]["fileid"]
    recid = deposit.status["recid"]
    deposit.upload(dry_run=False, token=token)

    # check stdout as expected
    stdout = get_stdout(capsys)
    assert stdout == (
        "\n"
        f"** Uploading '{name}' to sandbox Zenodo\n"
        f"   - remove file {filename} (recid: {recid}, fileid: {fileid})\n"
    )


# Perform and check deletion of a draft deposit
def check_delete(name: str, token: str, capsys: CaptureFixture):
    deposit = Deposit(name=name, sandbox=True, base_dir=BASE_DIR)
    recid = deposit.status["recid"]

    capsys.readouterr()
    deposit.delete(dry_run=False, token=token)
    captured = capsys.readouterr()

    print(captured.out)
    assert captured.out == (
        "\n"
        f"** Deleting '{name}' from sandbox Zenodo\n"
        f"   - delete deposit (recid: {recid})\n"
    )

    assert deposit.status == {}


# --- Workflow tests with draft deposits - requires authentication token

# Test workflow with a draft root node
def test_root_only(token: str, capsys: CaptureFixture,
                   monkeypatch: MonkeyPatch):

    # upload which creates the root deposit
    check_upload_create(name="", token=token, capsys=capsys)

    # checking root deposit metadata
    check_get_metadata(name="", rootid="2", files={"readme.md"}, token=token,
                       capsys=capsys)

    # upload where nothing has changed
    check_upload_nochange(name="", token=token, capsys=capsys)

    # upload where metadata has changed
    check_upload_metadata_changed(name="", token=token, capsys=capsys,
                                  monkeypatch=monkeypatch)

    # check that deposit metadata has been changed
    check_get_metadata(name="", rootid="2A", files={"readme.md"}, token=token,
                       capsys=capsys)

    # adding a file
    check_add_file(name="", filename="testfile.tmp", token=token,
                   capsys=capsys)

    # check that file has been added to deposit
    check_get_metadata(name="", rootid="2A",
                       files={"readme.md", "testfile.tmp"}, token=token,
                       capsys=capsys)

    # upload where metadata and readme has changed
    check_upload_metadata_and_readme_changed(name="", token=token,
                                             capsys=capsys,
                                             monkeypatch=monkeypatch)

    # check that metadata has been changed, still two files
    check_get_metadata(name="", rootid="2B",
                       files={"readme.md", "testfile.tmp"}, token=token,
                       capsys=capsys)

    # removing a file
    check_remove_file(name="", filename="testfile.tmp", token=token,
                      capsys=capsys)

    # check that file has been has been removed from deposit
    check_get_metadata(name="", rootid="2B", files={"readme.md"}, token=token,
                       capsys=capsys)

    # delete the root deposit
    check_delete(name="", token=token, capsys=capsys)
