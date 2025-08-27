
# Integration Tests of CauaslIQ Pipeline for an isolated root node

from pytest import MonkeyPatch, fixture, skip
from _pytest.capture import CaptureFixture
from os import remove as remove_local_file
from json import dump

from tests.common import REPRO_TEST_DATA_DIR
from causaliq_repro.repro import get_zenodo_token
from causaliq_repro.deposit import Deposit, SANDBOX_URL

IS_DOCUMENTED_BY = {  # expected "is documented by" related link
    "https://orcid.org/0000-0002-7970-1453": {
        "relation": "isDocumentedBy",
        "resource_type": "other",
        "scheme": "url"
    }
}


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


# Helper function to generate expected stdout for related deposit changes
def related_link_stdout(to: str, related: dict, sandbox: bool) -> str:
    """
        Returns expected stdout for changes made to related deposits

        :param str to: name of deposit initially updated
        :param dict related: related deposits, {name: recid}
        :param bool sandbox: sanbox or live Zenodo being used

        :returns str: expected stdout for the related deposit changes
    """
    str = ''
    for name, recid in related.items():
        str += (f" * Updating related link to '{to}' in '{name}'"
                f" on {'sandbox' if sandbox else 'LIVE'} Zenodo\n"
                f"   - update metadata (recid: {recid})\n")
    return str


# Helper function writing empty sandbox status file
def write_empty_status_file(name: str, base_dir: str):
    """
        Write empty status file.

        :param str name: deposit name
        :param str base_dir: base directory for the local deposit files
    """
    try:
        with open(f"{base_dir}{name}/sandbox_status.json", "w") as f:
            dump({}, f, indent=4)
    except Exception as e:
        raise ValueError(f"Error writing status of {name}: {e}")


# Perform and check upload which creates a new draft deposit
def check_upload_create(name: str, related: dict, base_dir: str, token: str,
                        capsys: CaptureFixture) -> int:
    """
        Checks creation of a new draft deposit

        :param str name: deposit name
        :param dict related: {name: recid} of related deposits updated
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr

        :returns int: recid of created deposit
    """
    # Write an empty status file to force upload
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
    deposit.status = {}
    deposit._write_status()

    # Upload the initial deposit with just a readme file
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
    deposit.upload(dry_run=False, token=token)
    recid = deposit.status["recid"]
    fileid = deposit.status["files"]["readme.md.j2"]["fileid"]
    size = deposit.status["files"]["readme.md.j2"]["size"]

    # check stdout as expected
    stdout = get_stdout(capsys)
    assert stdout == (
        "\n"
        f"** Uploading '{name}' to sandbox Zenodo\n"
        f"   - create deposit (recid: {recid})\n"
        f"   - add file readme.md (recid: {recid}, "
        f"fileid: {fileid}, {size} bytes)\n"
    ) + related_link_stdout(to=name, related=related, sandbox=True)

    return recid


# Perform and check upload where nothing has changed
def check_upload_nochange(name: str, base_dir: str, token: str,
                          capsys: CaptureFixture):
    """
        Checks upload where nothing has changed

        :param str name: deposit name
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
    """

    # Upload the initial deposit with just a readme file
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
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
def check_upload_metadata_changed(name: str, base_dir: str, token: str,
                                  capsys: CaptureFixture,
                                  monkeypatch: MonkeyPatch):
    """
        Checks upload where metadata change is simulated

        :param str name: deposit name
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param MonkeyPatch monkeypatch: fixture to patch function behaviour
    """

    # Simulate the metadata being changed
    capsys.readouterr()
    monkeypatch.setattr(Deposit, "_identify_changes",
                        set_changes({'metadata': True}))
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
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
def check_upload_metadata_and_readme_changed(name: str, base_dir: str,
                                             token: str,
                                             capsys: CaptureFixture,
                                             monkeypatch: MonkeyPatch):
    """
        Checks upload where metadata and readme change has been simulated

        :param str name: deposit name
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param MonkeyPatch monkeypatch: fixture to patch function behaviour
    """
    # Simulate the metadata being changed
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)

    # patch _identify_changes return to signal metadata & readme change
    fileid1 = deposit.status["files"]["readme.md.j2"]["fileid"]
    monkeypatch.setattr(Deposit, "_identify_changes",
                        set_changes({"metadata": True,
                                     "files": ["readme.md.j2"],
                                     "deleted": [("readme.md.j2", fileid1)]}))
    deposit.metadata["title"] = deposit.metadata["title"].replace("2", "2B")

    deposit.upload(dry_run=False, token=token)
    recid = deposit.status["recid"]
    fileid2 = deposit.status["files"]["readme.md.j2"]["fileid"]
    size = deposit.status["files"]["readme.md.j2"]["size"]

    monkeypatch.undo()

    # check stdout as expected
    stdout = get_stdout(capsys)
    assert stdout == (
        "\n"
        "** Uploading '' to sandbox Zenodo\n"
        f"   - update metadata (recid: {recid})\n"
        f"   - remove file readme.md (recid: {recid}, fileid: {fileid1})\n"
        f"   - add file readme.md (recid: {recid}, fileid: {fileid2}"
        f", {size} bytes)\n"
    )


# Perform and check retrieving metadata
def check_get_metadata(name: str, rootid: str, files: set[str], related: dict,
                       base_dir: str, token: str, capsys: CaptureFixture):
    """
        Checks getting metadata of a draft deposit

        :param str name: deposit name
        :param str rootid: root id for this test
        :param set files: set of expected filenames in deposit
        :param dict related: related deposits that are expected
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
    """

    # Check the metadata of the deposit
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
    recid = deposit.status["recid"]
    info = deposit.get_metadata(dry_run=False, token=token)

    # check stdout as expected
    stdout = get_stdout(capsys)
    assert stdout == (
        "\n"
        f"** Get metadata of '{name}' from sandbox Zenodo\n"
        f"   - get metadata (recid: {recid})\n"
    )

    # check key elements of the retrieved metadata are as expected
    assert info["id"] == recid
    assert info["metadata"]["title"] == \
        f"CausalIQ Integration Test Root {rootid} Deposit"
    assert {f["filename"] for f in info["files"]} == files
    assert info['state'] == "unsubmitted"

    # check the related links are as expected
    actual = {r["identifier"]: {"relation": r["relation"],
                                "resource_type": r["resource_type"],
                                "scheme": r["scheme"]}
              for r in info["metadata"]["related_identifiers"]}
    assert actual == dict(related, **IS_DOCUMENTED_BY)


# Perform and check adding a file
def check_add_file(name: str, filename: str, base_dir: str, token: str,
                   capsys: CaptureFixture):
    """
        Checks adding a file to a draft deposit

        :param str name: deposit name
        :param str filename: local name of file to add
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
    """
    # Write new file to local directory
    with open(base_dir + filename, "w") as f:
        f.write("This is a test deposit file.\nLine 2 of the deposit file.\n")

    # Upload deposit which adds file
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
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
def check_remove_file(name: str, filename: str, base_dir: str, token: str,
                      capsys: CaptureFixture):
    """
        Checks removing a file from a draft deposit

        :param str name: deposit name
        :param str filename: local name of file to remove
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
    """
    # Remove file from local system
    remove_local_file(base_dir + filename)

    # Upload deposit which removes file
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
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
def check_delete(name: str, related: dict, base_dir: str, token: str,
                 capsys: CaptureFixture):
    """
        Checks deleting a draft deposit

        :param str name: deposit name
        :param dict related: {name: recid} of related deposits updated
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
    """
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
    recid = deposit.status["recid"]

    capsys.readouterr()
    deposit.delete(dry_run=False, token=token)

    stdout = get_stdout(capsys)
    assert stdout == (
        "\n"
        f"** Deleting '{name}' from sandbox Zenodo\n"
        f"   - delete deposit (recid: {recid})\n"
    ) + related_link_stdout(to=name, related=related, sandbox=True)

    assert deposit.status == {}


# --- Workflow tests with draft deposits - requires authentication token

# Test workflow with a draft root node
def test_root_only(token: str, capsys: CaptureFixture,
                   monkeypatch: MonkeyPatch):
    base_dir = REPRO_TEST_DATA_DIR + "integration/root2/"

    # Ensure empty status files are present
    write_empty_status_file(name="", base_dir=base_dir)

    # upload which creates the root deposit
    check_upload_create(name="", related={}, base_dir=base_dir, token=token,
                        capsys=capsys)

    # checking root deposit metadata
    check_get_metadata(name="", rootid="2", files={"readme.md"}, related={},
                       base_dir=base_dir, token=token, capsys=capsys)

    # upload where nothing has changed
    check_upload_nochange(name="", base_dir=base_dir, token=token,
                          capsys=capsys)

    # upload where metadata has changed
    check_upload_metadata_changed(name="", base_dir=base_dir, token=token,
                                  capsys=capsys, monkeypatch=monkeypatch)

    # check that deposit metadata has been changed
    check_get_metadata(name="", rootid="2A", files={"readme.md"}, related={},
                       base_dir=base_dir, token=token, capsys=capsys)

    # adding a file
    check_add_file(name="", filename="testfile.tmp", base_dir=base_dir,
                   token=token, capsys=capsys)

    # check that file has been added to deposit
    check_get_metadata(name="", rootid="2A",
                       files={"readme.md", "testfile.tmp"}, related={},
                       base_dir=base_dir,
                       token=token, capsys=capsys)

    # upload where metadata and readme has changed
    check_upload_metadata_and_readme_changed(name="", base_dir=base_dir,
                                             token=token, capsys=capsys,
                                             monkeypatch=monkeypatch)

    # check that metadata has been changed, still two files
    check_get_metadata(name="", rootid="2B",
                       files={"readme.md", "testfile.tmp"}, related={},
                       base_dir=base_dir, token=token, capsys=capsys)

    # removing a file
    check_remove_file(name="", filename="testfile.tmp", base_dir=base_dir,
                      token=token, capsys=capsys)

    # check that file has been has been removed from deposit
    check_get_metadata(name="", rootid="2B", files={"readme.md"}, related={},
                       base_dir=base_dir, token=token, capsys=capsys)

    # delete the root deposit
    check_delete(name="", related={}, base_dir=base_dir, token=token,
                 capsys=capsys)


# Test workflow with a draft root and hub
def test_root_and_hub(token: str, capsys: CaptureFixture):
    base_dir = REPRO_TEST_DATA_DIR + "integration/root3/"

    # Ensure empty status files are present
    write_empty_status_file(name="", base_dir=base_dir)
    write_empty_status_file(name="hub", base_dir=base_dir)

    # upload which creates the root deposit
    root_id = check_upload_create(name="", related={}, base_dir=base_dir,
                                  token=token, capsys=capsys)
    root_url = (SANDBOX_URL.replace("api", "") +
                f"records/{root_id}?preview=1")

    # checking root deposit metadata
    check_get_metadata(name="", rootid="3", files={"readme.md"}, related={},
                       base_dir=base_dir, token=token, capsys=capsys)

    # upload which creates the hub deposit
    hub_id = check_upload_create(name="hub", related={'': root_id},
                                 base_dir=base_dir, token=token, capsys=capsys)
    hub_url = (SANDBOX_URL.replace("api", "") +
               f"records/{hub_id}?preview=1")

    # checking hub deposit metadata - related should reference root
    related = {
        root_url: {"relation": "isPartOf",
                   "resource_type": "other",
                   "scheme": "url"
                   }
    }
    check_get_metadata(name="hub", rootid="3 Hub", files={"readme.md"},
                       related=related, base_dir=base_dir, token=token,
                       capsys=capsys)

    # checking root deposit metadata - related should now reference hub
    related = {
        hub_url: {"relation": "hasPart",
                  "resource_type": "other",
                  "scheme": "url"
                  }
    }
    check_get_metadata(name="", rootid="3", files={"readme.md"},
                       related=related, base_dir=base_dir, token=token,
                       capsys=capsys)

    # delete the hub deposit
    check_delete(name="hub", related={'': root_id}, base_dir=base_dir,
                 token=token, capsys=capsys)

    # check root deposit has hub related link removed
    check_get_metadata(name="", rootid="3", files={"readme.md"}, related={},
                       base_dir=base_dir, token=token, capsys=capsys)

    # delete the root deposit
    check_delete(name="", related={}, base_dir=base_dir, token=token,
                 capsys=capsys)


# Test workflow with a draft root, hub and leaves
def test_root_hub_and_leaves(token: str, capsys: CaptureFixture):
    base_dir = REPRO_TEST_DATA_DIR + "integration/root4/"

    # Ensure empty status files are present
    write_empty_status_file(name="", base_dir=base_dir)
    write_empty_status_file(name="hub", base_dir=base_dir)
    write_empty_status_file(name="hub/leaf1", base_dir=base_dir)
    write_empty_status_file(name="hub/leaf2", base_dir=base_dir)

    # --- upload which creates the root deposit

    root_id = check_upload_create(name="", related={}, base_dir=base_dir,
                                  token=token, capsys=capsys)
    root_url = (SANDBOX_URL.replace("api", "") +
                f"records/{root_id}?preview=1")

    # checking root deposit metadata
    check_get_metadata(name="", rootid="4", files={"readme.md"}, related={},
                       base_dir=base_dir, token=token, capsys=capsys)

    # --- upload which creates the hub deposit

    hub_id = check_upload_create(name="hub", related={'': root_id},
                                 base_dir=base_dir, token=token, capsys=capsys)
    hub_url = (SANDBOX_URL.replace("api", "") +
               f"records/{hub_id}?preview=1")

    # checking hub deposit metadata - related should reference root
    related = {
        root_url: {"relation": "isPartOf",
                   "resource_type": "other",
                   "scheme": "url"
                   }
    }
    check_get_metadata(name="hub", rootid="4 Hub", files={"readme.md"},
                       related=related, base_dir=base_dir, token=token,
                       capsys=capsys)

    # checking root deposit metadata - related should now reference hub
    related = {
        hub_url: {"relation": "hasPart",
                  "resource_type": "other",
                  "scheme": "url"
                  }
    }
    check_get_metadata(name="", rootid="4", files={"readme.md"},
                       related=related, base_dir=base_dir, token=token,
                       capsys=capsys)

    # --- upload which creates the leaf1 deposit

    leaf1_id = check_upload_create(name="hub/leaf1", related={'hub': hub_id},
                                   base_dir=base_dir, token=token,
                                   capsys=capsys)
    leaf1_url = (SANDBOX_URL.replace("api", "") +
                 f"records/{leaf1_id}?preview=1")

    # check metadata of leaf1 deposit
    related = {
        hub_url: {"relation": "isPartOf",
                  "resource_type": "other",
                  "scheme": "url"
                  }
    }
    check_get_metadata(name="hub/leaf1", rootid="4 Leaf 1",
                       files={"readme.md"}, related=related, base_dir=base_dir,
                       token=token, capsys=capsys)

    # check metadata of hub deposit - related should reference root and leaf1
    related = {
        root_url: {"relation": "isPartOf",
                   "resource_type": "other",
                   "scheme": "url"
                   },
        leaf1_url: {"relation": "hasPart",
                    "resource_type": "other",
                    "scheme": "url"
                    },
    }
    check_get_metadata(name="hub", rootid="4 Hub",
                       files={"readme.md"}, related=related, base_dir=base_dir,
                       token=token, capsys=capsys)

    # --- upload which creates the leaf2 deposit

    leaf2_id = check_upload_create(name="hub/leaf2", related={'hub': hub_id},
                                   base_dir=base_dir, token=token,
                                   capsys=capsys)
    leaf2_url = (SANDBOX_URL.replace("api", "") +
                 f"records/{leaf2_id}?preview=1")

    # check metadata of leaf2 deposit
    related = {
        hub_url: {"relation": "isPartOf",
                  "resource_type": "other",
                  "scheme": "url"
                  }
    }
    check_get_metadata(name="hub/leaf2", rootid="4 Leaf 2",
                       files={"readme.md"}, related=related, base_dir=base_dir,
                       token=token, capsys=capsys)

    # --- delete the leaf1 deposit

    check_delete(name="hub/leaf1", related={'hub': hub_id},
                 base_dir=base_dir, token=token, capsys=capsys)

    # check hub deposit has leaf1 link removed
    related = {
        root_url: {"relation": "isPartOf",
                   "resource_type": "other",
                   "scheme": "url"
                   },
        leaf2_url: {"relation": "hasPart",
                    "resource_type": "other",
                    "scheme": "url"}
    }
    check_get_metadata(name="hub", rootid="4 Hub",
                       files={"readme.md"}, related=related, base_dir=base_dir,
                       token=token, capsys=capsys)

    # --- delete the hub deposit

    check_delete(name="hub", related={'': root_id, 'hub/leaf2': leaf2_id},
                 base_dir=base_dir, token=token, capsys=capsys)

    # check root deposit has hub related link removed
    check_get_metadata(name="", rootid="4", files={"readme.md"}, related={},
                       base_dir=base_dir, token=token, capsys=capsys)

    # check leaf2 deposit has hub related link removed
    check_get_metadata(name="hub/leaf2", rootid="4 Leaf 2",
                       files={"readme.md"}, related={}, base_dir=base_dir,
                       token=token, capsys=capsys)

    # --- delete the leaf2 deposit

    check_delete(name="hub/leaf2", related={}, base_dir=base_dir, 
                 token=token, capsys=capsys)

    # --- delete the root deposit

    check_delete(name="", related={}, base_dir=base_dir, token=token,
                 capsys=capsys)
