
# Helpers for integration tests

from json import dump
from pytest import MonkeyPatch
from _pytest.capture import CaptureFixture
from os import remove as remove_local_file

from causaliq_repro.deposit import Deposit

capture_disabled = False  # set by fixture in conftest

IS_DOCUMENTED_BY = {  # expected "is documented by" related link
    "https://orcid.org/0000-0002-7970-1453": {
        "relation": "isDocumentedBy",
        "resource_type": "other",
        "scheme": "url"
    }
}


# Helper function writing empty sandbox status file
def init_status_file(name: str, base_dir: str):
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


# returns captured output, optionally echoing that output to stdout
def get_stdout(capsys: CaptureFixture, echo_output: bool = True) -> str:
    """
        Returns captured output (since previous call), optionally echoing
        the output to the screen.

        :param CaptureFixture capsys: in-built fixture providing output capture
        :param bool echo_output: whether captured output should be echoed out

        :returns str: the captured output
    """
    stdout = capsys.readouterr().out
    if echo_output is True:
        with capsys.disabled():
            print(stdout, end="")
    return stdout


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


# Perform and check upload which creates a new draft deposit
def upload_create(name: str, base_dir: str, token: str, capsys: CaptureFixture,
                  echo_output: bool) -> int:
    """
        Checks creation of a new draft deposit

        :param str name: deposit name
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param bool echo_output: whether output should be echoed to screen

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
    stdout = get_stdout(capsys, echo_output=echo_output)
    assert stdout == (
        "\n"
        f"** Uploading '{name}' to sandbox Zenodo\n"
        f"   - create deposit (recid: {recid})\n"
        f"   - add file readme.md (recid: {recid}, "
        f"fileid: {fileid}, {size} bytes)\n"
    )

    return recid


# Perform and check upload where nothing has changed
def upload_nochange(name: str, base_dir: str, token: str,
                    capsys: CaptureFixture, echo_output: bool):
    """
        Checks upload where nothing has changed

        :param str name: deposit name
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param bool echo_output: whether output should be echoed to screen
    """

    # Upload the initial deposit with just a readme file
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
    deposit.upload(dry_run=False, token=token)
    recid = deposit.status["recid"]

    # check stdout as expected
    stdout = get_stdout(capsys, echo_output=echo_output)
    assert stdout == (
        "\n"
        "** Uploading '' to sandbox Zenodo\n"
        f"   - no changes made (recid: {recid})\n"
    )


# Perform and check upload where metadata has changed
def upload_metadata_changed(name: str, base_dir: str, token: str,
                            capsys: CaptureFixture, monkeypatch: MonkeyPatch,
                            echo_output: bool,
                            title_mod: tuple[str] = tuple()):
    """
        Checks upload where metadata change is simulated

        :param str name: deposit name
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param MonkeyPatch monkeypatch: fixture to patch function behaviour
        :param bool echo_output: whether output should be echoed to screen
        :param tuple title_mod: string replacement in deposit title
    """

    # Simulate the metadata being changed
    capsys.readouterr()
    monkeypatch.setattr(Deposit, "_identify_changes",
                        set_changes({'metadata': True}))
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
    was_published = deposit.status["published"]

    if len(title_mod) == 2:
        deposit.metadata["title"] = \
            deposit.metadata["title"].replace(title_mod[0], title_mod[1])

    deposit.upload(dry_run=False, token=token)
    monkeypatch.undo()
    recid = deposit.status["recid"]

    # check stdout as expected
    stdout = get_stdout(capsys, echo_output=echo_output)
    new_version = (f"   - create new version (recid: {recid}, version: " +
                   f"{deposit.status['version']})\n" if was_published else "")
    assert stdout == (
        "\n"
        f"** Uploading '{name}' to sandbox Zenodo\n{new_version}"
        f"   - update metadata (recid: {recid})\n"
    )


# Perform and check upload where metadata and readme has changed
def upload_metadata_and_readme_changed(name: str, base_dir: str, token: str,
                                       capsys: CaptureFixture,
                                       monkeypatch: MonkeyPatch,
                                       echo_output: bool = False,
                                       title_mod: tuple[str] = tuple()):
    """
        Checks upload where metadata and readme change has been simulated

        :param str name: deposit name
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param MonkeyPatch monkeypatch: fixture to patch function behaviour
        :param bool echo_output: whether output should be echoed to screen
        :param tuple title_mod: string replacement in deposit title
    """
    # Simulate the metadata being changed
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
    was_published = deposit.status["published"]

    # patch _identify_changes return to signal metadata & readme change
    fileid1 = deposit.status["files"]["readme.md.j2"]["fileid"]
    monkeypatch.setattr(Deposit, "_identify_changes",
                        set_changes({"metadata": True,
                                     "files": ["readme.md.j2"],
                                     "deleted": [("readme.md.j2", fileid1)]}))
    if len(title_mod) == 2:
        deposit.metadata["title"] = \
            deposit.metadata["title"].replace(title_mod[0], title_mod[1])

    deposit.upload(dry_run=False, token=token)
    recid = deposit.status["recid"]
    fileid2 = deposit.status["files"]["readme.md.j2"]["fileid"]
    size = deposit.status["files"]["readme.md.j2"]["size"]

    monkeypatch.undo()

    # check stdout as expected
    stdout = get_stdout(capsys, echo_output=echo_output)
    new_version = (f"   - create new version (recid: {recid}, version: " +
                   f"{deposit.status['version']})\n" if was_published else "")
    assert stdout == (
        "\n"
        f"** Uploading '{name}' to sandbox Zenodo\n{new_version}"
        f"   - update metadata (recid: {recid})\n"
        f"   - remove file readme.md (recid: {recid}, fileid: {fileid1})\n"
        f"   - add file readme.md (recid: {recid}, fileid: {fileid2}"
        f", {size} bytes)\n"
    )


# Perform and check adding a file
def add_file(name: str, filename: str, base_dir: str, token: str,
             capsys: CaptureFixture, echo_output: bool):
    """
        Checks adding a file to a draft deposit

        :param str name: deposit name
        :param str filename: local name of file to add
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param bool echo_output: whether output should be echoed to screen
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
    stdout = get_stdout(capsys, echo_output=echo_output)
    assert stdout == (
        "\n"
        f"** Uploading '{name}' to sandbox Zenodo\n"
        f"   - add file {filename} (recid: {recid}, "
        f"fileid: {fileid}, {size} bytes)\n"
    )


# Perform and check removing a file
def remove_file(name: str, filename: str, base_dir: str, token: str,
                capsys: CaptureFixture, echo_output: bool):
    """
        Checks removing a file from a draft deposit

        :param str name: deposit name
        :param str filename: local name of file to remove
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param bool echo_output: whether output should be echoed to screen
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
    stdout = get_stdout(capsys, echo_output=echo_output)
    assert stdout == (
        "\n"
        f"** Uploading '{name}' to sandbox Zenodo\n"
        f"   - remove file {filename} (recid: {recid}, fileid: {fileid})\n"
    )


# Perform and check deletion of a draft deposit
def delete(name: str, base_dir: str, token: str, capsys: CaptureFixture,
           echo_output: bool):
    """
        Checks deleting a draft deposit

        :param str name: deposit name
        :param dict related: {name: recid} of related deposits updated
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param bool echo_output: whether output should be echoed to screen
    """
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
    recid = deposit.status["recid"]

    capsys.readouterr()
    deposit.delete(dry_run=False, token=token)

    stdout = get_stdout(capsys, echo_output=echo_output)
    assert stdout == (
        "\n"
        f"** Deleting '{name}' from sandbox Zenodo\n"
        f"   - delete deposit (recid: {recid})\n"
    )

    assert deposit.status == {}


# Perform and check publication of a draft deposit
def publish(name: str, base_dir: str, token: str, capsys: CaptureFixture,
            echo_output: bool, title_mod: tuple[str] = tuple()):
    """
        Checks publishing a draft deposit

        :param str name: deposit name
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param bool echo_output: whether output should be echoed to screen
        :param tuple title_mod: string replacement in deposit title
    """
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
    recid = deposit.status["recid"]
    if len(title_mod) == 2:
        deposit.metadata["title"] = \
            deposit.metadata["title"].replace(title_mod[0], title_mod[1])

    capsys.readouterr()
    deposit.publish(dry_run=False, token=token)

    stdout = get_stdout(capsys, echo_output=echo_output)
    assert stdout == (
        "\n"
        f"** Publishing '{name}' on sandbox Zenodo\n"
        f"   - publish deposit (recid: {recid})\n"
    )

    assert deposit.status["published"] is True


# Perform and check retrieving metadata
def get_metadata(name: str, rootid: str, files: set[str], related: dict,
                 base_dir: str, token: str, capsys: CaptureFixture,
                 echo_output: bool = False, state: str = "unsubmitted"):
    """
        Checks getting metadata of a draft deposit

        :param str name: deposit name
        :param str rootid: root id for this test
        :param set files: set of expected filenames in deposit
        :param dict related: related deposits that are expected
        :param str base_dir: local base (i.e., top-level) directory
        :param str token: Zenodo authentication token
        :param CaptureFixture capsys: fixture to capture stdout & stderr
        :param bool echo_output: whether output should be echoed to screen
        :param str echo_output: expected state of deposit
    """

    # Check the metadata of the deposit
    capsys.readouterr()
    deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
    recid = deposit.status["recid"]
    info = deposit.get_metadata(dry_run=False, token=token)

    # check stdout as expected
    stdout = get_stdout(capsys, echo_output)
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
    assert info['state'] == state

    # check the related links are as expected
    actual = {r["identifier"]: {"relation": r["relation"],
                                "resource_type": r["resource_type"],
                                "scheme": r["scheme"]}
              for r in info["metadata"]["related_identifiers"]}
    assert actual == dict(related, **IS_DOCUMENTED_BY)

    return info
