
# Unit Tests of CauaslIQ Pipeline repro.py command line processing

from pytest import fixture, raises, MonkeyPatch
from argparse import Namespace

from causaliq_repro.repro import validate_args
from causaliq_repro.deposit import ZENODO_DIR


# fixture returning a valid set of arguments
@fixture
def args():
    return Namespace(
        operation="upload",
        target="root",
        zenodo="sandbox",
        token="dummy_token",
        run=False
    )


# Mock Deposit class where status can be explicitly set
class MockDeposit():
    def __init__(self, status: dict, name: str, sandbox: bool, base_dir: str):
        self.name = name
        self.base = base_dir
        self.sandbox = sandbox
        self.status = status


# Returns a mock unsubmitted deposit
def mock_unsubmitted(name: str, sandbox: bool, base_dir: str):
    return MockDeposit({}, name, sandbox, base_dir)


# Returns a mock draft deposit
def mock_draft(name: str, sandbox: bool, base_dir: str):
    return MockDeposit({"recid": 1, "published": False}, name, sandbox,
                       base_dir)


# Returns a mock published deposit
def mock_published(name: str, sandbox: bool, base_dir: str):
    return MockDeposit({"recid": 1, "published": True}, name, sandbox,
                       base_dir)


# --- Failure cases

# validate_args must be called with argument
def test_bad_args_fails():
    with raises(TypeError):
        validate_args()


# learn not yet supported
def test_learn_unsupported_fails(args):
    args.operation = "learn"
    assert validate_args(args) is None


# analyse not yet supported
def test_analysis_unsupported_fails(args):
    args.operation = "analyse"
    assert validate_args(args) is None


# cannot publish a deposit which has not been uploaded
def test_publish_unsubmitted_fails(args: Namespace, monkeypatch: MonkeyPatch):
    args.operation = "publish"
    monkeypatch.setattr("causaliq_repro.repro.Deposit", mock_unsubmitted)
    result = validate_args(args)

    assert result is None


# cannot upload a non-existent deposit
def test_nonexistent_deposit_fails(args: Namespace):
    args.target = "non_existent"
    result = validate_args(args)

    assert result is None


# cannot publish a deposit which has already been published
def test_publish_published_fails(args: Namespace, monkeypatch: MonkeyPatch):
    args.operation = "publish"
    monkeypatch.setattr("causaliq_repro.repro.Deposit", mock_published)
    result = validate_args(args)

    assert result is None


# cannot delete a deposit which has not been uploaded
def test_delete_unsubmitted_fails(args: Namespace, monkeypatch: MonkeyPatch):
    args.operation = "delete"
    monkeypatch.setattr("causaliq_repro.repro.Deposit", mock_unsubmitted)
    result = validate_args(args)

    assert result is None


# cannot publish a deposit which has already been published
def test_delete_published_fails(args: Namespace, monkeypatch: MonkeyPatch):
    args.operation = "delete"
    monkeypatch.setattr("causaliq_repro.repro.Deposit", mock_published)
    result = validate_args(args)

    assert result is None


# --- Successful cases

def test_upload_unsubmitted_ok(args: Namespace, monkeypatch: MonkeyPatch):
    monkeypatch.setattr("causaliq_repro.repro.Deposit", mock_unsubmitted)
    result = validate_args(args)

    expected = Namespace(**vars(args))
    expected.target = ""
    expected.base_dir = ZENODO_DIR
    assert result == expected


def test_upload_draft_ok(args: Namespace, monkeypatch: MonkeyPatch):
    monkeypatch.setattr("causaliq_repro.repro.Deposit", mock_draft)
    result = validate_args(args)

    expected = Namespace(**vars(args))
    expected.target = ""
    expected.base_dir = ZENODO_DIR
    assert result == expected


def test_upload_published_ok(args: Namespace, monkeypatch: MonkeyPatch):
    monkeypatch.setattr("causaliq_repro.repro.Deposit", mock_published)
    result = validate_args(args)

    expected = Namespace(**vars(args))
    expected.target = ""
    expected.base_dir = ZENODO_DIR
    assert result == expected


def test_publish_draft_ok(args: Namespace, monkeypatch: MonkeyPatch):
    args.operation = "publish"
    monkeypatch.setattr("causaliq_repro.repro.Deposit", mock_draft)
    result = validate_args(args)

    expected = Namespace(**vars(args))
    expected.target = ""
    expected.base_dir = ZENODO_DIR
    assert result == expected


def test_delete_draft_ok(args: Namespace, monkeypatch: MonkeyPatch):
    args.operation = "delete"
    monkeypatch.setattr("causaliq_repro.repro.Deposit", mock_draft)
    result = validate_args(args)

    expected = Namespace(**vars(args))
    expected.target = ""
    expected.base_dir = ZENODO_DIR
    assert result == expected
