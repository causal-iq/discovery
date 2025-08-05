
# Functional Tests of CauaslIQ Pipeline Deposit instantiation

import pytest

from tests.common import REPRO_TEST_DATA_DIR
from causaliq_repro.deposit import Deposit

BASE_DIR = REPRO_TEST_DATA_DIR + "functional/deposit/init/"


# --- Failed instantions of deposits

# Non-existant name
def test_bad_name_error():
    with pytest.raises(FileNotFoundError):
        Deposit(name="bad_name", live=False, base_dir=BASE_DIR)


# Missing metadata.json.j2 template
def test_no_metadata_error():
    with pytest.raises(FileNotFoundError):
        Deposit(name="", live=True, base_dir=BASE_DIR + "no_metadata/")


# Binary metadata.json.j2 template
def test_binary_metadata_error():
    with pytest.raises(ValueError):
        Deposit(name="", live=True, base_dir=BASE_DIR + "binary_metadata/")


# Bad JSON metadata.json.j2 template
def test_bad_json_metadata_error():
    with pytest.raises(ValueError):
        Deposit(name="", live=True, base_dir=BASE_DIR + "bad_json_metadata/")


# Missing readme.md.j2 template
def test_no_readme_error():
    with pytest.raises(FileNotFoundError):
        Deposit(name="", live=True, base_dir=BASE_DIR + "no_readme/")


# Missing zenodo_status JSON file
def test_no_zenodo_status_error():
    with pytest.raises(FileNotFoundError):
        Deposit(name="", live=True, base_dir=BASE_DIR + "no_zenodo_status/")


# Binary zenodo_status JSON file
def test_binary_zenodo_status_error():
    with pytest.raises(ValueError):
        Deposit(name="", live=True,
                base_dir=BASE_DIR + "binary_zenodo_status/")


# Bad JSON zenodo_status JSON file
def test_bad_json_zenodo_status_error():
    with pytest.raises(ValueError):
        Deposit(name="", live=True,
                base_dir=BASE_DIR + "bad_json_zenodo_status/")


# Missing sandbox_status JSON file
def test_no_sandbox_status_error():
    with pytest.raises(FileNotFoundError):
        Deposit(name="", live=False,
                base_dir=BASE_DIR + "no_sandbox_status/")


# --- Successful instantiation of a Deposit

# Instantiation of the live root object
def test_live_root():
    Deposit(name="/", live=True)
