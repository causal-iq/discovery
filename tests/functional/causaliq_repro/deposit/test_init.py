
# Functional Tests of CauaslIQ Pipeline Deposit instantiation

import pytest

from tests.common import TEST_DATA_BASE
from causaliq_repro.deposit import Deposit


# --- Failed instantions of deposits

# Non-existant name
def test_bad_name_error():
    with pytest.raises(FileNotFoundError):
        Deposit(name="deposit/init/bad_name", live=False)


# Missing metadata.json.j2 template
def test_no_metadata_error():
    with pytest.raises(FileNotFoundError):
        Deposit(name="deposit/init/no_metadata", live=True,
                base_dir=TEST_DATA_BASE)


# Binary metadata.json.j2 template
def test_binary_metadata_error():
    with pytest.raises(ValueError):
        Deposit(name="deposit/init/binary_metadata", live=True,
                base_dir=TEST_DATA_BASE)


# Bad JSON metadata.json.j2 template
def test_bad_json_metadata_error():
    with pytest.raises(ValueError):
        Deposit(name="deposit/init/bad_json_metadata", live=True,
                base_dir=TEST_DATA_BASE)


# Missing readme.md.j2 template
def test_no_readme_error():
    with pytest.raises(FileNotFoundError):
        Deposit(name="deposit/init/no_readme", live=True,
                base_dir=TEST_DATA_BASE)


# Missing zenodo_status JSON file
def test_no_zenodo_status_error():
    with pytest.raises(FileNotFoundError):
        Deposit(name="deposit/init/no_zenodo_status", live=True,
                base_dir=TEST_DATA_BASE)


# Binary zenodo_status JSON file
def test_binary_zenodo_status_error():
    with pytest.raises(ValueError):
        Deposit(name="deposit/init/binary_zenodo_status", live=True,
                base_dir=TEST_DATA_BASE)


# Bad JSON zenodo_status JSON file
def test_bad_json_zenodo_status_error():
    with pytest.raises(ValueError):
        Deposit(name="deposit/init/bad_json_zenodo_status", live=True,
                base_dir=TEST_DATA_BASE)


# Missing sandbox_status JSON file
def test_no_sandbox_status_error():
    with pytest.raises(FileNotFoundError):
        Deposit(name="deposit/init/no_sandbox_status", live=False,
                base_dir=TEST_DATA_BASE)


# --- Successful instantiation of a Deposit

# Instantiation of the live root object
def test_live_root():
    Deposit(name="", live=True)
