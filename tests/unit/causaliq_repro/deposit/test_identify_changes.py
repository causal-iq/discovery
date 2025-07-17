
# Unit test Deposit._identify_changes

from causaliq_repro.deposit import Deposit, METADATA_TEMPLATE, README_TEMPLATE


# New root deposit
def test_root_create():
    deposit = Deposit.__new__(Deposit)
    deposit.status = {
        "id": None
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_1",
        README_TEMPLATE: "CHECKSUM_README_1"
    }

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": [
            README_TEMPLATE
        ],
        "deleted": [],
        "status": {
            "id": None,
            "checksum": "CHECKSUM_METADATA_1",
            "files": {
                README_TEMPLATE: "CHECKSUM_README_1"
            }
        }
    }


# Unchanged root deposit
def test_root_unchanged():
    deposit = Deposit.__new__(Deposit)
    deposit.status = {
        "id": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: "CHECKSUM_README_1"
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_1",
        README_TEMPLATE: "CHECKSUM_README_1"
    }

    changed = deposit._identify_changes(checksums)

    assert changed is None


# Updating root metadata
def test_root_update_metadata():
    deposit = Deposit.__new__(Deposit)
    deposit.status = {
        "id": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: "CHECKSUM_README_1"
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_2",
        README_TEMPLATE: "CHECKSUM_README_1"
    }

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": [],
        "deleted": [],
        "status": {
            "id": 1234567,
            "published": False,
            "checksum": "CHECKSUM_METADATA_2",
            "files": {
                README_TEMPLATE: "CHECKSUM_README_1"
            }
        }
    }


# Updating root readme
def test_root_update_readme():
    deposit = Deposit.__new__(Deposit)
    deposit.status = {
        "id": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: "CHECKSUM_README_1"
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_1",
        README_TEMPLATE: "CHECKSUM_README_2"
    }

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": False,
        "files": [README_TEMPLATE],
        "deleted": [],
        "status": {
            "id": 1234567,
            "published": False,
            "checksum": "CHECKSUM_METADATA_1",
            "files": {
                README_TEMPLATE: "CHECKSUM_README_2"
            }
        }
    }


# Updating root metadata and readme
def test_root_update_metadata_and_readme():
    deposit = Deposit.__new__(Deposit)
    deposit.status = {
        "id": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: "CHECKSUM_README_1"
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_2",
        README_TEMPLATE: "CHECKSUM_README_2"
    }

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": [README_TEMPLATE],
        "deleted": [],
        "status": {
            "id": 1234567,
            "published": False,
            "checksum": "CHECKSUM_METADATA_2",
            "files": {
                README_TEMPLATE: "CHECKSUM_README_2"
            }
        }
    }


# New dataset deposit
def test_dataset_create():
    deposit = Deposit.__new__(Deposit)
    deposit.status = {
        "id": None
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_1",
        README_TEMPLATE: "CHECKSUM_README_1",
        "network.dsc": "CHECKSUM_DSC_1"
    }

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": [
            README_TEMPLATE,
            "network.dsc"
        ],
        "deleted": [],
        "status": {
            "id": None,
            "checksum": "CHECKSUM_METADATA_1",
            "files": {
                README_TEMPLATE: "CHECKSUM_README_1",
                "network.dsc": "CHECKSUM_DSC_1"
            }
        }
    }


# Unchanged dataset deposit
def test_dataset_unchanged():
    deposit = Deposit.__new__(Deposit)
    deposit.status = {
        "id": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: "CHECKSUM_README_1",
            "network.dsc": "CHECKSUM_DSC_1"
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_1",
        README_TEMPLATE: "CHECKSUM_README_1",
        "network.dsc": "CHECKSUM_DSC_1"
    }

    changed = deposit._identify_changes(checksums)

    assert changed is None


# Add file to dataset
def test_dataset_add_file():
    deposit = Deposit.__new__(Deposit)
    deposit.status = {
        "id": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: "CHECKSUM_README_1",
            "network.dsc": "CHECKSUM_DSC_1"
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_2",
        README_TEMPLATE: "CHECKSUM_README_2",
        "network.dsc": "CHECKSUM_DSC_1",
        "network.xdsl": "CHECKSUM_XDSL_1"
    }

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": [
            README_TEMPLATE,
            "network.xdsl"
        ],
        "deleted": [],
        "status": {
            "id": 1234567,
            "published": False,
            "checksum": "CHECKSUM_METADATA_2",
            "files": {
                README_TEMPLATE: "CHECKSUM_README_2",
                "network.dsc": "CHECKSUM_DSC_1",
                "network.xdsl": "CHECKSUM_XDSL_1"
            }
        }
    }


# Delete file from dataset
def test_dataset_delete_file():
    deposit = Deposit.__new__(Deposit)
    deposit.status = {
        "id": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: "CHECKSUM_README_1",
            "network.dsc": "CHECKSUM_DSC_1",
            "network.xdsl": "CHECKSUM_XDSL_1"
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_2",
        README_TEMPLATE: "CHECKSUM_README_2",
        "network.dsc": "CHECKSUM_DSC_1",
    }

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": [
            README_TEMPLATE,
        ],
        "deleted": [
            "network.xdsl"
        ],
        "status": {
            "id": 1234567,
            "published": False,
            "checksum": "CHECKSUM_METADATA_2",
            "files": {
                README_TEMPLATE: "CHECKSUM_README_2",
                "network.dsc": "CHECKSUM_DSC_1"
            }
        }
    }
