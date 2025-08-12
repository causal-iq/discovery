
# Unit test Deposit._identify_changes

from causaliq_repro.deposit import Deposit, METADATA_TEMPLATE, README_TEMPLATE


# New root deposit
def test_root_create(monkeypatch):
    deposit = Deposit.__new__(Deposit)
    deposit.base = ""
    deposit.name = ""
    deposit.status = {}
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_1",
        README_TEMPLATE: "CHECKSUM_README_1"
    }
    monkeypatch.setattr("causaliq_repro.deposit.getsize", lambda _: 107)

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": {
            README_TEMPLATE
        },
        "deleted": set(),
        "status": {
            "checksum": "CHECKSUM_METADATA_1",
            "files": {
                README_TEMPLATE: {"checksum": "CHECKSUM_README_1",
                                  "size": 107}
            }
        }
    }


# Unchanged root deposit
def test_root_unchanged(monkeypatch):
    deposit = Deposit.__new__(Deposit)
    deposit.base = ""
    deposit.name = ""
    deposit.status = {
        "recid": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: {
                "checksum": "CHECKSUM_README_1",
                "size": 107,
                "fileid": "pretend-file-id"
            }
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_1",
        README_TEMPLATE: "CHECKSUM_README_1"
    }
    monkeypatch.setattr("causaliq_repro.deposit.getsize", lambda _: 107)

    changed = deposit._identify_changes(checksums)

    assert changed is None


# Updating root metadata
def test_root_update_metadata_only(monkeypatch):
    deposit = Deposit.__new__(Deposit)
    deposit.base = ""
    deposit.name = ""
    deposit.status = {
        "recid": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: {"checksum": "CHECKSUM_README_1",
                              "size": 107,
                              "fileid": "readme-md-fileid"}
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_2",
        README_TEMPLATE: "CHECKSUM_README_1"
    }
    monkeypatch.setattr("causaliq_repro.deposit.getsize", lambda _: 107)

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": set(),
        "deleted": set(),
        "status": {
            "recid": 1234567,
            "published": False,
            "checksum": "CHECKSUM_METADATA_2",
            "files": {
                README_TEMPLATE: {"checksum": "CHECKSUM_README_1",
                                  "size": 107,
                                  "fileid": "readme-md-fileid"}
            }
        }
    }


# Updating root readme
def test_root_update_readme(monkeypatch):
    deposit = Deposit.__new__(Deposit)
    deposit.base = ""
    deposit.name = ""
    deposit.status = {
        "recid": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: {"checksum": "CHECKSUM_README_1",
                              "size": 107,
                              "fileid": "readme-md-fileid"}
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_1",
        README_TEMPLATE: "CHECKSUM_README_2"
    }
    monkeypatch.setattr("causaliq_repro.deposit.getsize", lambda _: 107)

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": False,
        "files": {README_TEMPLATE},
        "deleted": {(README_TEMPLATE, "readme-md-fileid")},
        "status": {
            "recid": 1234567,
            "published": False,
            "checksum": "CHECKSUM_METADATA_1",
            "files": {
                README_TEMPLATE: {"checksum": "CHECKSUM_README_2",
                                  "size": 107,
                                  "fileid": "readme-md-fileid"}
            }
        }
    }


# Updating root metadata and readme
def test_root_update_metadata_and_readme(monkeypatch):
    deposit = Deposit.__new__(Deposit)
    deposit.base = ""
    deposit.name = ""
    deposit.status = {
        "recid": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: {
                "checksum": "CHECKSUM_README_1",
                "size": 107,
                "fileid": "readme-md-fileid"
            }
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_2",
        README_TEMPLATE: "CHECKSUM_README_2"
    }
    monkeypatch.setattr("causaliq_repro.deposit.getsize", lambda _: 107)

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": {README_TEMPLATE},
        "deleted": {(README_TEMPLATE, "readme-md-fileid")},
        "status": {
            "recid": 1234567,
            "published": False,
            "checksum": "CHECKSUM_METADATA_2",
            "files": {
                README_TEMPLATE: {
                    "checksum": "CHECKSUM_README_2",
                    "size": 107,
                    "fileid": "readme-md-fileid"
                }
            }
        }
    }


# New dataset deposit
def test_dataset_create(monkeypatch):
    deposit = Deposit.__new__(Deposit)
    deposit.base = ""
    deposit.name = ""
    deposit.status = {}
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_1",
        README_TEMPLATE: "CHECKSUM_README_1",
        "network.dsc": "CHECKSUM_DSC_1"
    }
    monkeypatch.setattr("causaliq_repro.deposit.getsize", lambda _: 1024)

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": {
            README_TEMPLATE,
            "network.dsc"
        },
        "deleted": set(),
        "status": {
            "checksum": "CHECKSUM_METADATA_1",
            "files": {
                README_TEMPLATE: {
                    "checksum": "CHECKSUM_README_1",
                    "size": 1024
                },
                "network.dsc": {
                    "checksum": "CHECKSUM_DSC_1",
                    "size": 1024
                }
            }
        }
    }


# Unchanged dataset deposit
def test_dataset_unchanged(monkeypatch):
    deposit = Deposit.__new__(Deposit)
    deposit.base = ""
    deposit.name = ""
    deposit.status = {
        "recid": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: {
                "checksum": "CHECKSUM_README_1",
                "size": 3000,
                "fileid": "readme-md-fileid"
            },
            "network.dsc": {
                "checksum": "CHECKSUM_DSC_1",
                "size": 3000,
                "fileid": "network-dsc-fileid"
            }
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_1",
        README_TEMPLATE: "CHECKSUM_README_1",
        "network.dsc": "CHECKSUM_DSC_1"
    }
    monkeypatch.setattr("causaliq_repro.deposit.getsize", lambda _: 3000)

    changed = deposit._identify_changes(checksums)

    assert changed is None


# Add file to dataset
def test_dataset_add_file(monkeypatch):
    deposit = Deposit.__new__(Deposit)
    deposit.base = ""
    deposit.name = ""
    deposit.status = {
        "recid": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: {
                "checksum": "CHECKSUM_README_1",
                "size": 2048,
                "fileid": "readme-md-fileid"
            },
            "network.dsc": {
                "checksum": "CHECKSUM_DSC_1",
                "size": 2048,
                "fileid": "network-dsc-fileid"
            }
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_2",
        README_TEMPLATE: "CHECKSUM_README_2",
        "network.dsc": "CHECKSUM_DSC_1",
        "network.xdsl": "CHECKSUM_XDSL_1"
    }
    monkeypatch.setattr("causaliq_repro.deposit.getsize", lambda _: 2048)

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": {
            README_TEMPLATE,
            "network.xdsl"
        },
        "deleted": {(README_TEMPLATE, "readme-md-fileid")},
        "status": {
            "recid": 1234567,
            "published": False,
            "checksum": "CHECKSUM_METADATA_2",
            "files": {
                README_TEMPLATE: {
                    "checksum": "CHECKSUM_README_2",
                    "size": 2048,
                    "fileid": "readme-md-fileid"
                },
                "network.dsc": {
                    "checksum": "CHECKSUM_DSC_1",
                    "size": 2048,
                    "fileid": "network-dsc-fileid"
                },
                "network.xdsl": {
                    "checksum": "CHECKSUM_XDSL_1",
                    "size": 2048
                }
            }
        }
    }


# Delete file from dataset
def test_dataset_delete_file(monkeypatch):
    deposit = Deposit.__new__(Deposit)
    deposit.base = ""
    deposit.name = ""
    deposit.status = {
        "recid": 1234567,
        "published": False,
        "checksum": "CHECKSUM_METADATA_1",
        "files": {
            README_TEMPLATE: {
                "checksum": "CHECKSUM_README_1",
                "size": 107,
                "fileid": "readme-md-fileid"
            },
            "network.dsc": {
                "checksum": "CHECKSUM_DSC_1",
                "size": 107,
                "fileid": "network-dsc-fileid"
            },
            "network.xdsl": {
                "checksum": "CHECKSUM_XDSL_1",
                "size": 107,
                "fileid": "network-xdsl-fileid"
            }
        }
    }
    checksums = {
        METADATA_TEMPLATE: "CHECKSUM_METADATA_2",
        README_TEMPLATE: "CHECKSUM_README_2",
        "network.dsc": "CHECKSUM_DSC_1",
    }
    monkeypatch.setattr("causaliq_repro.deposit.getsize", lambda _: 107)

    changed = deposit._identify_changes(checksums)

    assert changed == {
        "metadata": True,
        "files": {
            README_TEMPLATE,
        },
        "deleted": {
            (README_TEMPLATE, "readme-md-fileid"),
            ("network.xdsl", "network-xdsl-fileid")
        },
        "status": {
            "recid": 1234567,
            "published": False,
            "checksum": "CHECKSUM_METADATA_2",
            "files": {
                README_TEMPLATE: {
                    "checksum": "CHECKSUM_README_2",
                    "size": 107,
                    "fileid": "readme-md-fileid"
                },
                "network.dsc": {
                    "checksum": "CHECKSUM_DSC_1",
                    "size": 107,
                    "fileid": "network-dsc-fileid"
                }
            }
        }
    }
