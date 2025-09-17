
# Integration Tests of CauaslIQ Pipeline for an isolated root node

from pytest import MonkeyPatch
from _pytest.capture import CaptureFixture

from tests.common import REPRO_TEST_DATA_DIR
import tests.causaliq_repro.integration.helpers as helpers


# --- Workflow tests with draft deposits - requires authentication token

# Test workflow with a draft root node
def test_root_only(token: str, capsys: CaptureFixture,
                   monkeypatch: MonkeyPatch, echo_output: bool):
    base_dir = REPRO_TEST_DATA_DIR + "integration/root2/"

    # Ensure empty status files are present
    helpers.init_status_file(name="", base_dir=base_dir)

    # upload which creates the root deposit
    helpers.upload_create(name="", base_dir=base_dir, token=token,
                          capsys=capsys, echo_output=echo_output)

    # checking root deposit metadata
    helpers.get_metadata(name="", rootid="2", files={"readme.md"}, related={},
                         base_dir=base_dir, token=token, capsys=capsys,
                         echo_output=echo_output)

    # upload where nothing has changed
    helpers.upload_nochange(name="", base_dir=base_dir, token=token,
                            capsys=capsys, echo_output=echo_output)

    # upload where metadata has changed
    helpers.upload_metadata_changed(name="", base_dir=base_dir, token=token,
                                    capsys=capsys, monkeypatch=monkeypatch,
                                    echo_output=echo_output,
                                    title_mod=("2", "2A"))

    # check that deposit metadata has been changed
    helpers.get_metadata(name="", rootid="2A", files={"readme.md"}, related={},
                         base_dir=base_dir, token=token, capsys=capsys,
                         echo_output=echo_output)

    # adding a file
    helpers.add_file(name="", filename="testfile.tmp", base_dir=base_dir,
                     token=token, capsys=capsys, echo_output=echo_output)

    # check that file has been added to deposit
    helpers.get_metadata(name="", rootid="2A",
                         files={"readme.md", "testfile.tmp"}, related={},
                         base_dir=base_dir, token=token, capsys=capsys,
                         echo_output=echo_output)

    # upload where metadata and readme has changed
    helpers.upload_metadata_and_readme_changed(name="", base_dir=base_dir,
                                               token=token, capsys=capsys,
                                               monkeypatch=monkeypatch,
                                               echo_output=echo_output,
                                               title_mod=("2", "2B"))

    # check that metadata has been changed, still two files
    helpers.get_metadata(name="", rootid="2B",
                         files={"readme.md", "testfile.tmp"}, related={},
                         base_dir=base_dir, token=token, capsys=capsys,
                         echo_output=echo_output)

    # removing a file
    helpers.remove_file(name="", filename="testfile.tmp", base_dir=base_dir,
                        token=token, capsys=capsys, echo_output=echo_output)

    # check that file has been has been removed from deposit
    helpers.get_metadata(name="", rootid="2B", files={"readme.md"}, related={},
                         base_dir=base_dir, token=token, capsys=capsys,
                         echo_output=echo_output)

    # delete the root deposit
    helpers.delete(name="", base_dir=base_dir, token=token, capsys=capsys,
                   echo_output=echo_output)


# Test workflow with a draft root and hub
def test_root_and_hub(token: str, capsys: CaptureFixture, echo_output: bool):
    base_dir = REPRO_TEST_DATA_DIR + "integration/root3/"

    # Ensure empty status files are present
    helpers.init_status_file(name="", base_dir=base_dir)
    helpers.init_status_file(name="hub", base_dir=base_dir)

    # upload which creates the root deposit
    helpers.upload_create(name="", base_dir=base_dir, token=token,
                          capsys=capsys, echo_output=echo_output)

    # checking root deposit metadata
    helpers.get_metadata(name="", rootid="3", files={"readme.md"}, related={},
                         base_dir=base_dir, token=token, capsys=capsys,
                         echo_output=echo_output)

    # upload which creates the hub deposit
    helpers.upload_create(name="hub", base_dir=base_dir, token=token,
                          capsys=capsys, echo_output=echo_output)

    # checking hub deposit metadata
    helpers.get_metadata(name="hub", rootid="3 Hub", files={"readme.md"},
                         related={}, base_dir=base_dir, token=token,
                         capsys=capsys, echo_output=echo_output)

    # delete the hub deposit
    helpers.delete(name="hub", base_dir=base_dir, token=token, capsys=capsys,
                   echo_output=echo_output)

    # check root deposit metadata again
    helpers.get_metadata(name="", rootid="3", files={"readme.md"}, related={},
                         base_dir=base_dir, token=token, capsys=capsys,
                         echo_output=echo_output)

    # delete the root deposit
    helpers.delete(name="", base_dir=base_dir, token=token, capsys=capsys,
                   echo_output=echo_output)


# Test workflow with a draft root, hub and leaves
def test_root_hub_and_leaves(token: str, capsys: CaptureFixture,
                             echo_output: bool):
    base_dir = REPRO_TEST_DATA_DIR + "integration/root4/"

    # Ensure empty status files are present
    helpers.init_status_file(name="", base_dir=base_dir)
    helpers.init_status_file(name="hub", base_dir=base_dir)
    helpers.init_status_file(name="hub/leaf1", base_dir=base_dir)
    helpers.init_status_file(name="hub/leaf2", base_dir=base_dir)

    # --- upload which creates the root deposit

    helpers.upload_create(name="", base_dir=base_dir, token=token,
                          capsys=capsys, echo_output=echo_output)

    # checking root deposit metadata
    helpers.get_metadata(name="", rootid="4", files={"readme.md"}, related={},
                         base_dir=base_dir, token=token, capsys=capsys,
                         echo_output=echo_output)

    # --- upload which creates the hub deposit

    helpers.upload_create(name="hub", base_dir=base_dir, token=token,
                          capsys=capsys, echo_output=echo_output)

    # checking hub deposit metadata
    helpers.get_metadata(name="hub", rootid="4 Hub", files={"readme.md"},
                         related={}, base_dir=base_dir, token=token,
                         capsys=capsys, echo_output=echo_output)

    # checking root deposit metadata
    helpers.get_metadata(name="", rootid="4", files={"readme.md"},
                         related={}, base_dir=base_dir, token=token,
                         capsys=capsys, echo_output=echo_output)

    # --- upload which creates the leaf1 deposit

    helpers.upload_create(name="hub/leaf1", base_dir=base_dir, token=token,
                          capsys=capsys, echo_output=echo_output)

    # check metadata of leaf1 deposit
    helpers.get_metadata(name="hub/leaf1", rootid="4 Leaf 1",
                         files={"readme.md"}, related={}, base_dir=base_dir,
                         token=token, capsys=capsys, echo_output=echo_output)

    # check metadata of hub deposit
    helpers.get_metadata(name="hub", rootid="4 Hub",
                         files={"readme.md"}, related={}, base_dir=base_dir,
                         token=token, capsys=capsys, echo_output=echo_output)

    # --- upload which creates the leaf2 deposit

    helpers.upload_create(name="hub/leaf2", base_dir=base_dir, token=token,
                          capsys=capsys, echo_output=echo_output)

    # check metadata of leaf2 deposit
    helpers.get_metadata(name="hub/leaf2", rootid="4 Leaf 2",
                         files={"readme.md"}, related={}, base_dir=base_dir,
                         token=token, capsys=capsys, echo_output=echo_output)

    # --- delete the leaf1 deposit

    helpers.delete(name="hub/leaf1", base_dir=base_dir, token=token,
                   capsys=capsys, echo_output=echo_output)

    # check hub deposit
    helpers.get_metadata(name="hub", rootid="4 Hub",
                         files={"readme.md"}, related={}, base_dir=base_dir,
                         token=token, capsys=capsys, echo_output=echo_output)

    # --- delete the hub deposit

    helpers.delete(name="hub", base_dir=base_dir, token=token, capsys=capsys,
                   echo_output=echo_output)

    # check root deposit
    helpers.get_metadata(name="", rootid="4", files={"readme.md"}, related={},
                         base_dir=base_dir, token=token, capsys=capsys,
                         echo_output=echo_output)

    # check leaf2 deposit
    helpers.get_metadata(name="hub/leaf2", rootid="4 Leaf 2",
                         files={"readme.md"}, related={}, base_dir=base_dir,
                         token=token, capsys=capsys, echo_output=echo_output)

    # --- delete the leaf2 deposit

    helpers.delete(name="hub/leaf2", base_dir=base_dir, token=token,
                   capsys=capsys, echo_output=echo_output)

    # --- delete the root deposit

    helpers.delete(name="", base_dir=base_dir, token=token, capsys=capsys,
                   echo_output=echo_output)
