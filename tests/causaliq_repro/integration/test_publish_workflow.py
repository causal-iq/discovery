
# Integration Tests of CauaslIQ Pipeline deposit publish

from pytest import fixture, skip, raises, MonkeyPatch
from _pytest.capture import CaptureFixture
from os.path import isfile

from tests.common import REPRO_TEST_DATA_DIR
from causaliq_repro.deposit import Deposit, ZenodoError
from causaliq_repro.repro import get_zenodo_token

import tests.causaliq_repro.integration.helpers as helpers


@fixture
def check_setup():
    """
    """
    base_dir = REPRO_TEST_DATA_DIR + "integration/root1/"
    deposits = {"", "hub1", "hub1/leaf1"}

    # Firstly ensure there are status files in case this is a virgin run
    print("\n\n")
    for name in deposits:
        _name = name if name == "" else name + "/"
        if not isfile(f"{base_dir}{_name}sandbox_status.json"):
            print(f"** Creating empty status file for '{name}'\n")
            helpers.init_status_file(name=name, base_dir=base_dir)

    # Now check whether deposits are on the sandbox - will skip test if so

    deposits_present = False
    token = get_zenodo_token(sandbox=True)
    print("Checking if deposits from previous runs on sandbox")
    for name in deposits:
        deposit = Deposit(name=name, sandbox=True, base_dir=base_dir)
        if "recid" in deposit.status:
            try:
                deposit.get_metadata(dry_run=False, token=token)
                deposits_present = True
                break
            except ZenodoError:
                pass

    if deposits_present:
        print("Root1 deposits on sandbox - skip test")
        skip("Skipping test - deposits on sandbox - manually clear them")
    else:
        print("No root1 deposits on sandbox - can run test")

    return base_dir


# -- Publish workflow test - skipped if deposits on Sandbox or authentication
# -- token not present

# Check a typical creation, publication and modification workflow
def test_publish_workflow(token: str, capsys: CaptureFixture,
                          monkeypatch: MonkeyPatch, echo_output: bool,
                          check_setup: str):

    base_dir = check_setup
    print("Integration test of publish workflow ...\n")

    # *** First upload and publish the placeholder hub ***

    # upload which creates the hub deposit
    helpers.upload_create(name="hub1", base_dir=base_dir, token=token,
                          capsys=capsys, echo_output=echo_output)

    # checking hub deposit metadata
    helpers.get_metadata(name="hub1", rootid="1 Hub V1", files={"readme.md"},
                         related={}, base_dir=base_dir, token=token,
                         capsys=capsys, echo_output=echo_output)

    # publish version 1 of the hub
    helpers.publish(name="hub1", base_dir=base_dir, token=token,
                    capsys=capsys, echo_output=echo_output)

    # checking hub deposit metadata - state should now be "done"
    hub = helpers.get_metadata(name="hub1", rootid="1 Hub V1",
                               files={"readme.md"}, related={},
                               base_dir=base_dir, token=token, capsys=capsys,
                               echo_output=echo_output, state="done")

    # check cannot publish hub once it is already published
    with raises(ValueError):
        helpers.publish(name="hub1", base_dir=base_dir, token=token,
                        capsys=capsys, echo_output=echo_output)

    # *** Upload and publish leaf 1 ***

    # upload leaf1 which will references published hub
    helpers.upload_create(name="hub1/leaf1", base_dir=base_dir, token=token,
                          capsys=capsys, echo_output=echo_output)

    # publish version 1 of leaf 1
    helpers.publish(name="hub1/leaf1", base_dir=base_dir, token=token,
                    capsys=capsys, echo_output=echo_output)

    # check leaf1 metadata - hub should be a related deposit
    related = {
        hub["conceptdoi"]: {
            "relation": "isPartOf",
            "resource_type": "other",
            "scheme": "doi"
        }
    }
    leaf1 = helpers.get_metadata(name="hub1/leaf1", rootid="1 Leaf 1 V1",
                                 files={"readme.md"}, related=related,
                                 base_dir=base_dir, token=token, capsys=capsys,
                                 echo_output=echo_output, state="done")

    # *** Upload and publish root ***

    # upload and publish root which references hub
    helpers.upload_create(name="", base_dir=base_dir, token=token,
                          capsys=capsys, echo_output=echo_output)

    # publish version 1 of root
    helpers.publish(name="", base_dir=base_dir, token=token,
                    capsys=capsys, echo_output=echo_output)

    # check root metadata - hub should be a related deposit
    related = {
        hub["conceptdoi"]: {
            "relation": "hasPart",
            "resource_type": "other",
            "scheme": "doi"
        }
    }
    root = helpers.get_metadata(name="", rootid="1 V1",
                                files={"readme.md"}, related=related,
                                base_dir=base_dir, token=token, capsys=capsys,
                                echo_output=echo_output, state="done")

    # *** Modify hub so it references root and leaf 1

    # Modify metadata and readme (simulated), which includes new related items
    helpers.upload_metadata_and_readme_changed(name="hub1", base_dir=base_dir,
                                               token=token, capsys=capsys,
                                               monkeypatch=monkeypatch,
                                               echo_output=echo_output,
                                               title_mod=("V1", "V2"))

    # publish version 2 of the hub (sends metadata so modify title again)
    helpers.publish(name="hub1", base_dir=base_dir, token=token,
                    capsys=capsys, echo_output=echo_output,
                    title_mod=("V1", "V2"))

    # checking hub deposit metadata - state "unsubmitted", and hub and
    # root and title is V2
    related = {
        root["conceptdoi"]: {
            "relation": "isPartOf",
            "resource_type": "other",
            "scheme": "doi"
        },
        leaf1["conceptdoi"]: {
              "relation": "hasPart",
              "resource_type": "dataset",
              "scheme": "doi"
        }
    }
    hub = helpers.get_metadata(name="hub1", rootid="1 Hub V2",
                               files={"readme.md"}, related=related,
                               base_dir=base_dir, token=token, capsys=capsys,
                               echo_output=echo_output, state="done")
