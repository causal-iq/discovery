
# Ensures test deposits are on Sandbox in case Sandbox has been cleared

import pytest

from tests.common import REPRO_TEST_DATA_DIR
from causaliq_repro.repro import get_zenodo_token
from causaliq_repro.deposit import Deposit


def ensure_deposit_present(name: str, token: str):
    """
        Check if integration test deposit present on Sandbox, and recreate
        it if it is missing

        :param str name: name of the deposit
        :param str token: Zenodo authentication token
    """
    # See if deposit present on sandbox by trying to download "readme.md"
    try:
        base_dir = REPRO_TEST_DATA_DIR + "integration/"
        deposit = Deposit(name=name, live=False, base_dir=base_dir)
        deposit.download(file="readme.md", dry_run=False, token=token)
    #     print(f"   - '{name}' present on sandbox Zenodo")

    # if deposit not on sandbox, upload to recreate it
    except ValueError:
    #     print(f"   - '{name}' not on sandbox Zenodo - recreating it ...")
        deposit.status = {}
        deposit.upload(dry_run=False, token=token)


@pytest.fixture(scope="session", autouse=True)
def global_setup():
    # Perform setup
    # print("\n\nGlobal setup: Ensuring required deposits on sandbox ...")

    token = get_zenodo_token(live=False)

    ensure_deposit_present(name="", token=token)
    ensure_deposit_present(name="/hub", token=token)

    yield
    # Perform teardown
    # print("\nGlobal teardown: Cleaning up resources")
