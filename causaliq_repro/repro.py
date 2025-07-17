from argparse import ArgumentParser, RawTextHelpFormatter
from os import getenv
from enum import Enum
from requests import get as get_url

ZENODO_LIVE_BASE = "https://zenodo.org/api"
ZENODO_SANDBOX_BASE = "https://sandbox.zenodo.org/api"
ZENODO_LIVE_ENV = "ZENODO_LIVE"
ZENODO_SANDBOX_ENV = "ZENODO_SANDBOX"
ZENODO_EMAIL_ENV = "ZENODO_EMAIL"


class StrEnum(str, Enum):
    pass


class ResourceType(StrEnum):
    OTHER = "other"
    DATASET = "dataset"


class RelationType(StrEnum):
    HAS_PART = "hasPart"
    IS_DERIVED_FROM = "isDerivedFrom"
    IS_REFERENCED_BY = "isReferencedBy"
    IS_SUPPLEMENT_TO = "isSupplementTo"


def run_repro():
    """
        Run the reproducibility job specified by command line arguments.
    """
    args = get_args()

    # Example usage
    print(f"Operation: {args.operation}")
    print(f"Target: {args.target}")
    print(f"Run: {args.run}")
    print(f"Zenodo: {args.zenodo}")


def get_zenodo_token(live: bool):
    """
        Get and check Zenodo access tokens for live and sandbox systems

        :param bool live: live token required, otherwise sandbox

        :returns str/None: validated token, otherwise None
    """
    # Obtain token and Zenodo user email from environment variables
    token = getenv(ZENODO_LIVE_ENV if live else ZENODO_SANDBOX_ENV)
    email = getenv(ZENODO_EMAIL_ENV)

    # Check that the token works
    if token is not None:
        url = (ZENODO_LIVE_BASE if live else ZENODO_SANDBOX_BASE) + "/me"
        response = get_url(url=url,
                           headers={"Authorization": f"Bearer {token}"})
        token = (token if response.status_code == 200
                 and response.json().get("email") == email else None)

    return token


def get_args():
    """
        Check and return the command line arguments
    """
    # See if this is an admin user
    admin = (get_zenodo_token(live=True) is not None
             and get_zenodo_token(live=False) is not None)

    # Set up the supported command line arguments
    parser = ArgumentParser(
        description="Reproduce or download assets for CausalIQ papers.",
        formatter_class=RawTextHelpFormatter
    )
    op_choices = ["learn", "analyse", "download"]
    op_help = (
        "Operation to perform:\n"
        "  learn     Run learning and analysis\n"
        "  analyse   Run analysis\n"
        "  download  Download results and asset\n"
    )
    if admin is True:
        op_choices += ["upload"]
        op_help += "  upload    Upload asset"
    parser.add_argument("operation", choices=op_choices, help=op_help)

    parser.add_argument(
        "target",
        help=(
            "Path to the asset to reproduce\n" +
            "(e.g., papers/ijar_stable/fig1)"
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually perform the operation (omit for a dry run)"
    )
    if admin:
        parser.add_argument(
            "--zenodo",
            choices=["sandbox", "live"],
            default="None",
            help="Whether to use live or sandbox Zenodo"
        )
    args = parser.parse_args()

    # set live/sandbox if not set, and add token to args
    args.zenodo = ("live" if admin is False else
                   ("sandbox" if args.operation == "upload" else "live"))
    args.token = get_zenodo_token(live=(args.zenodo == "live"))

    return args
