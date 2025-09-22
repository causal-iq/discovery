from argparse import ArgumentParser, RawTextHelpFormatter, Namespace
from os import getenv
from enum import Enum
from requests import get as get_url
from typing import Optional

from causaliq_repro.deposit import Deposit, ZENODO_DIR

ZENODO_LIVE_BASE = "https://zenodo.org/api"
ZENODO_SANDBOX_BASE = "https://sandbox.zenodo.org/api"
ZENODO_LIVE_ENV = "CAUSALIQ_ZENODO_LIVE"
ZENODO_SANDBOX_ENV = "CAUSALIQ_ZENODO_SANDBOX"
ZENODO_EMAIL_ENV = "CAUSALIQ_ZENODO_EMAIL"


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
    # get and check consistency the command line arguments
    args = get_args()

    # validate that required operation valid
    args = validate_args(args)
    if args is None:
        return

    # perform the requested operation
    sandbox = args.zenodo == "sandbox"
    dry_run = args.run is not True
    if dry_run is True:
        print("\n** This is a DRY-RUN - operations WOULD be:")
    deposit = Deposit(name=args.target, sandbox=sandbox,
                      base_dir=args.base_dir)
    if args.operation == "upload":
        deposit.upload(dry_run=dry_run, token=args.token)
    elif args.operation == "publish":
        deposit.publish(dry_run=dry_run, token=args.token)
    elif args.operation == "delete":
        deposit.delete(dry_run=dry_run, token=args.token)
    else:
        print(f"'{args.operation}' not yet supported")
    print("\nOperations complete\n")


def get_zenodo_token(sandbox: bool):
    """
        Get and check Zenodo access tokens for live and sandbox systems

        :param bool sandbox: sandbox token required, otherwise live

        :returns str/None: validated token, otherwise None
    """
    # Obtain token and Zenodo user email from environment variables
    token = getenv(ZENODO_SANDBOX_ENV if sandbox else ZENODO_LIVE_ENV)
    email = getenv(ZENODO_EMAIL_ENV)

    # Check that the token works
    if token is not None:
        url = (ZENODO_SANDBOX_BASE if sandbox else ZENODO_LIVE_BASE) + "/me"
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
    admin = (get_zenodo_token(sandbox=False) is not None
             and get_zenodo_token(sandbox=True) is not None)

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
        op_choices += ["upload", "publish", "delete"]
        op_help += "  upload    Upload deposit [admin only]\n"
        op_help += "  publish   Publish deposit [admin only]\n"
        op_help += "  delete    Delete draft deposit [admin only]\n"
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

    if not hasattr(args, "zenodo") or args.zenodo == "None":
        args.zenodo = ("live" if admin is False else
                       ("sandbox" if args.operation in
                        {"upload", "publish", "delete"} else "live"))
    args.token = get_zenodo_token(sandbox=(args.zenodo == "sandbox"))

    return args


def validate_args(args: Namespace,
                  base_dir: Optional[str] = None) -> Namespace:
    """
        Validate args take into account existing state of the target etc.

        :param Namespace args: arguments specified on c (timed out)ommand line
        :param str|None base_dir: optional base directory for testing

        :returns Namespace|None: modified and augmented process arguments
    """
    args.base_dir = ZENODO_DIR if base_dir is None else base_dir
    args.target = "" if args.target == "root" else args.target
    sandbox = args.zenodo == "sandbox"

    # try to instantiate the deposit which effectively checks the target name
    # and whether the mandatory files are present
    try:
        deposit = Deposit(name=args.target, sandbox=sandbox,
                          base_dir=args.base_dir)
    except Exception as e:
        print(f"Unknown target or invalid/missing files ({e})")
        return None

    # learn, analyse and download not supported yet
    if args.operation in {"learn", "analyse", "download"}:
        print(f"'{args.operation}' not supported yet")
        return None

    # delete and publish only allowed when deposit in draft status
    state = ("absent" if "recid" not in deposit.status
             else ("published" if deposit.status["published"] else "draft"))
    if args.operation == "publish" and state != "draft":
        print("Can only publish deposits in draft status")
        return None
    elif args.operation == "delete" and state != "draft":
        print("Can only delete deposits in draft status")
        return None

    return args
