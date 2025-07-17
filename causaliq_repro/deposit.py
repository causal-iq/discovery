# Class encapsulating Zenodo deposits

from os import scandir
from hashlib import new as new_hashlib
import requests

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from json import load, loads, JSONDecodeError

METADATA_TEMPLATE = "metadata.json.j2"
README_TEMPLATE = "readme.md.j2"

ZENODO_DIR = "causaliq_repro/zenodo/"  # where Zenodo template files located

ZENODO_URL = None  # safety for now, "https://zenodo.org/api"
SANDBOX_URL = "https://sandbox.zenodo.org/api"

LICENSE = "CC-BY-4.0"
CREATOR_ORCID = "0000-0002-7970-1453"
COMMON_METADATA = {
    "creator_name": "Kitson, Neville Kenneth",
    "creator_orcid": CREATOR_ORCID,
    "creator_affiliation": "see ORCID",
    "license": LICENSE,
    "upload_type": "other",
    "keywords": [
        "Bayesian Networks",
        "Causal Discovery",
        "Reproducibility",
        "Open Science"
    ],
    "related_identifiers": [
        {
            "identifier": f"https://orcid.org/{CREATOR_ORCID}",
            "relation": "isDocumentedBy",
            "scheme": "URL"
        }
    ],
    "language": "eng",
    "access_right": "open"
}

COMMON_README = {
    "license": LICENSE,
    "contact_name": "Dr. Ken Kitson",
    "creator_orcid": CREATOR_ORCID,
    "repo_name": "CausalIQ GitHub repository",
    "repo_url": "https://github.com/causal-iq/discovery"
}


class Deposit:
    """
        Zenodo deposit
    """
    def __init__(self, name: str, live: bool, base_dir: str = ""):
        """
            Instantiate specified deposit from template file

            :param str name: (pathlike) name of the resource e.g. "/data/asia"
            :param bool live: whether this is a live deposit, else sandbox
            :param str base_dir: use to change file directory for testing
        """
        self.name = name
        self.base = base_dir + ZENODO_DIR
        self.live = live

        self._render_jinja2_templates()
        self._read_status_json()

    def _render_jinja2_templates(self):
        """
            Render the JinJa2 templates for metadata.json and readme.md

            :raises FileNotFoundError: if file does not exist
            :raises ValueError: if file has bad contents
        """
        try:
            env = Environment(loader=FileSystemLoader(self.base + self.name))

            template = env.get_template(METADATA_TEMPLATE)
            rendered_metadata = template.render(**COMMON_METADATA)
            self.metadata = loads(
                rendered_metadata
            )  # Parse JSON into Python structure

            template = env.get_template(README_TEMPLATE)
            self.readme = template.render(**COMMON_README)

        except TemplateNotFound as e:
            raise FileNotFoundError(f"Template not found: {self.name} ({e})")

        except JSONDecodeError as e:
            raise ValueError(f"Metadata not valid JSON: {self.name} ({e})")

        except Exception as e:
            raise ValueError(f"Error rendering {self.name}: {e})")

    def _read_status_json(self):
        """
            Read the appropriate status JSON file to check status of
            deposit on either live or sandbox Zenodod
        """
        path = self.base + self.name + ("/zenodo" if self.live else
                                        "/sandbox") + "_status.json"
        try:
            with open(path, "r") as f:
                self.status = load(f)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Status not found in {self.name}: {e}")

        except JSONDecodeError as e:
            raise ValueError(f"Status not valid JSON: {self.name} ({e})")

        except Exception as e:
            raise ValueError(f"Error reading status of {self.name}: {e}")

    def _get_checksums(self, folder):
        """
            Get names and checksums of files in specified local folder
        """
        def file_checksum(path, algo="sha256", chunk_size=8192):
            h = new_hashlib(algo)
            with open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    h.update(chunk)
            return h.hexdigest()

        results = {}
        for entry in scandir(folder):
            if entry.is_file():
                if entry.name in {'zenodo_status.json', 'sandbox_status.json'}:
                    continue
                checksum = file_checksum(entry.path)
                results.update({entry.name: checksum})
        return results

    def _identify_changes(self, checksums: dict):
        """
            Identify changes required to deposits based on existing status
            and checksums of local files

            :param dict checksums: checksums of local files

            :returns dict/None: {"metadata": bool, has metadata changed,
                                 "status": dict, contents of new status record}
        """
        status = {
            "id": self.status["id"]
        }
        if "published" in self.status:
            status["published"] = self.status["published"]
        changed = {}

        # See if metadata needs to be changed
        changed["metadata"] = (self.status["id"] is None or
                               self.status["checksum"] !=
                               checksums[METADATA_TEMPLATE])
        status["checksum"] = checksums[METADATA_TEMPLATE]

        # Check for new and modified files
        status["files"] = {}
        changed["files"] = []
        for file, checksum in checksums.items():
            if file == METADATA_TEMPLATE:
                continue
            status['files'][file] = checksum
            if (self.status["id"] is None
                    or file not in self.status["files"]
                    or self.status["files"][file] != checksum):
                changed["files"].append(file)

        # Check for deleted files
        changed["deleted"] = []
        if self.status["id"] is not None:
            for file in self.status["files"]:
                if file not in checksums:
                    changed["deleted"].append(file)

        changed.update({"status": status})
        return changed if changed['status'] != self.status else None

    def upload(self, dry_run: bool, auth_token: str):
        """
            Ensures deposit on Zenodo matches local files

            :param bool dry_run: whether this ia just a dry run
            :param str auth_token: token required to update Zenodo
        """
        MSG = "\n** Uploading '{}' to {} Zenodo"
        print(MSG.format(self.name, "LIVE" if self.live else "sandbox"))

        # get names of local files and their checksums
        checksums = self._get_checksums(self.base + self.name)

        # Identify changes required to deposit
        changes = self._identify_changes(checksums)

        if changes is None:
            print(f"'{self.name}' unchanged - nothing to upload")
        else:
            # create the deposit on Zenodo if necessary
            if self.status["id"] is None:
                self._create_draft_deposit(dry_run, auth_token)

    def _create_draft_deposit(self, dry_run: bool, auth_token: str):
        """
            Create a draft deposit on Zenodo sandbox without files

            :param bool dry_run: whether this ia just a dry run
            :param str auth_token: token required to update Zenodo

            :raises ValueError: if the API request fails
        """
        MSG = "   - draft deposit created (id: {})"
        if dry_run is True:
            self.status["id"] = -1
            self.status["published"] = False
            print(MSG.format(self.status["id"]))
            return

        response = requests.get(SANDBOX_URL + "/deposit/depositions",
                                headers={"Authorization":
                                         f"Bearer {auth_token}"})
        if response.status_code != 200:
            raise ValueError("Invalid token or insufficient permissions")

        url = SANDBOX_URL + "/deposit/depositions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}"
        }
        data = {
            "metadata": self.metadata  # Use pre-rendered metadata
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code != 201:
            error_message = (
                f"Failed to create deposit: {response.status_code} "
                f"{response.text}"
            )
            raise ValueError(error_message)

        deposit_info = response.json()
        self.status["id"] = deposit_info['id']
        self.status["published"] = False
        print(MSG.format(self.status["id"]))

    def delete(self, dry_run: bool, auth_token: str):
        """
        Delete a (draft) deposit from Zenodo.

        :param int deposit_id: ID of the deposit to delete
        :param str auth_token: Authentication token for Zenodo
        :param bool sandbox: Whether to use the sandbox environment

        :raises ValueError: if the API request fails
        """
        print(f"\n** Deleting '{self.name}' " +
              f"from {'LIVE' if self.live else 'sandbox'} Zenodo")

        if dry_run is False:
            base_url = ZENODO_URL if self.live else SANDBOX_URL
            url = f"{base_url}/deposit/depositions/{self.status['id']}"

            headers = {
                "Authorization": f"Bearer {auth_token}"
            }

            response = requests.delete(url, headers=headers)

            if response.status_code != 204:
                error_message = (
                    f"Failed to delete deposit: {response.status_code} "
                    f"{response.text}"
                )
                raise ValueError(error_message)

        print(f"   - draft deposit deleted (id: {self.status['id']})")
        self.status = {"id": None}
