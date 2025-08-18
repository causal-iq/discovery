# Class encapsulating Zenodo deposits

from os import scandir
from os.path import getsize
from enum import Enum
from hashlib import new as new_hashlib
import requests
from requests.models import Response

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from json import load, loads, dump, JSONDecodeError


class StrEnum(str, Enum):
    pass


class ZenodoOp(StrEnum):  # Operations possible on Zenodo deposits
    CREATE = "create deposit"
    UPDATE = "update metadata"
    PUBLISH = "publish deposit"
    DOWNLOAD_FILE = "download file"
    GET_METADATA = "get metadata"
    DELETE = "delete deposit"
    ADD_FILE = "add file"
    REMOVE_FILE = "remove file"


class ResourceType(StrEnum):  # Supported resource types
    OTHER = "other"


class SchemeType(StrEnum):  # Scheme type of identifier (URL or DOI)
    URL = "URL"
    DOI = "DOI"


class RelationType(StrEnum):  # Supported relations between CausalIQ deposits
    IS_DOCUMENTED_BY = "isDocumentedBy"
    IS_PART_OF = "isPartOf"
    HAS_PART = "hasPart"


METADATA_TEMPLATE = "metadata.json.j2"
README_TEMPLATE = "readme.md.j2"

ZENODO_DIR = "causaliq_repro/zenodo/"  # where Zenodo template files located

ZENODO_URL = None  # safety for now, "https://zenodo.org/api"
SANDBOX_URL = "https://sandbox.zenodo.org/api"

LICENSE = "CC-BY-4.0"
CREATOR_ORCID = "0000-0002-7970-1453"
COMMON_METADATA = {  # Common metadata in all CausalIQ deposits
    "creator_name": "Kitson, Neville Kenneth",
    "creator_orcid": CREATOR_ORCID,
    "creator_affiliation": "see ORCID",
    "license": LICENSE,
    "upload_type": ResourceType.OTHER.value,
    "keywords": [
        "Bayesian Networks",
        "Causal Discovery",
        "Reproducibility",
        "Open Science"
    ],
    "related_identifiers": [
        {
            "identifier": f"https://orcid.org/{CREATOR_ORCID}",
            "relation": RelationType.IS_DOCUMENTED_BY.value,
            "scheme": SchemeType.URL.value,
            "resource_type": ResourceType.OTHER.value
        }
    ],
    "language": "eng",
    "access_right": "open"
}

COMMON_README = {  # Common field values in all readme.md files
    "license": LICENSE,
    "contact_name": "Dr. Ken Kitson",
    "creator_orcid": CREATOR_ORCID,
    "repo_name": "CausalIQ GitHub repository",
    "repo_url": "https://github.com/causal-iq/discovery"
}


class ZenodoError(Exception):  # Signals unexpected response from Zenodo
    pass


class Deposit:
    """
        Zenodo deposit
    """
    def __init__(self, name: str, sandbox: bool, base_dir: str = ZENODO_DIR,
                 load_related: bool = True):
        """
            Instantiate specified deposit from local files

            :param str name: (pathlike) name of the resource e.g. "/data/asia"
            :param bool sandbox: whether this is a sandbox deposit, else live
            :param str base_dir: use to change file directory for testing
            :param bool load_related: whether to load links to related deposits
                                      - is set to False to stop infinite
                                        recursion when assembling related links
        """
        self.name = name
        self.base = base_dir
        self.sandbox = sandbox

        self._render_jinja2_templates()
        if load_related is True:
            self._set_related_identifiers()
        self._read_status()

    def upload(self, dry_run: bool, token: str):
        """
            Ensures deposit on Zenodo matches local files and metadata

            :param bool dry_run: whether this ia just a dry run
            :param str token: authentication token required to update Zenodo
        """
        MSG = "\n** Uploading '{}' to {} Zenodo"
        print(MSG.format(self.name, "sandbox" if self.sandbox else "LIVE"))

        # get names of local files and their checksums
        checksums = self._get_checksums(self.base + self.name)

        # Identify changes required to deposit
        changes = self._identify_changes(checksums)
        if changes is not None:
            self.status = dict(changes["status"])

        if changes is None:
            print(f"   - no changes made (recid: {self.status['recid']})")
        else:
            update_related = False
            # create the deposit on Zenodo if necessary
            if "recid" not in self.status:
                self._create_deposit(dry_run, token)
                update_related = True

            # or modify its metadata if necessary
            elif changes["metadata"] is True:
                self._update_deposit(dry_run, token)

            # remove deleted or changed files
            for file_data in changes["deleted"]:
                self._remove_file(dry_run, token, file_data)

            # upload new or changed files
            for file in changes["files"]:
                if file == "readme.md.j2":
                    fileid = self._add_file(dry_run, token,
                                            ("readme.md", self.readme))
                else:
                    with open(self.base + self.name + "/" + file, "rb") as fd:
                        fileid = self._add_file(dry_run, token, (file, fd))
                self.status["files"][file]["fileid"] = fileid

            # update the local status file if not a dry run
            if dry_run is False:
                self._write_status()

            # update links to this deposit in related deposits
            if update_related is True:
                self._update_related(to=self.name, dry_run=dry_run,
                                     token=token)

    def download(self, file: str, dry_run: bool, token: str):
        """
        Download a single file from Zenodo.

        :param str file: file to download from deposit
        :param bool dry_run: whether this ia just a dry run
        :param str token: Authentication token for Zenodo

        :raises ValueError: if the file download fails
        """
        operation = ZenodoOp.DOWNLOAD_FILE
        print(f"\n** Download {self.name}/{file}" +
              f" from {'sandbox' if self.sandbox else 'LIVE'} Zenodo")

        if dry_run is False:
            response = requests.get(**self._request(operation, token,
                                                    (file, None)))
            if response.status_code == 200:
                if file != "readme.md":
                    path = self.base + self.name + '/' + file
                    with open(path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                        print(f"Saved to {path}")
            else:
                error_message = (
                    f"Failed to download file: {response.status_code} "
                    f"{response.text}"
                )
                raise ValueError(error_message)

        loc_file = "readme.md.j2" if file == "readme.md" else file
        print(f"   - {operation.value} {file} (recid: {self.status['recid']}, "
              f" {self.status['files'][loc_file]['size']} bytes)")
        return

    def get_metadata(self, dry_run: bool, token: str):
        """
        Get metedata of a depsoit on Zenodo

        :param bool dry_run: whether this ia just a dry run
        :param str token: Authentication token for Zenodo

        :raises ValueError: if the file download fails
        """
        operation = ZenodoOp.GET_METADATA
        print(f"\n** Get metadata of '{self.name}'" +
              f" from {'sandbox' if self.sandbox else 'LIVE'} Zenodo")
        recid = self.status["recid"]

        if dry_run is False:
            response = requests.get(**self._request(operation, token))
            info = self._check_response(operation, response)
        else:
            info = None

        print(f"   - {operation.value} (recid: {recid})")
        return info

    def delete(self, dry_run: bool, token: str):
        """
        Delete a (draft) deposit from Zenodo.

        :param bool dry_run: whether this ia just a dry run
        :param str token: Authentication token for Zenodo

        :raises ValueError: if the API request fails
        """
        operation = ZenodoOp.DELETE
        print(f"\n** Deleting '{self.name}' " +
              f"from {'sandbox' if self.sandbox else 'LIVE'} Zenodo")
        recid = self.status["recid"]

        if dry_run is False:
            response = requests.delete(**self._request(operation, token))
            self._check_response(operation, response)
            self.status = {}
            self._write_status()

        print(f"   - {operation.value} (recid: {recid})")

        # update links in related deposits
        self._update_related(to=self.name, dry_run=dry_run, token=token)

    def _update_related(self, to: str, dry_run: bool, token: str):
        """
            Update the related links of all related deposits

            :param str to: links referring to this deposit must be updated
            :param bool dry_run: whether this ia just a dry run
            :param str token: Authentication token for Zenodo
        """
        # Update related links in parent deposit if there is one
        if to != "":
            parent = "/".join(self.name.split("/")[:-1])
            print(f" * Updating related link to '{to}' in '{parent}' " +
                  f"on {'sandbox' if self.sandbox else 'LIVE'} Zenodo")
            parent = Deposit(name=parent, sandbox=self.sandbox,
                             base_dir=self.base)
            parent._update_deposit(dry_run=dry_run, token=token)

    def _render_jinja2_templates(self):
        """
            Render the JinJa2 templates for metadata.json and readme.md

            :raises FileNotFoundError: if file does not exist
            :raises ValueError: if file has bad contents
        """
        try:
            env = Environment(loader=FileSystemLoader(self.base + self.name))

            # Render metadata template into Python dict structure
            template = env.get_template(METADATA_TEMPLATE)
            rendered_metadata = template.render(**COMMON_METADATA)
            self.metadata = loads(
                rendered_metadata
            )

            # Render the readme.md template into markdown
            template = env.get_template(README_TEMPLATE)
            self.readme = template.render(**COMMON_README)

        except TemplateNotFound as e:
            raise FileNotFoundError(f"Template not found: {self.name} ({e})")

        except JSONDecodeError as e:
            raise ValueError(f"Metadata not valid JSON: {self.name} ({e})")

        except Exception as e:
            raise ValueError(f"Error rendering {self.name}: {e})")

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
        status = dict(self.status)
        changed = {}

        # See if metadata needs to be changed
        changed["metadata"] = ("recid" not in self.status or
                               self.status["checksum"] !=
                               checksums[METADATA_TEMPLATE])
        status["checksum"] = checksums[METADATA_TEMPLATE]

        # Check for new and modified files
        status["files"] = {}
        changed["files"] = set()
        changed["deleted"] = set()
        for file, checksum in checksums.items():
            # metadata template file is never added to deposit
            if file == METADATA_TEMPLATE:
                continue

            # record latest checksum and size in new status dict
            status['files'][file] = {"checksum": checksum,
                                     "size": getsize(self.base + self.name +
                                                     "/" + file)}

            # replicate the fileid of already ploaded file into new status
            if ("files" in self.status and file in self.status["files"]
                    and "fileid" in self.status["files"][file]):
                status["files"][file]["fileid"] = (self.status["files"]
                                                   [file]["fileid"])

            # the file is new or modified so add it to changed["files"]
            if ("recid" not in self.status
                    or file not in self.status["files"]
                    or self.status["files"][file]["checksum"] != checksum):
                changed["files"].add(file)

                # if existing file being modified must be deleted first
                if "recid" in self.status and file in self.status["files"]:
                    changed["deleted"].add((file,
                                            self.status["files"]
                                            [file]["fileid"]))

        # Check for deleted files
        if "recid" in self.status:
            for file in self.status["files"]:
                if file not in checksums:
                    changed["deleted"].add((file, self.status["files"]
                                            [file]["fileid"]))

        changed.update({"status": status})
        return changed if changed['status'] != self.status else None

    def _related_info(self, name: str, relation: RelationType):
        """
            Returns info for a related deposit in Zenodo metadata format

            :param str name: deposit name
            :param RelationType relation: the relation type

            :returns str/None: related identifier if deposit on Zenodo
        """
        # get status and resource_type of the deposit
        deposit = Deposit(name=name, sandbox=self.sandbox,
                          base_dir=self.base, load_related=False)
        status = deposit.status
        resource_type_str = deposit.metadata["upload_type"]

        # return None if deposit not on Zenodo
        if "recid" not in status:
            return None

        # Use DOI if deposit published, otherwise URL
        if "doi" in status:
            id = status["doi"]
            scheme = SchemeType.DOI.value
        else:
            id = ((ZENODO_URL if self.sandbox is False
                   else SANDBOX_URL).replace("api", "records") +
                  f"/{status['recid']}?preview={status['version']}")
            scheme = SchemeType.URL.value

        return {
            "identifier": id,
            "relation": relation.value,
            "scheme": scheme,
            "resource_type": resource_type_str
        }

    def _set_related_identifiers(self):
        """
            Sets the related identifiers of parent and children
        """
        # Add in the related identifier of parent on Zenodo (if any)
        if self.name != "":
            parent = "/".join(self.name.split("/")[:-1])
            parent = self._related_info(name=parent,
                                        relation=RelationType.IS_PART_OF)
            if parent is not None:
                self.metadata["related_identifiers"].append(parent)

        # Add in the related identifiers of any children on Zenodo
        for child in scandir(self.base + self.name):
            if not child.is_dir():
                continue
            child = self._related_info(name=child.name,
                                       relation=RelationType.HAS_PART)
            if child is not None:
                self.metadata["related_identifiers"].append(child)

    def _create_deposit(self, dry_run: bool, token: str):
        """
            Create a draft deposit on Zenodo sandbox without files

            :param bool dry_run: whether this ia just a dry run
            :param str token: token required to update Zenodo

            :raises ZenodoError: if the API request fails
        """
        operation = ZenodoOp.CREATE
        if dry_run is False:
            response = requests.post(**self._request(operation, token))
            info = self._check_response(operation, response)
            self.status["recid"] = info['id']
            self.status["conceptid"] = info["conceptrecid"]
        else:
            self.status["recid"] = -1  # pretend id for dry runs
            self.status["conceptid"] = -2  # pretend conceptid
        self.status["published"] = False
        self.status["version"] = 1

        print(f"   - {operation.value} (recid: {self.status['recid']})")

    def _add_file(self, dry_run: bool, token: str, file_data: tuple):
        """
            Add a file to a draft deposit

            :param bool dry_run: whether this ia just a dry run
            :param str token: token required to update Zenodo
            :param tuple file_data: (str: file name,
                                     str/file descriptor: content or pointer)

            :raises ZenodoError: if the API request fails

            :returns str: the Zenodo fileid
       """
        operation = ZenodoOp.ADD_FILE
        if dry_run is False:
            response = requests.post(**self._request(operation, token,
                                                     file_data=file_data))
            self._check_response(operation, response)
            fileid = response.json()["id"]
            recid = self.status["recid"]
        else:
            fileid = "not-applicable-for-dry-run"
            recid = -1

        loc_name = ("readme.md.j2" if file_data[0] == "readme.md"
                    else file_data[0])
        print(f"   - {operation.value} {file_data[0]} " +
              f"(recid: {recid}, fileid: {fileid}, "
              f"{self.status['files'][loc_name]['size']} bytes)")
        return fileid

    def _remove_file(self, dry_run: bool, token: str, file_data: tuple):
        """
            Remove a file from a draft deposit

            :param bool dry_run: whether this ia just a dry run
            :param str token: token required to update Zenodo
            :param tuple file_data: (str: file name,
                                     str: file id)

            :raises ZenodoError: if the API request fails
       """
        operation = ZenodoOp.REMOVE_FILE
        if dry_run is False:
            response = requests.delete(**self._request(operation, token,
                                                       file_data=file_data))
            self._check_response(operation, response)
            recid = self.status["recid"]
        else:
            recid = -1

        loc_name = ("readme.md" if file_data[0] == "readme.md.j2"
                    else file_data[0])
        print(f"   - {operation.value} {loc_name} " +
              f"(recid: {recid}, fileid: {file_data[1]})")

    def _update_deposit(self, dry_run: bool, token: str):
        """
            Update metadata of an existing draft deposit on Zenodo.

            :param bool dry_run: whether this ia just a dry run
            :param str token: Authentication token for Zenodo

            :raises ZenodoError: if the API request fails
        """
        operation = ZenodoOp.UPDATE
        if dry_run is False:
            response = requests.put(**self._request(operation, token))
            self._check_response(operation, response)

        print(f"   - {operation.value} (recid: {self.status['recid']})")

    def _request(self, operation: str, token: str, file_data: tuple = None):
        """
            Builds the Zenodo request - the URL, headers and content

            :param str operation: operation being performed
            :param str token: authentication token
            :param tuple file_data: (str: file name,
                                     str/file descriptor: content or pointer)

            :returns str: the complete Zenodo request URL
        """
        # construct request URL for specific operation
        base_url = ZENODO_URL if self.sandbox is False else SANDBOX_URL
        if operation == ZenodoOp.DOWNLOAD_FILE:
            # req_url = "/need-url-format-for-published-file"
            req_url = (f"/records/{self.status['recid']}/draft/files/" +
                       f"{file_data[0]}/content")
        else:
            req_url = "/deposit/depositions"
            if "recid" in self.status:
                req_url += f"/{self.status['recid']}"
        request = {"url": base_url + req_url}

        # supply authentication token
        headers = {"Authorization": f"Bearer {token}" if token else None}

        # request JSON response for create, update and get_metadata
        if operation in {ZenodoOp.GET_METADATA, ZenodoOp.CREATE,
                         ZenodoOp.UPDATE}:
            headers.update({"Content-Type": "application/json"})

        # supply metadata information with create and update operation
        if operation in {ZenodoOp.CREATE, ZenodoOp.UPDATE}:
            request["json"] = {
                "metadata": self.metadata
            }

        # supply file name and contents when adding a file
        if operation == ZenodoOp.ADD_FILE:
            request["url"] += "/files"
            request["files"] = {"file": file_data}

        # supply file id when removing a file
        if operation == ZenodoOp.REMOVE_FILE:
            request["url"] += f"/files/{file_data[1]}"

        # Add stream flag when downloading a file
        if operation == ZenodoOp.DOWNLOAD_FILE:
            request["stream"] = True

        request["headers"] = headers
        return request

    def _check_response(self, operation: str, response: Response):
        """
            Checks the response status code is as expected

            :param str operation: operation being performed
            :param Response response: response to Zenodo request

            :raises ZenodoError: if expected status code not received
        """
        EXPECTED_STATUS = {
            ZenodoOp.CREATE: 201,
            ZenodoOp.UPDATE: 200,
            ZenodoOp.GET_METADATA: 200,
            ZenodoOp.DELETE: 204,
            ZenodoOp.ADD_FILE: 201,
            ZenodoOp.REMOVE_FILE: 204
        }
        if response.status_code != EXPECTED_STATUS[operation]:
            error_message = (
                f"{operation.value}: {response.status_code} "
                f"{response.text}"
            )
            raise ZenodoError(error_message)

        return (response.json() if operation not in
                {ZenodoOp.DELETE, ZenodoOp.REMOVE_FILE} else None)

    def _read_status(self):
        """
            Read the appropriate status JSON file to check status of
            deposit on either live or sandbox Zenodo
        """
        path = (
            self.base + self.name +
            ("/sandbox" if self.sandbox else "/zenodo") + "_status.json"
        )
        try:
            with open(path, "r") as f:
                self.status = load(f)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Status not found in {self.name}: {e}")

        except JSONDecodeError as e:
            raise ValueError(f"Status not valid JSON: {self.name} ({e})")

        except Exception as e:
            raise ValueError(f"Error reading status of {self.name}: {e}")

    def _write_status(self):
        """
            Write the current status to the appropriate status JSON file.

            :raises ValueError: if unable to write the status
        """
        path = (self.base + self.name +
                ("/sandbox" if self.sandbox else "/zenodo") + "_status.json")

        try:
            with open(path, "w") as f:
                dump(self.status, f, indent=4)

        except Exception as e:
            raise ValueError(f"Error writing status of {self.name}: {e}")
