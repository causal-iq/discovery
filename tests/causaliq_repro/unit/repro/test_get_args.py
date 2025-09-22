
# Unit Tests of CauaslIQ Pipeline repro.py command line processing

import sys
import pytest

from causaliq_repro.repro import get_args


# --- Invalid arguments

# Invalid operation raises SystemExit
def test_invalid_operation(monkeypatch):
    test_args = ["repro.py", "invalidop", "papers/ijar_stable/fig1"]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit):
        get_args()


# Missing target raises SystemExit
def test_missing_target(monkeypatch):
    test_args = ["repro.py", "learn"]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit):
        get_args()


# Unprivileged user help raises SystemExit
def test_user_help_1(monkeypatch):
    test_args = ["repro.py", "-h"]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr("causaliq_repro.repro.get_zenodo_token",
                        lambda *a, **kw: None)
    with pytest.raises(SystemExit):
        get_args()


# Unprivileged user help raises SystemExit
def test_user_help_2(monkeypatch):
    test_args = ["repro.py", "--help"]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr("causaliq_repro.repro.get_zenodo_token",
                        lambda *a, **kw: None)
    with pytest.raises(SystemExit):
        get_args()


# --- Unprivileged user, successful commands

# Unprivileged user defaults to "live" and dry run
def test_user_default(monkeypatch):
    test_args = ["repro.py", "analyse", "papers/ijar_stable/fig1"]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr("causaliq_repro.repro.get_zenodo_token",
                        lambda *a, **kw: None)
    args = get_args()
    assert args.operation == "analyse"
    assert args.zenodo == "live"
    assert args.target == "papers/ijar_stable/fig1"
    assert args.run is False
    assert args.token is None


# Unprivileged user doing download with actual run
def test_user_run(monkeypatch):
    test_args = ["repro.py", "download", "papers/ijar_stable/fig1", "--run"]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr("causaliq_repro.repro.get_zenodo_token",
                        lambda *a, **kw: None)
    args = get_args()
    assert args.operation == "download"
    assert args.zenodo == "live"
    assert args.target == "papers/ijar_stable/fig1"
    assert args.run is True
    assert args.token is None


# Unprivileged user doing learn and dry run
def test_user_learn(monkeypatch):
    test_args = ["repro.py", "learn", "papers/ijar_stable/fig1"]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr("causaliq_repro.repro.get_zenodo_token",
                        lambda *a, **kw: None)
    args = get_args()
    assert args.operation == "learn"
    assert args.zenodo == "live"
    assert args.target == "papers/ijar_stable/fig1"
    assert args.run is False
    assert args.token is None


# --- Unprivileged user tries forbidden action

# Unprivileged user tries upload
def test_user_upload_forbidden(monkeypatch):
    test_args = ["repro.py", "upload", "papers/ijar_stable/fig1"]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr("causaliq_repro.repro.get_zenodo_token",
                        lambda *a, **kw: None)
    with pytest.raises(SystemExit):
        get_args()


# Unprivileged user specifies Zenodo keyword
def test_user_zenodo_forbidden(monkeypatch):
    test_args = ["repro.py", "learn", "papers/ijar_stable/fig1",
                 "--zenodo", "live"]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr("causaliq_repro.repro.get_zenodo_token",
                        lambda *a, **kw: None)
    with pytest.raises(SystemExit):
        get_args()


# --- Privileged user, successful commands ---

# Admin specifies upload (defaults to sandbox)
def test_admin_sandbox_upload(monkeypatch):
    test_args = ["repro.py", "upload", "papers/ijar_stable/fig1"]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr("causaliq_repro.repro.get_zenodo_token",
                        lambda *a, **kw: "dummy token")
    args = get_args()
    assert args.operation == "upload"
    assert args.target == "papers/ijar_stable/fig1"
    assert args.run is False
    assert args.zenodo == "sandbox"
    assert args.token == "dummy token"


# Admin publishes to live
def test_admin_live_publish(monkeypatch):
    test_args = ["repro.py", "publish", "papers/ijar_stable/fig1",
                 "--zenodo", "live"]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr("causaliq_repro.repro.get_zenodo_token",
                        lambda *a, **kw: "dummy token")
    args = get_args()
    assert args.operation == "publish"
    assert args.target == "papers/ijar_stable/fig1"
    assert args.run is False
    assert args.zenodo == "live"
    assert args.token == "dummy token"


# Admin deletes from sandbox
def test_admin_sandbox_delete(monkeypatch):
    test_args = ["repro.py", "delete", "papers/ijar_stable/fig1",
                 "--zenodo", "sandbox"]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr("causaliq_repro.repro.get_zenodo_token",
                        lambda *a, **kw: "dummy token")
    args = get_args()
    assert args.operation == "delete"
    assert args.target == "papers/ijar_stable/fig1"
    assert args.run is False
    assert args.zenodo == "sandbox"
    assert args.token == "dummy token"
