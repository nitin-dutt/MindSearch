import os

import pytest

import manage_users as MU
from access_policy import AccessPolicy, PolicyError
from auth import UserStore, verify_password

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policies", "departmental.yaml"))


def test_add_stores_a_hash_not_the_password_and_can_log_in(tmp_path):
    path = str(tmp_path / "users.yaml")
    data = MU.add_user({"users": []}, POLICY, "alice", "a-long-password", "Manager", ["HR"], ["Admin"])
    MU.save(path, data)
    raw = open(path, encoding="utf-8").read()
    assert "a-long-password" not in raw and "scrypt$" in raw
    store = UserStore.from_yaml(path)
    assert store.authenticate("alice", "a-long-password").level == "Manager"
    assert store.authenticate("alice", "wrong-password!") is None


def test_adding_the_same_username_replaces_it():
    data = MU.add_user({"users": []}, POLICY, "bob", "first-password!", "Staff")
    data = MU.add_user(data, POLICY, "bob", "second-password!", "Manager")
    assert [u["level"] for u in data["users"]] == ["Manager"]
    assert verify_password("second-password!", data["users"][0]["password_hash"])


@pytest.mark.parametrize("kwargs", [dict(level="Intern"), dict(level="Staff", groups=["Marketing"]),
                                    dict(level="Staff", password="short")])
def test_invalid_users_are_rejected(kwargs):
    args = dict(password="a-long-password", level="Staff", groups=[])
    args.update(kwargs)
    with pytest.raises(PolicyError):
        MU.add_user({"users": []}, POLICY, "x", **args)


def test_remove_user():
    data = MU.add_user({"users": []}, POLICY, "carol", "a-long-password", "Staff")
    assert MU.remove_user(data, "carol") and data["users"] == []
    assert not MU.remove_user(data, "carol")
