"""The delete rule (alignment_plan SEC-12): a table of every role x uploader x mode, loading and validation of the `delete:` section,
and the start-up checks. No server and no models: this is the rule alone (the API tests cover it over HTTP)."""
import itertools
import logging
import os

import pytest

from access_policy import DELETE_MODES, AccessPolicy, DeleteDecision, PolicyError

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = [{"name": "Staff", "rank": 1}, {"name": "Manager", "rank": 2}, {"name": "Executive", "rank": 3}]


def make(mode=None, levels=LEVELS, super_roles=("Admin",)):
    data = {"organization": "T", "levels": levels, "groups": ["HR"], "super_roles": list(super_roles)}
    if mode:
        data["delete"] = {"mode": mode}
    return AccessPolicy.from_dict(data)


# who: (level, is_administrator). Administrators are given the LOWEST level on purpose: their power must not come from their level.
PEOPLE = {"staff": ("Staff", False), "manager": ("Manager", False), "exec": ("Executive", False), "admin": ("Staff", True)}


def person(policy, key):
    level, admin = PEOPLE[key]
    return policy.make_user(key, level, roles=["Admin"] if admin else [])


# (caller, uploader) -> (allowed, clause) in the default mode. `None` as uploader = the account no longer exists.
HIGHER_LEVELS = {
    ("staff", "staff"): (False, "uploader_cannot_delete_own"),
    ("manager", "manager"): (False, "uploader_cannot_delete_own"),
    ("exec", "exec"): (False, "uploader_cannot_delete_own"),
    ("manager", "staff"): (True, "higher_level"),
    ("exec", "staff"): (True, "higher_level"),
    ("exec", "manager"): (True, "higher_level"),
    ("staff", "manager"): (False, "not_higher_level"),
    ("staff", "exec"): (False, "not_higher_level"),
    ("manager", "exec"): (False, "not_higher_level"),
    ("staff", "admin"): (False, "uploader_is_administrator"),
    ("manager", "admin"): (False, "uploader_is_administrator"),
    ("exec", "admin"): (False, "uploader_is_administrator"),
    ("admin", "staff"): (True, "administrator"),
    ("admin", "manager"): (True, "administrator"),
    ("admin", "exec"): (True, "administrator"),
    ("admin", "admin"): (True, "administrator"),
    ("staff", None): (False, "uploader_unknown"),
    ("manager", None): (False, "uploader_unknown"),
    ("exec", None): (False, "uploader_unknown"),
    ("admin", None): (True, "administrator"),
}


def test_the_table_covers_every_caller_and_uploader():
    assert set(HIGHER_LEVELS) == set(itertools.product(PEOPLE, [*PEOPLE, None]))


@pytest.mark.parametrize("caller,uploader", sorted(HIGHER_LEVELS, key=str))
def test_default_mode_every_combination(caller, uploader):
    policy = make()
    assert policy.delete_mode == "higher_levels"                                       # a file without a `delete:` section behaves like this
    decision = policy.can_delete(person(policy, caller), person(policy, uploader) if uploader else None)
    assert decision == DeleteDecision(*HIGHER_LEVELS[(caller, uploader)])


@pytest.mark.parametrize("caller,uploader", sorted(HIGHER_LEVELS, key=str))
def test_admins_only_mode_every_combination(caller, uploader):
    policy = make("admins_only")
    decision = policy.can_delete(person(policy, caller), person(policy, uploader) if uploader else None)
    assert decision == (DeleteDecision(True, "administrator") if caller == "admin" else DeleteDecision(False, "admins_only"))


def test_nobody_but_an_administrator_can_delete_their_own_upload_in_any_mode():
    for mode in DELETE_MODES:
        policy = make(mode)
        for key in PEOPLE:
            me = person(policy, key)
            assert policy.can_delete(me, me).allowed is (key == "admin")


def test_a_higher_level_is_strictly_higher_equal_ranks_cannot_delete_each_others():
    policy = make(levels=[{"name": "A", "rank": 1}, {"name": "B", "rank": 1}, {"name": "C", "rank": 2}])
    a, b, c = (policy.make_user(n, n) for n in "ABC")
    assert policy.can_delete(a, b) == DeleteDecision(False, "not_higher_level")        # different names, same rank: still equals
    assert policy.can_delete(c, a) == DeleteDecision(True, "higher_level")


def test_a_second_administrator_role_also_counts_as_administrator_for_both_roles():
    policy = make(super_roles=("Admin", "Auditor"))
    auditor = policy.make_user("aud", "Staff", roles=["Auditor"])
    admin = policy.make_user("adm", "Staff", roles=["Admin"])
    assert policy.can_delete(auditor, admin) == DeleteDecision(True, "administrator")


# ------------------------------------------------------------------ loading and validation

def test_the_mode_is_read_from_the_delete_section():
    assert make("admins_only").delete_mode == "admins_only" and make("higher_levels").delete_mode == "higher_levels"


@pytest.mark.parametrize("delete", [{"mode": "everyone"}, {"mode": 1}, {"mode": ""}, {"mode": "higher_levels", "retention_days": 30},
                                    {"levels_may_delete_lower": True}, "admins_only", ["higher_levels"]])
def test_an_unknown_mode_or_key_is_refused_at_start_up_not_ignored(delete):
    with pytest.raises(PolicyError):
        AccessPolicy.from_dict({"levels": LEVELS, "super_roles": ["Admin"], "delete": delete})


@pytest.mark.parametrize("super_roles", [None, []])
def test_a_policy_without_an_administrator_role_is_refused(super_roles):
    data = {"levels": LEVELS}
    if super_roles is not None:
        data["super_roles"] = super_roles
    with pytest.raises(PolicyError, match="super_roles"):
        AccessPolicy.from_dict(data)


def test_a_single_level_policy_in_the_default_mode_warns_that_only_administrators_can_delete(caplog):
    with caplog.at_level(logging.WARNING, logger="mindsearch.security"):
        make(levels=[{"name": "Member", "rank": 0}])
        make("admins_only", levels=[{"name": "Member", "rank": 0}])
        make()
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1 and "only administrators can delete" in warnings[0]       # only the default mode on one level is surprising


@pytest.mark.parametrize("name", sorted(f for f in os.listdir(os.path.join(BACKEND, "org_policies")) if f.endswith(".yaml")))
def test_every_shipped_policy_file_still_loads_with_the_default_delete_rule(name):
    assert AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policies", name)).delete_mode == "higher_levels"


def test_the_rule_is_explained_in_plain_words_for_each_mode():
    assert "cannot delete their own document" in make().delete_rule_text()
    assert "Only an administrator" in make("admins_only").delete_rule_text()
