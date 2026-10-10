import glob
import itertools
import os

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from access_policy import AccessPolicy, DocACL, PolicyError, User, can_access  # noqa: E402

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_FILES = sorted(glob.glob(os.path.join(BACKEND, "org_policies", "*.yaml"))) + [os.path.join(BACKEND, "org_policy.yaml")]


def subsets(items, max_size):
    for n in range(0, max_size + 1):
        yield from itertools.combinations(items, n)


def user_grid(policy):
    for level in policy.level_names:
        for groups in subsets(policy.groups, len(policy.groups)):
            yield policy.make_user(f"{level}-{'+'.join(groups) or 'nogroup'}", level, groups)


def acl_grid(policy):
    for level in policy.level_names:
        for groups in subsets(policy.groups, 2):
            yield policy.make_acl(level, groups)
    # one document shared with a specific person regardless of level
    yield policy.make_acl(policy.level_names[-1], policy.groups[:1], users=["explicit-guest"])


# ------------------------------------------------------------ the rule itself

def test_rule_truth_table():
    mgr_hr = User("m", rank=2, groups=("HR",))
    emp = User("e", rank=1, groups=())
    assert can_access(mgr_hr, DocACL(min_rank=1))                       # higher rank reads lower
    assert not can_access(emp, DocACL(min_rank=2))                      # lower rank cannot read higher
    assert can_access(mgr_hr, DocACL(min_rank=2, groups=("HR",)))       # shared group
    assert not can_access(mgr_hr, DocACL(min_rank=2, groups=("Legal",)))  # right rank, wrong department
    assert can_access(emp, DocACL(min_rank=3, groups=("Legal",), users=("e",)))   # explicit grant
    assert can_access(User("root", 0, (), is_super=True), DocACL(min_rank=99, groups=("X",)))


def test_missing_acl_data_is_denied_by_default():
    assert not can_access(User("e", 1), DocACL(min_rank=2))


# -------------------------------------------------------- policy construction

def test_unknown_level_or_group_is_rejected():
    p = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policies", "departmental.yaml"))
    with pytest.raises(PolicyError):
        p.make_user("x", "Intern")
    with pytest.raises(PolicyError):
        p.make_acl("Staff", groups=["Marketing"])
    assert p.make_user("x", "staff", ["hr"]).groups == ("HR",)   # case-insensitive, normalized


def test_super_roles_come_from_the_policy_file():
    p = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policies", "matrix.yaml"))
    assert p.make_user("a", "Contributor", roles=["Auditor"]).is_super
    assert not p.make_user("b", "Contributor", roles=["Intern"]).is_super


def test_malformed_policy_is_rejected():
    with pytest.raises(PolicyError):
        AccessPolicy.from_dict({"levels": "oops"})
    with pytest.raises(PolicyError):
        AccessPolicy.from_dict({"levels": [{"name": "A", "rank": 1}, {"name": "a", "rank": 2}]})


# ------------------------- the database filter must agree with the rule, for every org

@pytest.mark.parametrize("policy_file", POLICY_FILES, ids=[os.path.basename(p) for p in POLICY_FILES])
def test_qdrant_filter_matches_the_rule_for_every_user_and_document(policy_file):
    policy = AccessPolicy.from_yaml(policy_file)
    acls = list(acl_grid(policy))
    client = QdrantClient(":memory:")
    client.create_collection("t", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    client.upsert("t", [PointStruct(id=i, vector=[1.0, 0.0], payload=acl.to_payload()) for i, acl in enumerate(acls)])

    users = list(user_grid(policy)) + [User("explicit-guest", rank=0, groups=())]
    checked = 0
    for user in users:
        expected = {i for i, acl in enumerate(acls) if can_access(user, acl)}
        hits, _ = client.scroll("t", scroll_filter=policy.qdrant_filter(user), limit=len(acls) + 5)
        assert {h.id for h in hits} == expected, f"{policy.organization}: filter disagrees with rule for {user}"
        checked += 1
    assert checked > 0


def test_super_user_gets_no_filter_and_sees_everything():
    p = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policy.yaml"))
    assert p.qdrant_filter(p.superuser()) is None


def test_points_without_acl_payload_are_invisible_to_normal_users():
    p = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policy.yaml"))
    client = QdrantClient(":memory:")
    client.create_collection("t", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    client.upsert("t", [PointStruct(id=1, vector=[1.0, 0.0], payload={"chunk_data": "legacy, no ACL fields"})])
    hits, _ = client.scroll("t", scroll_filter=p.qdrant_filter(p.make_user("e", "Employee")), limit=5)
    assert hits == []


def test_neo4j_predicate_uses_the_same_parameters_as_the_user():
    params = AccessPolicy.neo4j_params(User("u1", 2, ("HR", "Legal")))
    assert params == {"u_rank": 2, "u_groups": ["HR", "Legal"], "u_id": "u1"}
    assert "rel.min_rank <= $u_rank" in AccessPolicy.neo4j_edge_predicate("rel")
