"""Graph write/read behaviour against a real Neo4j. Skipped when Neo4j is unreachable.
Uses a unique ZZTEST_ name prefix and removes its own data; other graph data is untouched."""
import os
import uuid

import pytest

pytest.importorskip("neo4j")

from access_policy import AccessPolicy, DocACL, can_access  # noqa: E402
from kg_builder import KnowledgeGraphBuilder  # noqa: E402
from kg_extractors import Triple  # noqa: E402

PREFIX = "ZZTEST_"
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = AccessPolicy.from_yaml(os.path.join(BACKEND, "org_policies", "departmental.yaml"))
STAFF = POLICY.make_user("staff", "Staff")
EXEC = POLICY.make_user("exec", "Executive")
SUPER = POLICY.make_user("root", "Staff", roles=["Admin"])
EVERYONE = DocACL(min_rank=0)


@pytest.fixture
def kg():
    builder = KnowledgeGraphBuilder()
    try:
        builder.driver.verify_connectivity()
    except Exception:
        pytest.skip("Neo4j not reachable")
    yield builder
    with builder.driver.session() as s:
        s.run("MATCH (n:Entity) WHERE n.name STARTS WITH $p DETACH DELETE n", p=PREFIX)
        s.run("MATCH (d:Document) WHERE d.id STARTS WITH $p DETACH DELETE d", p=PREFIX)
    builder.close()


def T(s, r, o):
    return Triple(subject=PREFIX + s, relation=r, object=PREFIX + o)


def doc():
    return PREFIX + str(uuid.uuid4())


def facts(kg, names, user=SUPER, **kw):
    return set(kg.expand_entities([PREFIX + n for n in names], user, **kw))


# ------------------------------------------------------------ correctness (KG-02)

def test_two_hop_returns_each_hop_as_its_own_correct_directed_fact(kg):
    # A -USES-> B -BUILT_BY-> C
    kg.save_triples([T("A", "USES", "B"), T("B", "BUILT_BY", "C")], EVERYONE, doc())
    got = facts(kg, ["A"], STAFF)
    assert f"{PREFIX}A uses {PREFIX}B" in got
    assert f"{PREFIX}B built by {PREFIX}C" in got
    # the old query paired the FIRST relation with the FAR node, inventing this false fact:
    assert f"{PREFIX}A uses {PREFIX}C" not in got
    assert not any(f.startswith(f"{PREFIX}A built by") for f in got)


def test_direction_is_preserved_when_starting_from_the_object(kg):
    kg.save_triples([T("Book", "WRITTEN_BY", "Author")], EVERYONE, doc())
    got = facts(kg, ["Author"], STAFF)
    assert f"{PREFIX}Book written by {PREFIX}Author" in got
    assert f"{PREFIX}Author written by {PREFIX}Book" not in got


def test_document_nodes_never_leak_into_facts(kg):
    kg.save_triples([T("X", "USES", "Y")], EVERYONE, doc())
    got = facts(kg, ["X"], STAFF)
    assert got and not any("contains" in f.lower() or "None" in f for f in got)


def test_saving_the_same_document_twice_does_not_duplicate_relationships(kg):
    d = doc()
    kg.save_triples([T("P", "USES", "Q")], EVERYONE, d)
    kg.save_triples([T("P", "USES", "Q")], EVERYONE, d)
    with kg.driver.session() as s:
        n = s.run("MATCH (:Entity {name:$a})-[r:USES]->(:Entity {name:$b}) RETURN count(r) AS c",
                  a=PREFIX + "P", b=PREFIX + "Q").single()["c"]
    assert n == 1


# ------------------------------------------------- access control (SEC-02, SEC-03)

def test_restricted_hop_is_not_reachable_through_permitted_hops(kg):
    """The leak the old design had: walk from a public fact into an executive-only fact."""
    kg.save_triples([T("Pub", "USES", "Shared")], EVERYONE, doc())                          # visible to all
    kg.save_triples([T("Shared", "ACQUIRED", "SecretCo")], POLICY.make_acl("Executive"), doc())  # exec only
    staff_view = facts(kg, ["Pub"], STAFF)
    assert f"{PREFIX}Pub uses {PREFIX}Shared" in staff_view
    assert f"{PREFIX}Shared acquired {PREFIX}SecretCo" not in staff_view        # hop 2 filtered
    assert f"{PREFIX}Shared acquired {PREFIX}SecretCo" in facts(kg, ["Pub"], EXEC)  # allowed for exec


def test_entity_that_only_exists_in_restricted_documents_gives_nothing(kg):
    kg.save_triples([T("Sec", "USES", "Secret")], POLICY.make_acl("Executive"), doc())
    assert facts(kg, ["Sec"], STAFF) == set()
    assert facts(kg, ["Secret"], STAFF) == set()
    assert facts(kg, ["Sec"], EXEC) != set()


def test_department_walls_apply_to_graph_facts(kg):
    kg.save_triples([T("Payroll", "OWNED_BY", "Finance")], POLICY.make_acl("Staff", groups=["HR"]), doc())
    hr = POLICY.make_user("h", "Staff", ["HR"])
    legal_exec = POLICY.make_user("l", "Executive", ["Legal"])   # senior but wrong department
    assert facts(kg, ["Payroll"], hr) != set()
    assert facts(kg, ["Payroll"], legal_exec) == set()


def test_same_fact_in_two_documents_is_visible_if_either_is_readable(kg):
    kg.save_triples([T("M", "USES", "N")], POLICY.make_acl("Executive"), doc())
    kg.save_triples([T("M", "USES", "N")], EVERYONE, doc())
    assert f"{PREFIX}M uses {PREFIX}N" in facts(kg, ["M"], STAFF)


def test_super_role_sees_everything(kg):
    kg.save_triples([T("Z1", "USES", "Z2")], POLICY.make_acl("Executive", groups=["Legal"]), doc())
    assert facts(kg, ["Z1"], SUPER) != set()


def test_graph_filter_matches_the_rule_for_every_user_and_acl(kg):
    """Same exhaustive grid as the Qdrant test: the Cypher predicate must agree with can_access()."""
    from tests.test_access_policy import acl_grid, user_grid  # noqa: F401  (grid builders)
    acls = list(acl_grid(POLICY))
    for i, acl in enumerate(acls):
        kg.save_triples([T(f"E{i:03d}a", "REL", f"E{i:03d}b")], acl, doc())
    users = list(user_grid(POLICY))
    for user in users:
        expected = {f"{PREFIX}E{i:03d}a rel {PREFIX}E{i:03d}b" for i, a in enumerate(acls) if can_access(user, a)}
        got = {f for f in facts(kg, ["E"], user, limit=1000, start_limit=1000) if " rel " in f}
        assert got == expected, f"graph disagrees with the rule for {user}"


def test_relation_type_starting_with_a_digit_is_stored(kg):
    """Found by loading real data: Neo4j rejects unquoted types like 33_YARD_FIELD_GOAL."""
    kg.save_triples([T("Kicker", "33_YARD_FIELD_GOAL", "Super Bowl")], EVERYONE, doc())
    assert f"{PREFIX}Kicker 33 yard field goal {PREFIX}Super Bowl" in facts(kg, ["Kicker"], STAFF)


def test_retrying_the_same_document_does_not_duplicate_graph_facts(kg):
    d = doc()
    kg.save_triples([T("R1", "USES", "R2"), T("R2", "USES", "R3")], EVERYONE, d)
    kg.save_triples([T("R1", "USES", "R2"), T("R2", "USES", "R3")], EVERYONE, d)    # task retry
    with kg.driver.session() as s:
        n = s.run("MATCH (:Entity)-[r:USES {doc_id:$d}]->(:Entity) RETURN count(r) AS c", d=d).single()["c"]
    assert n == 2


# ------------------------------------------------ deleting a document's graph facts (INF-04)

def test_deleting_a_document_removes_only_its_own_facts_and_orphan_entities(kg):
    d1, d2 = doc(), doc()
    kg.save_triples([T("Shared", "USES", "OnlyInOne"), T("Shared", "USES", "AlsoInBoth")], EVERYONE, d1)
    kg.save_triples([T("Shared", "USES", "AlsoInBoth"), T("Shared", "USES", "OnlyInTwo")], EVERYONE, d2)

    report = kg.delete_document(d1)
    assert report["relationships"] == 2

    after = facts(kg, ["Shared"], SUPER)
    assert f"{PREFIX}Shared uses {PREFIX}OnlyInOne" not in after                 # gone with its document
    assert f"{PREFIX}Shared uses {PREFIX}AlsoInBoth" in after                    # still supported by document 2
    assert f"{PREFIX}Shared uses {PREFIX}OnlyInTwo" in after
    with kg.driver.session() as s:
        left = s.run("MATCH (n:Entity {name:$n}) RETURN count(n) AS c", n=PREFIX + "OnlyInOne").single()["c"]
        shared = s.run("MATCH (n:Entity {name:$n}) RETURN count(n) AS c", n=PREFIX + "Shared").single()["c"]
        node = s.run("MATCH (d:Document {id:$i}) RETURN count(d) AS c", i=d1).single()["c"]
    assert left == 0 and shared == 1 and node == 0                               # orphan removed, shared kept


def test_deleting_a_document_twice_is_harmless(kg):
    d = doc()
    kg.save_triples([T("Q1", "USES", "Q2")], EVERYONE, d)
    kg.delete_document(d)
    again = kg.delete_document(d)
    assert again == {"relationships": 0, "entities_removed": 0}
