"""Policy-driven access control (alignment_plan SEC-02, SEC-03).

Each organization describes its own structure in a policy file (see org_policies/*.yaml):
  levels       an ordered clearance ladder with org-specific names (may be a single level)
  groups       departments / teams / projects
  super_roles  roles that bypass every check (their access should be audited)

A document carries an ACL: a minimum rank, optional groups, optional explicit users.
THE rule, implemented once in can_access() and compiled into both database filters:

    allowed  =  user is a super role
             OR user.id is listed in acl.users
             OR ( user.rank >= acl.min_rank
                  AND ( acl.groups is empty OR user shares a group with acl.groups ) )

Qdrant (vector search) and Neo4j (graph traversal) filters are generated from this one place, so the
two stores cannot disagree. Missing ACL data is denied (secure default).
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import yaml
from qdrant_client.models import (FieldCondition, Filter, IsEmptyCondition, MatchAny, MatchValue,
                                  PayloadField, Range)

DEFAULT_POLICY_PATH = os.getenv(
    "ORG_POLICY_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "org_policy.yaml"))


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class User:
    id: str
    rank: int
    groups: Tuple[str, ...] = ()
    is_super: bool = False


@dataclass(frozen=True)
class DocACL:
    min_rank: int = 0
    groups: Tuple[str, ...] = ()
    users: Tuple[str, ...] = ()

    def to_payload(self):
        """Stored on every Qdrant point and every Neo4j relationship."""
        return {"min_rank": self.min_rank, "groups": list(self.groups), "users": list(self.users)}


def can_access(user: User, acl: DocACL) -> bool:
    """The access rule in plain Python. Both database filters must agree with this function."""
    if user.is_super or user.id in acl.users:
        return True
    if user.rank < acl.min_rank:
        return False
    return not acl.groups or bool(set(acl.groups) & set(user.groups))


class AccessPolicy:
    def __init__(self, organization, levels, groups=(), super_roles=()):
        if not levels:
            raise PolicyError("a policy needs at least one level")
        self.organization = organization
        self._levels = {str(n).casefold(): (str(n), int(r)) for n, r in levels}
        if len(self._levels) != len(levels):
            raise PolicyError("duplicate level names")
        self.groups = tuple(groups)
        self.super_roles = tuple(str(r).casefold() for r in super_roles)

    # ------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data):
        try:
            levels = [(lv["name"], lv["rank"]) for lv in data["levels"]]
        except (KeyError, TypeError) as e:
            raise PolicyError(f"policy 'levels' must be a list of {{name, rank}}: {e}") from e
        return cls(data.get("organization", "unnamed"), levels, data.get("groups", []), data.get("super_roles", []))

    @classmethod
    def from_yaml(cls, path=None):
        with open(path or DEFAULT_POLICY_PATH, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    # ------------------------------------------------------- building users / ACLs
    @property
    def level_names(self):
        return [name for name, _ in sorted(self._levels.values(), key=lambda x: x[1])]

    @property
    def max_rank(self):
        return max(rank for _, rank in self._levels.values())

    def level_name(self, rank: int) -> str:
        """Name of the highest level whose rank does not exceed `rank` (used for default document levels)."""
        eligible = [(r, n) for n, r in self._levels.values() if r <= rank]
        return max(eligible)[1] if eligible else self.level_names[0]

    def rank_of(self, level) -> int:
        try:
            return self._levels[str(level).casefold()][1]
        except KeyError:
            raise PolicyError(f"unknown level '{level}'. Known levels: {', '.join(self.level_names)}") from None

    def _check_groups(self, groups):
        known = {g.casefold(): g for g in self.groups}
        out = []
        for g in groups or ():
            if str(g).casefold() not in known:
                raise PolicyError(f"unknown group '{g}'. Known groups: {', '.join(self.groups) or '(none)'}")
            out.append(known[str(g).casefold()])
        return tuple(dict.fromkeys(out))

    def make_user(self, user_id, level, groups=(), roles=()) -> User:
        is_super = any(str(r).casefold() in self.super_roles for r in roles or ())
        return User(id=str(user_id), rank=self.rank_of(level), groups=self._check_groups(groups), is_super=is_super)

    def make_acl(self, level, groups=(), users=()) -> DocACL:
        return DocACL(min_rank=self.rank_of(level), groups=self._check_groups(groups), users=tuple(users or ()))

    def superuser(self) -> User:
        """Internal identity for jobs that must see everything (evaluation, migrations)."""
        return User(id="system", rank=self.max_rank, groups=(), is_super=True)

    # --------------------------------------------------------- database filters
    def qdrant_filter(self, user: User) -> Optional[Filter]:
        """Qdrant payload filter equivalent to can_access(user, <point's ACL>). None = no restriction."""
        if user.is_super:
            return None
        group_ok = [IsEmptyCondition(is_empty=PayloadField(key="groups"))]
        if user.groups:
            group_ok.append(FieldCondition(key="groups", match=MatchAny(any=list(user.groups))))
        by_rank_and_group = Filter(must=[
            FieldCondition(key="min_rank", range=Range(lte=user.rank)),
            Filter(should=group_ok),
        ])
        explicit = FieldCondition(key="users", match=MatchValue(value=user.id))
        return Filter(should=[by_rank_and_group, explicit])

    @staticmethod
    def neo4j_edge_predicate(alias="rel") -> str:
        """Cypher boolean for ONE relationship. Use ALL(rel IN relationships(p) WHERE <this>) so that
        every hop of a path is checked, not only the starting node. Pair with neo4j_params()."""
        a = alias
        return (f"(({a}.min_rank <= $u_rank AND "
                f"(size({a}.groups) = 0 OR any(g IN {a}.groups WHERE g IN $u_groups))) "
                f"OR $u_id IN {a}.users)")

    @staticmethod
    def neo4j_params(user: User) -> dict:
        return {"u_rank": user.rank, "u_groups": list(user.groups), "u_id": user.id}


_default_policy = None


def get_default_policy() -> AccessPolicy:
    global _default_policy
    if _default_policy is None:
        _default_policy = AccessPolicy.from_yaml()
    return _default_policy
