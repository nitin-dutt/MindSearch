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

Deleting a document is a separate decision, made once in AccessPolicy.can_delete() (alignment_plan SEC-12). The uploader can NOT
delete their own document (someone might do it in anger or on the way out of the company); deletion belongs to someone above them:

    allowed  =  user is a super role (administrator)
             OR ( mode is 'higher_levels'
                  AND the uploader is a known, non-administrator user
                  AND user.rank > the uploader's rank )

and, before any of that, the caller must be allowed to READ the document (the API looks it up with the access filter first).
The mode comes from the policy file's optional `delete:` section: `higher_levels` (the default) or `admins_only`.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import yaml

from settings import get_settings
from qdrant_client.models import (FieldCondition, Filter, IsEmptyCondition, MatchAny, MatchValue,
                                  PayloadField, Range)

DEFAULT_POLICY_PATH = get_settings().org_policy_path
log = logging.getLogger("mindsearch.security")

DELETE_MODES = ("higher_levels", "admins_only")
DEFAULT_DELETE_MODE = "higher_levels"


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class DeleteDecision:
    """The answer to 'may this user delete this document?', and WHICH clause decided it (written to the audit log)."""
    allowed: bool
    rule: str


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
    def __init__(self, organization, levels, groups=(), super_roles=(), delete_mode=DEFAULT_DELETE_MODE):
        if not levels:
            raise PolicyError("a policy needs at least one level")
        if delete_mode not in DELETE_MODES:
            raise PolicyError(f"delete mode must be one of {', '.join(DELETE_MODES)}, got '{delete_mode}'")
        self.delete_mode = delete_mode
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
        delete = data.get("delete") or {}
        if not isinstance(delete, dict) or set(delete) - {"mode"}:
            raise PolicyError("policy 'delete' must be a section with only a 'mode' key (higher_levels or admins_only)")
        if not data.get("super_roles"):
            # Nobody could ever delete a document that its uploader's level cannot reach, and nobody could fix a mistake.
            raise PolicyError("a policy needs at least one entry in 'super_roles' (an administrator role)")
        policy = cls(data.get("organization", "unnamed"), levels, data.get("groups", []), data.get("super_roles", []),
                     delete.get("mode", DEFAULT_DELETE_MODE))
        if len(policy.level_names) == 1 and policy.delete_mode == "higher_levels":
            log.warning("policy '%s' has a single level, so nobody is 'higher' than anybody: only administrators can delete documents",
                        policy.organization)
        return policy

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

    def can_delete(self, user: User, uploader: Optional[User]) -> DeleteDecision:
        """May `user` delete a document that `uploader` uploaded? `uploader` is None when that account no longer exists.
        The caller has already checked that `user` may read the document. Returns the decision and the clause behind it."""
        if user.is_super:
            return DeleteDecision(True, "administrator")
        if self.delete_mode == "admins_only":
            return DeleteDecision(False, "admins_only")
        if uploader is None:
            return DeleteDecision(False, "uploader_unknown")
        if uploader.is_super:
            return DeleteDecision(False, "uploader_is_administrator")
        if user.id == uploader.id:
            return DeleteDecision(False, "uploader_cannot_delete_own")
        if user.rank > uploader.rank:
            return DeleteDecision(True, "higher_level")
        return DeleteDecision(False, "not_higher_level")

    def delete_rule_text(self) -> str:
        """The delete rule in force, in plain words (shown on the 'My access' page)."""
        if self.delete_mode == "admins_only":
            return "Only an administrator can delete a document. Nobody else can, including the person who uploaded it."
        return ("A document can be deleted by an administrator, or by someone of a higher level than the person who uploaded it. "
                "The uploader cannot delete their own document, and people of the same or a lower level cannot delete each other's. "
                "A document uploaded by an administrator can only be deleted by an administrator.")

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
