import logging
import os
import re
import threading
import uuid
from collections import defaultdict

from neo4j import GraphDatabase

from access_policy import AccessPolicy, DocACL, User
from settings import get_settings
from kg_extractors import extract_chunks, make_extractor_from_env

log = logging.getLogger("mindsearch.kg")

GRAPH_POOL_LIMIT = 50  # raw facts handed to the reranker, which keeps the best few
DEFAULT_HUB_LIMIT = 12  # an entity mentioned in more passages than this ("Americans", "1990") is too generic to follow


def entity_key(name: str) -> str:
    """Lookup form of an entity name: lowercase, single spaces, no surrounding quotes, no leading article."""
    quotes = " 	\"'"
    key = re.sub(r"\s+", " ", name.casefold()).strip(quotes)
    return re.sub(r"^(the|a|an) ", "", key).strip(quotes)


def chunk_title_names(chunk: str) -> list:
    """Names a passage is 'the page of': its leading title when it has one ("Kool Kojak. Allan P. Grigg, better known as..." or
    "Fort Lee, New Jersey. Fort Lee is a borough..."). Returns the title without a trailing "(film)" and its part before a comma.
    Empty when the passage has no short leading title."""
    head, sep, _ = chunk.strip().partition(". ")
    head = re.sub(r"\s*\([^)]*\)", "", head).strip()
    if not sep or not 3 <= len(head) <= 80 or len(head.split()) > 10:
        return []
    names = [head]
    if "," in head:
        names.append(head.split(",")[0].strip())
    return [n for n in dict.fromkeys(names) if len(n) >= 3]


class KnowledgeGraphBuilder:
    def __init__(self, uri=None, user=None, password=None, extractor=None):
        cfg = get_settings()
        neo4j_uri = uri or cfg.neo4j_uri
        neo4j_user = user or cfg.neo4j_user
        neo4j_pass = password or os.environ.get("NEO4J_PASSWORD")
        if not neo4j_pass:
            raise RuntimeError("NEO4J_PASSWORD is not set. Put it in .env (see .env.example).")
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self._extractor = extractor
        self._nlp = None
        self._nlp_lock = threading.Lock()

    # Heavy dependencies are created on first use so constructing the builder stays cheap
    # (the API builds one at startup, and tests can inject a fake extractor).
    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = make_extractor_from_env()
        return self._extractor

    @extractor.setter
    def extractor(self, value):
        self._extractor = value

    @property
    def nlp(self):
        if self._nlp is None:
            with self._nlp_lock:               # concurrent first requests must not both load the model
                if self._nlp is None:
                    import spacy
                    name = get_settings().spacy_model
                    try:
                        self._nlp = spacy.load(name)
                    except OSError:
                        from spacy.cli import download
                        download(name)
                        self._nlp = spacy.load(name)
        return self._nlp

    def close(self):
        if self.driver is not None:
            self.driver.close()

    # ------------------------------------------------------------ ingestion

    def process_chunks(self, chunks, acl: DocACL, doc_id=None, workers=None):
        """Extract triples chunk by chunk and write them to Neo4j. Returns ExtractionStats."""
        if workers is None:
            workers = get_settings().kg_extract_workers or (8 if self.extractor.name.startswith("vllm") else 1)
        mentions = []
        triples, stats = extract_chunks(self.extractor, chunks, workers=workers, mentions_out=mentions)
        mentions = self._mention_rows(chunks, mentions)
        log.info("KG extraction finished: %s", stats.to_dict())
        if triples or mentions:
            # the id must be the SAME document id the vector index uses: it is how a graph hit finds its passage
            self.save_triples(triples, acl, doc_id or str(uuid.uuid4()), mentions=mentions)
        return stats

    def _mention_rows(self, chunks, extracted):
        """(entity name, chunk index, is_page) for every entity a passage mentions. Three sources: entities of the LLM-extracted
        facts, named entities / proper-noun runs found directly in the text (so a name is linked even when the LLM extractor
        skipped it), and the passage's own leading title (it is the "page" of that entity)."""
        rows = {}
        def add(name, index, page=False):
            key = (name, int(index))
            rows[key] = rows.get(key, False) or page
        for name, index in extracted:
            add(name, index)
        for index, doc in enumerate(self.nlp.pipe(chunks, batch_size=64)):
            for ent in doc.ents:
                add(ent.text, index)
            run = []
            for tok in list(doc) + [None]:
                if tok is not None and tok.pos_ == "PROPN":
                    run.append(tok.text)
                else:
                    if run:
                        add(" ".join(run), index)
                    run = []
            for title in chunk_title_names(chunks[index]):
                add(title, index, page=True)
        return [(name, index, page) for (name, index), page in rows.items() if len(entity_key(name)) > 2]

    def ensure_schema(self):
        with self.driver.session() as session:
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            session.run("CREATE INDEX entity_key IF NOT EXISTS FOR (n:Entity) ON (n.key)")

    def save_triples(self, triples, acl: DocACL, doc_id, mentions=None):
        """Write triples in a few batched statements (one transaction), not one query per fact.

        Every relationship carries its document's ACL (min_rank, groups, users) and doc_id. There is one
        relationship per source document, so two documents with different permissions never merge into
        one edge, and a fact is visible exactly when at least one of its source documents is.
        """
        self.ensure_schema()
        mentions = [(m[0], m[1], bool(m[2]) if len(m) > 2 else False) for m in (mentions or [])]
        names = sorted({t.subject for t in triples} | {t.object for t in triples} | {m[0] for m in mentions})
        by_relation = defaultdict(list)
        for t in triples:
            by_relation[t.relation].append({"s": t.subject, "o": t.object})
        acl_props = acl.to_payload()
        keys = {n: entity_key(n) for n in names}
        mention_rows = [{"name": n, "chunk": int(i), "page": p} for n, i, p in mentions if n in keys]

        def write(tx):
            tx.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.min_rank = $min_rank, d.groups = $groups, d.users = $users
                """, doc_id=doc_id, **acl_props)
            tx.run("""
                UNWIND $names AS name
                MERGE (n:Entity {name: name})
                SET n.key = $keys[name]
                WITH n
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:CONTAINS]->(n)
                """, names=names, keys=keys, doc_id=doc_id)
            if mention_rows:
                # Entity -[MENTIONED_IN {chunk_index}]-> Document: which passage of the document mentions the entity.
                # Carries the document's access rule like every other relationship.
                tx.run("""
                    UNWIND $rows AS row
                    MATCH (n:Entity {name: row.name})
                    MATCH (d:Document {id: $doc_id})
                    MERGE (n)-[m:MENTIONED_IN {doc_id: $doc_id, chunk_index: row.chunk}]->(d)
                    SET m.min_rank = $min_rank, m.groups = $groups, m.users = $users,
                        m.page = coalesce(m.page, false) OR row.page
                    """, rows=mention_rows, doc_id=doc_id, **acl_props)
            for rel_type, rows in by_relation.items():
                # rel_type was normalized to [A-Z0-9_]+ by clean_triples, so it is safe to inline; the backticks
                # are required because a type may start with a digit (e.g. 33_YARD_FIELD_GOAL)
                tx.run(f"""
                    UNWIND $rows AS row
                    MATCH (s:Entity {{name: row.s}})
                    MATCH (o:Entity {{name: row.o}})
                    MERGE (s)-[r:`{rel_type}` {{doc_id: $doc_id}}]->(o)
                    SET r.min_rank = $min_rank, r.groups = $groups, r.users = $users
                    """, rows=rows, doc_id=doc_id, **acl_props)

        with self.driver.session() as session:
            session.execute_write(write)

    def delete_document(self, doc_id) -> dict:
        """Remove everything one document contributed: its relationships, its Document node, and any entity that no
        other document still uses. Idempotent."""
        def work(tx):
            entities = [r["name"] for r in tx.run(
                "MATCH (:Document {id: $id})-[:CONTAINS]->(n:Entity) RETURN n.name AS name", id=doc_id)]
            edges = tx.run("MATCH (:Entity)-[r {doc_id: $id}]->(:Entity) DELETE r RETURN count(r) AS c", id=doc_id).single()["c"]
            tx.run("MATCH (d:Document {id: $id}) DETACH DELETE d", id=doc_id)
            orphans = tx.run(
                "UNWIND $names AS name MATCH (n:Entity {name: name}) WHERE NOT (n)--() DELETE n RETURN count(n) AS c",
                names=entities).single()["c"]
            return {"relationships": edges, "entities_removed": orphans}

        with self.driver.session() as session:
            return session.execute_write(work)

    # ------------------------------------------------------------ retrieval

    def query_entity_keys(self, query: str) -> list:
        """Entity lookup keys for the names the question mentions: named entities and runs of proper nouns, plus noun
        phrases. Matching is by EXACT key later (no substring), so 'Show' no longer finds 'The Dr. Fad Show'."""
        doc = self.nlp(query)
        names = {ent.text for ent in doc.ents}
        run = []
        for tok in doc:
            if tok.pos_ == "PROPN":
                run.append(tok.text)
            else:
                if run:
                    names.add(" ".join(run))
                run = []
        if run:
            names.add(" ".join(run))
        names |= {nc.text for nc in doc.noun_chunks}
        return sorted({k for k in (entity_key(n) for n in names) if len(k) > 2})

    def get_graph_context(self, query: str, user: User, seed_keys=None) -> list:
        """Facts within 1-2 hops of the entities the question mentions (plus `seed_keys`, e.g. entities of the passages
        already retrieved), as readable sentences. Entities are matched by exact key."""
        keys = sorted(set(self.query_entity_keys(query)) | set(seed_keys or []))
        if not keys:
            return []
        return self.expand_keys(keys, user)

    def _access(self, alias, user):
        if user.is_super:
            return "true", {}
        return AccessPolicy.neo4j_edge_predicate(alias), AccessPolicy.neo4j_params(user)

    def entities_in_chunks(self, refs, user: User, max_entities=60) -> list:
        """Entity keys mentioned by the given passages. refs = [(doc_id, chunk_index)]. Only mentions the user may see."""
        if not refs:
            return []
        ok, params = self._access("m", user)
        cypher = f"""
            UNWIND $refs AS ref
            MATCH (e:Entity)-[m:MENTIONED_IN {{doc_id: ref.doc_id, chunk_index: ref.chunk}}]->(:Document)
            WHERE {ok} AND e.key IS NOT NULL
            RETURN DISTINCT e.key AS key LIMIT $limit
            """
        with self.driver.session() as session:
            return [r["key"] for r in session.run(cypher, refs=[{"doc_id": d, "chunk": int(c)} for d, c in refs],
                                                  limit=max_entities, **params)]

    def entities_by_chunk(self, refs, user: User) -> dict:
        """{(doc_id, chunk_index): set of entity keys it mentions}, for the passages the user may see."""
        if not refs:
            return {}
        ok, params = self._access("m", user)
        cypher = f"""
            UNWIND $refs AS ref
            MATCH (e:Entity)-[m:MENTIONED_IN {{doc_id: ref.doc_id, chunk_index: ref.chunk}}]->(:Document)
            WHERE {ok} AND e.key IS NOT NULL
            RETURN ref.doc_id AS doc_id, ref.chunk AS chunk, collect(DISTINCT e.key) AS keys
            """
        out = {}
        with self.driver.session() as session:
            for r in session.run(cypher, refs=[{"doc_id": d, "chunk": int(c)} for d, c in refs], **params):
                out[(r["doc_id"], int(r["chunk"]))] = set(r["keys"])
        return out

    def candidate_chunks(self, keys, user: User, hub_limit=DEFAULT_HUB_LIMIT, page_limit=5) -> dict:
        """Passages that mention any of the entities: {(doc_id, chunk_index): {"keys": set, "pages": set}}. "pages" are the
        entities this passage is the page of (its title). An entity found in more than `hub_limit` passages is skipped as
        a mention (too generic to follow) but its page, if any, is still returned."""
        if not keys:
            return {}
        ok, params = self._access("m", user)
        cypher = f"""
            UNWIND $keys AS key
            MATCH (e:Entity {{key: key}})-[m:MENTIONED_IN]->(:Document)
            WHERE {ok}
            WITH key, collect([m.doc_id, m.chunk_index, coalesce(m.page, false)]) AS refs
            UNWIND refs AS ref
            WITH key, ref, size(refs) AS total
            WHERE ref[2] OR total <= $hub_limit
            RETURN key, ref[0] AS doc_id, ref[1] AS chunk, ref[2] AS page
            """
        out = {}
        pages_per_key = defaultdict(int)
        with self.driver.session() as session:
            for r in session.run(cypher, keys=list(keys), hub_limit=hub_limit, **params):
                ref = (r["doc_id"], int(r["chunk"]))
                entry = out.setdefault(ref, {"keys": set(), "pages": set()})
                entry["keys"].add(r["key"])
                if r["page"]:
                    if pages_per_key[r["key"]] >= page_limit:
                        continue
                    pages_per_key[r["key"]] += 1
                    entry["pages"].add(r["key"])
        return out

    def chunk_refs_for_entities(self, keys, user: User, hub_limit=DEFAULT_HUB_LIMIT) -> dict:
        """Passages that mention each entity: {(doc_id, chunk_index): [entity keys it mentions]}. Entities that appear in
        more than `hub_limit` passages are skipped: they connect everything to everything and would drown the signal."""
        if not keys:
            return {}
        ok, params = self._access("m", user)
        cypher = f"""
            UNWIND $keys AS key
            MATCH (e:Entity {{key: key}})-[m:MENTIONED_IN]->(:Document)
            WHERE {ok}
            WITH key, collect([m.doc_id, m.chunk_index]) AS refs
            WHERE size(refs) <= $hub_limit
            UNWIND refs AS ref
            RETURN key, ref[0] AS doc_id, ref[1] AS chunk
            """
        out = defaultdict(list)
        with self.driver.session() as session:
            for r in session.run(cypher, keys=list(keys), hub_limit=hub_limit, **params):
                out[(r["doc_id"], int(r["chunk"]))].append(r["key"])
        return dict(out)

    def expand_entities(self, entity_names, user: User, limit=GRAPH_POOL_LIMIT, start_limit=25) -> list:
        """Facts reachable within 2 hops of the named entities, that `user` is allowed to see.

        Every relationship on every path is returned as its own directed triple
        (startNode -[type]-> endNode). Document nodes and CONTAINS edges are excluded so only
        Entity-to-Entity facts come back. Access is checked on EVERY relationship of the path
        (not just the starting node), using the predicate generated by AccessPolicy, so a restricted
        fact cannot be reached by walking through permitted ones.
        """
        if user.is_super:
            start_ok, path_ok = "true", "true"
        else:
            pred = AccessPolicy.neo4j_edge_predicate("rel")
            # a start entity is only useful if the user may see at least one of its relationships
            start_ok = f"EXISTS {{ MATCH (n)-[rel]-(:Entity) WHERE {pred} }}"
            path_ok = f"ALL(rel IN relationships(p) WHERE {pred})"
        cypher = f"""
            UNWIND $entities AS entity_name
            MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower(entity_name) AND {start_ok}
            WITH DISTINCT n LIMIT $start_limit
            MATCH p = (n)-[*1..2]-(m:Entity)
            WHERE ALL(x IN nodes(p) WHERE x:Entity) AND {path_ok}
            UNWIND relationships(p) AS rel
            WITH DISTINCT rel
            RETURN startNode(rel).name AS source, type(rel) AS relation, endNode(rel).name AS target
            LIMIT $limit
            """
        facts = []
        with self.driver.session() as session:
            for rec in session.run(cypher, entities=list(entity_names), limit=limit, start_limit=start_limit,
                                   **AccessPolicy.neo4j_params(user)):
                facts.append(f"{rec['source']} {rec['relation'].replace('_', ' ').lower()} {rec['target']}")
        return list(dict.fromkeys(facts))  # de-duplicate, keep order

    def expand_keys(self, keys, user: User, limit=GRAPH_POOL_LIMIT, start_limit=40) -> list:
        """Like expand_entities, but starts from entities whose key EQUALS one of `keys` (exact, not substring)."""
        if user.is_super:
            start_ok, path_ok, params = "true", "true", {}
        else:
            pred = AccessPolicy.neo4j_edge_predicate("rel")
            start_ok = f"EXISTS {{ MATCH (n)-[rel]-(:Entity) WHERE {pred} }}"
            path_ok = f"ALL(rel IN relationships(p) WHERE {pred})"
            params = AccessPolicy.neo4j_params(user)
        cypher = f"""
            MATCH (n:Entity) WHERE n.key IN $keys AND {start_ok}
            WITH DISTINCT n LIMIT $start_limit
            MATCH p = (n)-[*1..2]-(m:Entity)
            WHERE ALL(x IN nodes(p) WHERE x:Entity) AND {path_ok}
            UNWIND relationships(p) AS rel
            WITH DISTINCT rel
            RETURN startNode(rel).name AS source, type(rel) AS relation, endNode(rel).name AS target
            LIMIT $limit
            """
        facts = []
        with self.driver.session() as session:
            for rec in session.run(cypher, keys=list(keys), limit=limit, start_limit=start_limit, **params):
                facts.append(f"{rec['source']} {rec['relation'].replace('_', ' ').lower()} {rec['target']}")
        return list(dict.fromkeys(facts))
