import logging
import os
import threading
import uuid
from collections import defaultdict

from neo4j import GraphDatabase

from access_policy import AccessPolicy, DocACL, User
from settings import get_settings
from kg_extractors import extract_chunks, make_extractor_from_env

log = logging.getLogger("mindsearch.kg")

GRAPH_POOL_LIMIT = 50  # raw facts handed to the reranker, which keeps the best few


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
        triples, stats = extract_chunks(self.extractor, chunks, workers=workers)
        log.info("KG extraction finished: %s", stats.to_dict())
        if triples:
            self.save_triples(triples, acl, doc_id or str(uuid.uuid4()))
        return stats

    def ensure_schema(self):
        with self.driver.session() as session:
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")

    def save_triples(self, triples, acl: DocACL, doc_id):
        """Write triples in a few batched statements (one transaction), not one query per fact.

        Every relationship carries its document's ACL (min_rank, groups, users) and doc_id. There is one
        relationship per source document, so two documents with different permissions never merge into
        one edge, and a fact is visible exactly when at least one of its source documents is.
        """
        self.ensure_schema()
        names = sorted({t.subject for t in triples} | {t.object for t in triples})
        by_relation = defaultdict(list)
        for t in triples:
            by_relation[t.relation].append({"s": t.subject, "o": t.object})
        acl_props = acl.to_payload()

        def write(tx):
            tx.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.min_rank = $min_rank, d.groups = $groups, d.users = $users
                """, doc_id=doc_id, **acl_props)
            tx.run("""
                UNWIND $names AS name
                MERGE (n:Entity {name: name})
                WITH n
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:CONTAINS]->(n)
                """, names=names, doc_id=doc_id)
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

    def get_graph_context(self, query: str, user: User) -> list:
        """Entities mentioned in the query -> facts within 1-2 hops, as readable sentences."""
        doc = self.nlp(query)
        query_entities = {ent.text for ent in doc.ents}
        query_entities |= {tok.text for tok in doc if tok.pos_ in ("NOUN", "PROPN")}
        if not query_entities:
            return []
        return self.expand_entities(sorted(query_entities), user)

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
