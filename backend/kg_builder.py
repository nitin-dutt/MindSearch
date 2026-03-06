import spacy
from neo4j import GraphDatabase
import ollama
import json
import re

class KnowledgeGraphBuilder:
    def __init__(self, uri="neo4j://127.0.0.1:7687", user="neo4j", password="24112003"):
        """
        Initializes the Neo4j connection and loads the SpaCy NLP model.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Ensure the English model is available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spacy model 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def process_text(self, text: str):
        """
        Extracts entities and basic relations from text using an LLM (Ollama - llama3:8b)
        and saves to Neo4j.
        """
        prompt = f"""
        You are an expert Data Engineer building a Knowledge Graph.
        Extract entities and their relationships from the text below.
        
        Analyze the text and return the relationships as a JSON list of dictionaries.
        Each dictionary must have exactly three keys: 'subject', 'relation', 'object'.
        - 'subject' and 'object' should be the concise string names of entities (like a Product, Company, Concept, Person).
        - 'relation' should be a single uppercase word describing how they connect (e.g., USES, DEVELOPED_BY, ENHANCES).
        
        ONLY output valid JSON without markdown formatting. Do not output anything else.
        Example output:
        [
          {{"subject": "FastAPI", "relation": "BUILT_WITH", "object": "Python"}},
          {{"subject": "MindSearch", "relation": "USES", "object": "FAISS"}}
        ]
        
        Text to analyze:
        {text}
        """
        
        try:
            # We use the synchronous ollama client since this happens during chunk ingestion
            response = ollama.generate(model="llama3:8b", prompt=prompt)
            output = response.get("response", "")
            
            # Use regex to find the json array in case the LLM returned markdown backticks
            match = re.search(r'\[.*\]', output, re.DOTALL)
            if match:
                json_str = match.group(0)
                relations_data = json.loads(json_str)
            else:
                relations_data = json.loads(output)
                
            entities = set()
            relations = []
            
            for item in relations_data:
                sub = item.get("subject")
                rel = item.get("relation")
                obj = item.get("object")
                
                if sub and rel and obj:
                    entities.add(sub)
                    entities.add(obj)
                    relations.append((sub, rel, obj))
                    
             # We no longer have spacy NER labels, so we'll assign "Entity" to all of them
            entity_list = [(e, "Entity") for e in entities]
            
            # Save to Neo4j
            self.save_to_neo4j(entity_list, relations)
            return entity_list, relations
            
        except Exception as e:
            print(f"Error during LLM extraction: {e}")
            return [], []

    def get_graph_context(self, query: str) -> list:
        """
        Extracts entities from the user query and retrieves 1-2 hop context from Neo4j.
        """
        doc = self.nlp(query)
        # Extract named entities and salient nouns/proper nouns
        query_entities = set()
        for ent in doc.ents:
            query_entities.add(ent.text)
            
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                query_entities.add(token.text)
                
        if not query_entities:
            return []
            
        context_triples = []
        with self.driver.session() as session:
            # Cypher query to find nodes matching extracted entities and their 1-2 hop relationships
            cypher = """
            UNWIND $entities AS entity_name
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower(entity_name)
            MATCH (n)-[r*1..2]-(m)
            WITH n, r, m
            LIMIT 15
            RETURN n.name AS source, type(r[0]) AS relation, m.name AS target
            """
            
            result = session.run(cypher, entities=list(query_entities))
            
            for record in result:
                source = record["source"]
                relation = record["relation"]
                target = record["target"]
                context_triples.append(f"{source} {relation.replace('_', ' ').lower()} {target}")
                
        # Deduplicate and format
        unique_context = list(set(context_triples))
        return unique_context

    def save_to_neo4j(self, entities, relations):
        """
        Merges the extracted entities and relations into the Neo4j graph.
        """
        with self.driver.session() as session:
            # 1. Create Entity Nodes from SpaCy NER
            for ent_text, ent_label in entities:
                # Clean up labels to be valid Neo4j types (no spaces or special chars)
                label = "".join(e for e in ent_label if e.isalnum())
                if not label: label = "Concept"
                
                # Merge on the base 'Entity' label, then add the specific label
                query = f"""
                MERGE (n:Entity {{name: $name}})
                SET n:{label}
                """
                session.run(query, name=ent_text)

            # 2. Create Relations (Subject-Verb-Object)
            for sub, rel, obj in relations:
                # Format relationship type to Neo4j standard (UPPERCASE_WITH_UNDERSCORES)
                rel_type = "".join(e for e in rel.upper() if e.isalnum() or e == '_')
                if not rel_type: rel_type = "RELATED_TO"
                
                # Merge on the base 'Entity' label to link the graph together
                query = f"""
                MERGE (s:Entity {{name: $sub}})
                MERGE (o:Entity {{name: $obj}})
                MERGE (s)-[:{rel_type}]->(o)
                """
                session.run(query, sub=sub, obj=obj)
