from kg_builder import KnowledgeGraphBuilder

def main():
    print("Initializing Knowledge Graph Builder...")
    # Update password if you set a different one in Neo4j Desktop
    kg = KnowledgeGraphBuilder(uri="neo4j://127.0.0.1:7687", user="neo4j", password="24112003")
    
    sample_text = "MindSearch uses FastAPI for the backend. The system proposes a hybrid retrieval method. FAISS accelerates vector search."
    
    print(f"\nProcessing text: '{sample_text}'")
    entities, relations = kg.process_text(sample_text)
    
    print("\nExtracted Entities (from NER):")
    for ent in entities:
        print(f" - {ent[0]} ({ent[1]})")
        
    print("\nExtracted Relations (from syntax rules):")
    for sub, rel, obj in relations:
        print(f" - ({sub}) -[{rel}]-> ({obj})")
        
    print("\nData should now be in Neo4j! Open Neo4j Browser to verify.")
    kg.close()

if __name__ == "__main__":
    main()
