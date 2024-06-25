from py2neo import Graph
import json
import ast
import re

# Initialize Neo4j Graph connection
graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"),name="neo4j")

def preprocess(text):
    """Basic preprocessing to standardize text."""
    return re.sub(r'\W+', '', text).lower()

def safe_literal_eval(entities_str,filename):
    """Safely evaluate a string representation of a Python expression."""
    try:
        return ast.literal_eval(entities_str)
    except (ValueError, SyntaxError):
        print(f"Error converting entities_str to list in file: {filename}. Skipping this entry.")
        return []

def escape_string(text):
    """Escape single quotes in text for Cypher query."""
    return text.replace("'", "\\'")

def build_and_execute_queries(json_data):
    for item in json_data:
        synonyms = {preprocess(key): value for key, value in item.get('synonyms', {}).items()}
        predicates_to_check = {preprocess(pred) for pred in item.get('predicates_to_check', [])}
        
        filename = escape_string(item.get('filename', 'Unknown'))
        if filename == "Unknown":
            continue
        relationships = safe_literal_eval(item.get('relationships', '[]'), filename)
        if relationships is None:
            continue
        summary = escape_string(item.get('Summary', ''))
        for src, rel_type, tgt in relationships:
            src_pre = preprocess(src)
            tgt_pre = preprocess(tgt)
            rel_type_pre = preprocess(rel_type)

            src_synonym_data = synonyms.get(src_pre, {})
            tgt_synonym_data = synonyms.get(tgt_pre, {})

            if 'synonym' in src_synonym_data and src_synonym_data['synonym']:
                src_synonym = escape_string(src_synonym_data['synonym']).lower()
                src_type = 'no_manual'
            else:
                src_synonym = escape_string(src).lower()
                src_type = 'manual'

            src_taxonomy = escape_string(src_synonym_data.get('taxonomy', src))

            if 'synonym' in tgt_synonym_data and tgt_synonym_data['synonym']:
                tgt_synonym = escape_string(tgt_synonym_data['synonym']).lower()
                tgt_type = 'no_manual'
            else:
                tgt_synonym = escape_string(tgt).lower()
                tgt_type = 'manual' 

            tgt_taxonomy = escape_string(tgt_synonym_data.get('taxonomy', tgt))
            rel_type_type = 'manual' if rel_type_pre in predicates_to_check else 'no_manual'

            src_label = 'Manual' if src_type == 'manual' else src_taxonomy
            tgt_label = 'Manual' if tgt_type == 'manual' else tgt_taxonomy

            node_query_a = (
                f"MERGE (a:`{src_label}` {{synonym: '{src_synonym}'}}) "
                f"ON CREATE SET a.name = '{src_synonym}', a.entity = ['{escape_string(src)}'], a.taxonomy = '{src_taxonomy}', "
                f"a.type = '{src_type}', a.filenames = ['{filename}'] "
                f"ON MATCH SET a.entity = CASE WHEN '{escape_string(src)}' IN a.entity THEN a.entity ELSE a.entity + ['{escape_string(src)}'] END, "
                f"a.taxonomy = '{src_taxonomy}', a.type = '{src_type}', "
                f"a.filenames = CASE WHEN '{filename}' IN a.filenames THEN a.filenames ELSE a.filenames + ['{filename}'] END"
            )
            node_query_b = (
                f"MERGE (b:`{tgt_label}` {{synonym: '{tgt_synonym}'}}) "
                f"ON CREATE SET b.name = '{tgt_synonym}', b.entity = ['{escape_string(tgt)}'], b.taxonomy = '{tgt_taxonomy}', "
                f"b.type = '{tgt_type}', b.filenames = ['{filename}'] "
                f"ON MATCH SET b.entity = CASE WHEN '{escape_string(tgt)}' IN b.entity THEN b.entity ELSE b.entity + ['{escape_string(tgt)}'] END, "
                f"b.taxonomy = '{tgt_taxonomy}', b.type = '{tgt_type}', "
                f"b.filenames = CASE WHEN '{filename}' IN b.filenames THEN b.filenames ELSE b.filenames + ['{filename}'] END"
            )
            graph.run(node_query_a)
            graph.run(node_query_b)

            # Create/update relationships
            relationship_query = (
                f"MATCH (a:`{src_label}` {{synonym: '{src_synonym}'}}), (b:`{tgt_label}` {{synonym: '{tgt_synonym}'}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                f"ON CREATE SET r.type = '{rel_type_type}', r.summary = ['{summary}'], r.filenames = ['{filename}'] "
                f"ON MATCH SET r.type = '{rel_type_type}', "
                f"r.summary = CASE WHEN '{summary}' IN r.summary THEN r.summary ELSE r.summary + ['{summary}'] END, "
                f"r.filenames = CASE WHEN '{filename}' IN r.filenames THEN r.filenames ELSE r.filenames + ['{filename}'] END"
            )

            graph.run(relationship_query)

with open('./Summary_without_footer_FineTune_check10.json', 'r') as file:
    json_data = json.load(file)
    build_and_execute_queries(json_data)