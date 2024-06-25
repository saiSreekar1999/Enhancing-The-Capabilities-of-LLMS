import os
import re
import json
import openai
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
import pdfplumber
from openai import OpenAI
import pandas as pd
from llama_parse import LlamaParse
import nest_asyncio
nest_asyncio.apply()
from pinecone import Pinecone, ServerlessSpec
import logging
import ast
from langchain_community.callbacks import get_openai_callback
import concurrent.futures
import threading


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llama_api_key=os.getenv("LLAMA_API_KEY")
pinecone_api=os.getenv("PineCone_API")
client = OpenAI()
pc=Pinecone(api_key=pinecone_api)
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)
llm1 = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4-turbo", temperature=0)
llm2 = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o", temperature=0)
llm_finetune=ChatOpenAI(openai_api_key=openai_api_key,model="ft:gpt-3.5-turbo-1106:rady-msba-ucsd:entityepochs:9PKeyomv",temperature=0)



parser = StrOutputParser()


def pdf_llama_text(filepath):
    parser=LlamaParse(
        api_key=llama_api_key,
        result_type="text"
    )
    # Assuming 'parser' is an instance of a class defined in the llama_parse module
    md_documents = parser.load_data(
        file_path=filepath
    )
    return md_documents[0].text

summary_prompt=PromptTemplate(
    template="""You are an expert in summarizing complex legal and financial documents with a focus on specific entities such as Regulatory Bodies, Legislative Bodies and Laws, Types of Advisers, Financial Instruments, Compliance Terms, Market Participants, Financial Market Infrastructure, Advisory Services, Regulatory and Oversight Committees, and Legal Frameworks and Guidelines. Your expertise includes analyzing the relationships and interactions between these entities within the regulatory framework.Ensure the output is just summary text without any superfluous text.
    Pdf File content: {context}
    Question: {question}
    """,
    input_variables=["context","question"]
)

entities_prompt = PromptTemplate(
    template="""You are a highly specialized AI tasked with identifying and listing entities from summaries of Final Rule documents issued by the U.S. Securities and Exchange Commission (SEC). Your role is critical in ensuring the accurate representation of named entities for integration into a graph database. Focus strictly on entities that are directly mentioned in the summaries, avoiding extrapolation or prediction.Format the output as a list without any additional text or explanation. Follow these detailed instructions:

    1. *Entities Extraction*:
    - Extract specific names of entities within the following illustrative categories, understanding that other relevant entities might also be mentioned:
        a) Regulatory and Oversight Committees (examples: "Financial Stability Oversight Council")
        b) Legislative Bodies and Key Laws (examples: "U.S. Congress", "Investment Advisers Act of 1940", "Dodd-Frank Act")
        c) Financial Instruments (examples: "derivatives", "equity securities")
        d) Market Participants (examples: "retail investors", "institutional investors")
        e) Financial Market Infrastructure (examples: "New York Stock Exchange", "NASDAQ")
        f) Regulatory Bodies (examples: "Securities and Exchange Commission")
        g) Legal Frameworks (examples: "General Data Protection Regulation (GDPR)")

    2. *Accuracy Requirement*:
    - Only include entities explicitly mentioned in the summaries. Avoid including entities based on inference or common association with the domain. There must be a direct reference in the text for each entity listed.

    3. *Exclusion of Non-Entities*:
    - Exclude terms and concepts that are not named entities. Specifically, avoid listing:
        • Generic terms (examples: 'Mobile applications', 'Clients')
        • Generic processes or concepts (examples: 'financial planning', 'portfolio management', 'fiduciary duty', 'due diligence')
    - Note that these examples are not exhaustive. Do not include any generic financial terms, services, or broad industry terminologies unless they are part of a named entity or specific act cited in the text.

    4. *Output Format*:
    - Format output as list of Entities like this ["entity1","entity2","entity3",....] without any additional text.

    Summary: {context}
    Question: {question}
    """,
    input_variables=["context","question"]
)

entities_finetune_prompt = PromptTemplate(
    template=""" You are a highly specialized AI tasked with identifying and listing key entities from summaries of final rule documents issued by the U.S. Securities and Exchange Commission (SEC). Your focus should be on entities directly mentioned in the text. Avoid extrapolating or inferring entities not explicitly stated. Your output should list these entities in a straightforward, comma-separated format like ['entity1','entity2',....]. Key categories of entities to consider include: Regulatory Bodies (e.g., 'Securities and Exchange Commission'), Legislative Bodies and Key Laws (e.g., 'U.S. Congress', 'Investment Advisers Act of 1940', 'Dodd-Frank Act'), Financial Instruments (e.g., 'derivatives', 'equity securities'), Market Participants (e.g., 'retail investors', 'institutional investors'), Financial Market Infrastructure (e.g., 'New York Stock Exchange', 'NASDAQ'), Regulatory and Oversight Committees (e.g., 'Financial Stability Oversight Council'), Legal Frameworks (e.g., 'General Data Protection Regulation (GDPR)'). These categories are illustrative but not exhaustive. As you analyze the text, use your understanding of what constitutes an entity within these categories to identify any additional relevant entities not explicitly listed above.

        summary: {summary}
    """,
    input_variables=["summary"]
)

relationships_summary_prompt = PromptTemplate(
    template="""You are tasked as an expert in analyzing regulatory documents issued by the Securities and Exchange Commission (SEC). Your role is to extract key relationships from the provided summaries and lists of entities. Always generate relationships in a Triplet format suitable for storage in a graph database. Format output as list of Relationships without any additional text following below instructions.

        Instructions for Relationship Extraction:

        1) Identify Direct Relationships: 
            From the provided summary and list of entities, identify direct relationships that are clear and actionable. Focus on relationships that
            represent regulatory actions, compliance measures, legislative changes, and direct interactions between entities.
        2) Structure Relationships as Triplets:
            Format: ["Subject", "Predicate", "Object"]
            Example: ["Securities and Exchange Commission", "adopts", "Rule 605"]
            Ensure that both the Subject and the Object are entities explicitly mentioned in the list. Do not alter entity names in subject and object.The Predicate should clearly define the action or relationship between the Subject and the Object, using verbs such as "adopts", "amends", "requires", "provides feedback on"
            etc.
        3) Output Format:
            Present these relationships as a list of triplets. Ensure the output contains only these triplets without any additional text or explanation.
            Example Output Format: [["Securities and Exchange Commission", "adopts", "Rule 605"], ["Rule 605", "requires", "broker-dealers"], ...]
        summary: {summary}
        entities: {entities}
        question: {question}
    """,
    input_variables=["summary","entities","question"]
)
format_relationships_prompt=PromptTemplate(
    template=""" You are an Expert in Formatting Relationships into format suitable to store in neo4j graph database.Given a list of relationships, entities extracted from Final Rule documents published by S.E.C, please format each relationship for storage in a Neo4j graph database. Each relationship consists of a subject, predicate, and object. Format output as list of Relationships only without any additional text following below instructions.

        Instructions for Relationship Formatting:
        
        1. Predicates are Verbal Actions: Convert each predicate into a concise verb or verb phrase that clearly describes the action or relation between the subject and object. Use uppercase for the verbs to maintain consistency.
            Example: change predicate "has made a decision on"  to  "DECIDES_ON"
            ["Commission", "has made a decision on", "Rule 605"] changes to ["Commission", "DECIDES_ON", "Rule 605"]

        2. Remove Redundancy: Ensure that predicates do not include any redundant information from the subject or object. The predicate should only contain the action or relation, not repeat any part of the subject or object.
            Example: Change ["Rule 605", "Rule 605 impacts", "market centers"] to ["Rule 605", "IMPACTS", "market centers"].

        3. Simplify Objects and Subjects: If the object or subject contains complex or lengthy descriptions, simplify these to their basic entity form.
            Example: For complex objects, reduce descriptions to their essence.
            Before: ["Market Data Infrastructure (MDI) Rules", "EXPAND", "content of consolidated market data"]
            After: ["MDI Rules", "EXPANDS", "market data content"]

        4. Correct Relationship Structure: Confirm each relationship follows the structure: [Subject, Predicate, Object]. The subject and object should be properly identified entities, and the predicate should effectively describe their relationship.
            Example: Correct ["Rule 605, extends to, broker-dealers using data"] to ["Rule 605", "EXTENDS_TO", "broker-dealers"].

        5. Output Format: Present these relationships as a list of triplets. Ensure the output contains only these triplets without any additional text or explanation.
            Example Output Format: [["Securities and Exchange Commission", "ADOPTS", "Rule 605"], ["Rule 605", "REQUIRES", "broker-dealers"], ...]

        relationships: {relationships}
        entities: {entities}        
        question: {question}
        """,
    input_variables=["relationships","entities","question"]
)


summary_chain=LLMChain(llm=llm,prompt=summary_prompt,output_parser=parser)
entities_chain=LLMChain(llm=llm,prompt=entities_prompt,output_parser=parser)
relationships_chain=LLMChain(llm=llm2,prompt=relationships_summary_prompt,output_parser=parser)
format_relationships_chain=LLMChain(llm=llm2,prompt=format_relationships_prompt,output_parser=parser)
entities_finetune_chain=LLMChain(llm=llm_finetune,prompt=entities_finetune_prompt,output_parser=parser)


def get_pdf_text(pdf_file_path):
    text = ""
    with open(pdf_file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

def remove_footer_sections(text):
    # Compile a regex pattern to match the specific footer sections described
    # The pattern starts with numbers and includes subsequent lines starting with long spaces or more numbered lines
    pattern = re.compile(
        r'^\s*\d+\s+.*(?:\n(?:\s{2,}.*|\d+\s+.*))*', 
        re.MULTILINE
    )
    
    # Replace matched sections with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    # Optionally clean up extra blank lines that may remain
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)
    
    return cleaned_text

summary_total_tokens=0
summary_total_cost=0


def summary_extractor(pdf_text):
    global summary_total_tokens
    global summary_total_cost
    question = """ From the provided pdf File Content, create an elaborated summary focused on the entities and relationships of it without missing any important information."""
    with get_openai_callback() as cb:
        response = summary_chain.invoke(
            {
                "context": pdf_text,
                "question": question
            }
        )
        summary_total_tokens+=cb.total_tokens
        summary_total_cost+=cb.total_cost
    return response["text"]



def entity_extractor(summary):
    question = """ From the provided summary of the Final rule document published by U.S SEC identify all the entities. Do not miss any entities."""
    response=entities_chain.invoke(
        {
            "context": summary,
            "question": question
        }
    )
    return response["text"]

entity_total_tokens=0
entity_total_cost=0

def entity_finetune_extractor(summary):
    global entity_total_tokens
    global entity_total_cost
    
    question = """ From the provided summary of the Final rule document published by U.S SEC identify all the entities. Do not miss any entities."""
    with get_openai_callback() as cb:
        response=entities_finetune_chain.invoke(
            {
                "summary": summary,
                "question": question
            }
        )
        entity_total_tokens += cb.total_tokens
        entity_total_cost += cb.total_cost
    return response["text"]


def entity_raw_extractor(text):
    global entity_total_tokens
    global entity_total_cost

    question = """ From the provided pdf text of the Final rule document published by U.S SEC identify all the entities. Do not miss any entities."""
    with get_openai_callback() as cb:
        response=entities_raw_chain.invoke(
            {
                "text": text,
                "question": question
            }
        )
    entity_total_tokens += cb.total_tokens
    entity_total_cost += cb.total_cost
    return response["text"]

relationships_total_tokens=0
relationships_total_cost=0

def relationships_extractor(summary,entities):
    global relationships_total_tokens
    global relationships_total_cost
    question = """ From the provided Summary and Entities of the Final rule document published by U.S SEC identify only the Important Relationships to store them in graph database. Do not miss any."""
    with get_openai_callback() as cb:
        response=relationships_chain.invoke(
            {
                "summary": summary,
                "entities": entities,
                "question": question
            }
        )
        relationships_total_tokens += cb.total_tokens
        relationships_total_cost += cb.total_cost
    return response["text"]

def relationships_raw_extractor(rawtext,entities):
    global relationships_total_tokens
    global relationships_total_cost
    question = """ From the provided pdf raw text and Entities of the Final rule document published by U.S SEC identify all Relationships. Do not miss any."""
    with get_openai_callback() as cb:
        response=relationships_raw_chain.invoke(
            {
                "rawtext": rawtext,
                "entities": entities,
                "question": question
            }
        )
        relationships_total_tokens += cb.total_tokens
        relationships_total_cost += cb.total_cost
    return response["text"]


def format_relationships(entities, relationships):
    global relationships_total_tokens
    global relationships_total_cost    
    question = """ From the provided Entities and Relationships, Format each Relationship into ["subject","predicate","object"]. Make sure both subject and object are properly identified entities provided in the list """
    with get_openai_callback() as cb:
        response=format_relationships_chain.invoke(
            {
                "relationships": relationships,
                "entities": entities,
                "question": question
            }
        )
        relationships_total_tokens += cb.total_tokens
        relationships_total_cost += cb.total_cost        
    return response["text"]

def clean_and_parse_relationships(rawtext):
    # Replace newlines and escape sequences
    rawtext = rawtext.replace('\n', '').replace('\r', '').replace('\t', '')
    try:
        # Convert the cleaned string to a list
        return json.loads(rawtext)
    except json.JSONDecodeError:
        return []
    
def preprocess(text):
    """Basic preprocessing to standardize text."""
    return re.sub(r'\W+', '', text).lower()

def process_relationships(relationships, entities):
    updated_relationships = []
    non_entities = set()  # Set to hold non-matching entities
    combine_entities=set()
    combine_entities.update(entities)

    for relationship in relationships:
        if len(relationship) != 3:
            continue
        
        subject, predicate, obj = relationship
        obj_processed = preprocess(obj)
        sub_processed = preprocess(subject)
        obj_found = False
        sub_found = False
        combine_entities.add(subject)
        combine_entities.add(obj)
        # # Check subject for exact match
        # if sub_processed in entities:
        #     sub_found = True

        # # Check object for exact match
        # if obj_processed in entities:
        #     obj_found = True

        # If subject is not an exact match, check for substring matches
        if not sub_found:
            for entity in entities:
                entity_processed = preprocess(entity)
                if entity_processed in sub_processed:
                    sub_found = True
                    # subject=entity
                    if entity_processed != sub_processed:
                        non_entities.add(subject)
                        predicate = subject + " " + predicate
                    break

        # If object is not an exact match, check for substring matches
        if not obj_found:
            for entity in entities:
                entity_processed = preprocess(entity)
                if entity_processed in obj_processed:
                    obj_found = True
                    # obj=entity                    
                    if entity_processed != obj_processed:
                        non_entities.add(obj)
                        predicate += " " + obj
                    break

        # If no exact or substring match is found for object
        if not obj_found:
            non_entities.add(obj)
            predicate += " " + obj

        # If no exact or substring match is found for subject
        if not sub_found:
            non_entities.add(subject)
            predicate = subject + " " + predicate

        updated_relationships.append([subject, predicate.strip(), obj])

    return updated_relationships, list(combine_entities), list(non_entities)


def safe_literal_eval(entities_str):
    try:
        return ast.literal_eval(entities_str)
    except (ValueError, SyntaxError):
        print("Error converting entities_str to list. Proceeding with an empty list.")
        return []


def load_synonym_data(synonym_path,entities_to_check):
    with open(synonym_path, 'r',encoding='utf-8') as file:
        synonyms_data = json.load(file)
    synonym_dict = {item: details["synonym"] for item, details in synonyms_data.items()}
    new_entities = []
    for entity in entities_to_check:
        if entity not in synonym_dict:
            new_entities.append(entity)
    return synonym_dict,new_entities

def load_predicates_from_excel(file_path):
    df = pd.read_excel(file_path)
    df['Predicate'] = df['Predicate'].str.lower().str.strip()
    return df

def check_predicate_existence(predicate, new_predicate):
    normalized_predicate = new_predicate.lower().strip()
    lower_keys = {key.lower(): value for key, value in predicate.items()}
    if normalized_predicate in lower_keys:
        return True
    else:
        return False

def relationships_check(relationships,predicates_path,entities):
    with open(predicates_path, 'r',encoding='utf-8') as file:
        predicates_data = json.load(file)
    manual_relationships=[]
    predicates_to_check=[]
    relationships_list=safe_literal_eval(relationships)
    for sub,pred,obj in relationships_list:
        if not check_predicate_existence(predicates_data,pred):
            manual_relationships.append([sub,pred,obj])
            predicates_to_check.append(pred)
        elif sub not in entities or obj not in entities:
            manual_relationships.append([sub,pred,obj])
    return manual_relationships ,predicates_to_check

def no_summary_entities(summary,entities):
    entities_valid=[]
    non_entities=[]
    for entity in entities:
        if entity not in summary:
            non_entities.append(entity)
        else:
            entities_valid.append(entity)
    return entities_valid,non_entities

def update_taxonomy(json_path, taxonomy_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    with open(taxonomy_path, 'r', encoding='utf-8') as file:
        taxonomy_data = json.load(file)

    synonym_to_category = {}
    for category, synonyms in taxonomy_data.items():
        for synonym in synonyms:
            synonym_to_category[synonym.lower()] = category
    
    # Update each item with the taxonomy category for its synonyms
    for item in data:
        if 'synonyms' in item:
            # Update each synonym with its corresponding taxonomy category
            for synonym_key, synonym_value in item['synonyms'].items():
                # Find the taxonomy category for the synonym value
                category = synonym_to_category.get(synonym_value.lower(), "Unknown")
                # Update the synonym value with both synonym and category
                item['synonyms'][synonym_key] = {
                    "synonym": synonym_value,
                    "taxonomy": category
                }
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

synonyms_path='./synonyms.json'
taxonomy_path='./taxonomy.json'
predicates_path='./predicates.json'


def process_finetune_pdfs(root_folder,output_path):
    data = []
    global summary_total_cost
    global summary_total_tokens
    global relationships_total_cost
    global relationships_total_tokens
    global entity_total_tokens
    global synonym_total_tokens
    global synonym_total_cost

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                text = pdf_llama_text(file_path)
                clean_text=remove_footer_sections(text)
                summary = summary_extractor(clean_text)
                print(f"file name is {file_path}")
                entities = entity_finetune_extractor(summary)
                raw_relationships=relationships_extractor(summary,entities)
                entities_lower=[e.lower() for e in entities]
                relationships = clean_and_parse_relationships(raw_relationships)
                updated_relationships,combine_entities,non_entities = process_relationships(relationships,safe_literal_eval(entities))
                all_entities=safe_literal_eval(entities)+non_entities
                final_relationships=format_relationships(all_entities,updated_relationships)
                manual_check_entities,no_check_entities = synonym_check(combine_entities,synonyms_path,file_path) 
                manual_check_relationships,predicates_to_check = relationships_check(final_relationships,predicates_path,no_check_entities)
                data.append({
                    'filename': file,
                    'Summary': summary,
                    'Extracted entities': entities,
                    'Extracted_relationships': relationships,
                    'Updated_relationships': updated_relationships,
                    'relationships': final_relationships,
                    'not_extracted_entities': non_entities,
                    'entities': combine_entities,
                    'no_check_entities': no_check_entities,
                    'manual_check_entities': manual_check_entities,
                    'manual_check_relationships': manual_check_relationships,
                    "predicates_to_check": predicates_to_check
                })

        with open(output_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        similar_entities(output_path, synonyms_path)
        taxonomy_checker(synonyms_path, taxonomy_path)
        update_taxonomy(output_path, taxonomy_path)
    print(f"Summary details: {summary_total_cost} {summary_total_tokens}")
    print(f"Relationships details: {relationships_total_cost} {relationships_total_tokens}")
    print(f"Entities details:  {entity_total_cost} {entity_total_tokens}")
    print(f"Synonym details: {synonym_total_cost} {synonym_total_tokens}")
    print(f"Taxonomy details: {taxonomy_total_cost} {taxonomy_total_tokens}")


taxonomy_prompt= PromptTemplate(    
    template=""" You are an expert in structuring complex legal and financial information into a systematic taxonomy. Your task is to analyze the provided entities within the context of the security-based swap market, organizing them into a coherent taxonomy that reflects their functions and roles within the regulatory framework. The output should be in JSON format, presenting a structured taxonomy without any additional or superfluous text.

    First, attempt to categorize the entities into the following predefined top-level categories based on their most apparent roles and functions
    within the regulatory environment:
    - Regulatory Bodies
    - Financial Instruments
    - Market Participants
    - Legal Frameworks and Laws
    - Compliance and Reporting Requirements
    - Trading Practices and Rules
    - Risk Management and Oversight
    - Enforcement Actions
    - Regulatory Processes
    - Sector-Specific Regulations

    If an entity does not clearly fit into any of these categories, create a new category that best represents its role and function. 

    Construct a taxonomy in JSON format:
    - The top-level keys represent the main categories.
    - Each category may contain subcategories or a list of entities, based on their functional roles and operational context in the security-based swap market.
    - Ensure the hierarchical structure of the taxonomy is logical and organized, with entities grouped under the most relevant categories based on their functions and roles. Entities that require new categories should be logically incorporated into the structure, providing a clear rationale for their categorization.

    Entities: {entities}
    Relationships: {relationships}
    Question: {question}
    """,
    input_variables=["entities","relationships","question"],
)

refine_taxonomy_prompt = PromptTemplate(
    template=""" 
            You are an expert in structuring and refining taxonomies from the given extracted entities of Final Rule PDFs published by the S.E.C. Your current task involves analyzing given entities extracted from subsequent sections of an SEC final rule document and integrating them into an existing taxonomy(categories) or creating a new category for them. Please follow the instructions below and provide output as a list of entities and their taxonomy mapping without any additional or superfluous text.

        Instructions to follow:
        - Carefully review the provided existing taxonomy to understand its structure and the categories already in place.
        - Integrate new entities into the existing taxonomy. Assign each new entity to the appropriate category based on its role and function within the regulatory framework.
        - If an entity appears under different names but refers to the same concept, treat it as the same entity and categorize it accordingly to maintain consistency throughout the taxonomy.
        - Create a new category only if an entity does not fit into any existing categories and ensure it accurately reflects the entity's role within the regulatory context.
        - Maintain and enhance the logical and organized structure of the taxonomy, ensuring all entities are appropriately categorized and that the taxonomy reflects a comprehensive view of the regulatory environment.
        - The output should list each entity followed by its corresponding taxonomy category in a clear and concise format.

        Existing Taxonomy: {existing_taxonomy}
        New Entities: {new_entities}
        Question: {question}
    """,
    input_variables=["existing_taxonomy", "new_entities", "question"]
)

synonym_prompt= PromptTemplate(    
    template=""" You are an expert in standardizing complex legal and financial terminology from the given extracted entities of Final Rule PDFs published by the S.E.C. Your task involves analyzing given entities extracted from subsequent sections of an SEC final rule document and integrating them into an existing synonyms list. The entities may appear under different names but represent the same concept or item.

        Please update the synonyms in JSON format:
        - Carefully review the provided existing synonyms list to understand its structure and the synonyms already in place.
        - Integrate new entities into the existing synonyms list. Match each new entity with existing synonyms. If a suitable synonym is found, associate the new entity with it.
        - If no suitable synonym exists, create a new synonym that is concise and pertinent to financial and regulatory documents.
        - Ensure all entities are appropriately standardized and that the synonyms list reflects a comprehensive view of the regulatory environment.
        - Provide the new entities and their synonyms in JSON format without any additional text.

    existing_synonyms: {existing_synonyms}
    newlist: {newlist}
    question: {question}
    """,
    input_variables=["existing_synonyms","newlist","question"],

)
new_synonym_prompt= PromptTemplate(    
    template=""" You are an expert in standardizing complex legal and financial terminology from the given extracted entities of Final Rule PDFs published by the S.E.C. Your task involves analyzing given entities extracted from subsequent sections of an SEC final rule document and integrating them into an existing synonyms list. The entities may appear under different names but represent the same concept or item.Please follow below instructions and provide output as List of entities without any additional or superfluous text.

        Instructions to follow:
        - Carefully review the provided existing entities and synonyms list to understand its structure and the synonyms already in place.
        - Match each new entity with existing entitie. If a similar entity is found,assign same synonym to the new entity. 
        - If no similar entity exists, create a new synonym that is concise and pertinent to financial and regulatory documents.
        - Ensure all entities are appropriately standardized and their synonyms reflects a comprehensive view of the regulatory environment.
        - Provide only the new entities and their synonyms in List format without any additional text.

    existing_synonyms: {existing_synonyms}
    newlist: {newlist}
    question: {question}
    """,
    input_variables=["existing_synonyms","newlist","question"],

)
predicate_prompt= PromptTemplate(    
    template=""" 
            You are an expert in developing relationships for graph databases and are tasked with integrating new legal and financial
            terminology from extracted entities of Final Rule PDFs published by the S.E.C. Your role involves analyzing new predicates
            extracted from subsequent sections of an SEC final rule document and aligning them with an existing synonyms list. These
            predicates may appear under different names but represent the same concept or relationship. Follow the instructions below and
            provide output as List of only new predicates and their synonyms without any additional or superfluous text, formatted
            appropriately for graph database entry.

        Instructions to follow:
        - Carefully review the provided existing synonyms list to understand its structure and the relationships already established.
        - Match each new predicate with the existing synonyms. If a similar predicate is found, assign the same synonym to the new predicate.
        - If no similar predicate exists, create a new synonym that is concise and pertinent to financial and regulatory contexts.
        - Ensure all predicates are appropriately standardized and that their synonyms reflect a comprehensive understanding of the regulatory framework.
        - Format the output as a List of only new predicates and their synonyms ready for graph database integration Provide the output without any additional or superfluous text.

        Existing Predicates: {existing_predicates}
        New predicates list: {new_predicates}
        question: {question}
    """,
    input_variables=["existing_predicates", "new_predicates","question"]
)

synonym_chain = LLMChain(llm=llm1,prompt=new_synonym_prompt,output_parser=parser)
taxonomy_chain = LLMChain(llm=llm1,prompt=taxonomy_prompt,output_parser=parser)
refine_taxonomy_chain = LLMChain(llm=llm1,prompt=refine_taxonomy_prompt,output_parser=parser)
predicate_chain = LLMChain(llm=llm1,prompt=predicate_prompt,output_parser=parser)

synonym_total_tokens=0
synonym_total_cost=0

def synonym_extractor(existing_synonyms,newlist):
    global synonym_total_tokens
    global synonym_total_cost
    question = """Using the Existing synonyms, generate new synonyms for all the new entities. Format output as List Format containing only the new entities  along with their synonyms. Do not provide any additional text"""
    with get_openai_callback() as cb:
        response = synonym_chain.invoke(
            {
                "existing_synonyms": existing_synonyms,
                "newlist": newlist, 
                "question": question
            }
        )
        synonym_total_tokens += cb.total_tokens
        synonym_total_cost += cb.total_cost

    return response["text"]

def taxonomy_extractor(entities,relationships):
    question = """From the provided Entities and Relationships , generate a structured taxonomy by categorizing only Entities. Format output as json without any additional text"""
    with get_openai_callback() as cb:
        response = taxonomy_chain.invoke(
            {
                "entities": entities,
                "relationships": relationships, 
                "question": question
            }
        )
    return response["text"]

taxonomy_total_tokens=0
taxonomy_total_cost=0

def refine_taxonomy_extractor(new_entities,existing_taxonomy):
    global taxonomy_total_tokens
    global taxonomy_total_cost

    question = """From the provided New Entities generate a structured taxonomy by categorizing these entities into the given existing taxonomy framework. Format output as List Format containing only the new entities  along with their taxonomy. Do not provide any additional or superfluous tex"""

    with get_openai_callback() as cb:
        response = refine_taxonomy_chain.invoke(
            {
                "existing_taxonomy": existing_taxonomy,
                "new_entities": new_entities,
                # "new_relationships": new_relationships,
                "question": question
            }
        )
        taxonomy_total_tokens += cb.total_tokens
        taxonomy_total_cost += cb.total_cost
    
    return response["text"]

predicate_total_cost=0
predicate_total_tokens=0

def predicate_extractor(existing_predicates,newlist):
    global predicate_total_tokens
    global predicate_total_cost
    question = """Using the Existing Synonyms for the Predicates, generate new synonyms for all the new predicates. Format output as List Format containing only the new predicates along with their synonyms. Do not provide any additional text"""
    with get_openai_callback() as cb:
        response = predicate_chain.invoke(
            {
                "existing_predicates": existing_predicates,
                "new_predicates": newlist, 
                "question": question
            }
        )
        predicate_total_tokens += cb.total_tokens
        predicate_total_cost += cb.total_cost

    return response["text"]


def synonym_check(entities,synonym_path,filename):
    manual_check_entites=set()
    no_check_entities=set()
    with open(synonym_path, 'r',encoding='utf-8') as file:
        synonyms_data = json.load(file)
    for entity in entities:
        entity_lower=entity.lower()
        if entity_lower in synonyms_data:
            no_check_entities.add(entity_lower)
            # Check if the filename already exists in the list to avoid duplication
            if filename not in synonyms_data[entity_lower]['filenames']:
                synonyms_data[entity_lower]['filenames'].append(filename)
        else:
            manual_check_entites.add(entity)
    with open(synonym_path, 'w') as file:
        json.dump(synonyms_data, file, indent=4)
    return list(manual_check_entites), list(no_check_entities)

def process_entity(entity):
    embeddings = {}
    embeddings[entity] = client.embeddings.create(input=entity, model="text-embedding-3-large").data[0].embedding
    index = pc.Index("word-index")
    res = index.query(vector=[embeddings[entity]], top_k=1, include_metadata=True, namespace="word-embeddings")
    for match in res['matches']:
        if match["score"] > 0.70:
            return entity, match['metadata']['entity']
        else:
            return entity, ""

def process_entity_batch(batch):
    batch_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_entity, entity) for entity in batch]
        for future in concurrent.futures.as_completed(futures):
            entity, synonym = future.result()
            batch_results[entity] = synonym
    return batch_results

def process_item(item, synonyms_data, lock,gpt_entities):
    entities_with_synonyms = {}
    entities_list = item.get('entities', '[]')
    # entities_list = safe_literal_eval(entities_str)
    batches = [entities_list[i:i + 5] for i in range(0, len(entities_list), 5)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_entity_batch, batch) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            entities_with_synonyms.update(result)

    embeddings = {}
    with lock:
        for entity, synonym in entities_with_synonyms.items():
            if synonym == "":
                if entity not in gpt_entities:
                    gpt_entities[entity] = {
                        'filenames': [item['filename']],
                        'synonym': ''
                    }
                else:
                    gpt_entities[entity]['filenames'].append(item['filename'])
            else:
                entity_lower = entity.lower()
                if entity_lower in synonyms_data:
                    if 'filenames' not in synonyms_data[entity_lower]:
                        synonyms_data[entity_lower]['filenames'] = []
                    if item['filename'] not in synonyms_data[entity_lower]['filenames']:
                        synonyms_data[entity_lower]['filenames'].append(item['filename'])
                else:
                    embeddings[entity_lower] = client.embeddings.create(input=entity_lower, model="text-embedding-3-large").data[0].embedding
                    synonyms_data[entity_lower] = {
                        'synonym': synonym.lower(),
                        'filenames': [item['filename']]
                    }
    store_embeddings_vectordb(embeddings)

def similar_entities(json_path, synonyms_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    with open(synonyms_path, 'r', encoding='utf-8') as file:
        synonyms_data = json.load(file)

    existing_synonyms_list = {key: value["synonym"] for key, value in synonyms_data.items()}


    gpt_entities = {}
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_item, item, synonyms_data, lock, gpt_entities) for item in data]
        concurrent.futures.wait(futures)
    
    gpt_entities_list = list(gpt_entities.keys())
    
    # print(gpt_entities_list)
    gpt_Response=synonym_extractor(existing_synonyms_list,gpt_entities)
    
    gpt_Response_list=safe_literal_eval(gpt_Response)

    # print(gpt_Response_list)

    for entity_pair in gpt_Response_list:
        for entity,synonym in entity_pair.items():
            synonyms_data[entity.lower()] = {
                'synonym': synonym.lower(),
                'filenames': gpt_entities[entity]["filenames"]
            }

    with open(synonyms_path, 'w') as file:
        json.dump(synonyms_data, file, indent=4)

    for item in data:
        if 'entities' in item:
            item['synonyms'] = {entity: synonyms_data[entity.lower()]['synonym'] for entity in item['entities'] if entity.lower() in synonyms_data}
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)


def store_embeddings_vectordb(embeddings):
    try:
         index = pc.Index("word-index")
         for word,vector in embeddings.items():
             index.upsert(
                    vectors=[{
                        "id": word.lower(),
                        "values": vector,
                        "metadata": {"entity": word.lower()}
                    }],
                    namespace="word-embeddings"
                )             
    except Exception as e:
        logging.error(f"Error storing embeddings in Pinecone DB: {e}")

def store_synonyms(synonyms_path):
    with open(synonyms_path, 'r',encoding='utf-8') as file:
        synonyms_data = json.load(file)
    embeddings={}
    for item in synonyms_data:
        embeddings[item]=client.embeddings.create(input=item, model="text-embedding-3-large").data[0].embedding
    store_embeddings_vectordb(embeddings)

def parse_taxonomy_output(output):
    lines = output.split("\n")
    taxonomy_pairs = []
    for line in lines:
        line = line.lstrip('- ').strip()
        if ": " in line:
            entity, category = line.split(": ", 1)
            entity = entity.strip().lower()
            category = category.strip().title()
            taxonomy_pairs.append({entity: category})
    return taxonomy_pairs


def taxonomy_checker(synonyms_path,taxonomy_path):
    with open(synonyms_path, 'r',encoding='utf-8') as file:
        synonyms_data = json.load(file)
    with open(taxonomy_path, 'r',encoding='utf-8') as file:
        taxonomy_data = json.load(file)
    
    normalized_taxonomy_data = {key.lower(): value for key, value in taxonomy_data.items()}
    synonyms_list = [value["synonym"].lower() for key, value in synonyms_data.items()]
    existing_taxonomy=[value for key, value in taxonomy_data.items()]
    existing_taxonomy_list=[item.lower() for sublist in existing_taxonomy for item in sublist]

    new_synonyms=[]
    for synonym in synonyms_list:
        if synonym not in existing_taxonomy_list:
            new_synonyms.append(synonym)
    latest_taxonomy=refine_taxonomy_extractor(new_synonyms,taxonomy_data)

    latest_taxonomy_list=safe_literal_eval(latest_taxonomy)
    if not latest_taxonomy_list:
        print("Attempting to parse textual output...")
        latest_taxonomy_list = parse_taxonomy_output(latest_taxonomy)
    for entry in latest_taxonomy_list:
        for entity, category in entry.items():
            category_lower = category.lower()
            if category_lower in normalized_taxonomy_data:
                if entity.lower() not in [e.lower() for e in normalized_taxonomy_data[category_lower]]:
                    normalized_taxonomy_data[category_lower].append(entity.lower())
            else:
                normalized_taxonomy_data[category_lower] = [entity]
    
    with open(taxonomy_path, 'w', encoding='utf-8') as file:
        json.dump({k.title(): v for k, v in normalized_taxonomy_data.items()}, file, indent=4)

if __name__ == "__main__":
    process_finetune_pdfs('./Sample-data/34-97656','./Summary_without_footer_FineTune_check14.json')