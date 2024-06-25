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
import pdfplumber

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)
parser = JsonOutputParser()
section_prompt = PromptTemplate(
    template="""You are adept at extracting sections and their corresponding subsections from PDF files, capturing every detail. Please format the output in JSON. For each section, use the section title as the main key. This key should map to a dictionary with 'page', 'content', and 'subsections' if there are any as nested keys. 'page' should store the section's page number, 'content' should be left empty for now, and 'subsections' should contain any further nested subsections, each also structured with 'page', 'content', and 'subsections' as necessary. Ensure the output is a clean JSON representation without any superfluous text.
    Pdf File content: {context}
    Question: {question}
    """,
    input_variables=["context","question"],
    # partial_variables={"format_instructions": parser.get_format_instructions()}
)

section_chain=LLMChain(llm=llm,prompt=section_prompt,output_parser=parser)

def get_pdf_text(pdf_file_path):
    text = ""
    with open(pdf_file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

def pinfo_extractor(pdf_text):
    question = """ From the provided pdf File Content, extract all sections and its subsections from it without missing any."""
    response=section_chain.invoke(
        {
            "context": pdf_text,
            "question": question
        }
    )
    return response["text"]

def find_toc_page(file_path, search_text):
    reader = PdfReader(file_path)
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        text = page.extract_text()
        if text and search_text.lower() in text.lower():
            return i + 1
    return None

def extract_text_from_page_range(pdf_path, start_page, additional_pages=3):
    reader = PdfReader(pdf_path)
    extracted_text = ""

    # Adjust start_page from 1-indexed to 0-indexed for PyPDF2 usage
    start_page_index = start_page - 1

    # Determine the last page to extract text from
    end_page_index = start_page_index + additional_pages

    # Extract text from the start page to the end page, ensuring not to exceed the PDF's page count
    for i in range(start_page_index, min(end_page_index + 1, len(reader.pages))):
        page = reader.pages[i]
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"

    return extracted_text


def main(pdf_file_path):
    try:
        page_num=find_toc_page(pdf_file_path,"TABLE OF CONTENTS")
        print(f"Table of contents is {page_num}")
        text=extract_text_from_page_range(pdf_file_path,page_num)
        sections = pinfo_extractor(text)
        print(f"sections are {sections}")
        calculate_page_ranges(sections)
        print(f"updated secctions are {sections}")
        create_pdf_chunks(pdf_file_path, sections)
        json_output_path = os.path.join(os.path.splitext(pdf_file_path)[0], 'sections_summary.json')
        json_data = serialize_to_json(sections, json_output_path)
        print("JSON data:", json_data)

    except Exception as e:
        print(f"An error occurred: {e}")



def create_pdf_chunks(pdf_path, sections):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    directory_path = os.path.join(os.path.dirname(pdf_path), base_name)
    os.makedirs(directory_path, exist_ok=True)
    with open(pdf_path, "rb") as infile:
        reader = PdfReader(infile)
        for section, info in sections.items():
            writer = PdfWriter()
            
            # Get start and end pages from the page-range attribute
            start_page, end_page = info['page-range']
            start_page = start_page - 1  # Convert to 0-indexed
            end_page = end_page if end_page == 'end' else end_page - 1  # Convert to 0-indexed, if not 'end'

            # Add pages to the PDF writer from the specified range
            if end_page == 'end':
                page_numbers = range(start_page, len(reader.pages))
            else:
                page_numbers = range(start_page, end_page + 1)  # Include the end page
            
            for page_number in page_numbers:
                writer.add_page(reader.pages[page_number])

            # Create the output filename and path
            output_filename = f"{pdf_path}_{section.replace(' ', '_').replace('.', '').replace('/', '_')}.pdf"
            output_file_path = os.path.join(directory_path, output_filename)
            
            # Save the new PDF
            with open(output_file_path, "wb") as outfile:
                writer.write(outfile)
                # print(f"Created PDF chunk: {output_file_path}")
            
            info['file_path'] = output_file_path

            # Recursively handle subsections if they exist
            if 'subsections' in info and info['subsections']:
                create_pdf_chunks(pdf_path, info['subsections'])    

def calculate_page_ranges(sections, next_parent_section_page=None):
    section_titles = list(sections.keys())
    for i, section in enumerate(section_titles):
        # Determine the end page for this section based on the start of the next section
        # Adjust so that the range includes the first page of the next section
        if i + 1 < len(section_titles):
            # We'll include the start page of the next section in the range
            next_section_page = sections[section_titles[i + 1]]['page']
        else:
            # If there's no next section within the same level, we look to extend to the parent's next section start
            # Or just indicate 'end' if there's no parent or further sections
            next_section_page = next_parent_section_page if next_parent_section_page else 'end'

        # Set page range to extend from the current section's start page to the next section's start page
        sections[section]['page-range'] = (sections[section]['page'], next_section_page)

        # Recursively calculate page ranges for subsections
        if 'subsections' in sections[section] and sections[section]['subsections']:
            # The last subsection should extend to the next section of the current section or to the end if no more sections
            subsection_next_page = sections[section_titles[i + 1]]['page'] if i + 1 < len(section_titles) else next_parent_section_page
            calculate_page_ranges(sections[section]['subsections'], subsection_next_page)

def serialize_to_json(sections, output_path=None):
    json_data = json.dumps(sections, indent=4)
    if output_path:
        with open(output_path, 'w') as json_file:
            json_file.write(json_data)
    return json_data

if __name__ == "__main__":
    pdf_file_path = './34-100155.pdf'
    main(pdf_file_path)
