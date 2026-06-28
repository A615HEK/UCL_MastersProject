# 1. All imports
import pandas as pd
from shutil import which
from urllib.error import HTTPError
from time import sleep, perf_counter

# Reading the PDF by OCR - Preliminaries
import requests, io, pytesseract
from PIL import Image
from PyPDF2 import PdfReader
import fitz as fz #PyMuPdf

import os, ollama, json
from pydantic import BaseModel

# 2. Class definitions
# JSON Schema
# Defining Pydantic Schema
class ProductInfo(BaseModel):
    AI_Product_name:str
    AI_Company_name:str
    Sub_specialty:str
    Imaging_Modality:str
    Intended_Purpose:str
    Summary_passage_of_Clinical_use:str
    Summary_passage_of_Product_output:str
    CE_Classification:str
    FDA_Classification:str
    Manufacturer:str

# 3. Function definitions
def check_tesseract_engine():
    """
    Checks for the Tesseract OCR system binary
    """
    tesseract_path = which("tesseract")
    possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%USERPROFILE%\scoop\apps\tesseract\current\tesseract.exe"),
        ]
    if not tesseract_path:
        for considered_path in possible_paths:
            if os.path.exists(considered_path):
                pytesseract.pytesseract.tesseract_cmd = considered_path
                return True
    else:
        return True if tesseract_path else False
    # Tesseract not found in the system
    return False

def check_DB_updates(fda_db_path:str = "FDA_DB.csv"):
    """
    Checks the FDA Database, compares it with the local copy for
    any changes and updates the local DB copy if any differences
    are found
    """
    # Get the details of all devices
    some_ret = pd.read_html("https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices")
    dataframe = some_ret[0]

    # Save the details
    csvFileName = fda_db_path
    new_indices = []
    if os.path.exists(fda_db_path):
        stored_DF = pd.read_csv(csvFileName, index_col=None)
        # Check for differences
        for val in dataframe["Submission Number"].values:
            if val not in stored_DF["Submission Number"].values:
                new_indices.append(stored_DF["Submission Number"][stored_DF["Submission Number"] == val].index)
    if len(new_indices) == 0:
        print("No difference...")
    else:
        # TODO: Write only these lines to the DB
        dataframe.to_csv(csvFileName, index=False)

    # Read the details
    stored_DF = pd.read_csv(csvFileName, index_col=None)

    # Extracting the details of Pre-Market approved, Radiology devices alone
    required_data = stored_DF.loc[stored_DF["Panel (lead)"] == "Radiology"]
    required_data = required_data[required_data["Submission Number"].str.contains("K") == True]
    required_data.reset_index(inplace=True)
    required_data = required_data.drop("index", axis=1)
    return required_data

def scrape_pdfs(required_data, file_with_URLs:str = "pdf_urls.txt", devices_already_done:int = 0, devices_needed_currently:int = 20, sleep_time_recommended:int = 25):
    """
    Scrape pdf URL data and save them to a text file<br>
    sleep_time_recommended:int (in seconds) --> from robots.txt<br>
    devices_needed_currently:int --> Number of devices to be scraped in the current run
    """
    pdf_urls = []
    for i in range(devices_already_done, devices_already_done+devices_needed_currently):
        try:
            sleep(sleep_time_recommended)
            device_ID = required_data["Submission Number"][i]
            specific_device_FDA_URL = f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPMN/pmn.cfm?ID={device_ID}"
            some_ret = pd.read_html(specific_device_FDA_URL) # Pandas DataFrame object
            # For single digit years its just a single digit -> 4 not 04
            year_submitted = str(int(some_ret[7].loc[some_ret[7][0] == "Date Received"][1].values[0][-2:]))
            pdf_URL = f"https://www.accessdata.fda.gov/cdrh_docs/pdf{year_submitted}/{device_ID}.pdf"
            pdf_urls.append(pdf_URL)
            print(f"Finished device {i+1}...")
            if not ((i+1)%5):
                with open(file_with_URLs, "a") as f:
                    for i in range(len(pdf_urls)):
                        f.write(pdf_urls[i]+"\n")
                pdf_urls = []
        except HTTPError:
            print(HTTPError)

    with open(file_with_URLs, "a") as f:
        for i in range(len(pdf_urls)):
            f.write(pdf_urls[i]+"\n")

    with open(file_with_URLs) as f:
        pdf_urls = f.readlines()
        for pdf_url in pdf_urls:
            write_pdf_to_txt(pdf_url)

def get_pdf_data(file_URL:str) -> list[str]:
    """Pass file URL to get the contents of the document"""
    try:
        response = requests.get(url=file_URL)
        filestream = io.BytesIO(response.content)
        info_doc = []
        # Check for presence of OCR engine else use default pipelines
        pytesseract_flag = check_tesseract_engine()
        if pytesseract_flag:
            # OCR and reading the file contents
            doc = fz.open(stream=filestream, filetype="pdf")
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fz.Matrix(2,2))
                img = Image.open(io.BytesIO(pix.tobytes()))
                text = pytesseract.image_to_string(img)
                info_doc.append(f"\n--- Page {page_num+1} ---\n" + text)
            doc.close()
        else:
            pdf_file = PdfReader(filestream)
            info_doc = [f"\n--- Page{i+1} ---\n"+pdf_file.pages[i].extract_text() for i in range(len(pdf_file.pages))]
        return info_doc
    except HTTPError:
        print(HTTPError)
        return []

def remove_letter_page(info_doc:list) -> list[str]:
    """Removes the obligatory letter page(s) from the FDA"""
    letter_pages = []
    for page in info_doc:
        if ("enclosure" in page.lower()) or ("sincerely" in page.lower()):
            letter_pages.append(info_doc.index(page))
    if letter_pages[0] == 0:
        return info_doc[letter_pages[-1]+1:]
    else:
        return info_doc[:letter_pages[0]] + info_doc[letter_pages[-1]+1:]

def write_pdf_to_txt(file_URL:str):
    """
    Reads & writes the PDF in the specified URL to a .txt file <br>
    pytesseract_flag:bool --> True - If Tesseract exists;
                              False (Default) - If no Tesseract
    """
    # Create a directory to hold the scraped PDF data
    if not os.path.exists("./test_pdfs"):
        print("Creating Test Directory...")
        os.mkdir("./test_pdfs")
    pdf_text = "".join(remove_letter_page(get_pdf_data(file_URL)))
    file_name = file_URL.split("/")[-1][:-4] # Extracts just the file submission number for indexing and naming
    with open(f"./test_pdfs/pdf_{file_name}.txt", "w") as current_file:
        current_file.write(pdf_text)

def join_docs(article_parts, *papers_to_process) -> str:
    """
    Join all document sources and return them as a single string
    """
    processed_docs = ""
    processed_docs += "\n".join(article_parts)
    if papers_to_process:
        for doc_considered in papers_to_process:
            processed_docs += "".join(doc_considered)
    return processed_docs

def get_answers(model:str, doc_contents:str, prompt:str, times_to_run:int = 5, temperature:float = 0):
    for i in range(times_to_run):
        if temperature == None:
            response = ollama.generate(model = model,
                        prompt = f"--- Document Contents: ---\n{doc_contents}\n\n--- My Query: ---\n{prompt}",
                        format = "json")
        else:
            response = ollama.generate(model = model,
                                    prompt = f"--- Document Contents: ---\n{doc_contents}\n\n--- My Query: ---\n{prompt}",
                                    format = "json",
                                    options = {"temperature": temperature})
        print(f"--- Response from time {i+1}: ---")
        print(response.response)

def save_responses(data_to_save:pd.DataFrame, save_file_name:str):
    """
    Function to save LLM responses
    """
    if os.path.exists(save_file_name):
        data_to_save.to_csv(save_file_name, mode='a', index=False, header=False)
    else:
        data_to_save.to_csv(save_file_name, index=False)

def LLM_response(model:str, details_required:list, details_schema:dict, doc_contents:str, save_file_name:str = None, judge:bool = False, gen_response:str = None, options:dict = None, run_num:str = None):
    """
    Function to get and return the response generated from the
    specified LLM
    """
    if judge:
        prompt = "You are part of the judging module of an automated machine interface that specialises in scanning through FDA regulatory documents to find the requested fields that are present in the document provided to you and packaging these outputs into only valid JSON outputs. Given above are the contents from an FDA regulatory document on a particular product. I require the following details from the document: " + ", ".join(details_required) + f". Given below is the JSON response from the module that you need to judge: {gen_response}. For each of the fields comment only with True or False, True if the fields match with the information from the document provided and False otherwise. Give me the outputs in a JSON format only."
    else:
        prompt = "You are part of an automated machine interface that specialises in scanning through FDA regulatory documents to find the requested fields that are present in the document provided to you and packaging these outputs into only valid JSON outputs. Given above are the contents from various regulatory documents on a particular product. I require the following details from the document: " + ", ".join(details_required) + ". Return None for the CE_Classification if it is not specified. Give me these information in a JSON format only."
    if not options:
        options = {"temperature": 0}
    print(f"options: {options}")
    start_time = perf_counter()
    response = ollama.generate(model = model,
                            prompt = f"--- Document Contents: ---\n{doc_contents}\n\n--- My Query: ---\n{prompt}",
                            format = details_schema,
                            options = options)
    run_time = perf_counter()- start_time
    if save_file_name:
        responses_DF = pd.DataFrame(json.loads(response.response), index=[0])
        if run_num:
            responses_DF["Run"] = run_num
        save_responses(responses_DF, save_file_name)
    return response, run_time

def fields_to_change(response_dict) -> list:
    """
    Print and return the fields to be changed
    """
    judge_response_dict = json.loads(response_dict.response)
    fields_to_correct = []
    for info_field, value in judge_response_dict.items():
        if not value or value == "False":
            fields_to_correct.append(info_field)
    print(fields_to_correct)
    return fields_to_correct