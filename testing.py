import pandas as pd
from time import sleep, time
from urllib.error import HTTPError
# Reading the PDF by OCR - Preliminaries
import requests, io, pytesseract
import fitz as fz #PyMuPdf
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# LLM initialization
import ollama
model = "llama3.2:1b"

sleep_time_recommended = 25 # in seconds --> from robots.txt
devices_already_done = 20 # Done 25 --> Start from 25 for next batch
devices_needed_currently = 5
csvFileName = "dataframe.csv"
file_with_URLs = "pdf_urls.txt"

details_required = ["AI Product name", "AI Company name", "Sub-specialty", "Imaging Modality", "Intended Purpose", "Clinical use", "Product output", "CE Classification", "FDA Classification", "Manufacturer"]
prompt = "Given below are the contents from an FDA regulatory document. I require the following details from the document: " + ", ".join(details_required) + ". Give me these information in a JSON format only."

# # Get the details of all devices
# some_ret = pd.read_html("https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices")
# dataframe = some_ret[0]

# # Save the details
# dataframe.to_csv(csvFileName, index=False)

# # Read the details
# dataframe = pd.read_csv(csvFileName, index_col=None)

# # Extracting the details of the Radiology devices alone
# required_data = dataframe.loc[dataframe["Panel (lead)"] == "Radiology"]
# required_data.reset_index(inplace=True)

def scrape_data(file_with_URLs:str, devices_already_done:int, devices_needed_currently:int):
    """Script to scrape pdf URLs and save them to a text file"""
    global required_data
    pdf_urls = []
    for i in range(devices_already_done, devices_already_done+devices_needed_currently):
        try:
            sleep(sleep_time_recommended)
            device_ID = required_data["Submission Number"][i]
            specific_device_FDA_URL = f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPMN/pmn.cfm?ID={device_ID}"
            some_ret = pd.read_html(specific_device_FDA_URL) # Pandas DataFrame object
            # For single digit years its just single digit -> 4 not 04
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

def ocr_process(file_with_URLs: str, devices_needed_currently = 2):
    """Performs OCR to extract text from the PDFs considered"""
    # Read the file with the PDF URLs
    with open(file_with_URLs) as url_file:
        all_lines = url_file.readlines()

    ocr_texts = []
    # for i in range(devices_already_done, devices_already_done+devices_needed_currently):
    for i in range(devices_needed_currently):
        try:
            current_url = all_lines[i][:-1] # Skips the "\n" portion at the end of each line
            print(current_url)
            start_time = time()
            request = requests.get(current_url) # Request is made only here
            filestream = io.BytesIO(request.content)
            # OCR and reading the file contents
            doc = fz.open(stream=filestream, filetype="pdf")
            ocr_text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fz.Matrix(2,2))
                img = Image.open(io.BytesIO(pix.tobytes()))
                text = pytesseract.image_to_string(img)
                ocr_text += f"\n--- Page {page_num + 1} (OCR) ---\n" + text
            doc.close()
            ocr_texts.append(ocr_text)
            print(f"Finished Device {i+1}")
            processing_time = time() - start_time
            if (i+1 != devices_needed_currently) and (processing_time < sleep_time_recommended):
                sleep(sleep_time_recommended - processing_time)
        except HTTPError:
            print(HTTPError)
    return ocr_texts

def LLM_response(document_contents: str):
    """Function to get the LLM's Response"""
    global model, details_required, prompt
    response = ollama.chat(model=model, messages=[
        {
            "role": "user",
            "content": f"--- Document Contents: ---\n{document_contents}\n\n--- My Query: ---\n{prompt}"
        }
    ])
    return response.message.content

ocr_texts = ocr_process(file_with_URLs, 1)
for document_contents in ocr_texts:
    current_reponse = LLM_response(document_contents)
    print(current_reponse)