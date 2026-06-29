from shutil import which
# Reading the PDF by OCR - Preliminaries
import requests, io, pytesseract, os
from PIL import Image
import fitz as fz #PyMuPdf
from time import sleep
import pandas as pd

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (research scraper)"}
DEVICE_URL_TEMPLATE = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPMN/pmn.cfm"

def check_tesseract_engine() -> bool:
    """
    Checks for the Tesseract OCR system binary in the system PATH
    first and then common Windows install locations as a fallback

    Returns:<br>
    -> True - If Tesseract path was found<br>
    -> False - If Tesseract wasn't found in the pre-defined locations
    """
    tesseract_path = which("tesseract")
    if tesseract_path:
        return True

    possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%USERPROFILE%\scoop\apps\tesseract\current\tesseract.exe"),
        ]
    for candidate_path in possible_paths:
        if os.path.exists(candidate_path):
            pytesseract.pytesseract.tesseract_cmd = candidate_path
            return True

    # Tesseract not found in the system
    return False

def scrape_pdfs(required_data: pd.DataFrame, file_with_urls: str = "pdf_urls.txt", devices_already_done: int = 0, devices_needed_currently: int = 20, sleep_time_recommended: int = 25) -> list[str]:
    """
    Scrape pdf URL data and save them to a text file<br>
    sleep_time_recommended:int (in seconds) --> from robots.txt<br>
    devices_needed_currently:int --> Number of devices to be scraped in the current run
    """
    pdf_urls = []
    end_idx = devices_already_done+devices_needed_currently
    for device_idx in range(devices_already_done, end_idx):
        try:
            device_ID = required_data["Submission Number"][device_idx]
            specific_fda_device_lookup_url = DEVICE_URL_TEMPLATE + f"?ID={device_ID}"
            table_returned = pd.read_html(specific_fda_device_lookup_url) # Pandas DataFrame object
            # For single digit years its just a single digit -> 4 not 04
            date_received = table_returned[7].loc[table_returned[7][0] == "Date Received"][1]
            year_submitted = str(int(date_received.values[0][-2:]))
            pdf_url = f"https://www.accessdata.fda.gov/cdrh_docs/pdf{year_submitted}/{device_ID}.pdf"
            pdf_urls.append(pdf_url)
            print(f"[{device_idx+1}/{end_idx}] Resolved URL for {device_ID}")
            if not ((device_idx+1)%5):
                with open(file_with_urls, "a") as f:
                    for device_idx in range(len(pdf_urls)):
                        f.write(pdf_urls[device_idx]+"\n")
                pdf_urls = []
        except Exception as exc:
            # Loggin the error and continuing business as usual
            # One bad row doesn;t affect entire pipeline/other process
            print(f"[{device_idx+1}/{end_idx}] Failed to resolve URL: {exc}")
        finally:
            if device_idx != end_idx-1:
                sleep(sleep_time_recommended)

    # Record the scraped PDF URLs to a text file
    with open(file_with_urls, "a") as f:
        for pdf_url in pdf_urls:
            f.write(pdf_url + "\n")

    return pdf_urls

def get_with_retry(url: str, max_retries: int = 3, backoff_seconds: float = 10.0, **kwargs) -> requests.Response:
    """
    Fetches data with an exponential backoff-based retry loop
    """
    last_exc = None
    for attempt in range(1, max_retries+1):
        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            print(f" [retry {attempt}/{max_retries}] {url} failed: {exc}")
            if attempt < max_retries:
                sleep(backoff_seconds*attempt)
    return last_exc

def get_pdf_data(file_url: str) -> list[str]:
    """
    Pass file URL to fetch the contents of the PDF via OCR if Tesseract is
    available, falling back to PyPDF2 metadata-based extraction otherwise
    """
    response = get_with_retry(file_url)
    filestream = io.BytesIO(response.content)
    # Check for presence of OCR engine else use default pipelines
    if check_tesseract_engine():
        # OCR and reading the file contents
        doc = fz.open(stream=filestream, filetype="pdf")
        doc_pages = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fz.Matrix(2,2))
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = pytesseract.image_to_string(img)
            doc_pages.append(f"\n--- Page {page_num+1} ---\n" + text)
        doc.close()
    
    # If this portion runs -> Tesseract not found
    print("[warning] Tessearct not found - falling back to PyPDF2 text extraction (quality may degrade on scanned/encrypted PDFs)")
    from PyPDF2 import PdfReader
    pdf_file = PdfReader(filestream)
    # Safeguard against no text extracted -> Empty string in place
    doc_pages = [f"\n--- Page {i+1} ---\n" + (pdf_file.pages[i].extract_text() or "") for i in range(len(pdf_file.pages))]
    return doc_pages

def remove_letter_page(doc_pages: list[str]) -> list[str]:
    """
    Removes the obligatory letter page(s) from the FDA
    """
    letter_pages_indices = []
    for idx, page in enumerate(doc_pages):
        if ("enclosure" in page.lower()) or ("sincerely" in page.lower()):
            letter_pages_indices.append(idx)

    if not letter_pages_indices:
        print("[warning] No letter page was detected (OCR might have missed it) - Keeping all pages")
        return doc_pages
    if letter_pages_indices[0] == 0:
        return doc_pages[letter_pages_indices[-1]+1:]
    return doc_pages[:letter_pages_indices[0]] + doc_pages[letter_pages_indices[-1]+1:]

def write_pdf_to_txt(file_url: str, output_directory: str = "./test_pdfs") -> str:
    """
    Extracts, cleans and writes the PDF in the specified URL to a .txt file
    """
    # Create a directory to hold the scraped PDF data
    # exist_ok = True -> Prevents raising OSError if directory exists 
    os.makedirs(output_directory, exist_ok=True)
    pdf_text = "".join(remove_letter_page(get_pdf_data(file_url)))
    # Extract just the file submission number for indexing and naming
    submission_num = file_url.split("/")[-1].removesuffix(".pdf")
    file_path = os.path.join(output_directory, f"pdf_{submission_num}.txt")
    with open(file_path, "w") as current_file:
        current_file.write(pdf_text)
    return file_path

def scrape_and_extract_pdfs(required_data: pd.DataFrame, file_with_urls: str = "pdf_urls.txt", devices_already_done: int = 0, devices_needed_currently: int = 20, sleep_time_recommended: int = 25, output_directory: str = "./test_pdfs") -> list[str]:
    """
    End-End: Scraping -> Extraction pipeline<br>
    - Resolve URL for requested device range<br>
    - Extract Text using OCR (Tesseract) or PyPDF2 (Fallback)<br>
    - Write extracted contents to a text file<br>
    - Return the list of written text file paths
    """
    pdf_urls = scrape_pdfs(required_data, file_with_urls, devices_already_done, devices_needed_currently, sleep_time_recommended)

    # Read and write the scraped PDF contents to a text file
    written_paths = []
    for pdf_url in pdf_urls:
        try:
            file_path = write_pdf_to_txt(pdf_url, output_directory)
            print(f"Written {file_path} successfully")
            written_paths.append(file_path)
        except Exception as exc:
            print(f"Failed to extract {pdf_url}: {exc}")
    return written_paths
