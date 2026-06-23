import pandas as pd
from urllib.error import HTTPError
from time import sleep

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
    stored_DF = pd.read_csv(csvFileName, index_col=None)

    # Check for differences
    new_indices = []
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

def scrape_based_on_ID(required_FDA_doc_idx:list, required_data):
    """
    Scrape pdf URL data for particular device IDs and save them
    to a text file
    """
    pdf_urls = []
    sleep_time_recommended = 25 # in seconds --> from robots.txt
    file_with_URLs = "pdf_urls.txt"
    for doc_id in required_FDA_doc_idx:
        try:
            device_ID = required_data["Submission Number"][doc_id]
            specific_device_FDA_URL = f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPMN/pmn.cfm?ID={device_ID}"
            some_ret = pd.read_html(specific_device_FDA_URL) # Pandas DataFrame object
            # For single digit years its just a single digit -> 4 not 04
            year_submitted = str(int(some_ret[7].loc[some_ret[7][0] == "Date Received"][1].values[0][-2:]))
            pdf_URL = f"https://www.accessdata.fda.gov/cdrh_docs/pdf{year_submitted}/{device_ID}.pdf"
            pdf_urls.append(pdf_URL)
            print(f"Finished device with ID: {doc_id}...")
            if doc_id != required_FDA_doc_idx[-1]:
                sleep(sleep_time_recommended)
        except HTTPError:
            print(HTTPError)
    with open(file_with_URLs, "a") as f:
        for i in range(len(pdf_urls)):
            f.write(pdf_urls[i]+"\n")

def scrape_pdfs(required_data):
    """
    Scrape pdf URL data and save them to a text file
    """
    pdf_urls = []
    sleep_time_recommended = 25 # in seconds --> from robots.txt
    devices_already_done = 4 # Done 29 --> Start from 29 for next batch
    devices_needed_currently = 17
    file_with_URLs = "pdf_urls.txt"
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