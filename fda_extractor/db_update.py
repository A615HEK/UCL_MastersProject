import os
import pandas as pd

FDA_SOURCE_URL = "https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices"

def check_db_updates(fda_db_path: str = "FDA_DB.csv"):
    """
    Checks the FDA Database, compares it with the local mirror for
    any changes and updates the local DB copy if any differences
    are found
    """
    # Get the details of all devices
    table_returned = pd.read_html(FDA_SOURCE_URL)
    fresh_df = table_returned[0]

    # Save the details
    if os.path.exists(fda_db_path):
        stored_df = pd.read_csv(fda_db_path, index_col=None)
        new_submission_numbers = set(fresh_df["Submission Number"]) - set(stored_df["Submission Number"])
        # Check for differences
        if not new_submission_numbers:
            print("No difference...")
        else:
            print(f"Found {len(new_submission_numbers)} new device(s). Updating local DB...")
            fresh_df.to_csv(fda_db_path, index=False)
    else:
        fresh_df.to_csv(fda_db_path, index=False)

    # Read the details
    stored_df = pd.read_csv(fda_db_path, index_col=None)

    # Extracting the details of Pre-Market approved, Radiology devices alone
    required_data = stored_df.loc[stored_df["Panel (lead)"] == "Radiology"]
    required_data = required_data[required_data["Submission Number"].str.contains("K") == True]
    required_data.reset_index(inplace=True)
    required_data = required_data.drop("index", axis=1)
    return required_data
