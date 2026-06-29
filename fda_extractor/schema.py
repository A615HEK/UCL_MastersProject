from pydantic import BaseModel

# Defining Pydantic Schema
class ProductInfo(BaseModel):
    AI_Product_name: str
    AI_Company_name: str
    Sub_specialty: str
    Imaging_Modality: str
    Intended_Purpose: str
    Summary_passage_of_Clinical_use: str
    Summary_passage_of_Product_output: str
    CE_Classification: str
    FDA_Classification: str
    Manufacturer: str

# List of fields
DETAILS_REQUIRED = [
    "AI Product name",
    "AI Company name",
    "Sub-specialty",
    "Imaging Modality",
    "Intended Purpose",
    "Summary paragraph/sentence of Clinical use",
    "Summary paragraph/sentence of Product output",
    "CE Classification",
    "FDA Classification",
    "Manufacturer",
]

# JSON Schema
DETAILS_SCHEMA = ProductInfo.model_json_schema()