# Main imports
import pandas as pd
# For Data Visualisation
import matplotlib.pyplot as plt
from time import sleep, time
# Reading the PDF by OCR - Preliminaries
import requests, io, pytesseract
import fitz as fz #PyMuPdf
from PIL import Image
import json, os
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# LLM initialization
import ollama
# For structured output schema
from pydantic import BaseModel

# Imports from associated code
from associated_code import *

# DB Check and Updation
check_DB_updates()

# Scraping PDF URLs


# Reading PDFs


# Pre-processing


# RAG


## Chunker


## Embedding and Vector DB Creation


## Similarity Search and Retrieval


# JSON Schema


# LLMaJ


# LLMaJ w/ RAG
