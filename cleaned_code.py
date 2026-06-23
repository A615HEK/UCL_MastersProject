# Main imports
import pandas as pd
# For Data Visualisation
from time import perf_counter
import json

# TODO: Check if Tesseract is in the system
# if tesseract in system:
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract_flag = True

# Imports from associated code
from associated_code import check_DB_updates, scrape_pdfs, write_pdf_to_txt, ProductInfo, RAG, LLM_response, save_responses, timer_n_printer

# DB Check and Updation
required_data = check_DB_updates()

# Scraping PDF URLs
scrape_pdfs(required_data)

# Reading PDFs and Writing to text files + Preprocessing
write_pdf_to_txt(pytesseract_flag)

# LLMaJ w/ RAG
# LLM Definitions
details_required = ["AI Product name", "AI Company name", "Sub-specialty", "Imaging Modality", "Intended Purpose", "Summary paragraph/sentence of Clinical use", "Summary paragraph/sentence of Product output", "CE Classification", "FDA Classification", "Manufacturer"]
model_generator = "llama3.2:1b" # Generating LLM
model_judge = "llama3.1:8b" # Judging LLM
max_runs = 3 # Initial Run + 3 more iterations
gen_responses_file_name = "Response_RAG_LLMaJ.csv"
judge_responses_file_name = "Response_RAG_LLMaJ_Judge.csv"

details_schema = ProductInfo.model_json_schema() # Dict of JSON Schema
# TODO: Change this into a sensible index
for i in range(12, 21):
    gen_times = []
    judge_times = []

    with open(f"./test_pdfs/pdf_{i}.txt") as current_file:
        doc_contents = current_file.read()

    # RAG Portion - Get only related documents for generation
    relevant_documents = RAG(doc_contents)

    print("Initial run...")
    # Generating LLM
    response_generator = LLM_response(model_generator, details_required, details_schema, doc_contents, save_file_name=gen_responses_file_name, run_num="Initial Run")

    # Judging LLM
    response_judge = LLM_response(model_judge, details_required, details_schema, doc_contents, judge=True, gen_response=response_generator.response, save_file_name=judge_responses_file_name, run_num="Initial Run")

    # Print the fields to be changed
    judge_response_dict = json.loads(response_judge.response)
    fields_to_correct = []
    for info_field, value in judge_response_dict.items():
        if not value or value == "False":
            fields_to_correct.append(info_field)
    print(fields_to_correct)
    run_num = 1

    # TODO: Need to add decorator for timing and printing
    while (fields_to_correct and run_num<=max_runs):
        print(f"Iteration number: {run_num}")

        # Dynamically changing JSON schema to get only required information
        modified_details_schema = dict(details_schema)
        modified_details_schema["properties"] = {field_considered: modified_details_schema["properties"][field_considered] for field_considered in fields_to_correct}
        modified_details_schema["required"] = fields_to_correct

        # Iteration Generator
        response_generator = LLM_response(model_generator, fields_to_correct, modified_details_schema, doc_contents, save_file_name=gen_responses_file_name, run_num=run_num)
        # response.update(json.loads(response_generator.response))

        # Iteration Judge
        response_judge = LLM_response(model_judge, fields_to_correct, modified_details_schema, doc_contents, judge=True, gen_response=response_generator.response, save_file_name=judge_responses_file_name, run_num=run_num)
        # response.update(json.loads(response_judge.response))

        # Fields to Correct
        judge_response_dict = json.loads(response_judge.response)
        fields_to_correct = []
        for info_field, value in judge_response_dict.items():
            if not value or value == "False":
                fields_to_correct.append(info_field)
        print(fields_to_correct)

        run_num += 1

    times_dict = {"Avg_Gen_Time":sum(gen_times)/len(gen_times), "Min_Gen_Time":min(gen_times), "Max_Gen_Time":max(gen_times), "Avg_Judge_Time":sum(judge_times)/len(judge_times), "Min_Judge_Time":min(judge_times), "Max_Judge_Time":max(judge_times)}
    save_responses(pd.DataFrame(times_dict, index=[0]), "time_RAG_LLMaJ.csv")
    print("----------\n")