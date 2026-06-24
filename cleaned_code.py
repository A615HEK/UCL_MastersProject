# Main imports
import pandas as pd

# Imports from associated code
from associated_code import check_DB_updates, scrape_pdfs, write_pdf_to_txt, ProductInfo, RAG, LLM_response, save_responses, fields_to_change

# DB Check and Updation
required_data = check_DB_updates()

# Scraping PDF URLs
scrape_pdfs(required_data)

# Reading PDFs and Writing to text files + Preprocessing
write_pdf_to_txt()

# LLMaJ w/ RAG
# LLM Definitions
details_required = ["AI Product name", "AI Company name", "Sub-specialty", "Imaging Modality", "Intended Purpose", "Summary paragraph/sentence of Clinical use", "Summary paragraph/sentence of Product output", "CE Classification", "FDA Classification", "Manufacturer"]
model_generator = "llama3.2:1b" # Generating LLM
model_judge = "llama3.1:8b" # Judging LLM
max_runs = 3 # Initial Run + 3 more iterations
gen_responses_file_name = "Response_RAG_LLMaJ.csv"
judge_responses_file_name = "Response_RAG_LLMaJ_Judge.csv"

details_schema = ProductInfo.model_json_schema() # Dict of JSON Schema

for device in required_devices:
    gen_times = []
    judge_times = []

    with open(f"./test_pdfs/pdf_{device}.txt") as current_file:
        doc_contents = current_file.read()

    # RAG Portion - Get only related documents for generation
    relevant_documents = RAG(doc_contents)

    for run_num in range(max_runs):
        if fields_to_correct or run_num==0:
            if run_num == 0:
                print("Initial run...")
                modified_details_schema = details_schema
            else:
                print(f"Iteration number: {run_num}")
                # Dynamically changing JSON schema to get only required information
                modified_details_schema = dict(details_schema)
                modified_details_schema["properties"] = {field_considered: modified_details_schema["properties"][field_considered] for field_considered in fields_to_correct}
                modified_details_schema["required"] = fields_to_correct

            # Iteration Generator
            response_generator, gen_time = LLM_response(model_generator, fields_to_correct, modified_details_schema, doc_contents, save_file_name=gen_responses_file_name, run_num=run_num)
            gen_times.append(gen_time)
            print(f"Gen Time = {gen_time}")

            # Iteration Judge
            response_judge, judge_time = LLM_response(model_judge, fields_to_correct, modified_details_schema, doc_contents, judge=True, gen_response=response_generator.response, save_file_name=judge_responses_file_name, run_num=run_num)
            judge_times.append(judge_time)
            print(f"Judge Time = {judge_time}")

            # Fields to Correct
            fields_to_correct = fields_to_change(response_judge)

    times_dict = {"Avg_Gen_Time":sum(gen_times)/len(gen_times), "Min_Gen_Time":min(gen_times), "Max_Gen_Time":max(gen_times), "Avg_Judge_Time":sum(judge_times)/len(judge_times), "Min_Judge_Time":min(judge_times), "Max_Judge_Time":max(judge_times)}
    save_responses(pd.DataFrame(times_dict, index=[0]), "time_RAG_LLMaJ.csv")
    print("----------\n")