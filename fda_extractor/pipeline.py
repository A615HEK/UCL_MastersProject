# Imports from associated code
from .llmaj import run_llmaj_Loop, merge_final_outputs, IterationResult
from .rag import rag
from .schema import DETAILS_REQUIRED, DETAILS_SCHEMA

def process_document(doc_contents: str, model_generator: str = "llama3.2:1b", model_judge: str = "llama3.1:8b", max_runs: int = 3, num_chunks: int = 7) -> tuple[dict, list[IterationResult]]:
    """
    End-end Processing of an already extracted document text:<br>
    - RAG -> filters and passes only relevant documents<br>
    - Generating + Judging -> Iterative refinement loop<br>
    - Merge the individual iteration results into a single JSON dict
    """
    relevant_docs = rag(doc_contents, num_chunks=num_chunks)
    results = run_llmaj_Loop(relevant_docs,
                             model_generator=model_generator,
                             model_judge=model_judge,
                             max_runs=max_runs,
                             fields_required=DETAILS_REQUIRED,
                             schema=DETAILS_SCHEMA)
    final_output = merge_final_outputs(results)
    return final_output, results

# check_DB_updates, save_responses

# # DB Check and Updation
# required_data = check_DB_updates()

# # Scraping PDF URLs &
# # Reading PDFs and Writing to text files + Preprocessing
# # All bundled into 1 function call
# scrape_pdfs(required_data)

# # LLMaJ w/ RAG
# # LLM Definitions
# model_generator = "llama3.2:1b" # Generating LLM
# model_judge = "llama3.1:8b" # Judging LLM
# max_runs = 3 # Initial Run + 3 more iterations
# gen_responses_file_name = "Response_RAG_LLMaJ.csv"
# judge_responses_file_name = "Response_RAG_LLMaJ_Judge.csv"

# def LLMaJ_RAG_Loop(required_devices:list[str]):
#     """
#     Runs the LLMaJ + RAG Loop on each device in the given list<br>
#     required_devices: List of devices identified by Submission numbers, eg.: K240369
#     """
#     for device in required_devices:
#         gen_times = []
#         judge_times = []

#         try:
#             with open(f"./test_pdfs/pdf_{device}.txt") as current_file:
#                 doc_contents = current_file.read()

#         except FileNotFoundError:
#             pass