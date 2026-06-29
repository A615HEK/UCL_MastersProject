import ollama, json
import pandas as pd
from time import perf_counter
from dataclasses import dataclass, field
from .schema import DETAILS_REQUIRED, DETAILS_SCHEMA

JUDGE_PROMPT_TEMPLATE = "You are part of the judging module of an automated machine interface that specialises in scanning through FDA regulatory documents to find the requested fields that are present in the document provided to you and packaging these outputs into only valid JSON outputs. Given above are the contents from an FDA regulatory document on a particular product. I require the following details from the document: {details_required}. Given below is the JSON response from the module that you need to judge: {gen_response}. For each of the fields comment only with True or False, True if the fields match with the information from the document provided and False otherwise. Give me the outputs in a JSON format only."
GENERATOR_PROMPT_TEMPLATE = "You are part of an automated machine interface that specialises in scanning through FDA regulatory documents to find the requested fields that are present in the document provided to you and packaging these outputs into only valid JSON outputs. Given above are the contents from various regulatory documents on a particular product. I require the following details from the document: {details_required}. Return None for the CE_Classification if it is not specified. Give me these information in a JSON format only."

def LLM_response(model: str, details_required: list[str], details_schema: dict, doc_contents: str, judge: bool = False, gen_response: str | None = None, options: dict | None = None):
    """
    Function to get and return the response generated from the
    specified LLM using Ollama
    """
    if judge:
        prompt = JUDGE_PROMPT_TEMPLATE.format(details_required = ", ".join(details_required), gen_response = gen_response)
    else:
        prompt = GENERATOR_PROMPT_TEMPLATE.format(details_required = ", ".join(details_required))

    options = options or {"temperature": 0}
    print(f"options: {options}")
    response = ollama.generate(
                            model = model,
                            prompt = f"--- Document Contents: ---\n{doc_contents}\n\n--- My Query: ---\n{prompt}",
                            format = details_schema,
                            options = options)
    return response

def fields_to_change(judge_response_dict) -> list[str]:
    """
    Parses a judge response and returns the list of fields that were
    marked as False i.e., need to be regenerared/changed
    """
    judge_response_dict = json.loads(judge_response_dict.response)
    fields_to_correct = []
    for info_field, value in judge_response_dict.items():
        if not value or value == "False":
            fields_to_correct.append(info_field)
    return fields_to_correct

def modified_schema(details_schema: dict, fields_to_correct:list[str]) -> dict:
    """
    Returns a copy of the schema restricted to only the given fields
    for the iterative-correction passes
    """
    modified_details_schema = dict(details_schema)
    modified_details_schema["properties"] = {field_considered: modified_details_schema["properties"][field_considered] for field_considered in fields_to_correct}
    modified_details_schema["required"] = fields_to_correct
    return modified_details_schema

@dataclass
class IterationResult:
    run_num: int
    generator_response: str
    judge_response: str
    gen_time: float
    judge_time: float
    fields_generated: list[str]
    fields_flagged_by_judge: list[str] = field(default_factory=list)

def run_llmaj_Loop(relevant_docs: str, model_generator: str = "llama3.2:1b", model_judge: str = "llama3.1:8b", max_runs: int = 3, fields_required: list[str] | None = None, schema: dict | None = None) -> list[IterationResult]:
    """
    Runs the LLMaJ + RAG Loop on each device in the given list<br>
    required_devices: List of devices identified by Submission numbers, eg.: K240369
    """
    fields_required = fields_required or DETAILS_REQUIRED
    details_schema = schema or DETAILS_SCHEMA

    results: list[IterationResult] = []
    current_fields = fields_required
    current_schema = details_schema
    run_num = 0
    gen_times = []
    judge_times = []
    fields_to_correct = []

    # Initial run + extra iteration runs for correction
    # until no mistakes are judged by LLM or
    # max runs are reached, whichever is smaller
    while True:
        if run_num == 0:
            print("Initial run...")
        else:
            print(f"Iteration number: {run_num}")

        # Iteration Generator
        start_time = perf_counter()
        generator_response = LLM_response(model_generator, current_fields, current_schema, relevant_docs)
        gen_time = perf_counter() - start_time
        gen_times.append(gen_time)
        print(f"Gen Time = {gen_time}")

        # Iteration Judge
        start_time = perf_counter()
        judge_response = LLM_response(model_judge, current_fields, current_schema, relevant_docs, judge=True, gen_response=generator_response.response)
        judge_time = perf_counter() - start_time
        judge_times.append(judge_time)
        print(f"Judge Time = {judge_time}")

        # Fields which were flagged as being wrong by the judge
        fields_to_correct = fields_to_change(judge_response)

        # Appending results
        results.append(IterationResult(
            run_num=run_num,
            generator_response=generator_response,
            judge_response=judge_response,
            # gen_time=gen_time,
            # judge_time=judge_time,
            fields_generated=current_fields,
            fields_flagged_by_judge=fields_to_correct))

        # Dynamically changing JSON schema to get only required information
        current_fields = fields_to_correct
        current_schema = modified_schema(DETAILS_SCHEMA, fields_to_correct)

        # Loop logic
        run_num += 1
        if not fields_to_correct or run_num>max_runs:
            break

    times_dict = {"Avg_Gen_Time":sum(gen_times)/len(gen_times), "Min_Gen_Time":min(gen_times), "Max_Gen_Time":max(gen_times), "Avg_Judge_Time":sum(judge_times)/len(judge_times), "Min_Judge_Time":min(judge_times), "Max_Judge_Time":max(judge_times)}
    save_responses(pd.DataFrame(times_dict, index=[0]), "time_RAG_LLMaJ.csv")
    print("----------\n")

    # Return all the raw results
    # Merging to be handled by pipeline as per use case
    return results

def merge_final_outputs(results: list[IterationResult]) -> dict:
    """
    Merges the individual iteration results into a coherent output
    """
    if not results:
        raise ValueError("No iteration results to merge")

    # Initial response
    final_dict = json.loads(results[0].generator_response)
    if len(results) > 1:
        for iteration in results[1:]:
            corrected = json.loads(iteration.generator_response)
            for field_name in iteration.fields_generated:
                if field_name in corrected:
                    final_dict[field_name] = corrected[field_name]
    return final_dict

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

def save_responses(data_to_save: pd.DataFrame, save_file_name: str):
    """
    Function to save LLM responses as .csv files
    """
    import os
    if os.path.exists(save_file_name):
        data_to_save.to_csv(save_file_name, mode='a', index=False, header=False)
    else:
        data_to_save.to_csv(save_file_name, index=False)

# def save_responses(data_to_save: pd.DataFrame, save_file_name, run_num: int):
#     import os
#     if os.path.exists(save_file_name):
#         responses_DF = pd.DataFrame(json.loads(response.response), index=[0])
#         if run_num:
#             responses_DF["Run"] = run_num
#         save_responses(responses_DF, save_file_name)
#     else: