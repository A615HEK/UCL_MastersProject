import json, sys, os
import fda_extractor.llmaj as llmaj_mod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class FakeResponse:
    def __init__(self, response_str):
        self.response = response_str

def make_fake_generate(judge_sequence, gen_sequence):
    call_counts = {}
    def fake_generate(model, prompt: str, format, options):
        call_counts[model] = call_counts.get(model, 0)+1
        idx = call_counts[model]-1
        if ("judge" in prompt.lower()) or ("judging" in  prompt.lower()):
            return FakeResponse(json.dumps(judge_sequence[idx]))
        return FakeResponse(json.dumps(gen_sequence[idx]))
    return fake_generate

def test_loop_runs_without_nameerror_on_first_pass():
    judge_sequence = [{f"field_{i}": "True"} for i in range(3)]
    gen_sequence = [{f"field_{i}": f"value_{i}"} for i in range(3)]
    llmaj_mod.ollama.generate = make_fake_generate(judge_sequence, gen_sequence)
    schema = {"properties": {"name": {}, "company": {}}, "required": []}
    results = llmaj_mod.run_llmaj_loop("doc", fields_required=["field_0", "field_1", "field_2"], schema={"properties": {f"field_{i}": {} for i in range(3)}, "required": []}, max_iterations=3)
    assert len(results) == 1, f"Expected 1 iteration, got {len(results)}"
    assert results[0].fields_flagged_by_judge == []
    print("PASS: test_loop_runs_without_nameerror_on_first_pass")

def test_merge_uses_most_recent_generation_per_field():
    """Test for merge_final_output: fields corrected in a later
    iteration must override the initial (wrong) value, and fields
    never flagged must be preserved from the initial run."""
    judge_sequence = [
        {"name": "False", "company": "True"}, # run 0: name is wrong
        {"name": "True"}                      # run 1: name now correct
    ]
    gen_sequence = [
        {"name": "Wrong Name", "company": "Correct Co"}, # run 0 generation
        {"name": "Right Name"},                          # run 1 generation (only "name" requested)
    ]
    llmaj_mod.ollama.generate = make_fake_generate(judge_sequence, gen_sequence)
    schema = {"properties": {"name": {}, "company": {}}, "required": []}
    results = llmaj_mod.run_llmaj_loop("doc", fields_required=["name", "company"], schema=schema, max_iterations=3)
    assert len(results) == 2, f"Expected 2 iterations, got {len(results)}"
    final = llmaj_mod.merge_final_outputs(results)
    assert final["name"] == "Right Name", f"Expected corrected name, got {final['name']!r}"
    assert final["company"] == "Correct Co", f"Expected preserved company, got {final['company']!r}"
    print("PASS: test_merge_uses_most_recent_generation_per_field")

def test_loop_respects_max_iterations_even_if_judge_keeps_flagging():
    """If the judge never converges, the loop must still terminate at
    max_iterations and not run forever."""
    # Judge always flags "name" as wrong, for every call
    judge_sequence = [{"name": "False"} for _ in range(10)]
    gen_sequence = [{"name": f"attempt_{i}"} for i in range(10)]
    llmaj_mod.ollama.generate = make_fake_generate(judge_sequence, gen_sequence)

    schema = {"properties": {"name": {}}, "required": []}
    results = llmaj_mod.run_llmaj_loop("doc", fields_required=["name"], schema=schema, max_iterations=3)
    # initial run (run_num=0) + 3 correction iterations = 4 total
    assert len(results) == 4, f"Expected 4 iterations (1 initial + 3 max), got {len(results)}"
    print("PASS: test_loop_respects_max_iterations_even_if_judge_keeps_flagging")

if __name__ == "__main__":
    test_loop_runs_without_nameerror_on_first_pass()
    test_merge_uses_most_recent_generation_per_field()
    test_loop_respects_max_iterations_even_if_judge_keeps_flagging()
    print("\nAll tests passed.")