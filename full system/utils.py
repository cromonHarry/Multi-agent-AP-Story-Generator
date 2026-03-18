import json
import re

def parse_json_response(gpt_output: str) -> dict:
    result_str = gpt_output.strip()
    # GPT sometimes wraps JSON in markdown code blocks, so strip them if present
    if result_str.startswith("```") and result_str.endswith("```"):
        result_str = re.sub(r'^```[^\n]*\n', '', result_str)
        result_str = re.sub(r'\n```$', '', result_str)
        result_str = result_str.strip()
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        print(f"JSON Parsing Error. Raw output:\n{result_str}")
        return {}
