import json
import re

def parse_json_response(gpt_output: str) -> dict:
    """Helper to parse JSON from GPT output which might include markdown code blocks."""
    result_str = gpt_output.strip()
    if result_str.startswith("```") and result_str.endswith("```"):
        # Remove opening ```json or ``` and closing ```
        result_str = re.sub(r'^```[^\n]*\n', '', result_str)
        result_str = re.sub(r'\n```$', '', result_str)
        result_str = result_str.strip()
    try:
        return json.loads(result_str)
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error. Raw output:\n{result_str}")
        # Return empty structure or re-raise depending on strictness
        return {}