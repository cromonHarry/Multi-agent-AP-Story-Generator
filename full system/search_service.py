from openai import OpenAI
from tavily import TavilyClient
from config import SYSTEM_PROMPT, AP_MODEL_STRUCTURE
import time

class SearchService:
    def __init__(self, openai_key, tavily_key):
        self.client = OpenAI(api_key=openai_key)
        self.tavily_client = TavilyClient(api_key=tavily_key)

    def generate_question(self, start_node: str, target_node: str, tech_topic: str, era_context: str) -> str:
        """
        Generates a search query to find the relationship between start_node and target_node
        in the context of the specific technology and era.
        """
        # Find the arrow definition that connects these two nodes
        arrow_name = None
        arrow_desc = ""
        for name, info in AP_MODEL_STRUCTURE["arrows"].items():
            if info["from"] == start_node and info["to"] == target_node:
                arrow_name = name
                arrow_desc = info["description"]
                break
        
        if not arrow_name:
            # Fallback if direct arrow not found (though it should be based on logic)
            arrow_desc = f"relationship between {start_node} and {target_node}"

        prompt = f"""
You are a query generator. Generate ONE specific, high-quality search query to investigate the AP Model relationship:
"{arrow_name}" (from "{start_node}" to "{target_node}").
Description: {arrow_desc}

Context:
- Detailed description: {tech_topic}
- Time Period: {era_context}

The query should be designed to find real-world examples or factual data confirming this relationship in that specific era.
Output ONLY the query string. No quotes, no explanations.
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def search_tavily(self, query: str) -> str:
        """Executes the search and returns the answer/content."""
        try:
            # Add delay to avoid rate limits if running rapidly
            time.sleep(0.5) 
            response = self.tavily_client.search(query=query, include_answer="advanced", search_depth="advanced", max_results=10)
            answer = response.get('answer', '')
            if answer: 
                return answer
            results = response.get('results', [])
            if results:
                return results[0].get('content', "No detailed information found.")
            return "No information found."
        except Exception as e:
            return f"Search failed: {str(e)}"

    def synthesize_node_data(self, start_node, target_node, arrow_name, search_result) -> dict:
        """
        Uses GPT to convert the raw search result into the structured AP node/arrow format.
        We need to extract data for the Arrow and potentially the Target Node.
        """
        prompt = f"""
Based on the search result below, summarize the findings for the AP Model Arrow "{arrow_name}" 
(which goes from "{start_node}" to "{target_node}").

Search Result:
{search_result}

Output in valid JSON format:
{{
    "arrow_type": "{arrow_name}",
    "definition": "Concise explanation of how this connection works in this context (max 30 words)",
    "example": "Specific real-world example found in the text",
    "target_node_content": "Content derived for the object '{target_node}' based on this relationship"
}}
"""
        from utils import parse_json_response
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return parse_json_response(response.choices[0].message.content)