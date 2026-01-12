
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FunctionCaller:
    """
    Utilities for function calling with SLMs.
    """
    
    @staticmethod
    def format_prompt(tools: List[Dict], query: str) -> str:
        """
        Format tools and query into a prompt the model can understand.
        Uses a standard structure often used by function-calling models.
        """
        tool_desc = json.dumps(tools, indent=2)
        return (
            f"Attributes:\n{tool_desc}\n\n"
            f"Query: {query}\n"
            "Return a JSON object with 'name' and 'arguments' keys if a tool should be called, else plain text."
        )

    @staticmethod
    def parse_response(response: str) -> Dict[str, Any]:
        """
        Try to parse JSON from the response.
        Handles:
        - Plain JSON: {"name": "foo", ...}
        - Markdown JSON: ```json ... ```
        - JSON wrapped in text
        """
        try:
            # 1. Try to find JSON within markdown code blocks first
            import re
            code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
            match = re.search(code_block_pattern, response, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 2. Try to find the outermost JSON object
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
                
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
            
        return {"content": response}

def get_function_call(model_response: str) -> Dict[str, Any]:
    """Extract function call from text."""
    return FunctionCaller.parse_response(model_response)
