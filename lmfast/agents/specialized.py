
from lmfast.agents.core import Agent
import subprocess
import sys

def execute_python(code: str) -> str:
    """Execute python code and return stdout."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)

class CodeAgent(Agent):
    """
    Agent specialized for writing and executing code.
    """
    def __init__(self, model_generate_fn):
        super().__init__(
            model_generate_fn,
            tools=[execute_python],
            system_prompt=(
                "You are an expert Python coding assistant."
                "You can write and execute code using the 'execute_python' tool."
                "Always check your code before running."
            )
        )

class DataAgent(Agent):
    """
    Agent specialized for data analysis.
    """
    def __init__(self, model_generate_fn, df=None):
        # We could inject dataframe context here
        super().__init__(
            model_generate_fn,
            tools=[execute_python],
            system_prompt=(
                "You are a Data Scientist."
                "You analyze data by writing Python code."
                "Assume pandas is available."
            )
        )
