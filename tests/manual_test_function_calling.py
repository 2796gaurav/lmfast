import logging
import unittest

from lmfast.agents.core import Agent
from lmfast.function_calling import FunctionCaller

# Configure logging
logging.basicConfig(level=logging.INFO)


class TestFunctionCalling(unittest.TestCase):
    def setUp(self):
        # Define a tool
        def calculator(expression: str) -> str:
            return str(eval(expression))

        self.tools = [calculator]
        self.mock_responses = []
        self.response_index = 0

    def mock_generate_fn(self, prompt: str) -> str:
        if self.response_index < len(self.mock_responses):
            resp = self.mock_responses[self.response_index]
            self.response_index += 1
            return resp
        return "Final Answer"

    def test_plain_json_parsing(self):
        response = '{"tool": "calculator", "args": {"expression": "1+1"}}'
        parsed = FunctionCaller.parse_response(response)
        self.assertEqual(parsed["tool"], "calculator")
        self.assertEqual(parsed["args"]["expression"], "1+1")

    def test_markdown_json_parsing(self):
        response = '```json\n{"tool": "calculator", "args": {"expression": "2+2"}}\n```'
        parsed = FunctionCaller.parse_response(response)
        self.assertEqual(parsed["tool"], "calculator")
        self.assertEqual(parsed["args"]["expression"], "2+2")

    def test_agent_loop(self):
        # Scenario: User asks "Calc 3+3", Model calls tool, Tool returns 6, Model says "It is 6"
        self.mock_responses = [
            '```json\n{"tool": "calculator", "args": {"expression": "3+3"}}\n```',  # Turn 1
            "The answer is 6.",  # Turn 2
        ]

        agent = Agent(self.mock_generate_fn, tools=self.tools)
        final = agent.run("What is 3+3?")

        # Verify history
        # 0: User input
        # 1: Assistant tool call
        # 2: System tool output
        # 3: Assistant final answer
        self.assertEqual(len(agent.history), 4)
        self.assertIn("Tool Output: 6", agent.history[2]["content"])
        self.assertEqual(final, "The answer is 6.")


if __name__ == "__main__":
    unittest.main()
