import logging
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel

from lmfast.function_calling import FunctionCaller

logger = logging.getLogger(__name__)


class Tool(BaseModel):
    """
    A tool that an agent can extract and use.
    Wraps a python function.
    """

    name: str
    description: str
    func: Callable
    parameters: dict[str, Any]

    @classmethod
    def from_function(cls, func: Callable) -> "Tool":
        """Create a Tool from a python function."""
        name = func.__name__
        description = func.__doc__ or "No description provided."

        # Simple type hint extraction (could be more robust with inspect)
        type_hints = get_type_hints(func)
        params = {}
        for k, v in type_hints.items():
            if k != "return":
                params[k] = str(v)

        return cls(name=name, description=description, func=func, parameters=params)

    def run(self, **kwargs) -> Any:
        try:
            return self.func(**kwargs)
        except Exception as e:
            return f"Error executing tool {self.name}: {e}"


class Agent:
    """
    Base Agent class.

    Interacts with an LLM (passed as a generate function or wrapper)
    and can execute tools.
    """

    def __init__(
        self,
        model_generate_fn: Callable[[str], str],
        tools: list[Callable] | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.generate_fn = model_generate_fn
        self.tools = {t.__name__: Tool.from_function(t) for t in (tools or [])}
        self.system_prompt = system_prompt
        self.history = []

    def _build_prompt(self, user_input: str) -> str:
        """Construct the prompt with tool definitions."""
        tool_desc = "\n".join(
            [f"- {t.name}: {t.description} Params: {t.parameters}" for t in self.tools.values()]
        )

        prompt = f"{self.system_prompt}\n\n"
        if self.tools:
            prompt += f"Available Tools:\n{tool_desc}\n\n"
            prompt += "To use a tool, you MUST respond with a JSON object:\n"
            prompt += '```json\n{"tool": "tool_name", "args": {"arg1": "value"}}\n```\n'
            prompt += "If no tool is needed, just respond with plain text.\n\n"

        for msg in self.history[-5:]:  # Keep short history context
            prompt += f"{msg['role']}: {msg['content']}\n"

        prompt += f"User: {user_input}\nAssistant:"
        return prompt

    def run(self, user_input: str) -> str:
        """Run the agent loop."""
        self.history.append({"role": "User", "content": user_input})

        prompt = self._build_prompt(user_input)

        # Max turns to prevent infinite loops
        max_turns = 5
        current_turn = 0

        while current_turn < max_turns:
            # Generate response
            response = self.generate_fn(prompt)

            # Check for tool call
            parsed = FunctionCaller.parse_response(response)

            # If it's a tool call
            if "tool" in parsed and parsed["tool"] in self.tools:
                tool_name = parsed["tool"]
                tool_args = parsed.get("args", {})

                logger.info(f"Agent calling tool: {tool_name} with {tool_args}")

                # Execute tool
                try:
                    result = self.tools[tool_name].run(**tool_args)
                except Exception as e:
                    result = f"Error: {str(e)}"

                # Update history and prompt
                observation = f"Tool Output: {result}"

                # Append to history so the model sees what happened
                self.history.append({"role": "Assistant", "content": response})
                self.history.append({"role": "System", "content": observation})

                prompt += f"\n{response}\nObservation: {observation}\nAssistant:"
                current_turn += 1
                continue

            else:
                # legitimate final response
                self.history.append({"role": "Assistant", "content": response})
                return response

        return "Agent reached maximum turns without a final answer."
