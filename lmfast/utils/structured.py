import json
import logging
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def generate_structured(
    model_generate_fn: Callable[[str], str],
    prompt: str,
    response_model: type[T],
    max_retries: int = 3,
) -> T | None:
    """
    Generate a response that conforms to a Pydantic model.

    Args:
        model_generate_fn: Function that accepts a prompt and returns a string.
        prompt: The user prompt.
        response_model: Pydantic model class defining the desired schema.
        max_retries: Number of retries on schema validation failure.

    Returns:
        Instance of response_model or None if failed.
    """
    schema = response_model.model_json_schema()

    # Construct system instruction
    instruction = (
        f"\nYou must output a valid JSON object matching this schema:\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```\n"
        f"Do not output any text other than the JSON object."
    )

    full_prompt = f"{prompt}\n{instruction}\nJSON Response:"

    for attempt in range(max_retries):
        try:
            response_text = model_generate_fn(full_prompt)

            # Use our robust parser from function_calling
            from lmfast.function_calling import FunctionCaller

            parsed_json = FunctionCaller.parse_response(response_text)

            # Validate with Pydantic
            if "content" in parsed_json and len(parsed_json) == 1:
                # It failed to parse as JSON or returned plain text
                raise ValueError("Model did not return JSON")

            obj = response_model.model_validate(parsed_json)
            return obj

        except Exception as e:
            logger.warning(f"Structured generation failed (attempt {attempt+1}): {e}")
            # Optional: Feedback to model could be added here

    return None
