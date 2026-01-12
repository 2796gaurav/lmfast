"""
Data loading and preprocessing utilities.

Provides efficient data loading and formatting for SLM training.
"""

import logging
from pathlib import Path
from typing import Any

from slmflow.core.config import ChatTemplate

logger = logging.getLogger(__name__)


# Chat templates for instruction formatting
CHAT_TEMPLATES = {
    ChatTemplate.ALPACA: """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}""",
    ChatTemplate.CHATML: """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>""",
    ChatTemplate.LLAMA2: """<s>[INST] {instruction} [/INST] {response}</s>""",
    ChatTemplate.MISTRAL: """<s>[INST] {instruction} [/INST]{response}</s>""",
    ChatTemplate.VICUNA: """USER: {instruction}
ASSISTANT: {response}""",
}


def load_dataset(
    source: str | Path | list[dict],
    *,
    split: str = "train",
    streaming: bool = False,
    **kwargs,
) -> Any:
    """
    Load a dataset from various sources.

    Supports:
    - HuggingFace Hub datasets
    - Local JSON/JSONL files
    - Local CSV files
    - Python list of dicts

    Args:
        source: Dataset source (HF ID, file path, or data)
        split: Dataset split to load
        streaming: Whether to stream the dataset
        **kwargs: Additional kwargs for datasets.load_dataset

    Returns:
        HuggingFace Dataset

    Example:
        >>> # From HuggingFace
        >>> ds = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
        >>>
        >>> # From local JSON
        >>> ds = load_dataset("./my_data.json")
        >>>
        >>> # From Python list
        >>> ds = load_dataset([{"text": "Example 1"}, {"text": "Example 2"}])
    """
    from datasets import Dataset
    from datasets import load_dataset as hf_load_dataset

    # Handle list of dicts
    if isinstance(source, list):
        logger.info(f"Loading dataset from list: {len(source)} samples")
        return Dataset.from_list(source)

    source_str = str(source)

    # Handle local files
    if Path(source_str).exists():
        logger.info(f"Loading dataset from local file: {source_str}")

        if source_str.endswith(".json"):
            return hf_load_dataset("json", data_files=source_str, split=split, **kwargs)
        elif source_str.endswith(".jsonl"):
            return hf_load_dataset("json", data_files=source_str, split=split, **kwargs)
        elif source_str.endswith(".csv"):
            return hf_load_dataset("csv", data_files=source_str, split=split, **kwargs)
        elif Path(source_str).is_dir():
            # Assume it's a dataset directory
            return hf_load_dataset(source_str, split=split, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {source_str}")

    # Handle HuggingFace Hub
    logger.info(f"Loading dataset from HuggingFace Hub: {source_str}")
    return hf_load_dataset(source_str, split=split, streaming=streaming, **kwargs)


def format_chat_template(
    example: dict,
    template: ChatTemplate = ChatTemplate.ALPACA,
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
) -> dict:
    """
    Format a single example using a chat template.

    Args:
        example: Dataset example
        template: Chat template to use
        instruction_field: Field containing the instruction
        input_field: Field containing additional input
        output_field: Field containing the output

    Returns:
        Example with "text" field added
    """
    instruction = example.get(instruction_field, "")
    input_text = example.get(input_field, "")
    output = example.get(output_field, "")

    # Combine instruction and input if present
    if input_text:
        full_instruction = f"{instruction}\n\n{input_text}"
    else:
        full_instruction = instruction

    # Get template
    template_str = CHAT_TEMPLATES.get(template, CHAT_TEMPLATES[ChatTemplate.ALPACA])

    # Format
    text = template_str.format(instruction=full_instruction, response=output)

    return {"text": text}


def prepare_dataset(
    dataset: Any,
    tokenizer: Any,
    *,
    text_field: str = "text",
    max_seq_length: int = 2048,
    template: ChatTemplate | None = None,
    instruction_field: str = "instruction",
    output_field: str = "output",
    input_field: str = "input",
    num_proc: int = 4,
) -> Any:
    """
    Prepare a dataset for training.

    Handles:
    - Applying chat templates if needed
    - Ensuring text field exists
    - Filtering by length

    Args:
        dataset: Input dataset
        tokenizer: Tokenizer for length calculation
        text_field: Field containing text (created if template used)
        max_seq_length: Maximum sequence length
        template: Chat template to apply
        instruction_field: Instruction field name
        output_field: Output field name
        input_field: Input field name
        num_proc: Number of processes for mapping

    Returns:
        Prepared dataset
    """
    logger.info(f"Preparing dataset: {len(dataset)} samples")

    # Apply chat template if specified
    if template is not None:
        logger.info(f"Applying chat template: {template}")

        def apply_template(example):
            return format_chat_template(
                example,
                template=template,
                instruction_field=instruction_field,
                input_field=input_field,
                output_field=output_field,
            )

        dataset = dataset.map(
            apply_template,
            num_proc=num_proc,
            desc="Applying template",
        )
        text_field = "text"

    # Check if text field exists
    if text_field not in dataset.column_names:
        # Try to create from instruction/output format
        if instruction_field in dataset.column_names:
            logger.info(f"Creating '{text_field}' from instruction format")
            dataset = dataset.map(
                lambda x: format_chat_template(x, ChatTemplate.ALPACA),
                num_proc=num_proc,
            )
        else:
            raise ValueError(
                f"Text field '{text_field}' not found. " f"Available: {dataset.column_names}"
            )

    # Filter by length (approximate)
    def is_valid_length(example):
        text = example[text_field]
        # Rough estimate: 4 chars per token
        estimated_tokens = len(text) / 4
        return estimated_tokens <= max_seq_length * 1.5

    original_len = len(dataset)
    dataset = dataset.filter(is_valid_length, num_proc=num_proc)
    filtered_len = len(dataset)

    if filtered_len < original_len:
        logger.info(f"Filtered {original_len - filtered_len} samples exceeding max length")

    logger.info(f"Dataset prepared: {len(dataset)} samples")
    return dataset


class DataCollator:
    """
    Data collator for SLM training.

    Handles tokenization and padding for causal language modeling.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_seq_length: int = 2048,
        text_field: str = "text",
        mlm: bool = False,
    ):
        """
        Initialize the data collator.

        Args:
            tokenizer: Tokenizer to use
            max_seq_length: Maximum sequence length
            text_field: Field containing text
            mlm: Whether to use masked language modeling
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_field = text_field
        self.mlm = mlm

    def __call__(self, examples: list[dict]) -> dict:
        """Collate examples into a batch."""
        # Get texts
        texts = [ex[self.text_field] for ex in examples]

        # Tokenize
        batch = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

        # For causal LM, labels = input_ids
        batch["labels"] = batch["input_ids"].clone()

        return batch
