import json
import logging
import sys
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Mock lmfast.setup_colab_env to do nothing
import lmfast

lmfast.setup_colab_env = MagicMock()


def run_notebook(path):
    logging.info(f"Running {path}...")
    try:
        with open(path) as f:
            nb = json.load(f)

        code_cells = [c["source"] for c in nb.get("cells", []) if c["cell_type"] == "code"]
        full_code = ""

        for cell in code_cells:
            source = "".join(cell)
            # Filter out magics and pip installs
            clean_lines = [
                l for l in source.splitlines() if not l.strip().startswith(("!", "%", "pip"))
            ]
            full_code += "\n".join(clean_lines) + "\n\n"

        # Execute
        exec(full_code, {"__name__": "__main__"})
        logging.info(f"Successfully executed {path}")
        return True

    except Exception as e:
        logging.error(f"Failed to execute {path}: {e}")
        return False


if __name__ == "__main__":
    notebooks_to_test = [
        "/home/gaurav/small-idea/examples/17_function_calling_agent.ipynb",
        # We can try others, but they might require downloading models which is slow.
        # "examples/10_reasoning_agents.ipynb"
    ]

    failed = []
    for nb in notebooks_to_test:
        if not run_notebook(nb):
            failed.append(nb)

    if failed:
        sys.exit(1)
