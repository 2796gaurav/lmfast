import ast
import glob
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def audit_notebook(path):
    logging.info(f"Auditing {path}...")
    try:
        with open(path) as f:
            nb = json.load(f)
    except Exception as e:
        logging.error(f"  Failed to parse JSON: {e}")
        return False

    code_cells = [c["source"] for c in nb.get("cells", []) if c["cell_type"] == "code"]
    full_code = ""

    for cell in code_cells:
        # Join lines in cell
        cell_source = "".join(cell)
        full_code += cell_source + "\n\n"

    # Check for syntax errors
    try:
        ast.parse(full_code)
        logging.info("  Syntax OK")
        return True
    except SyntaxError:
        # Ignore magic commands commonly found in colab (lines starting with ! or %)
        lines = full_code.splitlines()
        clean_lines = [
            line for line in lines if not line.strip().startswith(("!", "%", "pip", "cd "))
        ]
        clean_code = "\n".join(clean_lines)
        try:
            ast.parse(clean_code)
            logging.info("  Syntax OK (after ignoring magics)")
            return True
        except SyntaxError as e2:
            logging.error(f"  Syntax Error: {e2}")
            return False


if __name__ == "__main__":
    notebooks = sorted(glob.glob("/home/gaurav/small-idea/examples/*.ipynb"))
    failed = []
    for nb in notebooks:
        if not audit_notebook(nb):
            failed.append(nb)

    if failed:
        logging.error(f"Failed notebooks: {failed}")
        exit(1)
    else:
        logging.info("All notebooks passed syntax check.")
