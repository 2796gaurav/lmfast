
import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 LMFast: Democratized LLM Development on Colab T4\n",
    "\n",
    "Welcome to the official **LMFast** tutorial. In this notebook, you will learn how to:\n",
    "1. Setup a T4 GPU environment in seconds.\n",
    "2. Fine-tune a Small Language Model (SmolLM-135M) on your own data.\n",
    "3. Serve the model with an OpenAI-compatible API.\n",
    "4. Export for running locally with MCP (Model Context Protocol)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup\n",
    "First, we install LMFast and setup the environment."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install lmfast[all] mcp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import lmfast\n",
    "\n",
    "# Magic setup for Colab T4 (mounts drive, checks GPU, installs system deps)\n",
    "lmfast.setup_colab_env()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Train a Model\n",
    "We will fine-tune `HuggingFaceTB/SmolLM-135M` on a sample dataset. This takes < 15 mins on T4."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# One-line training\n",
    "trainer = lmfast.train(\n",
    "    model=\"HuggingFaceTB/SmolLM-135M\",\n",
    "    dataset=\"yahma/alpaca-cleaned\",\n",
    "    output_dir=\"./my_smollm\",\n",
    "    max_steps=100  # Set to 500+ for real training\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Inference & Serving\n",
    "Test the model immediately."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Quick generation attempt\n",
    "response = trainer.generate(\"Explain quantum computing in one sentence.\")\n",
    "print(response)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Start a temporary API server (OpenAI compatible)\n",
    "# You can use ngrok to expose this to the world if needed\n",
    "lmfast.serve(\"./my_smollm\", port=8000)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. MCP Integration (Model Context Protocol)\n",
    "LMFast supports MCP out of the box. You can download this model and run it locally with:\n",
    "\n",
    "```bash\n",
    "pip install lmfast[mcp]\n",
    "lmfast serve --model ./my_smollm --mcp\n",
    "```\n",
    "\n",
    "Then connect it to Claude Desktop or Cursor!"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open("examples/lmfast_colab.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook created at examples/lmfast_colab.ipynb")
