
import logging
import sys
import os
import shutil

# Add project root to path
sys.path.append("/home/gaurav/small-idea")

from lmfast.rag import LightweightRAG, create_rag
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_cycle():
    logger.info("Testing RAG cycle...")
    
    # Mock model for CPU testing (since we might not have GPU or big model files)
    class MockModel:
        def generate(self, prompt, **kwargs):
            return f"Generated answer for prompt: {prompt[:20]}..."

    # Initialize RAG with mock model and CPU-friendly embedding model
    # using a very small embedding model for speed
    rag = LightweightRAG(
        model=MockModel(),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=100,
        chunk_overlap=10
    )
    
    # Add documents
    docs = [
        "LMFast is a Python library for training small language models.",
        "It supports quantization and LoRA fine-tuning.",
        "RAG stands for Retrieval-Augmented Generation.",
        "Colab T4 is a free GPU instance provided by Google."
    ]
    
    rag.add_documents(docs)
    logger.info(f"Added {len(docs)} documents.")
    
    # Test Retrieval
    query = "What is LMFast?"
    results = rag.retrieve(query, top_k=2)
    logger.info(f"Retrieval results for '{query}':")
    for r in results:
        logger.info(f" - {r[0]} (score: {r[1]:.4f})")
    
    assert len(results) > 0
    assert "LMFast" in results[0][0]
    
    # Test Retrieval with file
    with open("test_doc.txt", "w") as f:
        f.write("This is a test document from a file.\nIt contains information about file loading.")
    
    rag.add_file("test_doc.txt")
    results_file = rag.retrieve("file loading", top_k=1)
    assert len(results_file) > 0
    assert "file loading" in results_file[0][0]
    
    # Test Query (Generation)
    answer = rag.query("Tell me about LMFast")
    logger.info(f"Query answer: {answer}")
    assert "Generated answer" in answer
    
    # Test Save/Load
    save_dir = "./test_rag_index"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        
    rag.save(save_dir)
    logger.info(f"Saved RAG index to {save_dir}")
    
    rag_loaded = LightweightRAG(
        model=MockModel(),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    rag_loaded.load(save_dir)
    logger.info("Loaded RAG index")
    
    results_loaded = rag_loaded.retrieve("What is LMFast?", top_k=1)
    assert len(results_loaded) > 0
    assert "LMFast" in results_loaded[0][0]
    
    # Cleanup
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    if os.path.exists("test_doc.txt"):
        os.remove("test_doc.txt")
        
    logger.info("RAG Test Passed Successfully!")

if __name__ == "__main__":
    test_rag_cycle()
