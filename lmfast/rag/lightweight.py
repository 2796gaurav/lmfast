"""
Lightweight RAG (Retrieval-Augmented Generation) for Small Language Models.

Designed for:
- Low memory footprint
- Colab T4 compatibility
- Simple, intuitive API
- Integration with LMFast inference

Example:
    >>> from lmfast.rag import LightweightRAG
    >>> rag = LightweightRAG("./my_model")
    >>> rag.add_documents(["Document 1 content", "Document 2 content"])
    >>> answer = rag.query("What is in document 1?")
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SimpleEmbedder:
    """
    Lightweight embedder using sentence-transformers.

    Falls back to TF-IDF if sentence-transformers unavailable.
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.use_tfidf = False
        self.tfidf_vectorizer = None

        self._load_model()

    def _load_model(self):
        """Load embedding model with fallback to TF-IDF."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded SentenceTransformer: {self.model_name}")

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to TF-IDF. Install with: pip install sentence-transformers"
            )
            self.use_tfidf = True
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.tfidf_vectorizer = TfidfVectorizer(max_features=384)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.use_tfidf:
            if not hasattr(self.tfidf_vectorizer, "vocabulary_"):
                # First time - fit and transform
                embeddings = self.tfidf_vectorizer.fit_transform(texts).toarray()
            else:
                embeddings = self.tfidf_vectorizer.transform(texts).toarray()
            return embeddings.astype(np.float32)
        else:
            return self.model.encode(texts, convert_to_numpy=True)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.use_tfidf:
            return self.tfidf_vectorizer.max_features or 384
        return self.model.get_sentence_embedding_dimension()


class SimpleIndex:
    """
    Simple vector index using numpy or FAISS.

    Uses numpy for small collections, FAISS for larger ones.
    """

    def __init__(self, embedding_dim: int, use_faiss: bool = True):
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss
        self.index = None
        self.embeddings = []

        if use_faiss:
            try:
                import faiss

                self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product
                logger.info("Using FAISS index")
            except ImportError:
                logger.warning(
                    "FAISS not installed. Using numpy. Install with: pip install faiss-cpu"
                )
                self.use_faiss = False

    def add(self, embeddings: np.ndarray):
        """Add embeddings to index."""
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)

        if self.use_faiss:
            self.index.add(normalized.astype(np.float32))
        else:
            self.embeddings.append(normalized)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        normalized_query = query_embedding / (norm + 1e-10)

        if self.use_faiss:
            scores, indices = self.index.search(
                normalized_query.reshape(1, -1).astype(np.float32), k
            )
            return scores[0], indices[0]
        else:
            # Numpy fallback
            all_embeddings = np.vstack(self.embeddings)
            scores = np.dot(all_embeddings, normalized_query.flatten())
            top_k_indices = np.argsort(scores)[-k:][::-1]
            return scores[top_k_indices], top_k_indices

    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        if self.use_faiss:
            return self.index.ntotal
        return sum(e.shape[0] for e in self.embeddings) if self.embeddings else 0


class LightweightRAG:
    """
    Minimal RAG for SLMs.

    Designed for:
    - Low memory footprint
    - Colab T4 compatibility
    - Simple API

    Example:
        >>> from lmfast.rag import LightweightRAG
        >>> from lmfast.inference import SLMServer
        >>>
        >>> # Create RAG with model
        >>> rag = LightweightRAG("./my_model")
        >>>
        >>> # Add documents
        >>> rag.add_documents([
        ...     "LMFast is a framework for training small language models.",
        ...     "RAG combines retrieval with generation for better answers."
        ... ])
        >>>
        >>> # Query
        >>> answer = rag.query("What is LMFast?")
        >>> print(answer)
    """

    def __init__(
        self,
        model: str | Any,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 3,
        use_faiss: bool = True,
    ):
        """
        Initialize RAG system.

        Args:
            model: LMFast model path, SLMServer instance, or generate function
            embedding_model: Sentence transformer model for embeddings
            chunk_size: Size of document chunks in characters
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
            use_faiss: Use FAISS for indexing (faster, requires install)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Initialize embedder
        self.embedder = SimpleEmbedder(embedding_model)

        # Initialize index
        self.index = SimpleIndex(self.embedder.embedding_dim, use_faiss)

        # Store documents
        self.documents: list[str] = []
        self.metadata: list[dict] = []

        # Initialize model
        self._init_model(model)

        logger.info(f"LightweightRAG initialized with {embedding_model}")

    def _init_model(self, model: str | Any):
        """Initialize the generation model."""
        if callable(model):
            # It's already a generate function
            self.generate_fn = model
            self.model = None
        elif isinstance(model, str):
            # It's a path - load SLMServer
            from lmfast.inference.server import SLMServer

            self.model = SLMServer(model)
            self.generate_fn = lambda p, **kw: self.model.generate(p, **kw)
        else:
            # Assume it's an SLMServer or similar
            self.model = model
            self.generate_fn = lambda p, **kw: model.generate(p, **kw)

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                # Find last period or newline
                last_break = max(
                    chunk.rfind("."), chunk.rfind("\n"), chunk.rfind("!"), chunk.rfind("?")
                )
                if last_break > self.chunk_size // 2:
                    chunk = chunk[: last_break + 1]
                    end = start + last_break + 1

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]  # Remove empty chunks

    def add_documents(
        self, documents: list[str], metadata: list[dict] | None = None, chunk: bool = True
    ) -> int:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            chunk: Whether to chunk documents (default True)

        Returns:
            Number of chunks added
        """
        if metadata is None:
            metadata = [{"doc_id": i} for i in range(len(documents))]

        all_chunks = []
        all_metadata = []

        for i, doc in enumerate(documents):
            if chunk:
                chunks = self._chunk_text(doc)
            else:
                chunks = [doc]

            for j, c in enumerate(chunks):
                all_chunks.append(c)
                all_metadata.append({**metadata[i], "chunk_id": j, "source_doc": i})

        # Generate embeddings
        embeddings = self.embedder.encode(all_chunks)

        # Add to index
        self.index.add(embeddings)

        # Store documents
        self.documents.extend(all_chunks)
        self.metadata.extend(all_metadata)

        logger.info(f"Added {len(all_chunks)} chunks from {len(documents)} documents")
        return len(all_chunks)

    def add_file(self, file_path: str, **kwargs) -> int:
        """
        Add a file to the knowledge base.

        Supports: .txt, .md, .pdf (requires pypdf)
        """
        path = Path(file_path)

        if path.suffix.lower() in [".txt", ".md"]:
            content = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            content = self._read_pdf(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        return self.add_documents([content], metadata=[{"source": str(path)}], **kwargs)

    def _read_pdf(self, path: Path) -> str:
        """Read PDF file."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("pypdf required for PDF files. Install with: pip install pypdf")

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float, dict]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results (default: self.top_k)

        Returns:
            List of (chunk_text, score, metadata) tuples
        """
        k = top_k or self.top_k

        # Embed query
        query_embedding = self.embedder.encode([query])[0]

        # Search
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores, indices):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score), self.metadata[idx]))

        return results

    def query(
        self,
        question: str,
        top_k: int | None = None,
        max_new_tokens: int = 256,
        return_sources: bool = False,
        system_prompt: str | None = None,
    ) -> str | tuple[str, list[str]]:
        """
        Query the RAG system.

        Args:
            question: User question
            top_k: Number of chunks to retrieve
            max_new_tokens: Max tokens to generate
            return_sources: Whether to return source chunks
            system_prompt: Optional custom system prompt

        Returns:
            Generated answer, optionally with source chunks
        """
        # Retrieve relevant chunks
        results = self.retrieve(question, top_k)
        context_chunks = [r[0] for r in results]
        context = "\n\n---\n\n".join(context_chunks)

        # Build prompt
        if system_prompt is None:
            system_prompt = "Answer the question based on the provided context. If the context doesn't contain the answer, say so."

        prompt = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""

        # Generate answer
        answer = self.generate_fn(prompt, max_new_tokens=max_new_tokens)

        if return_sources:
            return answer, context_chunks
        return answer

    @classmethod
    def from_files(
        cls, model: str | Any, file_paths: list[str], **kwargs
    ) -> "LightweightRAG":
        """
        Create RAG from files.

        Args:
            model: Model path or instance
            file_paths: List of file paths to index
            **kwargs: Additional arguments for LightweightRAG

        Returns:
            Initialized LightweightRAG with files indexed
        """
        rag = cls(model, **kwargs)

        for path in file_paths:
            rag.add_file(path)

        return rag

    @classmethod
    def from_directory(
        cls,
        model: str | Any,
        directory: str,
        extensions: list[str] = [".txt", ".md"],
        recursive: bool = True,
        **kwargs,
    ) -> "LightweightRAG":
        """
        Create RAG from all files in a directory.

        Args:
            model: Model path or instance
            directory: Directory path
            extensions: File extensions to include
            recursive: Search recursively
            **kwargs: Additional arguments for LightweightRAG

        Returns:
            Initialized LightweightRAG with directory indexed
        """
        dir_path = Path(directory)

        if recursive:
            files = []
            for ext in extensions:
                files.extend(dir_path.rglob(f"*{ext}"))
        else:
            files = []
            for ext in extensions:
                files.extend(dir_path.glob(f"*{ext}"))

        return cls.from_files(model, [str(f) for f in files], **kwargs)

    def save(self, path: str):
        """Save RAG state to disk."""
        import json
        import pickle

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save documents and metadata
        with open(save_path / "documents.json", "w") as f:
            json.dump(
                {
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "config": {
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "top_k": self.top_k,
                    },
                },
                f,
            )

        # Save embeddings
        if self.index.use_faiss:
            import faiss

            faiss.write_index(self.index.index, str(save_path / "index.faiss"))
        else:
            with open(save_path / "embeddings.pkl", "wb") as f:
                pickle.dump(self.index.embeddings, f)

        logger.info(f"RAG saved to {path}")

    def load(self, path: str):
        """Load RAG state from disk."""
        import json
        import pickle

        load_path = Path(path)

        # Load documents and metadata
        with open(load_path / "documents.json") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]

        # Load embeddings
        if (load_path / "index.faiss").exists():
            import faiss

            self.index.index = faiss.read_index(str(load_path / "index.faiss"))
            self.index.use_faiss = True
        else:
            with open(load_path / "embeddings.pkl", "rb") as f:
                self.index.embeddings = pickle.load(f)
                self.index.use_faiss = False

        logger.info(f"RAG loaded from {path}")


# Convenience function
def create_rag(
    model: str | Any,
    documents: list[str] | None = None,
    files: list[str] | None = None,
    directory: str | None = None,
    **kwargs,
) -> LightweightRAG:
    """
    Create a RAG system with one function call.

    Args:
        model: Model path or instance
        documents: List of document strings
        files: List of file paths
        directory: Directory to index
        **kwargs: Additional RAG arguments

    Returns:
        Initialized LightweightRAG

    Example:
        >>> rag = create_rag(
        ...     model="./my_model",
        ...     documents=["Doc 1", "Doc 2"],
        ...     top_k=5
        ... )
    """
    if directory:
        return LightweightRAG.from_directory(model, directory, **kwargs)
    elif files:
        return LightweightRAG.from_files(model, files, **kwargs)
    else:
        rag = LightweightRAG(model, **kwargs)
        if documents:
            rag.add_documents(documents)
        return rag
