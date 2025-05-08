import os
import logging
from typing import List, Generator
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chunking.cluster_semantic_chunker import ClusterSemanticChunker
from docling.datamodel.base_models import Page, SplitPage
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger("semantic_text_splitter")

# Azure OpenAI setup
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2025-03-01-preview"

embedding_function = AzureOpenAIEmbeddingFunction(
    api_key=subscription_key,
    api_version=api_version,
    azure_endpoint=endpoint,
    model="text-embedding-3-large"
)

# Initialize ChromaDB client and collection
chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
client = Client(Settings(persist_directory=chroma_path))
collection = client.get_or_create_collection("rag_chunks")

class SemanticTextSplitter:
    """
    Drop-in replacement for old TextSplitter that uses semantic clustering and ChromaDB.
    Returns SplitPage objects like the original implementation.
    """
    def __init__(self, max_chunk_size: int = 500):
        self.chunker = ClusterSemanticChunker(
            embedding_function=embedding_function,
            max_chunk_size=max_chunk_size
        )

    def split_pages(self, pages: List[Page]) -> Generator[SplitPage, None, None]:
        """
        Split pages using semantic chunking, return generator of SplitPage.

        Args:
            pages (List[Page]): Input document pages.

        Yields:
            SplitPage: Semantically split chunks with page_num and text.
        """
        full_text = "".join(page.text for page in pages)

        if not full_text.strip():
            logger.warning("No text found in pages.")
            return

        chunks = self.chunker.split_text(full_text)
        embeddings = [self.chunker.embedding_function(chunk)[0] for chunk in chunks]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )

        for chunk in chunks:
            yield SplitPage(page_num=0, text=chunk)

    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve most similar chunks from ChromaDB given a query.

        Args:
            query (str): The input query string.
            top_k (int): Number of top similar results to retrieve.

        Returns:
            List[str]: List of retrieved text chunks.
        """
        query_embedding = self.chunker.embedding_function(query)[0]
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)
        return results['documents']
