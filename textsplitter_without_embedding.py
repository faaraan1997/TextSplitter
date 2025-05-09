import os
import logging
from typing import List, Generator
from chunking.cluster_semantic_chunker import ClusterSemanticChunker
from docling.datamodel.base_models import Page, SplitPage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger("semantic_text_splitter")

# Azure OpenAI setup
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2025-03-01-preview"

from openai import AzureOpenAI
embedding_function = AzureOpenAIEmbeddingFunction(
    api_key=subscription_key,
    api_version=api_version,
    azure_endpoint=endpoint,
    model="text-embedding-3-large"
)

class SemanticTextSplitter:
    """
    Drop-in replacement for old TextSplitter that uses semantic clustering.
    Returns SplitPage objects like the original implementation.
    Does NOT perform embedding or storage — keeps logic pure.
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

        for chunk in chunks:
            yield SplitPage(page_num=0, text=chunk)
