"""
Vector Store implementation using Pinecone for semantic search.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config.settings import get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Pinecone-based vector store for semantic search operations.
    """
    
    def __init__(self):
        """Initialize the vector store with Pinecone configuration."""
        self.settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.settings.openai_api_key,
            model="text-embedding-ada-002"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize Pinecone
        self._initialize_pinecone()
        
    def _initialize_pinecone(self) -> None:
        """Initialize Pinecone client and index."""
        try:
            pinecone.init(
                api_key=self.settings.pinecone_api_key,
                environment=self.settings.pinecone_environment
            )
            
            # Check if index exists, create if not
            if self.settings.pinecone_index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {self.settings.pinecone_index_name}")
                pinecone.create_index(
                    name=self.settings.pinecone_index_name,
                    dimension=self.settings.embedding_dimension,
                    metric="cosine"
                )
            
            # Connect to the index
            self.index = pinecone.Index(self.settings.pinecone_index_name)
            self.vectorstore = Pinecone(
                index=self.index,
                embedding=self.embeddings,
                text_key="text"
            )
            
            logger.info("Pinecone vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts to add
            metadata: Optional metadata for each document
            
        Returns:
            List of document IDs
        """
        try:
            # Split documents into chunks
            doc_chunks = []
            for i, doc in enumerate(documents):
                chunks = self.text_splitter.split_text(doc)
                for j, chunk in enumerate(chunks):
                    doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                    doc_metadata.update({"chunk_id": j, "doc_id": i})
                    doc_chunks.append(Document(page_content=chunk, metadata=doc_metadata))
            
            # Add to vector store
            ids = self.vectorstore.add_documents(doc_chunks)
            logger.info(f"Added {len(doc_chunks)} document chunks to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        try:
            k = k or self.settings.top_k_results
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of tuples containing documents and their similarity scores
        """
        try:
            k = k or self.settings.top_k_results
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search with score failed: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from vector store")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dictionary containing index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            logger.info("Retrieved index statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise
