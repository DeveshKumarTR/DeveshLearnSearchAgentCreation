"""
Main Search Agent implementation integrating vector store and prompt pipeline.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain.schema import Document

from .vector_store import VectorStore
from .prompt_pipeline import PromptPipeline
from .utils import setup_logging, format_search_results, extract_keywords
from config.settings import get_settings, validate_settings

logger = logging.getLogger(__name__)


class SearchAgent:
    """
    Main Search Agent class that orchestrates vector search and prompt processing.
    """
    
    def __init__(
        self, 
        use_chat_model: bool = True,
        streaming: bool = False,
        auto_expand_queries: bool = True
    ):
        """
        Initialize the Search Agent.
        
        Args:
            use_chat_model: Whether to use ChatOpenAI for better conversation handling
            streaming: Whether to enable streaming responses
            auto_expand_queries: Whether to automatically expand search queries
        """
        # Setup logging
        settings = get_settings()
        setup_logging(settings.log_level)
        
        # Validate configuration
        if not validate_settings():
            raise ValueError("Invalid configuration. Please check your environment variables.")
        
        # Initialize components
        self.settings = settings
        self.auto_expand_queries = auto_expand_queries
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        self.vector_store = VectorStore()
        
        # Initialize prompt pipeline
        logger.info("Initializing prompt pipeline...")
        self.prompt_pipeline = PromptPipeline(
            use_chat_model=use_chat_model,
            streaming=streaming
        )
        
        # Create default search chain
        self._setup_default_chains()
        
        logger.info("Search Agent initialized successfully")
    
    def _setup_default_chains(self) -> None:
        """Set up default prompt chains for search operations."""
        try:
            # Create search chain
            if self.prompt_pipeline.use_chat_model:
                search_prompt = self.prompt_pipeline.create_chat_search_prompt()
            else:
                search_prompt = self.prompt_pipeline.create_search_prompt()
            
            self.prompt_pipeline.create_chain(
                chain_name="search",
                prompt=search_prompt,
                output_key="answer"
            )
            
            logger.info("Default search chains created")
            
        except Exception as e:
            logger.error(f"Failed to setup default chains: {e}")
            raise
    
    def add_documents(
        self, 
        documents: List[str], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            
        Returns:
            List of document IDs
        """
        try:
            logger.info(f"Adding {len(documents)} documents to knowledge base")
            
            # Add documents to vector store
            doc_ids = self.vector_store.add_documents(documents, metadata)
            
            logger.info(f"Successfully added {len(documents)} documents")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search(
        self, 
        query: str,
        k: Optional[int] = None,
        return_sources: bool = True,
        filter_dict: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive search using vector similarity and LLM processing.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            return_sources: Whether to include source documents in response
            filter_dict: Optional metadata filter for search
            similarity_threshold: Minimum similarity score for results
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        try:
            logger.info(f"Processing search query: {query[:100]}...")
            
            # Expand query if enabled
            expanded_queries = [query]
            if self.auto_expand_queries:
                expanded_queries.extend(self.prompt_pipeline.expand_query(query))
                logger.info(f"Expanded to {len(expanded_queries)} queries")
            
            # Perform vector search for all queries
            all_results = []
            k_per_query = max(1, (k or self.settings.top_k_results) // len(expanded_queries))
            
            for expanded_query in expanded_queries:
                results = self.vector_store.similarity_search_with_score(
                    query=expanded_query,
                    k=k_per_query,
                    filter_dict=filter_dict
                )
                all_results.extend(results)
            
            # Filter by similarity threshold if specified
            if similarity_threshold:
                all_results = [
                    (doc, score) for doc, score in all_results 
                    if score >= similarity_threshold
                ]
            
            # Remove duplicates and sort by score
            unique_results = {}
            for doc, score in all_results:
                doc_key = doc.page_content[:100]  # Use first 100 chars as key
                if doc_key not in unique_results or score > unique_results[doc_key][1]:
                    unique_results[doc_key] = (doc, score)
            
            # Convert back to list and sort by score
            final_results = sorted(
                unique_results.values(), 
                key=lambda x: x[1], 
                reverse=True
            )[:k or self.settings.top_k_results]
            
            if not final_results:
                return {
                    "answer": "I couldn't find any relevant information for your query. Please try rephrasing your question or adding more context.",
                    "sources": [],
                    "metadata": {
                        "query": query,
                        "expanded_queries": expanded_queries,
                        "documents_found": 0,
                        "search_successful": False
                    }
                }
            
            # Format context for LLM
            context_docs = [doc for doc, _ in final_results]
            context = self.prompt_pipeline.format_context(context_docs)
            
            # Generate answer using LLM
            answer = self.prompt_pipeline.run_chain(
                chain_name="search",
                inputs={"query": query, "context": context}
            )
            
            # Format response
            response = {
                "answer": answer,
                "metadata": {
                    "query": query,
                    "expanded_queries": expanded_queries,
                    "documents_found": len(final_results),
                    "search_successful": True,
                    "keywords": extract_keywords(query)
                }
            }
            
            if return_sources:
                response["sources"] = format_search_results(final_results, include_scores=True)
            
            logger.info(f"Search completed successfully with {len(final_results)} relevant documents")
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "answer": f"An error occurred during search: {str(e)}",
                "sources": [],
                "metadata": {
                    "query": query,
                    "error": str(e),
                    "search_successful": False
                }
            }
    
    def simple_search(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Perform a simple vector similarity search without LLM processing.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k or self.settings.top_k_results
            )
            logger.info(f"Simple search returned {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Simple search failed: {e}")
            raise
    
    def get_similar_documents(
        self, 
        query: str, 
        k: Optional[int] = None,
        with_scores: bool = False
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Get similar documents with optional similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            with_scores: Whether to include similarity scores
            
        Returns:
            List of documents or document-score tuples
        """
        try:
            if with_scores:
                return self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k or self.settings.top_k_results
                )
            else:
                return self.vector_store.similarity_search(
                    query=query,
                    k=k or self.settings.top_k_results
                )
                
        except Exception as e:
            logger.error(f"Failed to get similar documents: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the knowledge base.
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            self.vector_store.delete_documents(ids)
            logger.info(f"Deleted {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary containing various statistics
        """
        try:
            index_stats = self.vector_store.get_index_stats()
            
            stats = {
                "index_stats": index_stats,
                "configuration": {
                    "model": self.settings.openai_model,
                    "temperature": self.settings.temperature,
                    "max_tokens": self.settings.max_tokens,
                    "top_k_results": self.settings.top_k_results,
                    "auto_expand_queries": self.auto_expand_queries
                }
            }
            
            logger.info("Retrieved knowledge base statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise
    
    def reset_conversation(self) -> None:
        """Reset the conversation memory."""
        self.prompt_pipeline.reset_memory()
        logger.info("Conversation memory reset")
    
    def update_settings(self, **kwargs) -> None:
        """
        Update agent settings dynamically.
        
        Args:
            **kwargs: Settings to update
        """
        try:
            for key, value in kwargs.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
                    logger.info(f"Updated setting {key} to {value}")
                else:
                    logger.warning(f"Unknown setting: {key}")
                    
        except Exception as e:
            logger.error(f"Failed to update settings: {e}")
            raise
