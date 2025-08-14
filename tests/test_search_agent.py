"""
Unit tests for the Search Agent.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Mock environment variables for testing
@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'PINECONE_API_KEY': 'test_pinecone_key',
        'PINECONE_ENVIRONMENT': 'test_env',
        'PINECONE_INDEX_NAME': 'test_index',
        'OPENAI_API_KEY': 'test_openai_key',
        'OPENAI_MODEL': 'gpt-3.5-turbo',
        'LOG_LEVEL': 'INFO'
    }):
        yield


class TestSearchAgent:
    """Test cases for the SearchAgent class."""
    
    @patch('src.search_agent.VectorStore')
    @patch('src.search_agent.PromptPipeline')
    def test_search_agent_initialization(self, mock_prompt_pipeline, mock_vector_store):
        """Test SearchAgent initialization."""
        from search_agent import SearchAgent
        
        # Mock the components
        mock_vector_store.return_value = Mock()
        mock_prompt_pipeline.return_value = Mock()
        
        # Initialize agent
        agent = SearchAgent()
        
        # Verify components were initialized
        assert agent.vector_store is not None
        assert agent.prompt_pipeline is not None
        mock_vector_store.assert_called_once()
        mock_prompt_pipeline.assert_called_once()
    
    @patch('src.search_agent.VectorStore')
    @patch('src.search_agent.PromptPipeline')
    def test_add_documents(self, mock_prompt_pipeline, mock_vector_store):
        """Test adding documents to the knowledge base."""
        from search_agent import SearchAgent
        
        # Mock the vector store
        mock_vs_instance = Mock()
        mock_vs_instance.add_documents.return_value = ['doc1', 'doc2']
        mock_vector_store.return_value = mock_vs_instance
        mock_prompt_pipeline.return_value = Mock()
        
        # Initialize agent and add documents
        agent = SearchAgent()
        documents = ["Test document 1", "Test document 2"]
        metadata = [{"source": "test1"}, {"source": "test2"}]
        
        result = agent.add_documents(documents, metadata)
        
        # Verify documents were added
        assert result == ['doc1', 'doc2']
        mock_vs_instance.add_documents.assert_called_once_with(documents, metadata)
    
    @patch('src.search_agent.VectorStore')
    @patch('src.search_agent.PromptPipeline')
    def test_simple_search(self, mock_prompt_pipeline, mock_vector_store):
        """Test simple vector search without LLM processing."""
        from search_agent import SearchAgent
        from langchain.schema import Document
        
        # Mock the vector store
        mock_vs_instance = Mock()
        mock_docs = [
            Document(page_content="Test content 1", metadata={"source": "test1"}),
            Document(page_content="Test content 2", metadata={"source": "test2"})
        ]
        mock_vs_instance.similarity_search.return_value = mock_docs
        mock_vector_store.return_value = mock_vs_instance
        mock_prompt_pipeline.return_value = Mock()
        
        # Initialize agent and perform search
        agent = SearchAgent()
        results = agent.simple_search("test query", k=2)
        
        # Verify search results
        assert len(results) == 2
        assert results[0].page_content == "Test content 1"
        mock_vs_instance.similarity_search.assert_called_once_with(query="test query", k=2)
    
    @patch('src.search_agent.VectorStore')
    @patch('src.search_agent.PromptPipeline')
    def test_search_with_llm(self, mock_prompt_pipeline, mock_vector_store):
        """Test comprehensive search with LLM processing."""
        from search_agent import SearchAgent
        from langchain.schema import Document
        
        # Mock the vector store
        mock_vs_instance = Mock()
        mock_docs_with_scores = [
            (Document(page_content="Relevant content", metadata={"source": "test"}), 0.9)
        ]
        mock_vs_instance.similarity_search_with_score.return_value = mock_docs_with_scores
        mock_vector_store.return_value = mock_vs_instance
        
        # Mock the prompt pipeline
        mock_pp_instance = Mock()
        mock_pp_instance.expand_query.return_value = ["expanded query 1"]
        mock_pp_instance.format_context.return_value = "Formatted context"
        mock_pp_instance.run_chain.return_value = "Generated answer"
        mock_prompt_pipeline.return_value = mock_pp_instance
        
        # Initialize agent and perform search
        agent = SearchAgent()
        result = agent.search("test query")
        
        # Verify search result
        assert result['answer'] == "Generated answer"
        assert result['metadata']['search_successful'] is True
        assert result['metadata']['documents_found'] == 1
        assert 'sources' in result
    
    @patch('src.search_agent.VectorStore')
    @patch('src.search_agent.PromptPipeline')
    def test_search_no_results(self, mock_prompt_pipeline, mock_vector_store):
        """Test search behavior when no relevant documents are found."""
        from search_agent import SearchAgent
        
        # Mock the vector store to return no results
        mock_vs_instance = Mock()
        mock_vs_instance.similarity_search_with_score.return_value = []
        mock_vector_store.return_value = mock_vs_instance
        
        # Mock the prompt pipeline
        mock_pp_instance = Mock()
        mock_pp_instance.expand_query.return_value = []
        mock_prompt_pipeline.return_value = mock_pp_instance
        
        # Initialize agent and perform search
        agent = SearchAgent()
        result = agent.search("irrelevant query")
        
        # Verify no results response
        assert "couldn't find any relevant information" in result['answer']
        assert result['metadata']['search_successful'] is False
        assert result['metadata']['documents_found'] == 0


class TestVectorStore:
    """Test cases for the VectorStore class."""
    
    @patch('src.vector_store.pinecone')
    @patch('src.vector_store.OpenAIEmbeddings')
    @patch('src.vector_store.Pinecone')
    def test_vector_store_initialization(self, mock_pinecone_langchain, mock_embeddings, mock_pinecone):
        """Test VectorStore initialization."""
        from vector_store import VectorStore
        
        # Mock Pinecone
        mock_pinecone.list_indexes.return_value = ['test_index']
        mock_index = Mock()
        mock_pinecone.Index.return_value = mock_index
        
        # Mock embeddings
        mock_embeddings.return_value = Mock()
        
        # Mock Pinecone LangChain wrapper
        mock_pinecone_langchain.return_value = Mock()
        
        # Initialize vector store
        vs = VectorStore()
        
        # Verify initialization
        assert vs.index is not None
        assert vs.vectorstore is not None
        mock_pinecone.init.assert_called_once()
    
    @patch('src.vector_store.pinecone')
    @patch('src.vector_store.OpenAIEmbeddings')
    @patch('src.vector_store.Pinecone')
    def test_add_documents_chunking(self, mock_pinecone_langchain, mock_embeddings, mock_pinecone):
        """Test document chunking and addition."""
        from vector_store import VectorStore
        
        # Mock setup
        mock_pinecone.list_indexes.return_value = ['test_index']
        mock_pinecone.Index.return_value = Mock()
        mock_embeddings.return_value = Mock()
        
        mock_vectorstore = Mock()
        mock_vectorstore.add_documents.return_value = ['id1', 'id2']
        mock_pinecone_langchain.return_value = mock_vectorstore
        
        # Initialize and test
        vs = VectorStore()
        documents = ["Short doc", "This is a longer document that might need chunking"]
        result = vs.add_documents(documents)
        
        # Verify documents were processed
        assert result == ['id1', 'id2']
        mock_vectorstore.add_documents.assert_called_once()


class TestPromptPipeline:
    """Test cases for the PromptPipeline class."""
    
    @patch('src.prompt_pipeline.ChatOpenAI')
    def test_prompt_pipeline_initialization(self, mock_chat_openai):
        """Test PromptPipeline initialization."""
        from prompt_pipeline import PromptPipeline
        
        # Mock ChatOpenAI
        mock_chat_openai.return_value = Mock()
        
        # Initialize pipeline
        pp = PromptPipeline()
        
        # Verify initialization
        assert pp.llm is not None
        assert pp.memory is not None
        assert pp.use_chat_model is True
        mock_chat_openai.assert_called_once()
    
    @patch('src.prompt_pipeline.ChatOpenAI')
    def test_create_search_prompt(self, mock_chat_openai):
        """Test search prompt creation."""
        from prompt_pipeline import PromptPipeline
        
        # Mock ChatOpenAI
        mock_chat_openai.return_value = Mock()
        
        # Initialize pipeline and create prompt
        pp = PromptPipeline()
        prompt = pp.create_search_prompt()
        
        # Verify prompt structure
        assert prompt is not None
        assert "query" in prompt.input_variables
        assert "context" in prompt.input_variables
    
    @patch('src.prompt_pipeline.ChatOpenAI')
    @patch('src.prompt_pipeline.LLMChain')
    def test_create_chain(self, mock_llm_chain, mock_chat_openai):
        """Test chain creation."""
        from prompt_pipeline import PromptPipeline
        
        # Mock dependencies
        mock_chat_openai.return_value = Mock()
        mock_chain_instance = Mock()
        mock_llm_chain.return_value = mock_chain_instance
        
        # Initialize pipeline and create chain
        pp = PromptPipeline()
        prompt = pp.create_search_prompt()
        chain = pp.create_chain("test_chain", prompt)
        
        # Verify chain creation
        assert chain == mock_chain_instance
        assert "test_chain" in pp.chains
        mock_llm_chain.assert_called_once()


class TestUtils:
    """Test cases for utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        from utils import setup_logging
        
        # Test that setup_logging doesn't raise an exception
        try:
            setup_logging("INFO")
            setup_logging("DEBUG")
            assert True
        except Exception:
            assert False, "setup_logging should not raise an exception"
    
    def test_generate_document_id(self):
        """Test document ID generation."""
        from utils import generate_document_id
        
        # Test basic ID generation
        content = "Test document content"
        doc_id1 = generate_document_id(content)
        doc_id2 = generate_document_id(content)
        
        # Same content should generate same ID
        assert doc_id1 == doc_id2
        
        # Different content should generate different IDs
        doc_id3 = generate_document_id("Different content")
        assert doc_id1 != doc_id3
    
    def test_truncate_text(self):
        """Test text truncation."""
        from utils import truncate_text
        
        # Test text within limit
        short_text = "Short text"
        assert truncate_text(short_text, 100) == short_text
        
        # Test text exceeding limit
        long_text = "This is a very long text that exceeds the maximum length limit"
        truncated = truncate_text(long_text, 20)
        assert len(truncated) <= 20
        assert truncated.endswith("...")
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        from utils import extract_keywords
        
        text = "Machine learning is a subset of artificial intelligence that uses algorithms"
        keywords = extract_keywords(text, max_keywords=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert "machine" in keywords or "learning" in keywords
    
    def test_validate_api_keys(self):
        """Test API key validation."""
        from utils import validate_api_keys
        
        # Test with mock environment variables
        with patch.dict(os.environ, {'TEST_KEY': 'test_value', 'EMPTY_KEY': ''}):
            result = validate_api_keys(['TEST_KEY', 'EMPTY_KEY', 'MISSING_KEY'])
            
            assert result['TEST_KEY'] is True
            assert result['EMPTY_KEY'] is False
            assert result['MISSING_KEY'] is False


# Test configuration
def test_settings_validation():
    """Test settings validation."""
    from config.settings import validate_settings
    
    # This test will pass with mocked environment variables
    assert validate_settings() is True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
