"""
Basic Search Agent Example

This example demonstrates how to use the Search Agent for basic operations.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from search_agent import SearchAgent


def main():
    """Main example function."""
    print("🔍 Basic Search Agent Example")
    print("=" * 50)
    
    try:
        # Initialize the search agent
        print("1. Initializing Search Agent...")
        agent = SearchAgent(
            use_chat_model=True,
            streaming=False,
            auto_expand_queries=True
        )
        print("✅ Search Agent initialized successfully!")
        
        # Add sample documents to the knowledge base
        print("\n2. Adding sample documents...")
        sample_documents = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, web development, and automation.",
            "Vector databases store data as high-dimensional vectors, enabling efficient similarity searches and semantic retrieval of information.",
            "Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language.",
            "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data."
        ]
        
        # Add metadata for documents
        metadata = [
            {"source": "AI Overview", "category": "machine_learning", "difficulty": "beginner"},
            {"source": "Programming Guide", "category": "programming", "difficulty": "beginner"},
            {"source": "Database Tutorial", "category": "databases", "difficulty": "intermediate"},
            {"source": "NLP Introduction", "category": "nlp", "difficulty": "beginner"},
            {"source": "Deep Learning Basics", "category": "deep_learning", "difficulty": "intermediate"}
        ]
        
        doc_ids = agent.add_documents(sample_documents, metadata)
        print(f"✅ Added {len(doc_ids)} documents to the knowledge base")
        
        # Perform searches
        print("\n3. Performing searches...")
        
        queries = [
            "What is machine learning?",
            "How to use Python for data science?",
            "Explain vector databases",
            "What are neural networks?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            result = agent.search(
                query=query,
                k=3,  # Return top 3 results
                return_sources=True
            )
            
            print(f"Answer: {result['answer']}")
            print(f"Documents found: {result['metadata']['documents_found']}")
            
            if result['sources']:
                print("Top sources:")
                for j, source in enumerate(result['sources'][:2], 1):
                    print(f"  {j}. {source['content'][:100]}...")
                    if 'similarity_score' in source:
                        print(f"     Similarity: {source['similarity_score']:.3f}")
        
        # Get knowledge base statistics
        print("\n4. Knowledge Base Statistics:")
        stats = agent.get_stats()
        print(f"Configuration: {stats['configuration']}")
        print(f"Index stats: {stats['index_stats']}")
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up your .env file with required API keys")
        print("2. Installed all dependencies: pip install -r requirements.txt")
        print("3. Configured Pinecone and OpenAI properly")


if __name__ == "__main__":
    main()
