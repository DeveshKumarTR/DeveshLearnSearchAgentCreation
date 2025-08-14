"""
Advanced Prompt Pipeline Example

This example demonstrates advanced features of the prompt pipeline including
custom chains, sequential processing, and query expansion.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from search_agent import SearchAgent
from prompt_pipeline import PromptPipeline
from vector_store import VectorStore
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import SequentialChain


def create_custom_analysis_chain(pipeline: PromptPipeline):
    """Create a custom analysis chain for detailed document analysis."""
    
    # Create custom prompt for document analysis
    analysis_template = """
    You are a document analysis expert. Analyze the following document and provide:
    1. Main topics and themes
    2. Key concepts and entities
    3. Relevance score (1-10) for the given query
    4. Summary in 2-3 sentences
    
    Query: {query}
    Document: {document}
    
    Analysis:
    """
    
    analysis_prompt = PromptTemplate(
        input_variables=["query", "document"],
        template=analysis_template
    )
    
    return pipeline.create_chain(
        chain_name="document_analysis",
        prompt=analysis_prompt,
        output_key="analysis"
    )


def create_synthesis_chain(pipeline: PromptPipeline):
    """Create a synthesis chain to combine multiple analyses."""
    
    synthesis_template = """
    Based on the following document analyses, create a comprehensive response that:
    1. Synthesizes information from all relevant sources
    2. Provides a well-structured answer to the query
    3. Acknowledges any limitations or gaps in the information
    4. Includes confidence level (Low/Medium/High)
    
    Original Query: {query}
    
    Document Analyses:
    {analyses}
    
    Comprehensive Response:
    """
    
    synthesis_prompt = PromptTemplate(
        input_variables=["query", "analyses"],
        template=synthesis_template
    )
    
    return pipeline.create_chain(
        chain_name="synthesis",
        prompt=synthesis_prompt,
        output_key="final_response"
    )


def main():
    """Main example function."""
    print("🚀 Advanced Prompt Pipeline Example")
    print("=" * 50)
    
    try:
        # Initialize components
        print("1. Initializing components...")
        
        # Initialize with streaming for real-time output
        agent = SearchAgent(
            use_chat_model=True,
            streaming=True,  # Enable streaming for better UX
            auto_expand_queries=True
        )
        
        pipeline = agent.prompt_pipeline
        vector_store = agent.vector_store
        
        print("✅ Components initialized!")
        
        # Add comprehensive sample documents
        print("\n2. Adding comprehensive knowledge base...")
        
        documents = [
            """
            Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
            capable of performing tasks that typically require human intelligence. These tasks include learning, 
            reasoning, problem-solving, perception, and language understanding. AI has applications in various 
            fields including healthcare, finance, transportation, and entertainment.
            """,
            """
            Machine Learning (ML) is a subset of AI that focuses on algorithms and statistical models that enable 
            computers to improve their performance on a task through experience. ML includes supervised learning, 
            unsupervised learning, and reinforcement learning. Popular algorithms include linear regression, 
            decision trees, neural networks, and support vector machines.
            """,
            """
            Deep Learning is a specialized area of machine learning that uses neural networks with multiple layers 
            (deep neural networks) to model and understand complex patterns in data. It has been particularly 
            successful in image recognition, natural language processing, and game playing. Popular frameworks 
            include TensorFlow, PyTorch, and Keras.
            """,
            """
            Natural Language Processing (NLP) is a field that combines computational linguistics with machine 
            learning and deep learning to help computers understand, interpret, and generate human language. 
            Applications include chatbots, translation systems, sentiment analysis, and text summarization.
            """,
            """
            Computer Vision is an interdisciplinary field that deals with how computers can gain high-level 
            understanding from digital images or videos. It seeks to automate tasks that the human visual 
            system can do, such as object detection, facial recognition, and autonomous driving.
            """
        ]
        
        metadata = [
            {"source": "AI Textbook", "chapter": "Introduction", "difficulty": "beginner", "field": "general_ai"},
            {"source": "ML Guide", "chapter": "Algorithms", "difficulty": "intermediate", "field": "machine_learning"},
            {"source": "DL Manual", "chapter": "Neural Networks", "difficulty": "advanced", "field": "deep_learning"},
            {"source": "NLP Handbook", "chapter": "Fundamentals", "difficulty": "intermediate", "field": "nlp"},
            {"source": "CV Tutorial", "chapter": "Basics", "difficulty": "intermediate", "field": "computer_vision"}
        ]
        
        doc_ids = agent.add_documents(documents, metadata)
        print(f"✅ Added {len(doc_ids)} documents")
        
        # Create custom chains
        print("\n3. Creating custom analysis pipeline...")
        
        analysis_chain = create_custom_analysis_chain(pipeline)
        synthesis_chain = create_synthesis_chain(pipeline)
        
        print("✅ Custom chains created!")
        
        # Demonstrate query expansion
        print("\n4. Demonstrating query expansion...")
        
        original_query = "How does machine learning work?"
        expanded_queries = pipeline.expand_query(original_query)
        
        print(f"Original query: {original_query}")
        print("Expanded queries:")
        for i, expanded in enumerate(expanded_queries, 1):
            print(f"  {i}. {expanded}")
        
        # Perform advanced search with custom pipeline
        print("\n5. Performing advanced search with custom analysis...")
        
        # Retrieve relevant documents
        relevant_docs = vector_store.similarity_search_with_score(
            query=original_query,
            k=3
        )
        
        print(f"Found {len(relevant_docs)} relevant documents")
        
        # Analyze each document individually
        analyses = []
        print("\nAnalyzing documents:")
        
        for i, (doc, score) in enumerate(relevant_docs, 1):
            print(f"\n--- Analyzing Document {i} (Score: {score:.3f}) ---")
            
            analysis = pipeline.run_chain(
                chain_name="document_analysis",
                inputs={
                    "query": original_query,
                    "document": doc.page_content
                }
            )
            
            analyses.append(f"Document {i} Analysis:\n{analysis}")
            print(f"Analysis: {analysis[:200]}...")
        
        # Synthesize final response
        print("\n6. Synthesizing comprehensive response...")
        
        final_response = pipeline.run_chain(
            chain_name="synthesis",
            inputs={
                "query": original_query,
                "analyses": "\n\n".join(analyses)
            }
        )
        
        print(f"\n📋 Comprehensive Response:")
        print(final_response)
        
        # Demonstrate conversation memory
        print("\n7. Demonstrating conversation memory...")
        
        # Reset memory to start fresh
        pipeline.reset_memory()
        
        # Multiple related queries to show memory in action
        conversation_queries = [
            "What is the difference between AI and ML?",
            "Can you give me examples of the concepts you just mentioned?",
            "How does this relate to deep learning?"
        ]
        
        for query in conversation_queries:
            print(f"\nQuery: {query}")
            result = agent.search(query, k=2, return_sources=False)
            print(f"Answer: {result['answer'][:300]}...")
        
        # Get final statistics
        print("\n8. Final Statistics:")
        stats = agent.get_stats()
        print(f"Total queries processed: Multiple")
        print(f"Knowledge base size: {stats['index_stats']}")
        
        print("\n✅ Advanced example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Verify your .env file contains all required API keys")
        print("2. Check your Pinecone index configuration")
        print("3. Ensure sufficient API credits/quota")
        print("4. Verify network connectivity")


if __name__ == "__main__":
    main()
