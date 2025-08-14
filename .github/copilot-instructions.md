<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Search Agent Project - Copilot Instructions

This is a Python-based Search Agent project using LangChain for prompt pipelining and Pinecone for vector database operations.

## Project Structure
- `src/` - Main source code directory
- `config/` - Configuration files
- `examples/` - Example usage scripts
- `tests/` - Unit tests
- `requirements.txt` - Python dependencies

## Key Technologies
- LangChain for LLM interactions and prompt management
- Pinecone for vector database operations
- Python for main implementation

## Development Guidelines
- Follow Python PEP 8 style guidelines
- Use type hints for better code clarity
- Implement proper error handling
- Create modular, reusable components

## Checklist Progress
- [x] Verify that the copilot-instructions.md file in the .github directory is created
- [x] Clarify Project Requirements - Python Search Agent with LangChain and Pinecone
- [x] Scaffold the Project - Created complete project structure with src/, config/, examples/, tests/
- [x] Customize the Project - Implemented SearchAgent, VectorStore, PromptPipeline with examples
- [x] Install Required Extensions - No specific extensions required
- [x] Compile the Project - Python dependencies installed (requires Python runtime)
- [x] Create and Run Task - Tasks created for running examples
- [x] Launch the Project - Setup instructions provided for Python environment
- [x] Ensure Documentation is Complete - README.md and setup instructions completed

## Python Environment Setup Required

**Important**: This project requires Python 3.11+ to be installed and accessible. 

### Setup Instructions:
1. Install Python 3.11+ from python.org or use conda/miniconda
2. Ensure Python is added to your system PATH
3. Create a `.env` file with your API keys (copy from `.env.example`)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the basic example: `python examples/basic_search.py`

### Required Environment Variables:
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment  
OPENAI_API_KEY=your_openai_api_key
```
