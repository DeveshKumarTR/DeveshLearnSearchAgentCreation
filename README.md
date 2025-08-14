# Search Agent with LangChain and Pinecone

A Python-based search agent that uses LangChain for prompt pipelining and Pinecone for vector database operations.

## Features

- **Semantic Search**: Powered by Pinecone vector database
- **Prompt Pipelining**: Advanced prompt management with LangChain
- **Modular Architecture**: Clean, extensible codebase
- **Configuration Management**: Environment-based configuration
- **Example Implementations**: Ready-to-use examples

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables (see Configuration section)

## Configuration

Create a `.env` file in the root directory with:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
OPENAI_API_KEY=your_openai_api_key
```

## Quick Start

```python
from src.search_agent import SearchAgent

# Initialize the search agent
agent = SearchAgent()

# Perform a search
results = agent.search("What is machine learning?")
print(results)
```

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── search_agent.py          # Main search agent class
│   ├── vector_store.py          # Pinecone vector operations
│   ├── prompt_pipeline.py       # LangChain prompt management
│   └── utils.py                 # Utility functions
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration management
├── examples/
│   ├── basic_search.py          # Basic search example
│   └── advanced_pipeline.py     # Advanced prompt pipeline example
├── tests/
│   ├── __init__.py
│   └── test_search_agent.py     # Unit tests
├── requirements.txt             # Python dependencies
└── .env.example                # Environment variables template
```

## Usage Examples

See the `examples/` directory for detailed usage examples.

## Testing

Run tests with:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License
