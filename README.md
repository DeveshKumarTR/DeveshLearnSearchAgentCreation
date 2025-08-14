# Search Agent with LangChain and Pinecone + Chat Agent with Ollama

A comprehensive AI assistant platform that combines semantic search capabilities with conversational AI through a modern Django web interface.

## Features

### Search Agent
- **Semantic Search**: Powered by Pinecone vector database
- **Prompt Pipelining**: Advanced prompt management with LangChain
- **Modular Architecture**: Clean, extensible codebase
- **Configuration Management**: Environment-based configuration
- **Example Implementations**: Ready-to-use examples

### Chat Agent (NEW!)
- **Local LLM Integration**: Ollama-powered conversational AI
- **Real-time Chat Interface**: Modern Django UI with WebSocket support
- **Conversation Management**: Persistent chat sessions and history
- **Search Augmentation**: Combines search results with chat responses
- **Model Switching**: Dynamic LLM model selection
- **REST API**: Full API support for external integrations
- **Responsive Design**: Mobile-friendly Bootstrap interface

## Installation

### Option 1: Automated Setup (Recommended)

1. Clone this repository
2. Run the automated setup:
   ```bash
   python setup_chat.py
   ```

### Option 2: Manual Setup

#### For Search Agent Only:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up environment variables (see Configuration section)

#### For Chat Agent + Django UI:
1. Install Ollama from https://ollama.ai
2. Install dependencies:
   ```bash
   cd django_ui
   pip install -r requirements.txt
   ```
3. Setup Django:
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

## Configuration

Create a `.env` file in the root directory with:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
OPENAI_API_KEY=your_openai_api_key
```

## Quick Start

### Search Agent
```python
from src.search_agent import SearchAgent

# Initialize the search agent
agent = SearchAgent()

# Perform a search
results = agent.search("What is machine learning?")
print(results)
```

### Chat Agent
```python
from chat_agent import ChatAgent, OllamaClient

# Initialize chat agent
client = OllamaClient()
agent = ChatAgent(client)

# Start conversation
response = agent.chat("Hello, how can you help me?")
print(response)
```

### Web Interface
1. Start Ollama: `ollama serve`
2. Install a model: `ollama pull llama2`
3. Start Django: `cd django_ui && python manage.py runserver`
4. Open http://localhost:8000 in your browser

## Project Structure

```
├── src/                         # Search Agent Core
│   ├── __init__.py
│   ├── search_agent.py          # Main search agent class
│   ├── vector_store.py          # Pinecone vector store
│   ├── prompt_pipeline.py       # LangChain prompt management
│   └── utils.py                 # Utility functions
├── chat_agent/                  # Chat Agent Package (NEW!)
│   ├── __init__.py
│   ├── chat_agent.py            # Main chat orchestrator
│   ├── ollama_client.py         # Ollama LLM interface
│   ├── conversation_manager.py  # Session management
│   └── README.md                # Chat agent documentation
├── django_ui/                   # Web Interface (NEW!)
│   ├── chat_ui/                 # Django project
│   ├── chat_app/                # Chat application
│   ├── templates/               # HTML templates
│   ├── static/                  # CSS/JS assets
│   ├── manage.py                # Django management
│   └── requirements.txt         # UI dependencies
├── examples/
│   ├── basic_search.py          # Simple search example
│   └── advanced_pipeline.py     # Advanced usage
├── tests/
│   ├── test_search_agent.py     # Unit tests
│   └── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration management
├── setup_chat.py                # Automated setup script
├── requirements.txt             # Core dependencies
└── README.md
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
