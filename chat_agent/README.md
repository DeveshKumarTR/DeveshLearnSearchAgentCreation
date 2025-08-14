# Chat Agent with Django UI

This package provides a conversational AI interface that combines the search agent capabilities with Ollama LLM integration through a modern Django web interface.

## Features

- **Ollama Integration**: Local LLM support with streaming responses
- **Conversation Management**: Persistent chat sessions with history
- **Search Augmentation**: Combines search results with conversational AI
- **Real-time Interface**: WebSocket-based chat with live typing indicators
- **Responsive Design**: Mobile-friendly Bootstrap interface
- **Model Switching**: Dynamic LLM model selection
- **REST API**: Full API support for external integrations

## Quick Start

### 1. Install Ollama
```bash
# Download and install Ollama from https://ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., llama2)
ollama pull llama2
```

### 2. Install Dependencies
```bash
cd django_ui
pip install -r requirements.txt
```

### 3. Setup Django
```bash
# Run migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser

# Start development server
python manage.py runserver
```

### 4. Access the Interface
Open http://localhost:8000 in your browser to access the chat interface.

## API Endpoints

### Chat Messages
- `POST /api/chat/message/` - Send chat message
- `GET /api/chat/conversations/` - List conversations
- `GET /api/chat/conversations/{id}/` - Get conversation details
- `DELETE /api/chat/conversations/{id}/` - Delete conversation

### Search Integration
- `POST /api/search/query/` - Perform search query
- `GET /api/search/history/` - Get search history

### System Management
- `GET /api/chat/models/` - List available models
- `POST /api/chat/models/switch/` - Switch active model
- `GET /api/health/` - Health check

## WebSocket Connection

Connect to `ws://localhost:8000/ws/chat/` for real-time chat functionality.

### WebSocket Message Format
```json
{
    "type": "chat_message",
    "message": "Your message here",
    "conversation_id": "optional-conversation-id",
    "model": "optional-model-name"
}
```

## Configuration

### Environment Variables
Create a `.env` file in the django_ui directory:

```env
# Django Settings
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama2

# Search Agent Configuration
SEARCH_ENABLED=True
PINECONE_API_KEY=your-pinecone-key
OPENAI_API_KEY=your-openai-key
```

### Django Settings
Key settings in `chat_ui/settings.py`:

```python
CHAT_AGENT_CONFIG = {
    'OLLAMA_HOST': 'http://localhost:11434',
    'DEFAULT_MODEL': 'llama2',
    'MAX_HISTORY_LENGTH': 50,
    'CONVERSATION_TIMEOUT': 3600,  # 1 hour
    'ENABLE_SEARCH': True,
}
```

## Usage Examples

### Python API
```python
from chat_agent import ChatAgent, OllamaClient

# Initialize chat agent
client = OllamaClient()
agent = ChatAgent(client)

# Start conversation
response = agent.chat("Hello, how can you help me?")
print(response)

# With search integration
response = agent.chat("Search for information about machine learning")
print(response)
```

### REST API
```bash
# Send chat message
curl -X POST http://localhost:8000/api/chat/message/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "conversation_id": "test-session"}'

# Get conversations
curl http://localhost:8000/api/chat/conversations/
```

## Development

### Running Tests
```bash
# Run Django tests
python manage.py test

# Run chat agent tests
cd ..
python -m pytest tests/
```

### Development Server
```bash
# Start Django development server
python manage.py runserver

# Start with specific settings
python manage.py runserver --settings=chat_ui.settings
```

## Troubleshooting

### Common Issues

1. **Ollama not running**: Ensure Ollama server is started (`ollama serve`)
2. **Model not found**: Pull the required model (`ollama pull <model-name>`)
3. **WebSocket connection failed**: Check CORS settings and WebSocket URL
4. **Search not working**: Verify Pinecone and OpenAI API keys

### Logs
Check Django logs for detailed error information:
```bash
# View Django logs
tail -f django_ui/logs/django.log
```

## Architecture

### Components
- **OllamaClient**: Interface to Ollama API
- **ConversationManager**: Session and history management
- **ChatAgent**: Main orchestrator combining search and chat
- **Django Views**: REST API endpoints
- **WebSocket Consumer**: Real-time communication
- **Chat Interface**: Responsive web UI

### Data Flow
1. User sends message via web interface
2. WebSocket consumer receives message
3. ChatAgent processes with Ollama LLM
4. Optional search augmentation
5. Streaming response back to user
6. Conversation saved to database

## License

This project is part of the DeveshLearnSearchAgentCreation repository.
