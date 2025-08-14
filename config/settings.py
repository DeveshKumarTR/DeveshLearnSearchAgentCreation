"""
Configuration management for the Search Agent.
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Pinecone Configuration
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_environment: str = Field(..., env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="search-agent-index", env="PINECONE_INDEX_NAME")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    
    # LangChain Configuration
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="search-agent", env="LANGCHAIN_PROJECT")
    
    # Application Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_tokens: int = Field(default=1000, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Vector Store Configuration
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings


def validate_settings() -> bool:
    """Validate that all required settings are present."""
    try:
        settings = get_settings()
        required_fields = ["pinecone_api_key", "pinecone_environment", "openai_api_key"]
        
        for field in required_fields:
            if not getattr(settings, field):
                raise ValueError(f"Missing required setting: {field}")
        
        return True
    except Exception as e:
        print(f"Settings validation failed: {e}")
        return False
