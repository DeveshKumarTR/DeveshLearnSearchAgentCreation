"""
Utility functions for the Search Agent.
"""

import logging
import json
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def generate_document_id(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a unique ID for a document based on its content and metadata.
    
    Args:
        content: Document content
        metadata: Optional metadata
        
    Returns:
        Unique document ID
    """
    # Create a string representation of content and metadata
    doc_string = content
    if metadata:
        doc_string += json.dumps(metadata, sort_keys=True)
    
    # Generate hash
    return hashlib.md5(doc_string.encode()).hexdigest()


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_search_results(results: List[Any], include_scores: bool = False) -> List[Dict[str, Any]]:
    """
    Format search results for display or further processing.
    
    Args:
        results: List of search results
        include_scores: Whether to include similarity scores
        
    Returns:
        Formatted search results
    """
    formatted_results = []
    
    for i, result in enumerate(results):
        formatted_result = {
            "rank": i + 1,
            "content": "",
            "metadata": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Handle different result formats
        if isinstance(result, tuple) and len(result) == 2:
            # Result with score
            doc, score = result
            formatted_result["content"] = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            formatted_result["metadata"] = doc.metadata if hasattr(doc, 'metadata') else {}
            if include_scores:
                formatted_result["similarity_score"] = float(score)
        else:
            # Regular document result
            formatted_result["content"] = result.page_content if hasattr(result, 'page_content') else str(result)
            formatted_result["metadata"] = result.metadata if hasattr(result, 'metadata') else {}
        
        formatted_results.append(formatted_result)
    
    return formatted_results


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text (simple implementation).
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction - in production, use more sophisticated methods
    import re
    
    # Remove special characters and convert to lowercase
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    
    # Split into words
    words = clean_text.split()
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'among', 'within', 'without', 'against', 'towards',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our',
        'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he',
        'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves'
    }
    
    # Filter and count words
    word_freq = {}
    for word in words:
        if len(word) > 2 and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]


def validate_api_keys(required_keys: List[str]) -> Dict[str, bool]:
    """
    Validate that required API keys are present.
    
    Args:
        required_keys: List of required environment variable names
        
    Returns:
        Dictionary mapping key names to validation status
    """
    import os
    
    validation_results = {}
    for key in required_keys:
        value = os.getenv(key)
        validation_results[key] = bool(value and value.strip())
    
    return validation_results


def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize an object to JSON, handling non-serializable types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string
    """
    def json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    try:
        return json.dumps(obj, default=json_serializer, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {str(e)}"})


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at word boundaries
        if end < len(text):
            # Look for the last space within the chunk
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunks.append(text[start:end])
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks


def calculate_similarity_threshold(scores: List[float], percentile: float = 0.7) -> float:
    """
    Calculate a similarity threshold based on score distribution.
    
    Args:
        scores: List of similarity scores
        percentile: Percentile to use for threshold (0.0 to 1.0)
        
    Returns:
        Similarity threshold
    """
    if not scores:
        return 0.0
    
    sorted_scores = sorted(scores, reverse=True)
    index = int(len(sorted_scores) * percentile)
    return sorted_scores[min(index, len(sorted_scores) - 1)]
