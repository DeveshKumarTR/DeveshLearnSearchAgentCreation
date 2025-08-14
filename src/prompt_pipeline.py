"""
Prompt Pipeline implementation using LangChain for advanced prompt management.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from config.settings import get_settings

logger = logging.getLogger(__name__)


class PromptPipeline:
    """
    Advanced prompt pipeline for managing complex prompt workflows.
    """
    
    def __init__(self, use_chat_model: bool = True, streaming: bool = False):
        """
        Initialize the prompt pipeline.
        
        Args:
            use_chat_model: Whether to use ChatOpenAI or regular OpenAI
            streaming: Whether to enable streaming responses
        """
        self.settings = get_settings()
        self.use_chat_model = use_chat_model
        self.streaming = streaming
        
        # Initialize callback manager for streaming
        self.callback_manager = None
        if streaming:
            self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Initialize LLM
        self._initialize_llm()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Store created chains for reuse
        self.chains: Dict[str, LLMChain] = {}
        
    def _initialize_llm(self) -> None:
        """Initialize the language model."""
        try:
            common_params = {
                "openai_api_key": self.settings.openai_api_key,
                "model_name": self.settings.openai_model,
                "temperature": self.settings.temperature,
                "max_tokens": self.settings.max_tokens,
            }
            
            if self.callback_manager:
                common_params["callback_manager"] = self.callback_manager
            
            if self.use_chat_model:
                self.llm = ChatOpenAI(**common_params)
            else:
                self.llm = OpenAI(**common_params)
                
            logger.info(f"Initialized {'Chat' if self.use_chat_model else ''}OpenAI model: {self.settings.openai_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def create_search_prompt(self) -> PromptTemplate:
        """Create a prompt template for search queries."""
        template = """
        You are an intelligent search assistant. Given the search query and relevant context documents, 
        provide a comprehensive and accurate answer.
        
        Search Query: {query}
        
        Relevant Context:
        {context}
        
        Instructions:
        1. Analyze the search query carefully
        2. Use the provided context to formulate your response
        3. If the context doesn't contain enough information, acknowledge this
        4. Provide a clear, well-structured answer
        5. Include relevant details and examples when available
        
        Answer:
        """
        
        return PromptTemplate(
            input_variables=["query", "context"],
            template=template
        )
    
    def create_chat_search_prompt(self) -> ChatPromptTemplate:
        """Create a chat prompt template for search queries."""
        system_template = """
        You are an intelligent search assistant with access to a knowledge base. Your role is to:
        1. Understand user search queries accurately
        2. Analyze provided context documents
        3. Synthesize information to provide comprehensive answers
        4. Acknowledge limitations when context is insufficient
        5. Maintain a helpful and professional tone
        """
        
        human_template = """
        Search Query: {query}
        
        Relevant Context:
        {context}
        
        Please provide a comprehensive answer based on the search query and context provided.
        """
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    def create_query_expansion_prompt(self) -> PromptTemplate:
        """Create a prompt template for query expansion."""
        template = """
        Given the following search query, generate 3-5 alternative phrasings or related queries 
        that could help retrieve more comprehensive results from a vector database.
        
        Original Query: {query}
        
        Alternative Queries:
        1.
        2.
        3.
        4.
        5.
        
        Provide only the alternative queries, one per line, without additional explanation.
        """
        
        return PromptTemplate(
            input_variables=["query"],
            template=template
        )
    
    def create_chain(
        self, 
        chain_name: str, 
        prompt: Union[PromptTemplate, ChatPromptTemplate],
        output_key: str = "result"
    ) -> LLMChain:
        """
        Create and store a reusable LLM chain.
        
        Args:
            chain_name: Unique name for the chain
            prompt: Prompt template to use
            output_key: Output variable name
            
        Returns:
            Created LLM chain
        """
        try:
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_key=output_key,
                memory=self.memory,
                verbose=True
            )
            
            self.chains[chain_name] = chain
            logger.info(f"Created chain: {chain_name}")
            return chain
            
        except Exception as e:
            logger.error(f"Failed to create chain {chain_name}: {e}")
            raise
    
    def create_sequential_pipeline(
        self, 
        chains: List[LLMChain],
        input_variables: List[str],
        output_variables: List[str]
    ) -> SequentialChain:
        """
        Create a sequential chain pipeline.
        
        Args:
            chains: List of LLM chains to execute in sequence
            input_variables: Input variables for the pipeline
            output_variables: Output variables from the pipeline
            
        Returns:
            Sequential chain pipeline
        """
        try:
            pipeline = SequentialChain(
                chains=chains,
                input_variables=input_variables,
                output_variables=output_variables,
                verbose=True
            )
            
            logger.info(f"Created sequential pipeline with {len(chains)} chains")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create sequential pipeline: {e}")
            raise
    
    def run_chain(
        self, 
        chain_name: str, 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a specific chain with given inputs.
        
        Args:
            chain_name: Name of the chain to run
            inputs: Input variables for the chain
            
        Returns:
            Chain output
        """
        try:
            if chain_name not in self.chains:
                raise ValueError(f"Chain '{chain_name}' not found")
            
            chain = self.chains[chain_name]
            result = chain.run(inputs)
            
            logger.info(f"Executed chain: {chain_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to run chain {chain_name}: {e}")
            raise
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a search query to generate alternative phrasings.
        
        Args:
            query: Original search query
            
        Returns:
            List of expanded queries
        """
        try:
            # Create or get query expansion chain
            if "query_expansion" not in self.chains:
                prompt = self.create_query_expansion_prompt()
                self.create_chain("query_expansion", prompt, "expanded_queries")
            
            result = self.run_chain("query_expansion", {"query": query})
            
            # Parse the result to extract individual queries
            expanded_queries = []
            if isinstance(result, str):
                lines = result.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(('1.', '2.', '3.', '4.', '5.')):
                        # Remove numbering if present
                        clean_line = line.lstrip('12345. ')
                        if clean_line and clean_line != query:
                            expanded_queries.append(clean_line)
            
            logger.info(f"Expanded query into {len(expanded_queries)} alternatives")
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []
    
    def format_context(self, documents: List[Any]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        try:
            context_parts = []
            for i, doc in enumerate(documents, 1):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                source_info = ""
                if metadata:
                    source_info = f" (Source: {metadata.get('source', 'Unknown')})"
                
                context_parts.append(f"Document {i}{source_info}:\n{content}\n")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to format context: {e}")
            return ""
    
    def reset_memory(self) -> None:
        """Reset the conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory reset")
