from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import json 

from src.data.indexing import load_vector_store
from src.config import OPENAI_API_KEY, DEFAULT_LLM_MODEL
from src.templates.prompt_templates import RAG_PROMPT_TEMPLATE

class RecipeChatbot:    
    def __init__(self, embeddings):
        self.vectorstore = load_vector_store(embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
        self.llm = ChatOpenAI(
            model_name=DEFAULT_LLM_MODEL, 
            api_key=OPENAI_API_KEY
            )
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_PROMPT_TEMPLATE),
            MessagesPlaceholder("chat_history"), # populate chat history
            ("user", "{input}")
        ])

        self.rag_chain = create_retrieval_chain(self.retriever, create_stuff_documents_chain(self.llm, self.chat_prompt))
        
        self.chat_history = []
    
    def chat(self, user_input: str, 
             use_chat_history: bool = True) -> tuple:
        """
        Generate response using RAG chain. 
        """
        if not use_chat_history:
            chat_history = []
        else: 
            chat_history = self.chat_history
            
        response = self.rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        if use_chat_history:
            self.chat_history.append(("user", user_input))
            self.chat_history.append(("assistant", response["answer"]))
        return {
            "answer": response["answer"],
            "context": response["context"]
            }
