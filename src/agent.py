from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.templates.prompt_templates import RAG_PROMPT_TEMPLATE, AGENT_PROMPT_TEMPLATE
from src.config import OPENAI_API_KEY, DEFAULT_LLM_MODEL
from src.data.indexing import load_vector_store


class RAGAgent:
    def __init__(self, embeddings):
        self.vectorstore = load_vector_store(embeddings)
        self.retriever = self.vectorstore.as_retriever()

        self.llm = ChatOpenAI(
            model_name=DEFAULT_LLM_MODEL, 
            api_key=OPENAI_API_KEY
        )
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_PROMPT_TEMPLATE),
            MessagesPlaceholder("chat_history"), 
            ("user", "{input}")
        ])
        self.rag_chain = create_retrieval_chain(
            self.retriever, create_stuff_documents_chain(self.llm, self.chat_prompt)
        )

        self.tools = [
            Tool(
                name="format_full_recipe",
                func=self.format_recipe,
                description="Formats and displays full recipe details from context. Use ONLY when user explicitly asks for the full recipe.",
                return_direct=True,
            )
        ]

        # Define the agent prompt
        self.agent_prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_PROMPT_TEMPLATE),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]).partial(context="{context}")

        # Define agent execution flow
        self.agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "context": lambda x: x["context"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | self.agent_prompt
            | self.llm.bind_tools(self.tools)
            | OpenAIFunctionsAgentOutputParser()
        )
        
        # AgentExecutor: decides whether to use RAG or call a tool
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

    def format_recipe(self, query: str) -> str:
        """Formats the full recipe details from RAG context"""
        if not hasattr(self, 'last_rag_context') or not self.last_rag_context:
            return "No recipe found. Please search for a recipe first."
            
        # Add debug print to see if this method is being called
        print(f"Formatting recipe: {self.last_rag_context[:100]}...")
        
        # Extract recipe details from context
        recipe_details = self.last_rag_context
        
        # Format the recipe in a nice way
        return f"### 📝 Full Recipe\n\n{recipe_details}\n\nBon appétit!"

    def run(self, user_input: str, chat_history: list) -> str:
        """
        The agent decides whether to:
          - Answer directly using retrieved RAG context, or
          - Call a tool to format and display the full recipe.
        """
        # Retrieve context using the RAG chain
        rag_result = self.rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        
        # Store the context for the format_recipe tool
        self.last_rag_context = rag_result["context"]
        
        # Run the agent with context - let the LLM decide whether to call the tool
        response = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history,
            "context": rag_result["context"]
        })

        # Debug print to see the response structure
        print(f"Agent response: {response}")
        
        # Return the agent's output
        if "output" in response and response["output"]:
            return response["output"]
        else:
            return "I couldn't find any relevant recipe information. Please try asking in a different way."
