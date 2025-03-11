import os
import sys
import pandas as pd
from textwrap import fill
from src.config import INPUT_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.embeddings import get_embedding_model
from src.utils.text_processing import format_response_context
from src.data.indexing import load_documents, create_vector_store
from src.data.preprocessing import preprocess_csv
from src.chatbot import RecipeChatbot
from src.utils.generation import call_llm_openrouter

def preprocess_data(input_file_path, output_file_path):
    """
    Preprocess the csv file with data and save to csv.
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(input_file_path)
    preprocess_csv(df, output_file_path)

def index_data(embeddings):
    """
    Index the data and store in Chroma database.
    """
    rag_path = os.path.join(PROCESSED_DATA_DIR, "Recipes_processed_rag.csv")
    docs = load_documents(rag_path, ["id", "author_note"])
    create_vector_store(docs, embeddings)

def start_chatbot(embeddings):
    """
    Start the chatbot in CLI. 
    """
    chatbot = RecipeChatbot(embeddings)
    
    print("\n\033[1;36mWelcome to the Recipe Chatbot!\033[0m")
    print("You can ask me about recipes, ingredients, or cooking techniques.")
    print("Type 'exit' to end the conversation, or 'help' for more options.\n")
    
    while True:
        try:
            user_input = input("\033[1;32mYou:\033[0m ")
            
            if user_input.lower() == "exit":
                print("\033[1;36mGoodbye! Happy cooking!\033[0m")
                break
            elif user_input.lower() == "help":
                print("\n\033[1;33mAvailable commands:\033[0m")
                print("- 'exit': End the conversation.")
                print("- 'help': Show this help message.")
                print("- You can also ask me anything related to recipes, ingredients, or cooking techniques.")
                continue
            
            response = chatbot.chat(user_input)
            print(f"\n\033[1;34mChatbot:\033[0m {fill(response['answer'])}")
        
        except KeyboardInterrupt:
            print("\n\033[1;36mGoodbye! Happy cooking!\033[0m")
            sys.exit(0)
        except Exception as e:
            print(f"\n\033[1;31mAn error occurred: {e}\033[0m")
            print("Please try again or type 'exit' to quit.")
            

def main():
    # testing
    # print(call_llm_openrouter("What is the capital of France?", "google/gemini-2.0-flash-001", {"answer": "Answer"}))
    embeddings = get_embedding_model() # loading embedding here to avoid multiple loads
    preprocess_data(
        os.path.join(INPUT_DATA_DIR, "Recipes.csv"), 
        os.path.join(PROCESSED_DATA_DIR, "Recipes_processed_rag.csv")
        )
    index_data(embeddings)
    start_chatbot(embeddings)

if __name__ == "__main__":
    main()
