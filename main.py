import os
import pandas as pd
from src.config import INPUT_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.embeddings import get_embedding_model
from src.data.indexing import load_documents, create_vector_store
from src.data.preprocessing import preprocess_csv
from src.utils.generation import call_llm_openrouter
from src.agent import RAGAgent

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
    chatbot = RAGAgent(embeddings)
    chat_history = []  # Store the conversation history

    print("\nWelcome to the Recipe Chatbot!")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = chatbot.run(user_input, chat_history)
        print(f"\nChatbot: {response}")

        # Update chat history
        chat_history.append(("user", user_input))
        chat_history.append(("assistant", response))


def main():
    # print(call_llm_openrouter("What is the capital of France?", "google/gemini-2.0-flash-001", {"answer": "Answer"}))     # testing
    embeddings = get_embedding_model() # loading embedding here to avoid multiple loads
    preprocess_data(
        os.path.join(INPUT_DATA_DIR, "Recipes.csv"), 
        os.path.join(PROCESSED_DATA_DIR, "Recipes_processed_rag.csv")
        )
    index_data(embeddings)
    start_chatbot(embeddings)

if __name__ == "__main__":
    main()
