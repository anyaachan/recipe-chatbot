import os
import pandas as pd
from src.config import INPUT_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.embeddings import get_embedding_model
from src.data.indexing import load_documents, create_vector_store
from src.chatbot import RecipeChatbot
from src.utils.text_processing import clean_html_tags, remove_hashtags, split_steps

def preprocess_data(input_file_path):
    """
    Preprocess the csv file with data and save to csv.
    More about steps can be found in explanatory notebook "01_data_preprocessing.ipynb".
    """
    rag_path = os.path.join(PROCESSED_DATA_DIR, "Recipes_processed_rag.csv")
    
    print("Loading and preprocessing data...")
    df = pd.read_csv(input_file_path)
    
    if "name.1" in df.columns:
        df = df.rename(columns={"name.1": "author_name"})
    
    if "name" in df.columns:
        df = df.rename(columns={"name": "recipe_name"})
    
    for column in df.columns:
        df[column] = df[column].apply(clean_html_tags)
        df[column] = df[column].apply(remove_hashtags)
    
    df["steps"] = df["steps"].str.replace('\r\n', '\n')
    df["steps"] = df["steps"].str.replace(r'\n\s*\n+', '\n\n', regex=True)
    df["steps"] = df["steps"].apply(split_steps, args=([r"\.,", r"\n\n", r"\n"],))
    
    df_for_rag = df.copy()
    
    if "author_note" in df_for_rag.columns:
        df_for_rag.drop(columns=["author_note"], inplace=True)
    
    df_for_rag.to_csv(rag_path, index=False)

def index_data(embeddings):
    """
    Index the data and store in Chroma database.
    """
    rag_path = os.path.join(PROCESSED_DATA_DIR, "Recipes_processed_rag.csv")
    docs = load_documents(rag_path)
    create_vector_store(docs, embeddings)

def start_chatbot():
    """
    Start the chatbot in CLI. 
    """
    chatbot = RecipeChatbot()
    
    print("\nWelcome to the Recipe Chatbot!")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        response = chatbot.chat(user_input)
        print(f"\nChatbot: {response}")

def main():
    embeddings = get_embedding_model()
    preprocess_data(os.path.join(INPUT_DATA_DIR, "Recipes.csv"))
    index_data(embeddings)
    start_chatbot()

if __name__ == "__main__":
    main()
