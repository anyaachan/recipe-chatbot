import pandas as pd
from src.utils.text_processing import clean_html_tags, remove_hashtags, split_steps

def preprocess_csv(df: pd.DataFrame, output_path: str) -> None:
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
    
    df_for_rag.to_csv(output_path, index=False)