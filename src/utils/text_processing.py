from bs4 import BeautifulSoup
import re
import pandas as pd
import json 

def clean_html_tags(data: str) -> str:
    """
    Remove HTML tags from textual data.
    """
    if not isinstance(data, str):
        return data  
        
    if not data:
        return
    
    soup = BeautifulSoup(data, "html.parser")
    return soup.get_text()

def remove_hashtags(text: str) -> str:
    """
    Remove hashtags from textual data.
    """
    if not isinstance(text, str):
        return text
        
    if not text:
        return
    
    return re.sub(r"#\w+", "", text)

def count_separator_rows(patterns: list, column: pd.Series) -> int:
    """
    Count rows in column with specified separators. 
    
    Prints the total number of rows in the column, the number of rows with each separator and the total number of rows with any of the separators.
    """
    
    column_len = len(column)
    print(f"Number of rows in column: {column_len}")

    rows_with_pattern_separators = {}
    for pattern in patterns:
        rows_with_pattern_separator = column.str.contains(pattern, regex=True, na=False)
        rows_with_pattern_separators[pattern] = rows_with_pattern_separator
        print(f"Number of rows with {repr(pattern)} separator: {rows_with_pattern_separator.sum()}")
        
    # series of false with len of step_column. we use OR on it and count the number of rows with any of the separators listed
    rows_with_either = pd.Series([False] * column_len)

    for pattern, pattern_matches in rows_with_pattern_separators.items():
        rows_with_either = rows_with_either | pattern_matches
        
    count_either = rows_with_either.sum()
    print(f"Number of rows with either ',.' or newlines: {count_either}")
    
    return count_either

def split_steps(steps: str, separators: list, ends_with_dot: bool = True) -> list:
    """
    Split textual data into a list based on provided separators.
    """
    if not isinstance(steps, str):
        return steps
    
    pattern = "|".join(separators)
    steps_list = re.split(pattern, steps)
    
    # ensure consistency, all steps end with a dot
    steps_list = [step.strip() for step in steps_list if step.strip()]
    
    if ends_with_dot:
        for i in range(len(steps_list)):
            if not steps_list[i].endswith('.'):
                steps_list[i] += '.'
    
    return steps_list

def find_max_len_doc(docs: list) -> tuple:
    """
    Find the document with the maximum length, in characters.
    """
    max_length = 0
    max_index = 0
    for i, doc in enumerate(docs):
        length = len(doc.page_content)
        if length > max_length:
            max_length = length
            max_index = i
            
    return max_length, max_index

def format_doc_for_display(doc: str) -> str:
    """
    Format document for display by chatbot.
    """
    return doc

def format_dictionary_pairs(dictionary: dict, number_of_last_pairs: int = 20) -> str:
    """
    Format dictionary pairs for integration into LLM context.
    """
    formatted_pairs = ""
    items = list(dictionary.items())
    last_n_items = items[-number_of_last_pairs:] if len(items) > number_of_last_pairs else items
    
    for key, value in last_n_items:
        formatted_pairs += f"Question:{key}. Answer: {value}\n\n"
    
    return formatted_pairs

def parse_context(context_str: str) -> dict:
    """
    Parse the context string into a dictionary.
    """
    context = {}
    lines = context_str.split('\n')
    for line in lines:
        if line.startswith("recipe_name:"):
            context["recipe_name"] = line.replace("recipe_name:", "").strip()
        elif line.startswith("author_name:"):
            context["author_name"] = line.replace("author_name:", "").strip()
        elif line.startswith("ingredients:"):
            context["ingredients"] = line.replace("ingredients:", "").strip()
        elif line.startswith("steps:"):
            steps_str = line.replace("steps:", "").strip()
            steps_list = re.findall(r"'(.*?)'", steps_str)
            context["steps"] = steps_list
            
    return context

def format_context(context) -> str:
    """
    Format the context dictionary into a user-friendly format.
    """
    context = parse_context(context)
    formatted_context = ""
    if "recipe_name" in context:
        formatted_context += f"\033[1;30mRecipe Name:\033[1;30m{context['recipe_name']}\n"
    if "author_name" in context:
        formatted_context += f"\033[1;30mAuthor Name:\033[1;30m{context['author_name']}\n"
    if "ingredients" in context:
        formatted_context += f"\033[1;30mIngredients:\033[1;30m{context['ingredients']}\n"
    if "steps" in context:
        formatted_context += "\033[1;30mSteps:\033[1;30m\n"
        for i, step in enumerate(context["steps"], 1):
            formatted_context += f"  {i}. {step}\n"
    return formatted_context
