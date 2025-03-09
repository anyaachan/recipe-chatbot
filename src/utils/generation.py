
import requests
from src.config import OPENROUTER_API_KEY

def call_llm_openrouter(prompt:str, model_name:str, json_schema:dict=None) -> str:
    """
    Call the OpenRouter API to generate text from a prompt using given model and optional schema.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    properties = {}
    for prop_name, description in json_schema.items():
        properties[prop_name] = {
            "type": "string",
            "description": description
        }
    
    if json_schema is None:
        payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}]
        }
    else:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": "true",
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": ["shouldRespond"],
                        "additionalProperties": "false"
                        }
                    }
                }
            }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions", 
        headers=headers, 
        json=payload
        )
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]    