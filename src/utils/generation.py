
import requests
from src.config import OPENROUTER_API_KEY, DEFAULT_QA_MODEL
import json
import aiohttp

def call_llm_openrouter(prompt:str, 
                        model_name:str = DEFAULT_QA_MODEL,
                        schema:dict=None) -> str:
    """
    Call the OpenRouter API to generate text from a prompt using given model and optional schema.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    properties = {}

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}]
        }
            
    if schema is not None:
        for prop_name, description in schema.items():
            properties[prop_name] = {
                "type": "string",
                "description": description
            }
            
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                    "additionalProperties": False
                    }
                }
            }
    # print(list(properties.keys()))
    # print(payload)
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions", 
        headers=headers, 
        json=payload
        )
    response.raise_for_status()
    
    if schema is None:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return_dict = {}
        content_str = response.json()["choices"][0]["message"]["content"]
        content_dict = json.loads(content_str)
        
        for prop_name in properties.keys():
            return_dict[prop_name] = content_dict[prop_name]
        
        return return_dict