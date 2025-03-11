from deepeval.models.base_model import DeepEvalBaseLLM
from src.config import DEFAULT_QA_MODEL
from src.utils.generation import call_llm_openrouter

class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(self, model_name=DEFAULT_QA_MODEL):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        model_name = self.load_model()
        return call_llm_openrouter(prompt, model_name)

    async def a_generate(self, prompt: str) -> str:            
        model_name = self.load_model()
        return call_llm_openrouter(prompt, model_name)

    def get_model_name(self):
        return DEFAULT_QA_MODEL