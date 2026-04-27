import google.generativeai as genai
# CHANGED: Import from .llm instead of .base
from websocietysimulator.llm.llm import LLMBase

class GeminiLLM(LLMBase):
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def __call__(self, prompt, **kwargs):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return ""