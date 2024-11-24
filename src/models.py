import os
import cohere
import openai
import requests

class LLMProvider:
    def __init__(self) -> None:
        self.ollama_endpoint = os.getenv('OLLAMA_ENDPOINT')
        self.llama_api_key = os.getenv('LLAMA_API_KEY')
        self.mistral_api_key = os.getenv('MISTRAL_API_KEY')
        self.cohere_api_key = os.getenv('COHERE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        openai.api_key = self.openai_api_key

    def __call__(self, model_name: str):
        model_classes = {
            "llama": LlamaModel,
            "mistral": MistralModel,
            "gpt": GPTModel,
            "cohere": CohereModel
        }

        model_class = model_classes.get(model_name.lower())
        if not model_class:
            raise ValueError(f"Model '{model_name}' is not supported.")
        
        return model_class(self)
    
class LLMModel:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
    
    def execute_prompt(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
class LlamaModel(LLMModel):
    def execute_prompt(self, prompt: str):
        payload = {"model": "llama3.2", "prompt": prompt}
        headers = {"Authorization": f"Bearer {self.provider.llama_api_key}"}
        try:
            response = requests.post(
                url=self.provider.ollama_endpoint, 
                json=payload, 
                headers=headers
            )
            response.raise_for_status()
            return response.json().get('data', [])
        except requests.exceptions.RequestException as e:
            print(f"LlamaModel Error: {e}")
            return None
        
class MistralModel(LLMModel):
    def execute_prompt(self, prompt: str):
        payload = {"model": "mistral", "prompt": prompt}
        headers = {"Authorization": f"Bearer {self.provider.mistral_api_key}"}
        try:
            response = requests.post(
                url=self.provider.ollama_endpoint, 
                json=payload, 
                headers=headers
            )
            response.raise_for_status()
            return response.json().get('data', [])
        except requests.exceptions.RequestException as e:
            print(f"MistralModel Error: {e}")
            return None
        
class CohereModel(LLMModel):
    def __init__(self, provider: LLMProvider) -> None:
        super().__init__(provider)
        self.client = cohere.Client()

    def execute_prompt(self, prompt: str):
        try:
            response = self.client.generate(prompt=prompt)
            return response.generations[0].text
        except requests.exceptions.RequestException as e:
            print(f"CohereModel Error: {e}")
            return None
        
class GPTModel(LLMModel):
    def execute_prompt(self, prompt: str):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a decision-making agent."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message['content']
        except openai.error.OpenAIError as e:
            print(f"GPTModel Error: {e}")
            return None