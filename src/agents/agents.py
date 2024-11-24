from llms.llm_provider import LLMProvider

class Agent:
    def __init__(self, model: str, name: str, role: str) -> None:
        provider = LLMProvider()
        self.provider = provider(model_name=model)
        self.name = name
        self.role = role

    def execute(self, prompt: str):
        try:
            prompt = self._extra_context(prompt=prompt)
            response =  self.provider.execute_prompt(prompt)
            return response
        except Exception as e:
            # TODO Log the e
            print(e)
            return None
    
    def _extra_context(self, prompt: str):
        return (
            "You are a helpful agent with the following information:\n"
            f"Your Name is: {self.name}\n"
            f"Your Role is: {self.role}\n"
            f"Additional Context: {prompt}\n"
        )