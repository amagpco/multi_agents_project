from llms.llm_provider import LLMProvider

class DataRetreiverAgent:
    def __init__(self, model: str) -> None:
        provider = LLMProvider()
        self.provider = provider(model_name=model)

    def retreive(self, query):
        try:
            prompt = self.retrieve_prompt_formatter(query)
            response =  self.provider.execute_prompt(prompt)
            return response
        except Exception as e:
            print(e)
            # TODO Log the e
            return None
        
    def retrieve_prompt_formatter(self, query: str) -> str:
        try:
            prompt = (
                f"You are an expert assistant providing detailed, accurate, and contextually relevant answers to the user's query: {query}."
                "Respond directly without introductory phrases like 'Here is' and avoid asking questions at the end of the response."
                "Provide a clear, structured response with examples, accurate facts, and actionable insights where appropriate."
            )
            return prompt
        except Exception as e:
            print(f"Error encountered in prompt formatting: {e}")
            return ""