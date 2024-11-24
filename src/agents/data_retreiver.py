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
                "You are an expert data retrieval assistant. Your role is to provide a long, detailed, "
                "and contextually relevant response to the user's query."
                "###DON'T include introductory phrases like 'Here is a ' or 'This is the information.' Instead, respond directly.\n\n"
                "### User Query:\n"
                f"{query}\n\n"
                "### Instructions:\n"
                "1. Begin with a brief introduction summarizing the query.\n"
                "2. Provide a detailed response broken into clear sections.\n"
                "   - Include examples, references, or relevant facts where appropriate.\n"
                "   - Ensure the information is accurate and up-to-date.\n"
                "3. Conclude actionable insights related to the query.\n\n"
                "### Notes:\n"
                "- Be concise but thorough.\n"
                "- Avoid unnecessary repetition.\n\n"
                "Now, respond to the user's query in detail."
                "#DON'T ASK ANY QUESTION AT THE END OF RESPONSE"
            )
            return prompt
        except Exception as e:
            print(f"Error encountered in prompt formatting: {e}")
            return ""