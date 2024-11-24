from llms.llm_provider import LLMProvider

class SummarizerAgent:
    def __init__(self, model: str) -> None:
        provider = LLMProvider()
        self.provider = provider(model_name=model)

    def summarize(self, query: str, retreived_data:str, feedback: str):
        try:
            prompt = self.summarize_prompt_formatter(query, retreived_data, feedback)
            return self.provider.execute_prompt(prompt)
        except Exception as e:
            # TODO Log e 
            return None

    def summarize_prompt_formatter(self, query: str, retrieved_data: str, feedback: str) -> str:
        try:
            prompt = (
                "You are an expert summarization assistant. Generate a concise, structured summary of the provided data "
                f"based on the user's query: '{query}', retrieved data: '{retrieved_data}', and feedback: '{feedback}'. "
                "Respond directly without introductory phrases or questions, and include key details and actionable insights."
            )
            return prompt
        except Exception as e:
            print(f"Error in Prompt Formatting: {e}")
            return ""
