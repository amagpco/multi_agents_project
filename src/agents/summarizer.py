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
                "You are an expert summarization assistant. Your task is to generate a concise and accurate summary "
                "of the provided retrieved data, considering the user's query and feedback."
                "###DON'T include introductory phrases like 'Here is a summary' or 'This is the information.' Instead, respond directly.\n\n"
                "### User Query:\n"
                f"{query}\n\n"
                "### Retrieved Data:\n"
                f"{retrieved_data}\n\n"
                "### User Feedback:\n"
                f"{feedback}\n\n"
                "### Instructions:\n"
                "1. Generate a summary that addresses the user's query explicitly.\n"
                "2. Ensure the summary includes key details from the retrieved data.\n"
                "3. Avoid unnecessary details or repetition.\n\n"
                "### Notes:\n"
                "- The summary should be concise, well-structured, and easy to read.\n"
                "- Use bullet points or paragraphs as appropriate.\n"
                "- Make sure to include any relevant points from the user's feedback.\n\n"
                "Now, generate a detailed and accurate summary based on the user's query, the retrieved data, and their feedback."
                "#DON'T ASK ANY QUESTION AT THE END OF RESPONSE"
            )
            return prompt
        except Exception as e:
            print(f"Error in Prompt Formatting: {e}")
            return ""