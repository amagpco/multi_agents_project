from llms.llm_provider import LLMProvider

class DecisionMakerAgent:
    def __init__(self, model: str) -> None:
        provider = LLMProvider()
        self.provider = provider(model_name=model)

    def decide(self, query: str, summary: str, feedback: str):
        try:
            prompt = self.decide_prompt_formatter(query, summary, feedback)
            return self.provider.execute_prompt(prompt)
        except Exception as e:
            # TODO Log e 
            print(f"Error rasied in executing prompt: {e}")
            return None   

    def decide_prompt_formatter(self, query: str, summary: str, feedback: str) -> str:
        try:
            prompt = (
                "You are an intelligent decision-making assistant. Your role is to analyze the user's query, "
                "the provided summary of retrieved data, and the user's feedback to deliver a short, actionable, "
                "and effective decision or recommendation."
                "###DON'T include introductory phrases like 'Here is a' or 'This is the information.' Instead, respond directly.\n\n"
                "### User Query:\n"
                f"{query}\n\n"
                "### Summary of Retrieved Data:\n"
                f"{summary}\n\n"
                "### User Feedback:\n"
                f"{feedback}\n\n"
                "### Instructions:\n"
                "1. Analyze the user's query and the provided summary carefully.\n"
                "2. Incorporate the user's feedback into your decision-making process.\n"
                "3. Provide a clear and concise decision or recommendation that directly addresses the query.\n"
                "4. Ensure the response is actionable, specific, and easy to understand.\n\n"
                "### Notes:\n"
                "- Keep the response brief but impactful.\n"
                "- Avoid unnecessary details or repetition.\n"
                "- Focus on delivering value to the user based on the query and context.\n\n"
                "Now, based on the provided details, generate a concise and effective decision or recommendation."
                "#DON'T ASK ANY QUESTION AT THE END OF RESPONSE"
            )
            return prompt
        except Exception as e:
            print(f"Error in Prompt Formatting: {e}")
            return ""