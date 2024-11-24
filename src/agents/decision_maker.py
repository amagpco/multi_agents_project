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
                f"You are an intelligent decision-making assistant. Analyze the user's query: '{query}', summary: '{summary}', "
                f"and feedback: '{feedback}' to deliver a concise, actionable recommendation. Respond directly without introductory phrases or questions."
            )
            return prompt
        except Exception as e:
            print(f"Error in Prompt Formatting: {e}")
            return ""
