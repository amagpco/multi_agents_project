from dotenv import load_dotenv
from agents.agents import Agent
from agents.summarizer import SummarizerAgent
from agents.decision_maker import DecisionMakerAgent
from utils.text_chunker import TextChunker
from utils.embedding_ranker import ChunkRanker

# Load environment variables
load_dotenv("../config/settings.env")

class WorkflowManager:
    def __init__(self, query: str) -> None:
        self.query = query
        self.data_retreiver = Agent(
            model="cohere",
            name="Data Retreiver",
            role="You are an expert assistant providing detailed, accurate, and contextually relevant answers to the user's query.",
        )
        self.summarizer = Agent(
            model="cohere",
            name="Summarizer",
            role="You are an expert summarization assistant. Generate a concise, structured summary of the provided data based on the user's query.",
        )
        self.decision_maker = Agent(
            model="cohere",
            name="Decision Maker",
            role="You are an intelligent decision-making assistant."
        )

        # self.text_chunker = TextChunker()
        # self.chunk_ranker = ChunkRanker()
        self.steps = {
            'retreived_data': None,
            'retreived_data_feedback': None,
            'chunks': None,
            'top_chunks': None,
            'summary': None,
            'summary_feedback': None,
            'final_decision': None
        }

    def collect_user_feedback(self, step: str, data: str):
        # TODO Implement this funtion for HITL 
        choice = input(f"Do you approve the {step}? (yes/no):")
        if choice.lower() == "yes":
            return ""
        else:
            improve_feedback = input(f"Your Feedback on {step}: ")
            return improve_feedback

    def execute(self):
        try:
            # Step 1: Retreive Data 
            retreive_data_prompt=(
                "Respond directly without introductory phrases like 'Here is' and avoid asking questions at the end of the response."
                "Provide a clear, structured response with examples, accurate facts, and actionable insights where appropriate."
                f"User Query is: {self.query}"
            )
            self.steps['retreived_data'] = self.data_retreiver.execute(retreive_data_prompt)
            print('\nRetreived Data: \n',self.steps['retreived_data'])

            # Get user feedback for retreived data
            self.steps['retreived_data_feedback'] = self.collect_user_feedback('retreived_data', self.steps['retreived_data'])
            print(self.steps['retreived_data_feedback'])
            
            # Step 2: Chunk Data
            # self.steps['chunks'] = self.text_chunker.chunk_text(self.steps['retreived_data'])
            # print(self.steps['chunks'])

            # Step 3: Rerank Chunked Data
            # self.steps['top_chunks'] = self.chunk_ranker.rank_chunks(self.query, self.steps['chunks'])
            # print(self.steps['top_chunks'])

            summarize_prompt=(
                f"User query: '{self.query}', retrieved data: '{self.steps['retreived_data']}', and feedback: '{self.steps['retreived_data_feedback']}'. "
                "Respond directly without introductory phrases or questions, and include key details and actionable insights."
            )
            # Step 4: Summarize Top Chunks 
            self.steps['summary'] = self.summarizer.execute(summarize_prompt)
            print('\n Summarized Data: \n',self.steps['summary'])

            # Get user feedback for summarized text
            self.steps['summary_feedback'] = self.collect_user_feedback('summary', self.steps['summary'])
            print(self.steps['summary_feedback'])
            

            decide_prompt=(
                f"You are an intelligent decision-making assistant. Analyze the user's query: '{self.query}', summary: '{self.steps['summary']}', "
                f"and feedback: '{self.steps['summary_feedback']}' to deliver a concise, actionable recommendation. Respond directly without introductory phrases or questions."
            )
            # Step 5: Make Final Decision
            self.steps['final_decision'] = self.decision_maker.execute(decide_prompt)
            print('\n Final Decision:\n ', self.steps['final_decision'])
            
            return self.steps['final_decision']

        except Exception as e:
            print(e)

def main():
    query = input("Enter your query: ")
    workflow_manager = WorkflowManager(query)
    workflow_manager.execute()

if __name__ == "__main__":
    main()