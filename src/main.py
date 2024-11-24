from dotenv import load_dotenv
from agents.data_retreiver import DataRetreiverAgent
from agents.summarizer import SummarizerAgent
from agents.decision_maker import DecisionMakerAgent
from utils.text_chunker import TextChunker
from utils.embedding_ranker import ChunkRanker

# Load environment variables
load_dotenv("../config/settings.env")

class WorkflowManager:
    def __init__(self, query: str) -> None:
        self.query = query
        self.data_retreiver = DataRetreiverAgent(model="cohere")
        self.summarizer = SummarizerAgent(model="cohere")
        self.decision_maker = DecisionMakerAgent(model="cohere")
        self.text_chunker = TextChunker()
        self.chunk_ranker = ChunkRanker()
        self.steps = {
            'retreived_data': None,
            'chunks': None,
            'top_chunks': None,
            'summary': None,
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
            self.steps['retreived_data'] = self.data_retreiver.retreive(query=self.query)
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

            # Step 4: Summarize Top Chunks 
            self.steps['summary'] = self.summarizer.summarize(query=self.query, retreived_data=self.steps['retreived_data'], feedback=self.steps['retreived_data_feedback'])
            print('\n Summarized Data: \n',self.steps['summary'])

            # Get user feedback for summarized text
            self.steps['summary_feedback'] = self.collect_user_feedback('summary', self.steps['summary'])
            print(self.steps['summary_feedback'])
            
            # Step 5: Make Final Decision
            self.steps['final_decision'] = self.decision_maker.decide(query=self.query, summary=self.steps['summary'], feedback=self.steps['summary_feedback'])
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