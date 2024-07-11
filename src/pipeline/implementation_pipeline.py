from components.vectodb import Database
from utils.config import *
from utils.utils import rag_finance


# database = Database(PINECONE_API)

finance = rag_finance(PINECONE_API, OPENAI_API_KEY)

agent = finance.agent()

agent.invoke("Does IBM give out any dividends? If so, how much?")


# database.delete_records()