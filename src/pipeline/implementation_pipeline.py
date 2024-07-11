from components.vectodb import Database
from utils.config import *
from utils.utils import rag_finance


# database = Database(PINECONE_API)

finance = rag_finance(PINECONE_API, OPENAI_API_KEY)

agent = finance.agent()

agent.invoke("Which stocks are performing the best right now?")


# database.delete_records()