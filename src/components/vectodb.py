from components.data_collection import * # Used for data collection
from utils.config import ALPHA_API # Used to configure api key
from pinecone import Pinecone # Used to create a vector database
from sentence_transformers import SentenceTransformer # Used to create embeddings
import torch # Used to perform torch operations

collection = data_collection(ALPHA_API) # Creating an object to collect the news summaries from the API

class Database:
    def __init__(self, api_key):
        self.api_key = api_key
        self.pinecone = Pinecone(api_key = api_key)
        self.index = self.pinecone.Index("financial-market-summarization")
    
    def upsert(self, ticker: str, topics: str, sort: str = "LATEST"):
        """
        A method used to upsert vector embeddings to the vector database
        Args: ticker - Name of the stock/ Index
              topics - Topics for the particular stock/ index
        """
        device = "cuda" if torch.cuda.is_available() else "cpu" # Consider GPU if available
        dense_model = SentenceTransformer("msmarco-bert-base-dot-v5", device = device) # Creating a Transformer model for dense embeddings
        data = collection.news_summary(ticker, topics, sort) # Gather the news summaries from the API
        upserts = [] # Creating a list to upsert the embeddings
        for i,j in zip(data["feed"], range(0, len(data["feed"]))): # Iterating over every news summary
            text = i["title"]+"."+i["summary"] # Adding more text for the embeddings
            upserts.append({
                "id": str(j),
                "values" : dense_model.encode(text),
                "metadata" : {"URL":i["url"], "context": text}    
            })
        self.index.upsert(upserts) # Upsert all the new summaries to the Vector database with embeddings and relevant metadata

    def delete_records(self):
        """
        Deleting all the records within the database for next instance
        """
        self.index.delete(delete_all=True)




        

