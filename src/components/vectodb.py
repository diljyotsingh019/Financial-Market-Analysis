from components.data_collection import *
from utils.config import ALPHA_API
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import torch

collection = data_collection(ALPHA_API)

class Database:
    def __init__(self, api_key):
        self.api_key = api_key
        self.pinecone = Pinecone(api_key = api_key)
        self.index = self.pinecone.Index("financial-market-summarization")
    
    def upsert(self, ticker: str, topics: str, sort: str = "LATEST"):
        device = "cuda" if torch.cuda.is_available() else "cpu" # Consider GPU if available
        dense_model = SentenceTransformer("msmarco-bert-base-dot-v5", device = device) # Creating a Transformer model for dense embeddings
        data = collection.news_summary(ticker, topics, sort)
        upserts = []
        for i,j in zip(data["feed"], range(0, len(data["feed"]))):
            text = i["title"]+"."+i["summary"]
            upserts.append({
                "id": str(j),
                "values" : dense_model.encode(text),
                "metadata" : {"URL":i["url"], "context": text}    
            })
        self.index.upsert(upserts)

    def delete_records(self):
        self.index.delete(delete_all=True)




        

