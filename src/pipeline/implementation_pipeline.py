from components.vectodb import Database
from utils.config import *
from langchain_huggingface import HuggingFaceEmbeddings # Used for converting retrieval queries into dense vector embeddings
from langchain_openai import ChatOpenAI # Used for text generation
from langchain.chains import RetrievalQA # Used for creating a RAG pipeline
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore as pc # Used for connecting the vectordatabase to the RAG pipeline
import os


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

database = Database(PINECONE_API)

# database.upsert("TSLA", "technology")
pinecone = Pinecone(PINECONE_API)
index = pinecone.Index("financial-market-summarization")

embeddings = HuggingFaceEmbeddings(model_name="msmarco-bert-base-dot-v5") # Creating a langchain's Sentence Transformer embedding function for the same transformer model
db = pc(index = index, 
        embedding = embeddings,
        text_key = "context")
llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo") # Creating a LLM object for RAG pipeline
pipeline = RetrievalQA.from_chain_type(llm = llm, chain_type= "stuff", retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})) # Generating RAG pipeline

result = pipeline.invoke("How is tesla stock performing")
print(result)