from components.vectodb import Database
from utils.config import *
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain_huggingface import HuggingFaceEmbeddings # Used for converting retrieval queries into dense vector embeddings
from langchain_openai import ChatOpenAI # Used for text generation
from langchain.chains import RetrievalQA # Used for creating a RAG pipeline
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore as pc # Used for connecting the vectordatabase to the RAG pipeline
import os
import json
from components.data_collection import data_collection

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class MarketNews(BaseTool):
    name = "Market News"
    description = "Use the entire user query to get the Financial market news from this tool. This tool returns a string"

    def _run(self, query):
        ai = ChatOpenAI(model = "gpt-4o", temperature=0)
        prompt = f"""
        Based on the query provided below, give me the topics and the name of the stock (eg - AAPL).
        Give me the topics only from the options provided below and give me a single name of the stock in the json format.
        Query - 
        {query}

        Topics available - 
        blockchain, earnings, financial_markets, manufacturing, real_estate, technology

        Output format - 
        1. stock_name
        2. topic

        Don't explain anything just give me the json output. Also, make sure the topics are a list. 
        Mention only one topic and don't give me a topic list, just give me just a string.
        """
        result = ai.invoke(prompt)
        result = json.loads(str(result[7:-3]))
        database = Database(PINECONE_API)
        database.upsert(result["stock_name"], result["topic"])
        pinecone = Pinecone(PINECONE_API)
        index = pinecone.Index("financial-market-summarization")
        embeddings = HuggingFaceEmbeddings(model_name="msmarco-bert-base-dot-v5") # Creating a langchain's Sentence Transformer embedding function for the same transformer model
        db = pc(index = index, 
                embedding = embeddings,
                text_key = "context")
        llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo") # Creating a LLM object for RAG pipeline
        pipeline = RetrievalQA.from_chain_type(llm = llm, chain_type= "stuff", retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 100})) # Generating RAG pipeline
        news = pipeline.invoke(query)
        return news

    def _arun(self, query: str):
        raise NotImplementedError("This tool doesn't support async")

class CompanyDividends(BaseTool):
    name = "Company Dividends"
    description = "Use the entire user query to get the dividends for a particular company from this tool. This tool returns a JSON object"

    def _run(self, query):
        ai = ChatOpenAI(model = "gpt-4o", temperature=0)
        prompt = f"""
        Based on the query provided below, give me the name of the stock (eg - AAPL).
        Give me a single name of the stock in the json format.
        Query - 
        {query}

        Output format - 
        1. stock_name

        Don't explain anything just give me the json output.
        """
        result = ai.invoke(prompt)
        result = json.loads(result.content[7:-3])
        collection = data_collection(ALPHA_API)
        dividends = collection.company_dividends(result["stock_name"])
        return dividends

    def _arun(self, query: str):
        raise NotImplementedError("This tool doesn't support async")

class CompanyOverview(BaseTool):
    name = "Company Overview"
    description = "Use the entire user query to get an overview for a particular company from this tool. This tool returns a JSON object"

    def _run(self, query):
        ai = ChatOpenAI(model = "gpt-4o", temperature=0)
        prompt = f"""
        Based on the query provided below, give me the name of the stock (eg - AAPL).
        Give me a single name of the stock in the json format.
        Query - 
        {query}

        Output format - 
        1. stock_name

        Don't explain anything just give me the json output.
        """
        result = ai.invoke(prompt)
        print(result.content)
        result = json.loads(result.content[7:-3])
        collection = data_collection(ALPHA_API)
        overview = collection.overview(result["stock_name"])
        return overview
    
class GainerLosers(BaseTool):
    name = "Gainers and Losers"
    description = "Use this tool to get the top gainers and losers in the market. This tool returns a JSON object"

    def _run(self, query: str):
        collection = data_collection(ALPHA_API)
        gl = collection.gainers_losers()
        return gl
    
    def _arun(self):
        raise NotImplementedError("This tool doesn't support async")


class rag_finance:
    def __init__(self, pinecone_api_key, openai_api_key):
        self.pinecone = Pinecone(pinecone_api_key)
        self.index = self.pinecone.Index("financial-market-summarization")
        os.environ["OPENAI_API_KEY"] = openai_api_key

    def agent(self):
        tools = [MarketNews(), GainerLosers(), CompanyDividends(), CompanyOverview()]
        llm = ChatOpenAI(model = "gpt-4o", temperature=0)
        finance_agent = initialize_agent(tools = tools,
                                         llm = llm,
                                         verbose = True, 
                                         max_iteration = 5
                                        )
        return finance_agent
