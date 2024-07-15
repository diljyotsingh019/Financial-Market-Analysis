from components.vectodb import Database
from utils.config import *
from langchain.tools import BaseTool
from typing import Any
from uuid import UUID
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
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
    name = "MarketNews"
    description = "You must use the entire user input to get the Financial market news from this tool"

    def _run(self, query):
        print(query)
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
        result = json.loads(result.content[7:-3])
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
    name = "CompanyDividends"
    description = "You must use the entire user input to get the dividends for a particular company from this tool"

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
        new_prompt = f"""Based on the API's JSON data, try to respond the user query
        
        Query - 
        {query}

        Json Data -
        {dividends}
        """
        response = ai.invoke(new_prompt)
        return response.content

    def _arun(self, query: str):
        raise NotImplementedError("This tool doesn't support async")

class CompanyOverview(BaseTool):
    name = "CompanyOverview"
    description = "You must use the entire user input to get the current overview for a particular company from this tool"

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
        new_prompt = f"""Based on the API's JSON data, try to respond the user query

        Query - 
        {query}

        Json Data -
        {overview}
        """
        response = ai.invoke(new_prompt)
        return response.content
    
class GainerLosers(BaseTool):
    name = "GainersLosers"
    description = "You must use this tool to get the top gainers and losers in the market"

    def _run(self, query: str):
        collection = data_collection(ALPHA_API)
        ai = ChatOpenAI(model = "gpt-4o", temperature=0)
        gl = collection.gainers_losers()
        new_prompt = f"""Based on the API's JSON data, try to respond the user query

        Query - 
        {query}

        Json Data -
        {gl}
        """
        response = ai.invoke(new_prompt)
        return response.content
    
    def _arun(self):
        raise NotImplementedError("This tool doesn't support async")

class handler(BaseCallbackHandler):
    def __init__(self):
        self.log_entries = []
    
    def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.log_entries.append({"Action": action})
        return super().on_agent_action(action, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.log_entries.append({"Finish": finish})
        return super().on_agent_finish(finish, run_id=run_id, parent_run_id=parent_run_id, **kwargs)


class rag_finance:
    def __init__(self, pinecone_api_key, openai_api_key):
        self.pinecone = Pinecone(pinecone_api_key)
        self.index = self.pinecone.Index("financial-market-summarization")
        os.environ["OPENAI_API_KEY"] = openai_api_key

    def agent(self):
        tools = [MarketNews(), GainerLosers(), CompanyDividends(), CompanyOverview()]
        llm = ChatOpenAI(model = "gpt-4o", temperature=0)
        react_prompt = PromptTemplate(input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'], 
                           template='''
                            You are very powerful assistant, that utilizes the tools to answer user queries.
                            Use the entire user query to get the result from the tool without changing a single word.

                            Answer the following questions as best you can. 
                                You have access to the following tools:
                                
                                {tools}

                            Use the following format:
                            
                            Question: the input question you must answer
                            Thought: you should always think about what to do
                            Action: the action to take, should be one of [{tool_names}]
                            Action Input: the input to the action
                            Observation: the result of the action
                            ... (this Thought/Action/Action Input/Observation can repeat N times)
                            Thought: I now know the final answer
                            Final Answer: the final answer to the original input question
                            
                            Begin!
                            
                            Question: {input}
                            Thought:{agent_scratchpad}''')
        finance_agent = create_react_agent(tools = tools,
                                         llm = llm, 
                                         prompt = react_prompt
                                        )
        agent_executor = AgentExecutor(
                                agent=finance_agent, tools=tools, verbose=True, handle_parsing_errors=True
                            )
        return agent_executor
    