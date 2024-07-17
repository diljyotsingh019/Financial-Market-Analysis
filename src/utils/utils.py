from components.vectodb import Database # Used to create a vector database
from utils.config import * # Used to gather all the API keys
from langchain.tools import BaseTool # Used to create a tool for the agent
from typing import Any # Used for typing operations
from uuid import UUID # Used for uuid operations
from langchain_core.agents import AgentAction, AgentFinish # Used to configure callback handler for the agent
from langchain_core.callbacks import BaseCallbackHandler # Used to create a callback handler for the agent
from langchain.agents import AgentExecutor, create_react_agent # Used to create a react agent
from langchain_core.prompts import PromptTemplate # Used to create a prompt for the agent
from langchain_huggingface import HuggingFaceEmbeddings # Used for converting retrieval queries into dense vector embeddings
from langchain_openai import ChatOpenAI # Used for text generation
from langchain.chains import RetrievalQA # Used for creating a RAG pipeline
from pinecone import Pinecone # Used to perform vector database operations
from langchain_pinecone import PineconeVectorStore as pc # Used for connecting the vectordatabase to the RAG pipeline
import os # Used for os operations
import json # Used for json operations
from components.data_collection import data_collection # Used to collect data from the API

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # Creating an environment to authenticate GPT models

# A tool used to answer user queries using the market news (RAG tool)
class MarketNews(BaseTool):
    name = "MarketNews"
    description = "You must use the entire user input to get the Financial market news from this tool"

    def _run(self, query):
        ai = ChatOpenAI(model = "gpt-4o", temperature=0) # Initialising the LLM
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
        result = ai.invoke(prompt) # Using the above prompt to get the stock name and the topic from the LLM
        result = json.loads(result.content[7:-3]) # Converting the text into json
        database = Database(PINECONE_API) # Creating an object for Vector Database upserts
        database.upsert(result["stock_name"], result["topic"]) # Creating a vector database using news summaries
        pinecone = Pinecone(PINECONE_API) # Connecting to the pinecone database
        index = pinecone.Index("financial-market-summarization") # Connecting to the right index within the database
        embeddings = HuggingFaceEmbeddings(model_name="msmarco-bert-base-dot-v5") # Creating a langchain's Sentence Transformer embedding function for the same transformer model
        db = pc(index = index, 
                embedding = embeddings,
                text_key = "context")
        llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo") # Creating a LLM object for the RAG pipeline
        pipeline = RetrievalQA.from_chain_type(llm = llm, chain_type= "stuff", retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 100})) # Generating RAG pipeline
        news = pipeline.invoke(query) # Using the RAG pipeline to answer user query
        return news 

    def _arun(self, query: str):
        raise NotImplementedError("This tool doesn't support async") # This tool doesn't support async method

# A tool used by the agent to extract company's dividend history
class CompanyDividends(BaseTool):
    name = "CompanyDividends"
    description = "You must use the entire user input to get the dividends for a particular company from this tool"

    def _run(self, query):
        ai = ChatOpenAI(model = "gpt-4o", temperature=0) # Initialising the above prompt
        prompt = f"""
        Based on the query provided below, give me the name of the stock (eg - AAPL).
        Give me a single name of the stock in the json format.
        Query - 
        {query}

        Output format - 
        1. stock_name

        Don't explain anything just give me the json output. Don't forget to prefix and end the output with ```json and ```.
        """
        result = ai.invoke(prompt) # Using the above prompt to retrieve the stock name from the given query
        result = json.loads(result.content[7:-3]) # Converting the text into json
        collection = data_collection(ALPHA_API) # Creating a data collection object to gather the json data
        dividends = collection.company_dividends(result["stock_name"]) # Using the company_dividends method to retrieve the data from the API
        new_prompt = f"""Based on the API's JSON data, try to respond the user query
        
        Query - 
        {query}

        Json Data -
        {dividends}
        """
        response = ai.invoke(new_prompt) # Answering the user query with the json data retrieved
        return response.content

    def _arun(self, query: str):
        raise NotImplementedError("This tool doesn't support async") # This tool doesn't support async method

class CompanyOverview(BaseTool):
    name = "CompanyOverview"
    description = "You must use the entire user input to get the current overview for a particular company from this tool"

    def _run(self, query):
        ai = ChatOpenAI(model = "gpt-4o", temperature=0) # Initialising the LLM
        prompt = f"""
        Based on the query provided below, give me the name of the stock (eg - AAPL).
        Give me a single name of the stock in the json format.
        Query - 
        {query}

        Output format - 
        1. stock_name

        Don't explain anything just give me the json output. Don't forget to prefix and end the output with ```json and ```.
        """
        result = ai.invoke(prompt) # Using the above prompt to retrieve the stock name from the query
        result = json.loads(result.content[7:-3]) # Converting the text object into json
        collection = data_collection(ALPHA_API) # Creating a data_collection object to retrieve data
        overview = collection.overview(result["stock_name"]) # Using the overview method to gather the data from the API
        new_prompt = f"""Based on the API's JSON data, try to respond the user query

        Query - 
        {query}

        Json Data -
        {overview}
        """
        response = ai.invoke(new_prompt) #Using the above prompt to answer the user query
        return response.content
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool doesn't support async") # This tool doesn't support async method

# A tool for the agent to retrieve the best and the worst performer in the market right now   
class GainerLosers(BaseTool):
    name = "GainersLosers"
    description = "You must use this tool to get the top gainers and losers in the market"

    def _run(self, query: str):
        collection = data_collection(ALPHA_API) # Creating a data_collection object to retrieve data
        ai = ChatOpenAI(model = "gpt-4o", temperature=0) # Initialising the LLM
        gl = collection.gainers_losers() # Retrieving the data from the API
        new_prompt = f"""Based on the API's JSON data, try to respond the user query

        Query - 
        {query}

        Json Data -
        {gl}
        """
        response = ai.invoke(new_prompt) # Using the above prompt to answer the user query
        return response.content
    
    def _arun(self):
        raise NotImplementedError("This tool doesn't support async") # This tool doesn't support async

# Creating a callback handler for the agent's actions and responses
class handler(BaseCallbackHandler):
    def __init__(self):
        self.log_actions = []
        self.log_finishes = []
    
    def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.log_actions.append({"Action": action}) # Logging agent's actions
        return super().on_agent_action(action, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.log_finishes.append({"Finish": finish}) # Logging agent's responses
        return super().on_agent_finish(finish, run_id=run_id, parent_run_id=parent_run_id, **kwargs)


class rag_finance:
    """
    A class for the RAG agent that answers Financial queries.
    """
    def __init__(self, pinecone_api_key, openai_api_key):
        self.pinecone = Pinecone(pinecone_api_key)
        self.index = self.pinecone.Index("financial-market-summarization")
        os.environ["OPENAI_API_KEY"] = openai_api_key

    def agent(self, handler):
        """
        A method to initialise the RAG agent
        Args: handler - Requires a callback handler to log agent's actions and responses
        Returns: returns the RAG agent
        """
        tools = [MarketNews(), GainerLosers(), CompanyDividends(), CompanyOverview()] # List of tools for the agent
        llm = ChatOpenAI(model = "gpt-4o", temperature=0) # Initialising the llm
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
                                        ) # Creating a react agent using the above prompt, llm and set of tools
        agent_executor = AgentExecutor(
                                agent=finance_agent, tools=tools, verbose=True, 
                                handle_parsing_errors=True, max_iterations= 15,
                                callbacks = [handler]
                            ) # Executing the react agent
        return agent_executor
    