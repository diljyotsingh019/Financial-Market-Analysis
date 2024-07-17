from components.vectodb import Database # Used for vector database operations
from components.data_collection import * # Used to collect data
from utils.config import * # Used to config API keys
from utils.utils import rag_finance, handler # Used to create an agent and the callback handler
import json # Used to perform json
from langchain_openai import ChatOpenAI # Used for Natural Language generation
from flask import Flask, redirect, url_for, render_template, request # Used to create Web UI
import os # Used for os operations

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # Creating an os environment to authenticate GPT models

app = Flask(__name__)
@app.route("/", methods = ["POST", "GET"])
def home():
    """
    A method that initialises the Flask Web UI
    Args: None
    Returns: - Renders home.html (Initial Web UI)
             - Redirects to the agent method once a query has been submitted
    """
    if request.method == "POST": # Checking if a query has been submitted or not
        global query
        query = request.form["askai"] # Creating a global query object
        return redirect(url_for("agent"))
    else:
        return render_template("home.html")

@app.route("/insights", methods = ["GET", "POST"])
def agent():
    """
    A method the utilizes financial RAG agent to answer user queries on the Flask Web UI
    Args: None
    Returns: - Renders webpage based on user queries
             - Redirects to the agent method once a query has been submitted
    """
    global query
    if request.method == "POST": # Checking if there's a new user query on the UI
        query = request.form["askai"] # Changing the global query object to the new query
        return redirect(url_for("agent"))
    else:
        try:
            database = Database(PINECONE_API) # Creating a database object for deleting the records
            database.delete_records() # Deleting the records
        except:
            pass 
        callback_handler = handler() # Creating a handler for logging agent actions
        finance = rag_finance(PINECONE_API, OPENAI_API_KEY) # Creating a rag_finance object to create an agent
        agent = finance.agent(callback_handler) # Creating a RAG agent
        agent_result = agent.invoke({"input": query}) # Answering the user query
        if not callback_handler.log_actions: # If the Agent doesn't require any tools to respond (Non-Financial Queries)
            return render_template("home.html") # Render initial Web UI
        else:      
            ai = ChatOpenAI(model = "gpt-4o", temperature=0) # Initialising the LLM
            prompt = f"""
            Based on the query provided below, give me the name of the stock (eg - AAPL).
            Give me a single name of the stock in the json format.
            Query - 
            {query}

            Output format - 
            1. stock_name

            Don't explain anything just give me the json output.
            """
            result = ai.invoke(prompt) # Using the above query to retrieve the stock name
            result = json.loads(result.content[7:-3]) # Converting the text object into json
            collection = data_collection(ALPHA_API) # Creating a data_collection object to retrieve data from the API
            actions = {"CompanyDividends": ["dividends.html",  collection.company_dividends(result["stock_name"])],
                "GainersLosers": ["gainers_losers.html", collection.gainers_losers()], 
                "CompanyOverview": ["overview.html", collection.overview(result["stock_name"])],
                "MarketNews": ["market_news.html", ""]} # Rendering different webpages based on agent's last action
            
            data = actions[callback_handler.log_actions[-1]["Action"].tool][1] # Gather data from the API based on agent's last action
            return render_template(actions[callback_handler.log_actions[-1]["Action"].tool][0], data = data, response = agent_result["output"], query = query)



if __name__ == "__main__":
    app.run(debug=True)