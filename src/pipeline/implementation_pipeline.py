from components.vectodb import Database
from components.data_collection import *
from utils.config import *
from utils.utils import rag_finance, handler
import json 
from langchain_openai import ChatOpenAI
from flask import Flask, redirect, url_for, render_template, request
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

app = Flask(__name__)
@app.route("/", methods = ["POST", "GET"])
def home():
    if request.method == "POST":
        global query
        query = request.form["askai"]
        return redirect(url_for("agent"))
    else:
        return render_template("home.html")

@app.route("/insights", methods = ["GET", "POST"])
def agent():
    global query
    if request.method == "POST":
        query = request.form["askai"]
        return redirect(url_for("agent"))
    else:
        try:
            database = Database(PINECONE_API)
            database.delete_records()
        except:
            pass 
        callback_handler = handler()
        finance = rag_finance(PINECONE_API, OPENAI_API_KEY)
        agent = finance.agent(callback_handler)
        agent_result = agent.invoke({"input": query})
        if not callback_handler.log_actions:
            return render_template("home.html")
        else:      
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
            actions = {"CompanyDividends": ["dividends.html",  collection.company_dividends(result["stock_name"])],
                "GainersLosers": ["gainers_losers.html", collection.gainers_losers()], 
                "CompanyOverview": ["overview.html", collection.overview(result["stock_name"])],
                "MarketNews": ["market_news.html", ""]}
            
            data = actions[callback_handler.log_actions[-1]["Action"].tool][1]
            return render_template(actions[callback_handler.log_actions[-1]["Action"].tool][0], data = data, response = agent_result["output"])



if __name__ == "__main__":
    app.run(debug=True)