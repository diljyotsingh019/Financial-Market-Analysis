from components.vectodb import Database
from utils.config import *
from utils.utils import rag_finance
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, render_template, request


# database = Database(PINECONE_API)

finance = rag_finance(PINECONE_API, OPENAI_API_KEY)

agent = finance.agent()

agent.invoke({"input":"Does IBM give out any dividends? If so, how much?"})


# database.delete_records()

app = Flask(__name__)
@app.route("/", methods = ["POST", "GET"])
def home():
    if request.method == "POST":
        return redirect(url_for("agent"))
    else:
        render_template("home.html")

@app.route("/insights", methods = ["GET", "POST"])
def agent():
    return render_template("dividends.html")


if __name__ == "__main__":
    app.run(debug=True)