# Advanced RAG Agent for Financial Market Analysis and News Summarization (Currently Working)
## Objective:
Develop an advanced Retrieval-Augmented Generation (RAG) agent designed to assist financial analysts by retrieving and summarizing relevant financial market data and news articles. The agent will fetch information from a large database of financial news and reports, providing concise summaries and insights to aid in investment decision-making.



### Linux Installation Steps (Ubuntu)
1. Create a virtual environment
```sh
virtualenv --python python3.10 rag-agent
```

2. Activate the enivronment
```sh
source rag-agent/bin/activate
```

3. Install the dependencies
```sh
pip install -r requirements.txt
```
4. Create a `config.py` file inside src/utils directory and add the following api keys
```sh
ALPHA_API = "<YOUR ALPHA VANTAGE API KEY>"
PINECONE_API = "<YOUR PINECONE API KEY>"
OPENAI_API_KEY = "<YOUR OPENAI API KEY>"
```

5. Execute the python script and open the flask UI on a web browser (Hosted Locally) 
```sh
python3 src/pipeline/implementation_pipeline.py
```

## Currently working on:
1. Logging and code documentation
2. Containerization (Docker)
3. Testing 

 