# Advanced RAG Agent for Financial Market Analysis and News Summarization
## Objective:
Develop an advanced Retrieval-Augmented Generation (RAG) agent designed to assist financial analysts by retrieving and summarizing relevant financial market data and news articles. The agent will fetch information from a large database of financial news and reports, providing concise summaries and insights to aid in investment decision-making.

## Installation Steps:
1. Clone the respository
```sh
git clone https://github.com/diljyotsingh019/Financial-Market-Analysis.git
```
2. Create a `config.py` file inside src/utils directory and add the following api keys
```sh
ALPHA_API = "<YOUR ALPHA VANTAGE API KEY>"
PINECONE_API = "<YOUR PINECONE API KEY>"
OPENAI_API_KEY = "<YOUR OPENAI API KEY>"
LOGFILE = "logs/main.log"
```
Now you can either create a docker image or install using a virtual environment

### Docker installation steps
Note: Docker file installs pytorch with cuda version 12.1, if you have cuda 11.8 kindly modify the docker file 
Ref: https://pytorch.org/get-started/locally/

1. Build the docker image 
```sh
docker build -t rag-agent .  
```
2. Run the container
```sh
docker run -p 5000:5000 rag-agent 
```
3. Open a web browser and start the UI
```sh
http://localhost:5000/
```

### Linux Installation Steps (Ubuntu)
1. Create a virtual environment
```sh
virtualenv --python python3.10 rag-agent
```

2. Activate the enivronment
```sh
source rag-agent/bin/activate
```

3. Install pytorch cuda version from https://pytorch.org/get-started/locally/
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Install the dependencies
```sh
pip install -r requirements.txt
```

5. Execute the python script and open the flask UI on a web browser (Hosted Locally) 
```sh
python3 src/pipeline/implementation_pipeline.py
```

6. Open a web browser
```sh
http://localhost:5000/
```

## Examples:
 ![dividends_example](https://github.com/user-attachments/assets/8e98b55d-1081-448a-bb44-1a44a6984564)
 
![gainer_losers](https://github.com/user-attachments/assets/cb78da3e-bb6c-4bd1-aa06-91e8782205de)

![market_news](https://github.com/user-attachments/assets/5fc763c4-d3d6-4423-88a7-3dcb640046c9)
