import requests

class data_collection:
    def __init__(self, api_key):
        self.api_key=api_key

    def overview(self, ticker):
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.api_key}'
        r = requests.get(url)
        data = r.json()
        return data

    def news_summary(self, ticker, topics, sort="LATEST"):
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_key}&topic={topics}&limit=1000&sort={sort}'
        r = requests.get(url)
        data = r.json()
        return data

    def gainers_losers(self):
        url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={self.api_key}'
        r = requests.get(url)
        data = r.json()
        return data 
    
    def company_dividends(self, ticker):
        url = f'https://www.alphavantage.co/query?function=DIVIDENDS&symbol={ticker}&apikey={self.api_key}'
        r = requests.get(url)
        data = r.json()
        return data