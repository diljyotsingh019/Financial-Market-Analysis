import requests

class data_collection:
    """
    A class to collect data from the API
    """
    def __init__(self, api_key):
        self.api_key=api_key

    def overview(self, ticker):
        """
        A method to extract company's overview
        Args: ticker - Name of the stock/ Index
        Returns - Json data from the API
        """
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.api_key}'
        r = requests.get(url)
        data = r.json()
        return data

    def news_summary(self, ticker, topics, sort="LATEST"):
        """
        A method to extract market news summaries
        Args: ticker - Name of the stock/ Index
              topics - Topics for the particular stock/ index
        Returns - Json data from the API
        """
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_key}&topic={topics}&limit=1000&sort={sort}'
        r = requests.get(url)
        data = r.json()
        return data

    def gainers_losers(self):
        """
        A method to extract top and the worst performers in the market right now
        Args: None
        Returns - Json data from the API
        """
        url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={self.api_key}'
        r = requests.get(url)
        data = r.json()
        return data 
    
    def company_dividends(self, ticker):
        """
        A method to extract a company's dividend history
        Args: ticker - Name of the stock/ Index
        Returns - Json data from the API
        """
        url = f'https://www.alphavantage.co/query?function=DIVIDENDS&symbol={ticker}&apikey={self.api_key}'
        r = requests.get(url)
        data = r.json()
        return data