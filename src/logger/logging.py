# Importing logging libraries
import logging
from logging.handlers import TimedRotatingFileHandler

#Creating a function to setup logs
def log_setup(logname, file):
    logger = logging.getLogger(logname) #creating an object for the desired logger
    logger.setLevel(logging.DEBUG) # Setting the level of logs (Levels: DEBUG --> INFO --> ERROR --> CRITICAL)
    handler = TimedRotatingFileHandler(file, "midnight", interval = 1, encoding = "utf-8") # Creating a handler to handle logs
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s") #Creating a formatter for logs
    handler.setFormatter(formatter) #Setting the formatter
    handler.suffix = "%Y%m%d" # Suffix log file names with year-month-day
    logger.addHandler(handler) # Adding handler to the logger object

    return logger