# Import required modules
import sys
import os
import json

# dotenv is used to load environment variables from a .env file (like MongoDB URL)
from dotenv import load_dotenv

# Load all variables from the .env file into the environment
load_dotenv()

# Get the MongoDB connection URL from the environment variable
MONGO_DB_URL = os.getenv("MONGO_DB_URL")  # Make sure your .env file has this variable
print(MONGO_DB_URL)  # For checking if the URL is loaded properly

# python package that provides root certificates
# used by python libraries that want to make secure (SSL/TLS) connection
# right now we are making a secure connection with MongoDB
# certifi gives path to SSL certificates needed for secure connection to services like MongoDB Atlas
import certifi
ca = certifi.where()

# pandas and numpy for data handling
import pandas as pd
import numpy as np

# pymongo is used to interact with MongoDB
import pymongo

# Custom exception and logging (you must have defined these in your project)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# Create a class to handle the data extraction and insertion process
class NetworkDataExtract():
    def __init__(self):
        try:
            # Constructor, right now it doesn't do anything
            pass
        except Exception as e:
            # Raise custom exception if something goes wrong
            raise NetworkSecurityException(e, sys)

    # This function reads a CSV file and converts it to a JSON-like format (list of dictionaries)
    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)  # Read CSV file using pandas
            data.reset_index(drop=True, inplace=True)  # Reset index to avoid unwanted columns
            records = list(json.loads(data.T.to_json()).values())  # Convert to list of JSON records
            return records  # Return the list of records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # This function connects to MongoDB and inserts the data into a specified collection
    def insert_data_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            # Connect to MongoDB using the connection URL
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)

            # Access the database
            self.database = self.mongo_client[self.database]

            # Access the collection within the database
            self.collection = self.database[self.collection]

            # Insert the list of records into the MongoDB collection
            self.collection.insert_many(self.records)

            # Return the number of records inserted
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

# This block will only run if the file is executed directly (not imported somewhere)
if __name__ == '__main__':
    # Path to the CSV file containing the data
    FILE_PATH = "Network_Data\\phisingData.csv"

    # Name of the database in MongoDB
    DATABASE = "TEJAS_ITANKAR_NETWORK_SECURITY_ML"

    # Name of the collection inside the database
    Collection = "NetworkData"

    # Create object of the NetworkDataExtract class
    networkobj = NetworkDataExtract()

    # Convert CSV file to JSON-like records
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)

    # Insert the records into MongoDB and get the count of inserted records
    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, Collection)

    # Print the number of inserted records
    print(no_of_records)
