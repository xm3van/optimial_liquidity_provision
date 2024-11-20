import os
import csv
import logging
from os.path import join
from typing import Optional, List, Dict
from dotenv import load_dotenv
from dune_client.client import DuneClient
from dune_client.query import QueryBase

# Load environment variables from .env file (optional)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str, query_id: int, api_key: Optional[str] = None, rerun: bool = False):
        """
        Initialize the DataLoader with a data source.

        Args:
            source (str): Path to the data file or API endpoint.
            query_id (int): The ID of the Dune query to execute if data is not available in the source.
            api_key (Optional[str]): Dune API key. Defaults to None, to use an environment variable.
            rerun (bool): Whether to re-run the query if cached data is available. Defaults to False.
        """
        self.data_dir = data_dir
        self.query_id = query_id
        self.path = join(self.data_dir, f"{str(self.query_id)}.csv")
        self.api_key = api_key or os.getenv("DUNE_KEY")
        self.rerun = rerun
    
    def load_data(self) -> Optional[List[Dict]]:
        """
        Load data from the source file if available, otherwise fetches data from the Dune API.

        Returns:
            Optional[List[Dict]]: Data rows from the source file or fetched from Dune API.
        """
        # Check if the source file exists
        if os.path.exists(self.path):
            logger.info(f"Loading data from {self.path}")
            return self._load_from_csv()
        else:
            logger.info(f"{self.path} not found. Fetching data from Dune API.")
            return self._fetch_and_save_data()
    
    def _load_from_csv(self) -> List[Dict]:
        """
        Load data from a CSV file.

        Returns:
            List[Dict]: Data rows loaded from the CSV file.
        """
        data = []
        with open(self.path, mode="r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                data.append(row)
        logger.info(f"Loaded {len(data)} rows from {self.path}")
        return data
    
    def _fetch_and_save_data(self) -> Optional[List[Dict]]:
        """
        Fetch data from Dune API and save it to the CSV file.

        Returns:
            Optional[List[Dict]]: Data rows fetched from Dune API.
        """
        data = self._fetch_dune_data()
        
        # Save data to CSV if fetch was successful
        if data:
            with open(self.path, mode="w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            logger.info(f"Data saved to {self.path}.")
        else:
            logger.warning("No data returned from the Dune API.")
        
        return data

    def _fetch_dune_data(self) -> Optional[List[Dict]]:
        """
        Fetches data from a Dune query and stores it as a CSV file in the specified directory
        if the file does not already exist or if rerun is True.
        
        Returns:
            Optional[List[Dict]]: List of rows containing the query result data, or None if the data 
                                  was already saved in a CSV file.
        
        Raises:
            ValueError: If the API key or query ID is not provided.
            Exception: For other errors during query execution or data retrieval.
        """
        if not self.api_key:
            raise ValueError("API key is required. Set it in .env or pass it as an argument.")
        
        # Initialize Dune client
        client = DuneClient(self.api_key)

        # Create a query object
        query = QueryBase(query_id=self.query_id)

        # Fetch the data
        try:
            if self.rerun:
                # Run the query if rerun is True
                results_response = client.run_query(query)
                logger.info("Data fetched successfully from a fresh execution.")
            else:
                # Fetch latest results if rerun is False
                results_response = client.get_latest_result(query)
                logger.info("Fetched the latest cached results.")

            # Access the rows directly from results_response
            rows = results_response.result.rows if results_response.result else []
            
            # Check if data is returned
            if not rows:
                logger.warning("No data returned from query.")
                return None
            
            return rows
        
        except Exception as e:
            logger.error(f"An error occurred while fetching data: {e}")
            return None

# # Example usage
# if __name__ == "__main__":
#     data_loader = DataLoader(data_dir="data/raw/", query_id=4251434, rerun=False)
#     data = data_loader.load_data()
#     if data:
#         for row in data:
#             print(row)
