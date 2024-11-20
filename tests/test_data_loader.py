import unittest
from unittest.mock import patch, mock_open, MagicMock
from src.data.data_loader import DataLoader  # Ensure this imports the correct path to your DataLoader

DEMO_ID = 4228314

class TestDataLoader(unittest.TestCase):
    @patch("os.path.exists")
    @patch("src.data.data_loader.DataLoader._fetch_dune_data")
    def test_load_data_from_csv(self, mock_fetch_dune_data, mock_exists):
        # Simulate that the CSV file exists
        mock_exists.return_value = True

        # Sample data to be returned by _load_from_csv
        csv_data = [{"col1": "value1", "col2": "value2"}]
        mock_csv_content = "col1,col2\nvalue1,value2\n"

        # Use mock_open to simulate reading from a CSV file
        with patch("builtins.open", mock_open(read_data=mock_csv_content)) as mock_file:
            data_loader = DataLoader(source="data/test_query.csv", query_id=DEMO_ID)
            data = data_loader.load_data()

            # Check that open was called with the source file
            mock_file.assert_called_once_with("data/test_query.csv", mode="r", newline="", encoding="utf-8")
            
            # Ensure fetch_dune_data is not called when CSV exists
            mock_fetch_dune_data.assert_not_called()

            # Verify that data loaded matches the expected CSV content
            self.assertEqual(data, csv_data)

    @patch("os.path.exists")
    @patch("src.data.data_loader.DataLoader._fetch_dune_data")
    def test_fetch_data_when_csv_missing(self, mock_fetch_dune_data, mock_exists):
        # Simulate that the CSV file does not exist
        mock_exists.return_value = False

        # Mock the response from fetch_dune_data
        api_data = [{"col1": "value1", "col2": "value2"}]
        mock_fetch_dune_data.return_value = api_data

        # Use mock_open to simulate writing to a CSV file
        with patch("builtins.open", mock_open()) as mock_file:
            data_loader = DataLoader(source="data/test_query.csv", query_id=DEMO_ID)
            data = data_loader.load_data()

            # Verify that fetch_dune_data was called
            mock_fetch_dune_data.assert_called_once()

            # Verify that data matches the mock API data
            self.assertEqual(data, api_data)

            # Verify the file write calls contain the expected CSV header and data rows
            handle = mock_file()
            written_content = [call[0][0].replace('\r\n', '\n') for call in handle.write.call_args_list]
            self.assertIn("col1,col2\n", written_content)
            self.assertIn("value1,value2\n", written_content)

    @patch("os.path.exists", return_value=False)
    @patch("src.data.data_loader.DataLoader._fetch_dune_data", return_value=None)
    def test_fetch_data_no_results(self, mock_fetch_dune_data, mock_exists):
        # Test case when no data is returned from Dune API
        data_loader = DataLoader(source="data/test_query.csv", query_id=DEMO_ID)
        data = data_loader.load_data()

        # Verify fetch_dune_data was called once
        mock_fetch_dune_data.assert_called_once()

        # Ensure data is None when no data is returned
        self.assertIsNone(data)

if __name__ == "__main__":
    unittest.main()
