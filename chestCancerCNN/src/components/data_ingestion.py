import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from src.entity.config_entity import DataIngestionConfig
from tqdm import tqdm
from src.utils.logging_setup import logger

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_data(self):
        """
        Downloads the dataset from Kaggle to the specified data directory.
        """
        try:
            logger.info(f"Attempting to download dataset: {self.config.dataset_name}")
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Download the dataset
            api.dataset_download_files(
                self.config.dataset_name,
                path=self.config.data_dir,
                unzip=False
            )
            
            logger.info(f"Dataset '{self.config.dataset_name}' downloaded successfully to {self.config.data_dir}")
            return str(self.config.data_dir)
            
        except Exception as e:
            logger.error(f"Error downloading dataset '{self.config.dataset_name}': {e}")
            raise e
    
    def extract_zip(self, zip_path: Path = None):
        """
        Extracts the downloaded zip file with a progress bar.
        
        Args:
            zip_path (Path, optional): The path to the zip file. 
                                       If None, it will search for a zip file in the data directory.
        """
        try:
            if zip_path is None:
                # Find the zip file in the data directory
                zip_files = list(self.config.data_dir.glob("*.zip"))
                if not zip_files:
                    logger.error(f"No zip file found in the data directory: {self.config.data_dir}")
                    return None
                zip_path = zip_files[0]
            
            logger.info(f"Beginning extraction of zip file: {zip_path.name}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files to extract for tqdm
                file_list = zip_ref.namelist()
                
                for file in tqdm(file_list, desc=f"Extracting {zip_path.name}"):
                    zip_ref.extract(file, self.config.extracted_dir)
            
            logger.info(f"Zip file '{zip_path.name}' extracted successfully to {self.config.extracted_dir}")
            return str(self.config.extracted_dir)
            
        except Exception as e:
            logger.error(f"Error extracting zip file '{zip_path.name if zip_path else 'N/A'}': {e}")
            raise e
    
    def initiate_data_ingestion(self):
        """
        Orchestrates the entire data ingestion process: download and extraction.
        """
        logger.info("Starting the data ingestion process.")
        
        # Download the data
        download_path = self.download_data()
        if download_path is None:
            logger.error("Data download failed, stopping ingestion.")
            return None
        
        # Extract the downloaded data
        extracted_path = self.extract_zip()
        if extracted_path is None:
            logger.error("Data extraction failed, stopping ingestion.")
            return None
            
        logger.info("Data ingestion completed successfully.")
        return extracted_path