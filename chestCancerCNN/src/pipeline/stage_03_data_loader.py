
from src.components.data_loader import DataLoader
from src.config.configuration import ConfigManager
from src.utils.logging_setup import logger

class DataLoaderPipeline:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.data_loader_config = self.config.get_data_loader_config()
        self.data_loader = DataLoader(config=self.data_loader_config)

    def run_pipeline(self):
        try:
            logger.info("Starting data loader pipeline")
            datasets, dataset_info = self.data_loader.load_and_prepare_datasets()
            logger.info("Data loader pipeline completed successfully")
            return datasets, dataset_info
        except Exception as e:
            logger.error(f"Error in data loader pipeline: {e}")
            raise e
        
