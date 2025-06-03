from src.components.data_validation import DataValidation
from src.config.configuration import ConfigManager
from src.utils.logging_setup import logger

class DataValidationPipeline:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.data_validation_config = self.config.get_data_validation_config()
        self.data_validation = DataValidation(config=self.data_validation_config)

    def run_pipeline(self):
        try:
            logger.info("Starting data validation pipeline")
            validation_status = self.data_validation.initiate_data_validation()
            logger.info(f"Data validation completed with status: {validation_status}")
            return validation_status
        except Exception as e:
            logger.error(f"Error in data validation pipeline: {e}")
            raise e