from src.components.data_augmentation import DataAugmentation
from src.config.configuration import ConfigManager
from src.utils.logging_setup import logger
from typing import Dict, Tuple, List
import tensorflow as tf

class DataAugmentationPipeline:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.augmentation_config = self.config.get_augmentation_config()
        self.data_augmentation = DataAugmentation(config=self.augmentation_config)

    def run_pipeline(self, datasets: Dict[str, tf.data.Dataset], dataset_info: Dict[str, Dict]) -> Tuple[List[tf.keras.Sequential], Dict[str, Dict]]:
        """
        Apply data augmentation to the datasets.
        
        Args:
            datasets: Dictionary of datasets (train, validation, test)
            dataset_info: Dictionary with dataset information
            
        Returns:
            Tuple containing:
                - List[tf.keras.Sequential]: A Keras Sequential model containing the configured augmentation layers.
                - Dictionary with dataset information (unchanged)
        """
        try:
            logger.info("Starting data augmentation pipeline")
            
            # Apply augmentation to datasets
            augmentation_layers = self.data_augmentation.initiate_data_augmentation(datasets)
            
            logger.info("Data augmentation pipeline completed successfully")
            return augmentation_layers, dataset_info
        except Exception as e:
            logger.error(f"Error in data augmentation pipeline: {e}")
            raise e
