from src.components.data_augmentation import DataAugmentation
from src.config.configuration import ConfigManager
from src.utils.logging_setup import logger
from typing import Dict, Tuple
import tensorflow as tf

class DataAugmentationPipeline:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.augmentation_config = self.config.get_augmentation_config()
        self.data_augmentation = DataAugmentation(config=self.augmentation_config)

    def run_pipeline(self, datasets: Dict[str, tf.data.Dataset], dataset_info: Dict[str, Dict]) -> Tuple[Dict[str, tf.data.Dataset], Dict[str, Dict]]:
        """
        Apply data augmentation to the datasets.
        
        Args:
            datasets: Dictionary of datasets (train, validation, test)
            dataset_info: Dictionary with dataset information
            
        Returns:
            Tuple containing:
                - Dictionary of augmented datasets
                - Dictionary with dataset information (unchanged)
        """
        try:
            logger.info("Starting data augmentation pipeline")
            
            # Apply augmentation to datasets
            augmented_datasets = self.data_augmentation.apply_augmentation(datasets)
            
            # Visualize augmentations on a sample of the training data
            if "train" in datasets:
                self.data_augmentation.visualize_augmentations(datasets["train"])
            
            logger.info("Data augmentation pipeline completed successfully")
            return augmented_datasets, dataset_info
        except Exception as e:
            logger.error(f"Error in data augmentation pipeline: {e}")
            raise e
