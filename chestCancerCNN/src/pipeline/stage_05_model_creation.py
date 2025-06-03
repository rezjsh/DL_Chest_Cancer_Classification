
from src.components.create_model import CreateModel
from src.config.configuration import ConfigManager
from src.utils.logging_setup import logger
from typing import Optional, Tuple
import tensorflow as tf

class ModelCreationPipeline:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.create_model_config = self.config.get_create_model_config()
        self.create_model = CreateModel(config=self.create_model_config,)

    def run_pipeline(self, input_shape: Tuple[int, int, int], num_classes: int, data_augmentation_model: Optional[tf.keras.Model] = None) -> tf.keras.Model:
        try:
            logger.info("Starting model creation pipeline")
            
            model = self.create_model.create_model(input_shape, num_classes, data_augmentation_model)
            logger.info("Model creation pipeline completed successfully")
            return model
        except Exception as e:
            logger.error(f"Error in model creation pipeline: {e}")
            raise e
    
