
from typing import List
from src.components.model_callbacks import ModelCallbacks
from src.config.configuration import ConfigManager
from src.utils.logging_setup import logger
import tensorflow as tf

class ModelCallbacksPipeline:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_callbacks_config = self.config.get_model_callbacks_config()
        self.model_callbacks = ModelCallbacks(config=self.model_callbacks_config)

    def run_pipeline(self) -> List[tf.keras.callbacks.Callback]:
        try:
            logger.info("Starting model callbacks pipeline")
            callbacks = self.model_callbacks.get_callbacks()
            logger.info("Model callbacks pipeline completed successfully")
            return callbacks
        except Exception as e:
            logger.error(f"Error in model callbacks pipeline: {e}")
            raise e
