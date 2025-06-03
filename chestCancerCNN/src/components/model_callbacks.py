import os
import tensorflow as tf
from typing import List, Optional
from src.entity.config_entity import CallbacksConfig 
from src.utils.logging_setup import logger 

class ModelCallbacks:
    """
    Manages the creation of Keras callbacks based on configuration.
    """
    def __init__(self, config: CallbacksConfig):
        """
        Initializes ModelCallbacks with configuration.

        Args:
            config (CallbacksConfig): Configuration dataclass for callbacks.
        """
        self.config = config
        logger.info(f"ModelCallbacks initialized with config: {self.config}")

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Creates and returns a list of Keras callbacks based on the configuration.

        Returns:
            List[tf.keras.callbacks.Callback]: A list of configured Keras callbacks.
        """
        callbacks_list: List[tf.keras.callbacks.Callback] = []

        if self.config.use_tensorboard and self.config.tensorboard_log_dir:
            os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.config.tensorboard_log_dir,
                histogram_freq=getattr(self.config, 'tensorboard_histogram_freq', 1) # Default to 1 if not specified
            )
            callbacks_list.append(tensorboard_callback)
            logger.info(f"TensorBoard callback configured. Log directory: {self.config.tensorboard_log_dir}")

        if self.config.use_model_checkpoint and self.config.checkpoint_model_filepath:
            os.makedirs(os.path.dirname(self.config.checkpoint_model_filepath), exist_ok=True)
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config.checkpoint_model_filepath,
                monitor=getattr(self.config, 'checkpoint_monitor', 'val_loss'),
                save_best_only=getattr(self.config, 'checkpoint_save_best_only', True),
                save_weights_only=getattr(self.config, 'checkpoint_save_weights_only', False),
                mode=getattr(self.config, 'checkpoint_mode', 'auto'),
                verbose=getattr(self.config, 'checkpoint_verbose', 1)
            )
            callbacks_list.append(checkpoint_callback)
            logger.info(f"ModelCheckpoint callback configured. Filepath: {self.config.checkpoint_model_filepath}")

        if self.config.use_early_stopping:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor=getattr(self.config, 'early_stopping_monitor', 'val_loss'),
                patience=getattr(self.config, 'early_stopping_patience', 10),
                min_delta=getattr(self.config, 'early_stopping_min_delta', 0),
                mode=getattr(self.config, 'early_stopping_mode', 'auto'),
                restore_best_weights=getattr(self.config, 'early_stopping_restore_best_weights', True),
                verbose=getattr(self.config, 'early_stopping_verbose', 1)
            )
            callbacks_list.append(early_stopping_callback)
            logger.info("EarlyStopping callback configured.")

        if self.config.use_reduce_lr_on_plateau:
            reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=getattr(self.config, 'reduce_lr_monitor', 'val_loss'),
                factor=getattr(self.config, 'reduce_lr_factor', 0.1),
                patience=getattr(self.config, 'reduce_lr_patience', 5),
                min_lr=getattr(self.config, 'reduce_lr_min_lr', 1e-6), # ensure min_lr is float
                verbose=getattr(self.config, 'reduce_lr_verbose', 1)
            )
            callbacks_list.append(reduce_lr_callback)
            logger.info("ReduceLROnPlateau callback configured.")
            
        if getattr(self.config, 'use_csv_logger', False) and getattr(self.config, 'csv_logger_filepath', None):
            os.makedirs(os.path.dirname(self.config.csv_logger_filepath), exist_ok=True)
            csv_logger_callback = tf.keras.callbacks.CSVLogger(
                filename=self.config.csv_logger_filepath,
                append=getattr(self.config, 'csv_logger_append', False)
            )
            callbacks_list.append(csv_logger_callback)
            logger.info(f"CSVLogger callback configured. Filepath: {self.config.csv_logger_filepath}")


        if not callbacks_list:
            logger.info("No Keras callbacks were configured or enabled.")
            
        return callbacks_list
