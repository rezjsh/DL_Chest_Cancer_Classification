from src.config.configuration import ConfigManager
from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.pipeline.stage_02_data_validation import DataValidationPipeline
from src.pipeline.stage_08_model_evaluation import ModelEvaluationPipeline
from src.pipeline.stage_07_model_trainer import ModelTrainerPipeline
# from src.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from src.pipeline.stage_03_data_loader import DataLoaderPipeline
from src.pipeline.stage_04_data_augmentation import DataAugmentationPipeline
from src.pipeline.stage_05_model_creation import ModelCreationPipeline
from src.pipeline.stage_06_model_callbacks import ModelCallbacksPipeline
from src.pipeline.stage_09_model_prediction import ModelPredictionPipeline
from src.utils.logging_setup import logger
import tensorflow as tf

if __name__ == '__main__':
    try:
        config_manager = ConfigManager()
        # # --- Data Ingestion Stage ---
        # STAGE_NAME = "Data Ingestion Stage"
        # logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # data_ingestion_pipeline = DataIngestionPipeline(config=config_manager)
        # data_ingestion_pipeline.run_pipeline()
        # logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # # --- Data Validation Stage ---
        # STAGE_NAME = "Data Validation Stage"
        # logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # data_validation_pipeline = DataValidationPipeline(config=config_manager)
        # data_validation_pipeline.run_pipeline()
        # logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")


        # --- Data Loader Stage ---
        STAGE_NAME = "Data Loader Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_loader_pipeline = DataLoaderPipeline(config=config_manager)
        datasets, dataset_info = data_loader_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Data Augmentation Stage ---
        STAGE_NAME = "Data Augmentation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_augmentation_pipeline = DataAugmentationPipeline(config=config_manager)
        augmented_datasets, dataset_info = data_augmentation_pipeline.run_pipeline(datasets, dataset_info)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Model Creation Stage ---
        STAGE_NAME = "Model Creation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_creation_pipeline = ModelCreationPipeline(config=config_manager)
        model = model_creation_pipeline.run_pipeline(dataset_info['image_shape'], dataset_info['num_classes'], augmented_datasets['train'])
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    #     # --- Model Callbacks Stage ---
    #     STAGE_NAME = "Model Callbacks Stage"
    #     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #     model_callbacks_pipeline = ModelCallbacksPipeline(config=config_manager)
    #     callbacks = model_callbacks_pipeline.run_pipeline()
    #     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    #     # --- Model Trainer Stage ---
    #     STAGE_NAME = "Model Trainer Stage"
    #     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #     model_trainer_pipeline = ModelTrainerPipeline(config=config_manager)
    #     training_history = model_trainer_pipeline.run_pipeline(augmented_datasets, dataset_info, model, callbacks)
    #     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    #     # --- Model Evaluation Stage ---
    #     STAGE_NAME = "Model Evaluation Stage"
    #     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #     model_evaluation_pipeline = ModelEvaluationPipeline(config=config_manager)
    #     evaluation_results = model_evaluation_pipeline.run_pipeline(model, augmented_datasets['test'], dataset_info['class_names'])
    #     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    # except Exception as e:
    #     logger.error(f"Error occurred during {STAGE_NAME} stage: {e}")
    #     raise e
    
    #     # --- Model Prediction Stage ---
    # try:
    #     STAGE_NAME = "Model Prediction Stage"
    #     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #     model_prediction_pipeline = ModelPredictionPipeline(config=config_manager)
    #     predictions = model_prediction_pipeline.run_pipeline(training_history["model_path"], augmented_datasets['test'], dataset_info['class_names'])
    #     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.error(f"Error occurred during {STAGE_NAME} stage: {e}")
        raise e