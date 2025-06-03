from pathlib import Path
from flask.cli import F
from src.constants.constants import *
from src.entity.config_entity import AugmentationConfig, CallbacksConfig, DataIngestionConfig, DataLoaderConfig, DataValidationConfig, CreateModelConfig, EvaluationConfig, ModelTrainerConfig, PredictionConfig
from src.utils.helpers import read_yaml_file, create_directory
from src.utils.logging_setup import logger
class ConfigManager:
    '''
    Manages the configuration for the project.
    '''
    def __init__(self, config_path:Path = CONFIG_FILE_PATH, params_path:Path = PARAMS_FILE_PATH):
        '''
        Initializes the ConfigManager with paths to the config and params YAML files.
        '''

        self.config = read_yaml_file(config_path)
        logger.info("Configuration loaded successfully.")
        self.params = read_yaml_file(params_path)
        logger.info("Parameters loaded successfully.")

    def get_data_ingestion_config(self):
        '''
        Returns the data ingestion configuration.
        '''
        config = self.config.dataset
        dirs_to_create = [
            Path(config.data_dir),
            Path(config.extracted_dir)
        ]
        create_directory(dirs_to_create)
        logger.info("Data ingestion directories created successfully.")
        get_data_ingestion_config = DataIngestionConfig(
            data_dir=config.data_dir,
            extracted_dir=config.extracted_dir,
            dataset_name=config.dataset_name
        )
        logger.info("Data ingestion configuration created successfully.")
        return get_data_ingestion_config

    def get_data_validation_config(self):
        '''
        Returns the data validation configuration.
        '''
        config = self.config.validation
        params = self.params.validation
        get_data_validation_config = DataValidationConfig(
            data_dir=config.data_dir,
            extracted_dir=config.extracted_dir,
            dataset_name=config.dataset_name,
            validation_report_file=config.validation_report_file,
            expected_splits=params.expected_splits,
            expected_classes=params.expected_classes,
            valid_extensions=params.valid_extensions
        )
        logger.info("Data validation configuration created successfully.")
        return get_data_validation_config

    def get_data_loader_config(self):
        '''
        Returns the data loader configuration.
        '''
        config = self.config.data_loader
        params = self.params.data_loader
        dirs_to_create = [
            Path(config.data_dir),
            Path(config.visualization_dir)
        ]
        create_directory(dirs_to_create)
        get_data_loader_config = DataLoaderConfig(
            data_dir=Path(config.data_dir),
            visualization_dir=Path(config.visualization_dir),
            image_size=(params.image_size_width, params.image_size_height),
            batch_size=params.batch_size,
            validation_split=params.validation_split,
            seed=params.seed,
            use_cache=params.use_cache,
            shuffle_buffer_size=params.shuffle_buffer_size,
            shuffle=params.shuffle,
            reshuffle_each_iteration=params.reshuffle_each_iteration,
            color_mode=params.color_mode,
            crop_to_aspect_ratio=params.crop_to_aspect_ratio,
            verbose=params.verbose,
            num_samples=params.num_samples
        )
        logger.info("Data loader configuration created successfully.")
        return get_data_loader_config

    def get_augmentation_config(self):
        '''
        Returns the data augmentation configuration.
        '''
        config = self.config.augmentation
        params = self.params.augmentation

        dirs_to_create = [
            Path(config.visualization_dir)
        ]
        create_directory(dirs_to_create)
        get_augmentation_config = AugmentationConfig(
            visualization_dir=Path(config.visualization_dir),
            seed=params.seed,
            random_flip=params.random_flip,
            random_rotation=params.random_rotation,
            random_zoom=params.random_zoom,
            random_translation=params.random_translation,
            random_brightness=params.random_brightness,
            random_contrast=params.random_contrast,
            apply_to_train=params.apply_to_train,
            num_examples=params.num_examples,
            num_augmentations=params.num_augmentations
        )
        logger.info("Data augmentation configuration created successfully.")
        return get_augmentation_config

    def get_create_model_config(self):
        '''
        Returns the create model configuration.
        '''
        config = self.config.create_model
        params = self.params.create_model

        dirs_to_create=[Path(config.model_dir)]
        create_directory(dirs_to_create)
        get_create_model_config = CreateModelConfig(
            root_dir=config.root_dir,
            model_dir=config.model_dir,
            model_name=params.model_name,
            use_augmentation=params.use_augmentation,
            optimizer=params.optimizer,
            learning_rate=params.learning_rate,
            loss=params.loss,
            use_lr_scheduler=params.use_lr_scheduler,
            lr_scheduler_type=params.lr_scheduler_type,
            monitor_metric=params.monitor_metric,
            patience=params.patience,
            min_lr=params.min_lr,
            use_tensorboard=params.use_tensorboard,
            freeze_base_model=params.freeze_base_model,
            fine_tune_num_layers=params.fine_tune_num_layers,
            image_channels=params.image_channels
        )
        logger.info("Create model configuration created successfully.")
        return get_create_model_config

    def get_model_callbacks_config(self):
        '''
        Returns the model callbacks configuration.
        '''
        config = self.config.callbacks
        params = self.params.callbacks

        dirs_to_create = [
            Path(config.tensorboard_log_dir),
            Path(config.checkpoint_model_filepath).parent,
            Path(config.csv_logger_filepath).parent
        ]
        create_directory(dirs_to_create)
        get_model_callbacks_config = CallbacksConfig(
            use_tensorboard=params.use_tensorboard,
            tensorboard_log_dir=config.tensorboard_log_dir,
            tensorboard_histogram_freq=params.tensorboard_histogram_freq,
            use_model_checkpoint=params.use_model_checkpoint,
            checkpoint_model_filepath=config.checkpoint_model_filepath,
            checkpoint_monitor=params.checkpoint_monitor,
            checkpoint_save_best_only=params.checkpoint_save_best_only,
            checkpoint_save_weights_only=params.checkpoint_save_weights_only,
            checkpoint_mode=params.checkpoint_mode,
            checkpoint_verbose=params.checkpoint_verbose,
            use_early_stopping=params.use_early_stopping,
            early_stopping_monitor=params.early_stopping_monitor,
            early_stopping_patience=params.early_stopping_patience,
            early_stopping_min_delta=params.early_stopping_min_delta,
            early_stopping_mode=params.early_stopping_mode,
            early_stopping_restore_best_weights=params.early_stopping_restore_best_weights,
            early_stopping_verbose=params.early_stopping_verbose,
            use_reduce_lr_on_plateau=params.use_reduce_lr_on_plateau,
            reduce_lr_monitor=params.reduce_lr_monitor,
            reduce_lr_factor=params.reduce_lr_factor,
            reduce_lr_patience=params.reduce_lr_patience,
            reduce_lr_min_lr=params.reduce_lr_min_lr,
            reduce_lr_verbose=params.reduce_lr_verbose,
            use_csv_logger=params.use_csv_logger,
            csv_logger_filepath=config.csv_logger_filepath,
            csv_logger_append=params.csv_logger_append
        )
        logger.info("Model callbacks configuration created successfully.")
        return get_model_callbacks_config

    def get_model_trainer_config(self):
        '''
        Returns the model trainer configuration.
        '''
        config = self.config.model_trainer
        params = self.params.model_trainer

        dirs_to_create = [
            Path(config.model_dir)
        ]
        create_directory(dirs_to_create)

        get_model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            model_dir=config.model_dir,
            model_name=config.model_name,
            use_pretrained=params.use_pretrained,
            freeze_base_model=params.freeze_base_model,
            optimizer=params.optimizer,
            learning_rate=params.learning_rate,
            loss=params.loss,
            use_lr_scheduler=params.use_lr_scheduler,
            lr_scheduler_type=params.lr_scheduler_type,
            monitor_metric=params.monitor_metric,
            patience=params.patience,
            min_lr=params.min_lr,
            use_tensorboard=params.use_tensorboard,
            use_auc=params.use_auc,
            use_precision_recall=params.use_precision_recall,
            save_best_model=params.save_best_model,
            early_stopping=params.early_stopping,
            epochs=params.epochs,
            batch_size=params.batch_size,
            dropout_rate=params.dropout_rate,
            dense_units=params.dense_units,
            use_batch_norm_head=params.use_batch_norm_head,
            use_augmentation_layers=params.use_augmentation_layers
        )
        logger.info("Model trainer configuration created successfully.")
        return get_model_trainer_config

    def get_model_evaluation_config(self):
        '''
        Returns the model evaluation configuration.
        '''
        config = self.config.evaluation
        params = self.params.evaluation

        dir_to_create = [
            Path(config.evaluation_results_filepath).parent,
            Path(config.evaluation_plots_dir).parent,
            Path(config.classification_report_filepath).parent,
            Path(config.confusion_matrix_filepath).parent,
            Path(config.roc_auc_curve_filepath).parent,
            Path(config.precision_recall_curve_filepath).parent,
            Path(config.f1_score_curve_filepath).parent,
            Path(config.accuracy_curve_filepath).parent,
            Path(config.loss_curve_filepath).parent,
            Path(config.training_curves_filepath).parent,
            Path(config.validation_curves_filepath).parent,
            Path(config.test_curves_filepath).parent
        ]
        create_directory(dir_to_create)

        get_model_evaluation_config = EvaluationConfig(
            evaluation_results_filepath=config.evaluation_results_filepath,
            evaluation_plots_dir=config.evaluation_plots_dir,
            evaluation_verbose=params.evaluation_verbose,
            generate_classification_report=params.generate_classification_report,
            classification_report_filepath=config.classification_report_filepath,
            generate_confusion_matrix=params.generate_confusion_matrix,
            confusion_matrix_filepath=config.confusion_matrix_filepath,
            generate_roc_auc_curve=params.generate_roc_auc_curve,
            roc_auc_curve_filepath=config.roc_auc_curve_filepath,
            generate_precision_recall_curve=params.generate_precision_recall_curve,
            precision_recall_curve_filepath=config.precision_recall_curve_filepath,
            generate_f1_score_curve=params.generate_f1_score_curve,
            f1_score_curve_filepath=config.f1_score_curve_filepath,
            generate_accuracy_curve=params.generate_accuracy_curve,
            accuracy_curve_filepath=config.accuracy_curve_filepath,
            generate_loss_curve=params.generate_loss_curve,
            loss_curve_filepath=config.loss_curve_filepath,
            generate_training_curves=params.generate_training_curves,
            training_curves_filepath=config.training_curves_filepath,
            generate_validation_curves=params.generate_validation_curves,
            validation_curves_filepath=config.validation_curves_filepath,
            generate_test_curves=params.generate_test_curves,
            test_curves_filepath=config.test_curves_filepath
        )
        logger.info("Model evaluation configuration created successfully.")
        return get_model_evaluation_config

    def get_prediction_config(self):
        '''
        Returns the model prediction configuration.
        '''
        config = self.config.prediction
        params = self.params.prediction

        dirs_to_create = [
            Path(config.predictions_filepath).parent
        ]
        create_directory(dirs_to_create)

        get_prediction_config = PredictionConfig(
            predictions_filepath=config.predictions_filepath,
            predictions_verbose=params.predictions_verbose,
            visualize_predictions=params.visualize_predictions,
            save_predictions=params.save_predictions
        )
        logger.info("Model prediction configuration created successfully.")
        return get_prediction_config