from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict


@dataclass
class DataIngestionConfig:
    data_dir: str
    extracted_dir: str
    dataset_name: str 

@dataclass
class DataValidationConfig:
    data_dir: str
    extracted_dir: str
    dataset_name: str
    validation_report_file: str
    expected_splits: list
    expected_classes: list
    valid_extensions: list

@dataclass
class DataLoaderConfig:
    data_dir: Path
    visualization_dir: Path
    image_size: Tuple[int, int]
    batch_size: int
    validation_split: float
    seed: int
    use_cache: bool
    shuffle_buffer_size: int
    shuffle: bool
    reshuffle_each_iteration: bool 
    color_mode: str
    crop_to_aspect_ratio: bool
    verbose: int
    num_samples: int
@dataclass
class AugmentationConfig:
    visualization_dir: Path
    seed: int
    random_flip: str
    random_rotation: float
    random_zoom: float
    random_translation: float
    random_brightness: float
    random_contrast: float
    apply_to_train: bool
    num_examples: int
    num_augmentations: int


@dataclass
class CreateModelConfig:
    root_dir: str
    model_dir: str
    model_name: str
    use_augmentation: bool = True # Whether to use augmentation in model creation. If True, the model will be created with augmentation layers. If False, the model will be created without augmentation layers. Default is True.
    optimizer: str = "adam"
    learning_rate: float = 0.001
    loss: str = "sparse_categorical_crossentropy"
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "reduce_on_plateau"
    monitor_metric: str = "val_loss"
    patience: int = 10
    min_lr: float = 1e-6
    use_tensorboard: bool = True
    freeze_base_model: bool = True

@dataclass
class CallbacksConfig:
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "artifacts/logs/tensorboard"
    tensorboard_histogram_freq: int = 1

    use_model_checkpoint: bool = True
    checkpoint_model_filepath: str = "artifacts/models/checkpoints/best_model.keras"
    checkpoint_monitor: str = "val_loss"
    checkpoint_save_best_only: bool = True
    checkpoint_save_weights_only: bool = False
    checkpoint_mode: str = "auto"
    checkpoint_verbose: int = 1

    use_early_stopping: bool = True
    early_stopping_monitor: str = "val_loss"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_mode: str = "auto"
    early_stopping_restore_best_weights: bool = True
    early_stopping_verbose: int = 1

    use_reduce_lr_on_plateau: bool = True
    reduce_lr_monitor: str = "val_loss"
    reduce_lr_factor: float = 0.2
    reduce_lr_patience: int = 5
    reduce_lr_min_lr: float = 1e-6
    reduce_lr_verbose: int = 1

    use_csv_logger: bool = True
    csv_logger_filepath: str = "artifacts/logs/training_log.csv"
    csv_logger_append: bool = False
    




@dataclass
class ModelTrainerConfig:
    root_dir: str
    model_dir: str
    model_name: str
    use_pretrained: bool = True
    freeze_base_model: bool = True
    optimizer: str = "adam"
    learning_rate: float = 0.001
    loss: str = "sparse_categorical_crossentropy"
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "reduce_on_plateau"
    monitor_metric: str = "val_loss"
    patience: int = 10
    min_lr: float = 1e-6
    use_tensorboard: bool = True
    use_auc: bool = True
    use_precision_recall: bool = True
    save_best_model: bool = True
    early_stopping: bool = True
    epochs: int = 50
    batch_size: int = 32
    dropout_rate: float = 0.3
    dense_units: int = 256
    use_batch_norm_head: bool = True
    use_augmentation_layers: bool = True


@dataclass
class EvaluationConfig:
    evaluation_results_filepath: str
    evaluation_plots_dir: str
    evaluation_verbose: int = 1

    generate_classification_report: bool = True
    classification_report_filepath: str = "classification_report.json"

    generate_confusion_matrix: bool = True
    confusion_matrix_filepath: str = "confusion_matrix.png"

    generate_roc_auc_curve: bool = True
    roc_auc_curve_filepath: str = "roc_auc_curve.png"


@dataclass
class PredictionConfig:
    predictions_filepath: str
    predictions_verbose: int = 1
    visualize_predictions: bool = True
    save_predictions: bool = True




