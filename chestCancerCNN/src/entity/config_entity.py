from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

from sympy import use


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
    use_augmentation: bool
    optimizer: str
    learning_rate: float
    loss: str
    use_lr_scheduler: bool
    lr_scheduler_type: str
    monitor_metric: str
    patience: int
    min_lr: float
    use_tensorboard: bool
    freeze_base_model: bool
    fine_tune_num_layers: int
    image_channels: int

@dataclass
class CallbacksConfig:
    use_tensorboard: bool
    tensorboard_log_dir: str
    tensorboard_histogram_freq: int

    use_model_checkpoint: bool
    checkpoint_model_filepath: str
    checkpoint_monitor: str
    checkpoint_save_best_only: bool
    checkpoint_save_weights_only: bool
    checkpoint_mode: str
    checkpoint_verbose: int

    use_early_stopping: bool
    early_stopping_monitor: str
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_mode: str
    early_stopping_restore_best_weights: bool
    early_stopping_verbose: int

    use_reduce_lr_on_plateau: bool
    reduce_lr_monitor: str
    reduce_lr_factor: float
    reduce_lr_patience: int
    reduce_lr_min_lr: float
    reduce_lr_verbose: int

    use_csv_logger: bool
    csv_logger_filepath: str
    csv_logger_append: bool




@dataclass
class ModelTrainerConfig:
    root_dir: str
    model_dir: str
    model_name: str
    use_pretrained: bool
    freeze_base_model: bool
    optimizer: str
    learning_rate: float
    loss: str
    use_lr_scheduler: bool
    lr_scheduler_type: str
    monitor_metric: str
    patience: int
    min_lr: float
    use_tensorboard: bool
    use_auc: bool
    use_precision_recall: bool
    save_best_model: bool
    early_stopping: bool
    epochs: int
    batch_size: int
    dropout_rate: float
    dense_units: int
    use_batch_norm_head: bool
    use_augmentation_layers: bool


@dataclass
class EvaluationConfig:
    evaluation_results_filepath: str
    evaluation_plots_dir: str
    evaluation_verbose: int

    generate_classification_report: bool
    classification_report_filepath: str

    generate_confusion_matrix: bool
    confusion_matrix_filepath: str

    generate_roc_auc_curve: bool
    roc_auc_curve_filepath: str


@dataclass
class PredictionConfig:
    predictions_filepath: str
    predictions_verbose: int
    visualize_predictions: bool
    save_predictions: bool




