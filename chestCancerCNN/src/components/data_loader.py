import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict
from src.entity.config_entity import DataLoaderConfig
from src.utils.logging_setup import logger


class DataLoader:
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.class_names = None
        
    def load_dataset(self) -> Dict[str, tf.data.Dataset]:
        """
        Load and prepare datasets using image_dataset_from_directory
        
        Returns:
            Dict containing train, validation, and test datasets
        """
        logger.info("Loading datasets from directory structure...")
        datasets = {}
        
        # Check if the data directory exists
        if not self.config.data_dir.exists():
            logger.error(f"Data directory {self.config.data_dir} does not exist")
            raise FileNotFoundError(f"Data directory {self.config.data_dir} does not exist")
        
        # Load training and validation datasets if using validation split
        if self.config.validation_split > 0:
            logger.info(f"Creating training dataset with validation_split={self.config.validation_split}")
            train_ds = tf.keras.utils.image_dataset_from_directory(
                directory=str(self.config.data_dir / "train"),
                validation_split=self.config.validation_split,
                color_mode= self.config.color_mode, 
                subset="training",
                seed=self.config.seed,
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                label_mode='categorical',
                shuffle=self.config.shuffle,
                crop_to_aspect_ratio=self.config.crop_to_aspect_ratio, 
                verbose = self.config.verbose
            )
            
            logger.info("Creating validation dataset from split")
            val_ds = tf.keras.utils.image_dataset_from_directory(
                directory=str(self.config.data_dir / "train"),
                validation_split=self.config.validation_split,
                color_mode= self.config.color_mode,
                subset="validation",
                seed=self.config.seed,
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                label_mode='categorical',
                shuffle=False,
                crop_to_aspect_ratio=self.config.crop_to_aspect_ratio, 
                verbose = self.config.verbose
            )
            
            datasets["train"] = train_ds
            datasets["validation"] = val_ds
            
            # Store class names
            self.class_names = train_ds.class_names
            logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")
            
        else:
            # If not using validation split, load train and validation separately from their respective directories
            logger.info("Loading training dataset from train directory")
            train_ds = tf.keras.utils.image_dataset_from_directory(
                directory=str(self.config.data_dir / "train"),
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                seed=self.config.seed,
                color_mode= self.config.color_mode,
                label_mode='categorical',
                shuffle=self.config.shuffle,
                crop_to_aspect_ratio=self.config.crop_to_aspect_ratio, 
                verbose = self.config.verbose 
            )
            
            # Check if validation directory exists
            val_path = self.config.data_dir / "valid"
            if val_path.exists():
                logger.info("Loading validation dataset from valid directory")
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    directory=str(val_path),
                    image_size=self.config.image_size,
                    color_mode= self.config.color_mode,
                    seed=self.config.seed,
                    batch_size=self.config.batch_size,
                    label_mode='categorical',
                    shuffle=False, 
                    crop_to_aspect_ratio=self.config.crop_to_aspect_ratio, 
                    verbose = self.config.verbose 
                )
                datasets["validation"] = val_ds
            else:
                logger.warning(f"Validation directory '{val_path}' not found. Skipping validation dataset loading.")
            
            datasets["train"] = train_ds
            
            # Store class names
            self.class_names = train_ds.class_names
            logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Check if test directory exists
        test_path = self.config.data_dir / "test"
        if test_path.exists():
            logger.info("Loading test dataset from test directory")
            test_ds = tf.keras.utils.image_dataset_from_directory(
                directory=str(test_path),
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                seed=self.config.seed,
                color_mode= self.config.color_mode,
                label_mode='categorical',
                shuffle=False,
                crop_to_aspect_ratio=self.config.crop_to_aspect_ratio, 
                verbose = self.config.verbose 
            )
            datasets["test"] = test_ds
        else:
            logger.warning(f"Test directory '{test_path}' not found. Skipping test dataset loading.")
            
        return datasets
    
    def configure_for_performance(self, datasets: Dict[str, tf.data.Dataset]) -> Dict[str, tf.data.Dataset]:
        """
        Configure datasets for optimal performance using map, cache, shuffle, and prefetch.
        
        Args:
            datasets: Dictionary of datasets to optimize
            
        Returns:
            Dictionary of optimized datasets
        """
        logger.info("Configuring datasets for performance...")
        optimized_datasets = {}
        
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Configure each dataset for performance
        for name, ds in datasets.items():
           
            # Cache the dataset to avoid re-execution of preprocessing
            # Caching is beneficial for datasets that fit in memory.
            # For very large datasets, consider caching to disk.
            if self.config.use_cache:
                logger.info(f"Caching dataset: {name}")
                ds = ds.cache()
            
            # Shuffle training data (not validation/test)
            if name == "train":
                logger.info(f"Shuffling training dataset with buffer size: {self.config.shuffle_buffer_size}")
                ds = ds.shuffle(buffer_size=self.config.shuffle_buffer_size, 
                                seed=self.config.seed,
                                reshuffle_each_iteration=self.config.reshuffle_each_iteration)
            
            # Use prefetching to overlap data preprocessing and model execution
            logger.info(f"Prefetching dataset: {name}")
            ds = ds.prefetch(buffer_size=AUTOTUNE)
            
            optimized_datasets[name] = ds
        
        logger.info("Datasets configured for performance")
        return optimized_datasets
    
    def get_dataset_info(self, datasets: Dict[str, tf.data.Dataset]) -> Dict[str, Dict]:
        """
        Get information about the datasets (number of batches, samples, etc.).
        
        Args:
            datasets: Dictionary of datasets
            
        Returns:
            Dictionary with dataset information
        """
        info = {}
        
        for name, ds in datasets.items():
            num_batches_tensor = tf.data.experimental.cardinality(ds)
            num_batches = num_batches_tensor.numpy() if num_batches_tensor != tf.data.UNKNOWN_CARDINALITY else -1
            
            approx_samples = num_batches * self.config.batch_size if num_batches != -1 else -1
            
            info[name] = {
                "num_batches": num_batches,
                "batch_size": self.config.batch_size,
                "approx_samples": approx_samples,
                "class_names": self.class_names,
                "num_classes": len(self.class_names) if self.class_names else None,
                "image_size": self.config.image_size
            }
        
        return info
    
    def load_and_prepare_datasets(self) -> Tuple[Dict[str, tf.data.Dataset], Dict[str, Dict]]:
        """
        Load and prepare all datasets in one call.
        
        Returns:
            Tuple containing:
                - Dictionary of optimized datasets
                - Dictionary with dataset information
        """
        logger.info("Starting dataset loading and preparation...")
        
        # Load raw datasets
        datasets = self.load_dataset()
        
        # Configure for performance
        optimized_datasets = self.configure_for_performance(datasets)
        
        # Visualize sample images from the datasets
        self.visualize_dataset_samples(datasets)
        
        # Get dataset info
        dataset_info = self.get_dataset_info(optimized_datasets)
        
        logger.info(f"Dataset preparation complete. Found {len(dataset_info)} splits.")
        for name, info in dataset_info.items():
            logger.info(f"  {name}: ~{info['approx_samples']} samples in {info['num_batches']} batches (Batch Size: {info['batch_size']})")
        
        return optimized_datasets, dataset_info
    
    def visualize_dataset_samples(self, datasets):
        """
        Visualize sample images from the datasets
        
        Args:
            datasets: Dictionary of datasets
        """
        logger.info("Visualizing dataset samples...")
        
        class_names = self.class_names
        
        for name, ds in datasets.items():
            plt.figure(figsize=(10, 10))
            plt.suptitle(f"Samples from {name} dataset")
            
            # Get a batch of data
            for images, labels in ds.take(1):
                # Display up to num_samples images
                for i in range(min(self.config.num_samples, len(images))):
                    plt.subplot(1, self.config.num_samples, i + 1)
                    plt.imshow(images[i].numpy())
                    
                    # Get the class name from one-hot encoded label
                    class_idx = np.argmax(labels[i])
                    class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
                    
                    plt.title(class_name, rotation=90, y=1)
                    plt.axis("off")
            
            # Save the figure
            plt.savefig(f"{self.config.visualization_dir}/{name}_samples.png")
            plt.close()
        
        logger.info("Dataset visualization completed")
