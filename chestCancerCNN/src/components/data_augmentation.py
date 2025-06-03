from typing import Dict, List
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from src.entity.config_entity import AugmentationConfig
from src.utils.logging_setup import logger

class DataAugmentation:
    """
    Class for applying data augmentation to image datasets using TensorFlow Keras preprocessing layers.
    It constructs a sequential model of augmentation layers based on the provided configuration
    and applies them, typically to the training dataset.
    """
    def __init__(self, config: AugmentationConfig):
        """
        Initializes the DataAugmentation class with a configuration.

        Args:
            config (AugmentationConfig): Configuration object containing augmentation parameters.
        """
        self.config = config
        

        
    def create_augmentation_layer(self) -> List[tf.keras.Sequential]:
        """
        Creates a sequential model of augmentation layers based on the instance's configuration.
        Each augmentation layer is added only if its corresponding configuration parameter
        is enabled and valid.

        Returns:
            tf.keras.Sequential: A Keras Sequential model containing the configured augmentation layers.
        """
        logger.info("Creating data augmentation layers...")
        augmentation_layers = []

        # Add random flip layers
        if self.config.random_flip:
            if "horizontal" in self.config.random_flip:
                augmentation_layers.append(
                    tf.keras.layers.RandomFlip("horizontal", seed=self.config.seed)
                )
                logger.debug("Added horizontal random flip layer.")
                
            if "vertical" in self.config.random_flip:
                augmentation_layers.append(
                    tf.keras.layers.RandomFlip("vertical", seed=self.config.seed)
                )
                logger.debug("Added vertical random flip layer.")
        
        # Add random rotation
        if self.config.random_rotation > 0:
            augmentation_layers.append(
                tf.keras.layers.RandomRotation(
                    self.config.random_rotation,
                    fill_mode='nearest',
                    interpolation='bilinear',
                    seed=self.config.seed
                )
            )
            logger.debug(f"Added random rotation layer with factor: {self.config.random_rotation}")
        
        # Add random zoom
        if self.config.random_zoom:
            augmentation_layers.append(
                tf.keras.layers.RandomZoom(
                    height_factor=self.config.random_zoom,
                    width_factor=self.config.random_zoom, # Often same as height_factor for square images
                    fill_mode='nearest',
                    interpolation='bilinear',
                    seed=self.config.seed
                )
            )
            logger.debug(f"Added random zoom layer with factor: {self.config.random_zoom}")
        
        # Add random translation
        if self.config.random_translation:
            augmentation_layers.append(
                tf.keras.layers.RandomTranslation(
                    height_factor=self.config.random_translation,
                    width_factor=self.config.random_translation, # Often same as height_factor
                    fill_mode='nearest',
                    interpolation='bilinear',
                    seed=self.config.seed
                )
            )
            logger.debug(f"Added random translation layer with factor: {self.config.random_translation}")
        
        # Add random brightness
        if self.config.random_brightness:
            augmentation_layers.append(
                tf.keras.layers.RandomBrightness(
                    factor=self.config.random_brightness,
                    seed=self.config.seed
                )
            )
            logger.debug(f"Added random brightness layer with factor: {self.config.random_brightness}")
        
        # Add random contrast
        if self.config.random_contrast:
            augmentation_layers.append(
                tf.keras.layers.RandomContrast(
                    factor=self.config.random_contrast,
                    seed=self.config.seed
                )
            )
            logger.debug(f"Added random contrast layer with factor: {self.config.random_contrast}")
        
        logger.info(f"Created augmentation model with {len(augmentation_layers)} layers.")
        if not augmentation_layers:
            logger.warning("No augmentation layers were enabled in the configuration.")
        return  tf.keras.Sequential(augmentation_layers)
            
        
    
    # def apply_augmentation(self, datasets: Dict[str, tf.data.Dataset]) -> Dict[str, tf.data.Dataset]:
    #     """
    #     Applies the configured augmentation layers to the specified datasets.
    #     Augmentation is typically only applied to the training data.

    #     Args:
    #         datasets (Dict[str, tf.data.Dataset]): A dictionary of datasets, where keys
    #                                               are dataset names (e.g., "train", "validation").

    #     Returns:
    #         Dict[str, tf.data.Dataset]: A dictionary of datasets with augmentation applied
    #                                    to the training dataset (if `apply_to_train` is True),
    #                                    otherwise, datasets are passed through unchanged.
    #     """
    #     logger.info("Applying data augmentation to datasets...")
        
    #     augmented_datasets = {}
    #     augmentation_model = self.create_augmentation_layer()
        
    #     # Only apply augmentation if there are layers to apply
    #     if not augmentation_model.layers:
    #         logger.warning("No augmentation layers created. Skipping augmentation application.")
    #         return datasets # Return original datasets if no augmentation layers
        
    #     for name, ds in datasets.items():
    #         # Only apply augmentation to training data if configured
    #         if name == "train" and self.config.apply_to_train:
    #             logger.info(f"Applying augmentation to '{name}' dataset.")
                
    #             # Define a function to apply augmentation to images only (not labels)
    #             def apply_augmentation_to_batch(images, labels):
    #                 # Ensure training=True is passed for random augmentation layers
    #                 augmented_images = augmentation_model(images, training=True)
    #                 return augmented_images, labels
                
    #             # Apply the augmentation function to the dataset
    #             augmented_ds = ds.map(
    #                 apply_augmentation_to_batch,
    #                 num_parallel_calls=tf.data.AUTOTUNE
    #             )
    #             augmented_datasets[name] = augmented_ds
    #         else:
    #             # Pass through other datasets unchanged (e.g., validation, test)
    #             logger.info(f"Skipping augmentation for '{name}' dataset.")
    #             augmented_datasets[name] = ds
        
    #     logger.info("Data augmentation application complete.")
    #     return augmented_datasets
    
    def visualize_augmentations(self, dataset: tf.data.Dataset, augmentation_layers: tf.keras.Sequential):
        """
        Visualizes the effect of augmentations on sample images from a dataset.
        It displays original images alongside several augmented versions and saves the plot.

        Args:
            dataset (tf.data.Dataset): The dataset containing images to augment and visualize.
                                       This should typically be a batch-prefetched dataset.
            augmentation_layers: tf.keras.Sequential: A Keras Sequential model containing the configured augmentation layers.
        """
        logger.info(f"Visualizing augmentations on {self.config.num_examples} examples...")
        


        if not augmentation_layers.layers:
            logger.warning("No augmentation layers to visualize. Skipping visualization.")
            return

        
        
        # Get a batch of images from the dataset
        images_found = False
        for images, labels in dataset.take(1): # Take only one batch
            if tf.size(images) == 0:
                logger.warning("No images found in the dataset for visualization.")
                break # Exit if the batch is empty
            
            images_found = True
            # Select a subset of images to visualize
            selected_images = images[:self.config.num_examples]
            
            # Create a figure for plotting
            fig = plt.figure(figsize=(self.config.num_augmentations * 3 + 3, self.config.num_examples * 3)) # Adjusted figure size for better display
            
            # For each selected image
            for i in range(min(self.config.num_examples, selected_images.shape[0])): # Ensure we don't exceed available images
                # Display the original image
                ax = fig.add_subplot(self.config.num_examples, self.config.num_augmentations + 1, i * (self.config.num_augmentations + 1) + 1)
                ax.imshow(selected_images[i].numpy())
                ax.set_title("Original")
                ax.axis("off")
                
                # Generate and display augmented versions
                for j in range(self.config.num_augmentations):
                    # Apply augmentation to the single image (expand and squeeze dims to maintain batch dimension)
                    augmented_image = augmentation_layers(
                       tf.cast(tf.expand_dims(selected_images[i], 0), tf.float32) , training=True
                    )
                    # augmented_image = augmentation_layers(
                    #     tf.expand_dims(selected_images[i], 0), 
                    #     training=True # Crucial for random layers to be active
                    # )
                    augmented_image = tf.squeeze(augmented_image, 0) # Remove batch dimension
                    
                    # Display the augmented image
                    ax = fig.add_subplot(self.config.num_examples, self.config.num_augmentations + 1, i * (self.config.num_augmentations + 1) + j + 2)
                    ax.imshow(augmented_image.numpy().astype("uint8"))
                    ax.set_title(f"Aug {j+1}")
                    ax.axis("off")
            
            # Save the figure
            plt.tight_layout()
            save_path = os.path.join(self.config.visualization_dir, "augmentation_examples.png")
            plt.savefig(save_path)
            plt.close(fig)
            
            logger.info(f"Augmentation visualization saved to {save_path}")
            break  # Only process one batch for visualization
        
        if not images_found:
            logger.warning("Visualization could not be generated as no images were available from the dataset.")

    def initiate_data_augmentation(self, datasets: Dict[str, tf.data.Dataset]) -> tf.keras.Sequential:
        """
        Orchestrates the entire augmentation process: creating and visualizing augmentation layers.
        """
        logger.info("Starting augmentation process.")
        
        # Create augmentation layers
        augmentation_layers = self.create_augmentation_layer()
        
        self.visualize_augmentations(datasets["train"], augmentation_layers)

        logger.info("augmentation process completed successfully.")
        return augmentation_layers