import os
import tensorflow as tf
from tensorflow.keras import layers, models
import importlib # Added for dynamic imports
import time

# Assuming these are correctly defined elsewhere
from src.entity.config_entity import CreateModelConfig
from src.utils.logging_setup import logger
from typing import List, Tuple, Dict, Optional, Union, Callable, Any

class CreateModel:
    """
    Class for creating and compiling deep learning models with dynamic imports and fine-tuning capabilities.
    """
    def __init__(self, config: CreateModelConfig):
        self.config = config
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None

        # Mappings for dynamic import: (module_path, object_name)
        self._MODEL_MODULE_MAP: Dict[str, Tuple[str, str]] = {
            'resnet50': ('tensorflow.keras.applications', 'ResNet50'),
            'resnet101': ('tensorflow.keras.applications', 'ResNet101'),
            'resnet152': ('tensorflow.keras.applications', 'ResNet152'),
            'vgg16': ('tensorflow.keras.applications', 'VGG16'),
            'vgg19': ('tensorflow.keras.applications', 'VGG19'),
            'mobilenetv2': ('tensorflow.keras.applications', 'MobileNetV2'),
            'mobilenetv3large': ('tensorflow.keras.applications', 'MobileNetV3Large'),
            'mobilenetv3small': ('tensorflow.keras.applications', 'MobileNetV3Small'),
            'densenet121': ('tensorflow.keras.applications', 'DenseNet121'),
            'densenet169': ('tensorflow.keras.applications', 'DenseNet169'),
            'densenet201': ('tensorflow.keras.applications', 'DenseNet201'),
            'efficientnetb0': ('tensorflow.keras.applications', 'EfficientNetB0'),
            'efficientnetb1': ('tensorflow.keras.applications', 'EfficientNetB1'),
            'efficientnetb2': ('tensorflow.keras.applications', 'EfficientNetB2'),
            'efficientnetb3': ('tensorflow.keras.applications', 'EfficientNetB3'),
            'inceptionv3': ('tensorflow.keras.applications', 'InceptionV3'),
            'xception': ('tensorflow.keras.applications', 'Xception'),
        }

        self._PREPROCESS_MODULE_MAP: Dict[str, Tuple[str, str]] = {
            'resnet50': ('tensorflow.keras.applications.resnet', 'preprocess_input'),
            'resnet101': ('tensorflow.keras.applications.resnet', 'preprocess_input'),
            'resnet152': ('tensorflow.keras.applications.resnet', 'preprocess_input'),
            'vgg16': ('tensorflow.keras.applications.vgg16', 'preprocess_input'),
            'vgg19': ('tensorflow.keras.applications.vgg19', 'preprocess_input'),
            'mobilenetv2': ('tensorflow.keras.applications.mobilenet_v2', 'preprocess_input'),
            'mobilenetv3large': ('tensorflow.keras.applications.mobilenet_v3', 'preprocess_input'),
            'mobilenetv3small': ('tensorflow.keras.applications.mobilenet_v3', 'preprocess_input'),
            'densenet121': ('tensorflow.keras.applications.densenet', 'preprocess_input'),
            'densenet169': ('tensorflow.keras.applications.densenet', 'preprocess_input'),
            'densenet201': ('tensorflow.keras.applications.densenet', 'preprocess_input'),
            'efficientnetb0': ('tensorflow.keras.applications.efficientnet', 'preprocess_input'),
            'efficientnetb1': ('tensorflow.keras.applications.efficientnet', 'preprocess_input'),
            'efficientnetb2': ('tensorflow.keras.applications.efficientnet', 'preprocess_input'),
            'efficientnetb3': ('tensorflow.keras.applications.efficientnet', 'preprocess_input'),
            'inceptionv3': ('tensorflow.keras.applications.inception_v3', 'preprocess_input'),
            'xception': ('tensorflow.keras.applications.xception', 'preprocess_input'),
        }
        
        self._loaded_model_constructors: Dict[str, Callable[..., tf.keras.Model]] = {}
        self._loaded_preprocess_fns: Dict[str, Callable] = {}

        logger.info(f"CreateModel initialized with config: {self.config}")

    def _import_object(self, model_name_lower: str, import_map: Dict[str, Tuple[str, str]], cache: Dict[str, Any], object_type: str) -> Optional[Any]:
        """Helper to dynamically import model constructors or preprocessing functions."""
        if model_name_lower in cache:
            return cache[model_name_lower]
        
        if model_name_lower in import_map:
            module_path, object_name = import_map[model_name_lower]
            try:
                module = importlib.import_module(module_path)
                obj = getattr(module, object_name)
                cache[model_name_lower] = obj
                logger.info(f"Successfully imported {object_type} '{object_name}' from '{module_path}'.")
                return obj
            except ImportError:
                logger.error(f"Could not import module '{module_path}' for {object_type} '{model_name_lower}'.")
            except AttributeError:
                logger.error(f"Could not find {object_type} '{object_name}' in module '{module_path}'.")
        return None

    def _get_model_constructor(self, model_name_lower: str) -> Optional[Callable[..., tf.keras.Model]]:
        return self._import_object(model_name_lower, self._MODEL_MODULE_MAP, self._loaded_model_constructors, "model constructor")

    def _get_preprocess_input_fn(self, model_name_lower: str) -> Optional[Callable]:
        return self._import_object(model_name_lower, self._PREPROCESS_MODULE_MAP, self._loaded_preprocess_fns, "preprocessing function")

    def create_model(self, input_shape: Tuple[int, int, int], num_classes: int, data_augmentation_model: Optional[tf.keras.Model]) -> tf.keras.Model:
        """
        Create and compile a model based on the configuration.

        Args:
            input_shape: Shape of input images (height, width, channels).
            num_classes: Number of output classes.
            data_augmentation_model: A Keras Model for data augmentation (optional).

        Returns:
            A compiled Keras model.
        
        Raises:
            ValueError: If an unknown model architecture is specified or cannot be loaded.
        """
        model_name_lower = self.config.model_name.lower()
        logger.info(f"Attempting to create model: {self.config.model_name}")
        input_shape = input_shape + (self.config.image_channels,)

        if model_name_lower == 'custom':
            self.model = self._create_custom_model(input_shape, num_classes, data_augmentation_model)
        elif model_name_lower in self._MODEL_MODULE_MAP:
            model_constructor = self._get_model_constructor(model_name_lower)
            preprocess_input_fn = self._get_preprocess_input_fn(model_name_lower)
            if not model_constructor:
                error_msg = f"Failed to load model constructor for '{self.config.model_name}'."
                logger.error(error_msg)
                raise ValueError(error_msg)
            self.model = self._create_transfer_learning_model(
                input_shape, num_classes, data_augmentation_model,
                model_constructor, preprocess_input_fn
            )
        else:
            supported_models = list(self._MODEL_MODULE_MAP.keys()) + ['custom']
            error_msg = (f"Unknown model architecture specified: '{self.config.model_name}'. "
                         f"Supported models are: {', '.join(supported_models)}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        optimizer = self._get_optimizer()
        loss_function = self.config.loss
        metrics_list = self._get_metrics()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics_list
        )

        logger.info(f"Model '{self.config.model_name}' compiled successfully.")
        logger.info(f"Optimizer: {optimizer.__class__.__name__}, Learning Rate: {getattr(self.config, 'learning_rate', 'N/A')}, "
                    f"Loss: {loss_function}, Metrics: {[m.name if hasattr(m, 'name') else str(m) for m in metrics_list]}")

        self._save_model_summary(self.model)
        self._plot_model_architecture(self.model)

        return self.model

    def _create_custom_model(self, input_shape: Tuple[int, int, int], num_classes: int, data_augmentation_model: Optional[tf.keras.Model]) -> tf.keras.Model:
        """
        Create a custom CNN model architecture using the Functional API.
        """
        logger.info("Creating custom CNN model architecture using Functional API.")
        
        inputs = tf.keras.Input(shape=input_shape, name="input_image")
        x = inputs

        use_augmentation = getattr(self.config, 'use_augmentation', False)
        if use_augmentation and data_augmentation_model:
            if isinstance(data_augmentation_model, tf.keras.Model):
                logger.info("Adding data augmentation pipeline to custom model.")
                x = data_augmentation_model(x, training=True) # Pass training flag if aug model supports it
            else:
                logger.warning("`data_augmentation_model` provided but not a valid Keras Model. Skipping augmentation.")
        
        x = layers.Rescaling(1./255, name="rescaling")(x)

        # Convolutional Base
        dropout_rate = getattr(self.config, 'dropout_rate', 0.2)
        dense_units = getattr(self.config, 'dense_units', 128)

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1a")(x)
        x = layers.BatchNormalization(name="bn1a")(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1b")(x)
        x = layers.BatchNormalization(name="bn1b")(x)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)
        x = layers.Dropout(dropout_rate, name="drop1")(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2a")(x)
        x = layers.BatchNormalization(name="bn2a")(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2b")(x)
        x = layers.BatchNormalization(name="bn2b")(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)
        x = layers.Dropout(dropout_rate, name="drop2")(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv3a")(x)
        x = layers.BatchNormalization(name="bn3a")(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv3b")(x)
        x = layers.BatchNormalization(name="bn3b")(x)
        x = layers.MaxPooling2D((2, 2), name="pool3")(x)
        x = layers.Dropout(dropout_rate, name="drop3")(x)

        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(dense_units, activation='relu', name="dense1")(x)
        x = layers.BatchNormalization(name="bn_dense")(x)
        x = layers.Dropout(dropout_rate, name="drop_dense")(x)
        outputs = layers.Dense(num_classes, activation='softmax', name="output_probabilities")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name="custom_cnn_model")
        logger.info(f"Created custom model using Functional API. Input: {inputs.shape}, Output: {outputs.shape}")
        return model
    
    def _create_transfer_learning_model(self,
                                        input_shape: Tuple[int, int, int],
                                        num_classes: int,
                                        data_augmentation_model: Optional[tf.keras.Model],
                                        model_constructor: Callable[..., tf.keras.Model],
                                        preprocess_input_fn: Optional[Callable]) -> tf.keras.Model:
        """
        Creates a model using a pre-trained backbone with a custom classification head.
        """
        model_name_lower = self.config.model_name.lower()
        logger.info(f"Creating transfer learning model with {self.config.model_name} backbone.")

        inputs = tf.keras.Input(shape=input_shape, name="input_image")
        x = inputs

        use_augmentation = getattr(self.config, 'use_augmentation', False)
        if use_augmentation and data_augmentation_model:
            if isinstance(data_augmentation_model, tf.keras.Model):
                logger.info(f"Adding data augmentation pipeline to {self.config.model_name} model.")
                x = data_augmentation_model(x, training=True) # Pass training flag
            else:
                logger.warning(f"`data_augmentation_model` not a valid Keras Model. Skipping augmentation.")
        
        if preprocess_input_fn:
            logger.info(f"Applying {model_name_lower}-specific preprocessing.")
            x = layers.Lambda(lambda tensor: preprocess_input_fn(tensor), name='preprocessing')(x)
        else:
            logger.warning(
                f"No specific preprocess_input function found for {model_name_lower}. "
                f"Ensure data is correctly preprocessed if needed (e.g., scaling to [0,1] or [-1,1]). "
                f"Some models include rescaling; others require specific handling."
            )
        
        try:
            base_model = model_constructor(
                weights='imagenet',
                include_top=False,
                input_tensor=x # Pass the processed tensor as input to the base model
            )
        except Exception as e:
            logger.error(f"Error creating base model {self.config.model_name} with input_tensor: {e}")
            logger.error(
                f"Ensure input_shape {input_shape} and augmentation/preprocessing (if any) "
                f"produce a tensor compatible with {self.config.model_name}."
            )
            raise

        # Fine-tuning and freezing logic
        freeze_base = getattr(self.config, 'freeze_base_model', True)
        if freeze_base:
            base_model.trainable = False
            logger.info(f"Base model '{self.config.model_name}' layers are FROZEN ({len(base_model.layers)} layers).")
        else:
            base_model.trainable = True # Make base model trainable to allow layer-specific control
            fine_tune_num = getattr(self.config, 'fine_tune_num_layers', 0)
            if fine_tune_num > 0 and fine_tune_num <= len(base_model.layers):
                logger.info(f"Fine-tuning the top {fine_tune_num} layers of {self.config.model_name}.")
                for layer in base_model.layers[:-fine_tune_num]:
                    layer.trainable = False
                for layer in base_model.layers[-fine_tune_num:]: # Ensure these are indeed trainable
                    layer.trainable = True # Explicitly set, though parent `base_model.trainable=True` should make them
                logger.info(f"Froze {len(base_model.layers) - fine_tune_num} bottom layers, "
                            f"top {fine_tune_num} layers are trainable.")
            elif fine_tune_num > len(base_model.layers):
                logger.warning(f"fine_tune_num_layers ({fine_tune_num}) > actual base model layers ({len(base_model.layers)}). "
                               f"All base model layers will be trainable.")
                # All layers remain trainable as per base_model.trainable = True
                logger.info(f"All layers of base model '{self.config.model_name}' are TRAINABLE.")
            else: # fine_tune_num is 0 or not specified
                logger.info(f"All layers of base model '{self.config.model_name}' are TRAINABLE.")
        
        processed_features = base_model.output

        dropout_rate = getattr(self.config, 'dropout_rate', 0.5) # Common default for head
        dense_units = getattr(self.config, 'dense_units', 512)  # Common default for head

        head = layers.GlobalAveragePooling2D(name='global_avg_pool')(processed_features)
        head = layers.Dropout(dropout_rate, name='top_dropout_1')(head) # Increased dropout for head
        head = layers.Dense(dense_units, activation='relu', name='dense_units')(head)
        head = layers.BatchNormalization(name='top_batchnorm')(head)
        head = layers.Dropout(dropout_rate, name='top_dropout_2')(head)
        outputs = layers.Dense(num_classes, activation='softmax', name='output_probabilities')(head)

        model = models.Model(inputs=inputs, outputs=outputs, name=f"{model_name_lower}_transfer_model")
        logger.info(f"Created transfer learning model '{self.config.model_name}'. Input: {inputs.shape}, Output: {outputs.shape}")
        return model

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        optimizer_name = getattr(self.config, 'optimizer', 'adam').lower()
        lr = getattr(self.config, 'learning_rate', 0.001)
        sgd_momentum = getattr(self.config, 'sgd_momentum', 0.9) 

        optimizer_map: Dict[str, Callable[..., tf.keras.optimizers.Optimizer]] = {
            'adam': lambda learning_rate_param: tf.keras.optimizers.Adam(learning_rate=learning_rate_param),
            'sgd': lambda learning_rate_param: tf.keras.optimizers.SGD(learning_rate=learning_rate_param, momentum=sgd_momentum),
            'rmsprop': lambda learning_rate_param: tf.keras.optimizers.RMSprop(learning_rate=learning_rate_param),
            'adagrad': lambda learning_rate_param: tf.keras.optimizers.Adagrad(learning_rate=learning_rate_param),
            'adadelta': lambda learning_rate_param: tf.keras.optimizers.Adadelta(learning_rate=learning_rate_param),
        }

        optimizer_constructor = optimizer_map.get(optimizer_name)
        if optimizer_constructor:
            return optimizer_constructor(learning_rate_param=lr)
        else:
            logger.warning(f"Unknown optimizer: '{optimizer_name}'. Using Adam with LR={lr} as default.")
            return tf.keras.optimizers.Adam(learning_rate=lr)
        
    def _get_metrics(self) -> List[Union[str, tf.keras.metrics.Metric]]:
        metrics_list: List[Union[str, tf.keras.metrics.Metric]] = ['accuracy']

        if getattr(self.config, 'use_auc', False):
            auc_curve = getattr(self.config, 'auc_curve', 'roc').lower()
            auc_summation_method = getattr(self.config, 'auc_summation_method', None)
            
            auc_kwargs = {'name': f'auc_{auc_curve}', 'curve': auc_curve}
            if auc_curve == 'pr' and auc_summation_method:
                auc_kwargs['summation_method'] = auc_summation_method
            
            logger.info(f"Adding AUC metric (curve: {auc_kwargs['curve']}"
                        f"{', summation: ' + auc_kwargs['summation_method'] if 'summation_method' in auc_kwargs else ''}).")
            metrics_list.append(tf.keras.metrics.AUC(**auc_kwargs)) # type: ignore

        if getattr(self.config, 'use_precision_recall', False):
            logger.info("Adding Precision and Recall metrics.")
            metrics_list.append(tf.keras.metrics.Precision(name='precision'))
            metrics_list.append(tf.keras.metrics.Recall(name='recall'))

        if getattr(self.config, 'use_f1_score', False): # Requires TF Addons or custom implementation for older TF
            try:
                # Attempt to use F1Score if available (e.g. TF 2.11+)
                metrics_list.append(tf.keras.metrics.F1Score(name='f1_score', average='macro')) # Use 'macro' for multi-class
                logger.info("Adding F1 Score metric (macro-averaged).")
            except AttributeError:
                logger.warning("tf.keras.metrics.F1Score not available in this TensorFlow version. Skipping F1 Score.")
                # You might need to install tensorflow-addons for F1Score in older TF versions
                # from tfa.metrics import F1Score
                # metrics_list.append(F1Score(num_classes=num_classes, average="micro/macro/weighted", name='f1_score'))
        return metrics_list

    def _ensure_model_dir_exists(self) -> Optional[str]:
        """Ensures the model directory from config exists and returns it."""
        model_dir = getattr(self.config, 'model_dir', None)
        if not model_dir:
            logger.warning("`model_dir` not specified in config. Cannot save model artifacts.")
            return None
        try:
            os.makedirs(model_dir, exist_ok=True)
            return model_dir
        except OSError as e:
            logger.error(f"Error creating model directory {model_dir}: {e}")
            return None

    def _plot_model_architecture(self, model: tf.keras.Model) -> None:
        model_dir = self._ensure_model_dir_exists()
        if not model_dir:
            return

        plot_filename = f"{self.config.model_name.lower()}_architecture.png"
        plot_path = os.path.join(model_dir, plot_filename)

        try:
            tf.keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, show_layer_activations=True, show_dtype=True)
            logger.info(f"Model architecture plot saved to {plot_path}")
        except Exception as e: # Catch broader errors, e.g., graphviz not installed
            logger.error(f"Failed to plot model architecture to {plot_path}: {e}. Ensure graphviz is installed.")


    def _save_model_summary(self, model: tf.keras.Model) -> None:
        model_dir = self._ensure_model_dir_exists()
        if not model_dir:
            return

        summary_filename = f"{self.config.model_name.lower()}_summary.txt"
        summary_path = os.path.join(model_dir, summary_filename)
        
        try:
            with open(summary_path, 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            logger.info(f"Model summary saved to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save model summary to {summary_path}: {e}")