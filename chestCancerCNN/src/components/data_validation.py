from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageFile 
import json
from src.entity.config_entity import DataValidationConfig
from tqdm import tqdm
from src.utils.logging_setup import logger

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.validation_report = {}
        self.all_image_paths = [] 

    def _check_path_exists_and_log(self, path: Path, description: str, is_critical: bool = True) -> bool:
        """Helper to check path existence and log messages."""
        if not path.exists():
            if is_critical:
                logger.error(f"{description} not found: {path}")
            else:
                logger.warning(f"{description} not found: {path}")
            return False
        logger.info(f"Found {description}: {path}")
        return True

    def validate_initial_directory_structure(self) -> bool:
        """
        Validate the existence of the base extracted directory and expected top-level splits.
        """
        logger.info("Validating initial directory structure (extracted_dir and top-level splits)...")
        overall_status = True

        if not self._check_path_exists_and_log(self.config.extracted_dir, "Extracted dataset root directory"):
            overall_status = False
            self.validation_report["initial_structure_validation"] = {"status": overall_status, "message": "Extracted dataset root directory not found."}
            return overall_status

        found_splits_status = {split: False for split in self.config.expected_splits}
        missing_splits = []
        for split in self.config.expected_splits:
            split_path = self.config.extracted_dir / split
            if split_path.exists():
                logger.info(f"Expected split directory found: {split_path}")
                found_splits_status[split] = True
            else:
                logger.warning(f"Expected split directory NOT found: {split_path}.")
                missing_splits.append(split)

        if missing_splits:
            overall_status = False # Set to false if any expected split is missing
            self.validation_report["initial_structure_validation"] = {
                "status": overall_status,
                "message": f"Some expected split directories are missing: {', '.join(missing_splits)}",
                "found_expected_splits": found_splits_status
            }
            logger.error(f"Initial directory structure validation failed due to missing splits.")
        else:
            self.validation_report["initial_structure_validation"] = {
                "status": overall_status,
                "message": "All expected split directories found.",
                "found_expected_splits": found_splits_status
            }
            logger.info("Initial directory structure validation completed successfully.")

        return overall_status

    def validate_image_files_and_discover_classes(self) -> bool:
        """
        Recursively finds all image files from extracted_dir, infers their class and split,
        and validates image integrity and dimensions.
        """
        logger.info("Recursively scanning for images and discovering classes from root...")
        validation_status = True
        image_stats = {
            "total_images_scanned": 0,
            "valid_images_processed": 0,
            "corrupted_or_invalid_images": 0,
            "zero_byte_files": 0,
            "image_errors_by_type": {},
            "dimensions": {"widths": [], "heights": []},
            "discovered_classes_by_split": {split: {} for split in self.config.expected_splits},
            "overall_unique_classes_discovered": [],
            "dimension_consistency": True,
            "actual_split_distribution": {split: 0 for split in self.config.expected_splits},
        }
        image_stats["discovered_classes_by_split"]["_other_"] = {}
        image_stats["actual_split_distribution"]["_other_"] = 0


        self.all_image_paths = []
        logger.debug(f"Starting recursive glob search from: {self.config.extracted_dir}")
        # Use rglob for more efficient recursive search
        for ext in self.config.valid_extensions:
            self.all_image_paths.extend(list(self.config.extracted_dir.rglob(f'*{ext}')))
            self.all_image_paths.extend(list(self.config.extracted_dir.rglob(f'*{ext.upper()}'))) # Also search for uppercase extensions

        image_stats["total_images_scanned"] = len(self.all_image_paths)

        if not self.all_image_paths:
            logger.error(f"No image files found recursively within '{self.config.extracted_dir}' "
                              f"with extensions: {', '.join(self.config.valid_extensions)}")
            self.validation_report["image_and_class_validation"] = {
                "status": False,
                "message": "No image files found.",
                "image_stats": image_stats
            }
            return False

        logger.info(f"Found {len(self.all_image_paths)} potential image files. Starting detailed validation...")

        first_image_dims = None
        unique_class_names_set = set()

        for img_path in tqdm(self.all_image_paths, desc="Validating images"):
            # Check if file is zero bytes
            if img_path.stat().st_size == 0:
                logger.error(f"Zero-byte file found: {img_path}. Skipping validation.")
                image_stats["zero_byte_files"] += 1
                image_stats["corrupted_or_invalid_images"] += 1
                image_stats["image_errors_by_type"]["ZeroByteFile"] = image_stats["image_errors_by_type"].get("ZeroByteFile", 0) + 1
                validation_status = False
                continue

            try:
                # Robust image opening and verification
                with Image.open(img_path) as img:
                    img.verify() # Verify file integrity (closes file)
                    img = Image.open(img_path) # Reopen for loading after verify
                    img.load()   # Load image data to catch issues

                    width, height = img.size
                    image_stats["dimensions"]["widths"].append(width)
                    image_stats["dimensions"]["heights"].append(height)

                    if first_image_dims is None:
                        first_image_dims = (width, height)
                    elif (width, height) != first_image_dims:
                        image_stats["dimension_consistency"] = False

                image_stats["valid_images_processed"] += 1

                # Infer class and split based on directory structure relative to extracted_dir
                relative_path = img_path.relative_to(self.config.extracted_dir)
                path_parts = relative_path.parts

                current_split = "_other_"
                class_name = "_unknown_class_" # Default unknown class

                # Find the split (e.g., 'train', 'test', 'valid')
                for s in self.config.expected_splits:
                    if s in path_parts:
                        split_index = path_parts.index(s)
                        current_split = s
                        # The class folder is usually directly after the split folder
                        if len(path_parts) > split_index + 1:
                            class_name = path_parts[split_index + 1]
                            # Basic check to ensure it's not the image file itself
                            if class_name.lower().endswith(tuple(self.config.valid_extensions)):
                                class_name = "_unknown_class_ (filename as class)"
                        break # Found the split, no need to check other splits

                # If no expected split was found, try to infer class from parent if it's directly under extracted_dir
                if current_split == "_other_" and len(path_parts) > 1:
                    # If image is directly under a folder in extracted_dir, that folder could be the class
                    # E.g., extracted_dir/class_name/image.jpg
                    if not any(part in self.config.expected_splits for part in path_parts[:-1]): # If no split was found
                        class_name = path_parts[0] # Assume the first folder is the class

                # Update class distribution and unique classes
                image_stats["discovered_classes_by_split"][current_split][class_name] = \
                    image_stats["discovered_classes_by_split"][current_split].get(class_name, 0) + 1
                image_stats["actual_split_distribution"][current_split] += 1
                unique_class_names_set.add(class_name)

            except UnidentifiedImageError as e:
                logger.error(f"Cannot identify image format or corrupted: {img_path}, Error: {e}")
                image_stats["corrupted_or_invalid_images"] += 1
                image_stats["image_errors_by_type"]["UnidentifiedImageError"] = image_stats["image_errors_by_type"].get("UnidentifiedImageError", 0) + 1
                validation_status = False
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
                image_stats["corrupted_or_invalid_images"] += 1
                error_type = type(e).__name__
                image_stats["image_errors_by_type"][error_type] = image_stats["image_errors_by_type"].get(error_type, 0) + 1
                validation_status = False

        image_stats["overall_unique_classes_discovered"] = sorted(list(unique_class_names_set))

        # Calculate dimension statistics if any valid images were processed
        if image_stats["valid_images_processed"] > 0:
            widths = np.array(image_stats["dimensions"]["widths"])
            heights = np.array(image_stats["dimensions"]["heights"])
            image_stats["avg_width"] = np.mean(widths).item() # .item() converts numpy scalar to Python scalar
            image_stats["avg_height"] = np.mean(heights).item()
            image_stats["min_width"] = np.min(widths).item()
            image_stats["min_height"] = np.min(heights).item()
            image_stats["max_width"] = np.max(widths).item()
            image_stats["max_height"] = np.max(heights).item()
        else:
            logger.warning("No valid image dimensions collected as no images were successfully processed.")
            image_stats["dimension_consistency"] = False # No consistency if no images

        # Clean up raw dimension lists to save report size
        del image_stats["dimensions"]["widths"]
        del image_stats["dimensions"]["heights"]

        # Final class validation against expected_classes from config
        all_discovered_classes = set(image_stats["overall_unique_classes_discovered"])
        all_expected_classes = set(self.config.expected_classes)

        missing_expected_classes = list(all_expected_classes - all_discovered_classes)
        unrecognized_discovered_classes = list(all_discovered_classes - all_expected_classes)

        class_validation_ok = (not missing_expected_classes) and (not unrecognized_discovered_classes)
        if not class_validation_ok:
            validation_status = False # If class mismatch, overall validation fails

        self.validation_report["image_and_class_validation"] = {
            "status": validation_status,
            "message": "Image and class validation completed. See image_stats for details.",
            "image_stats": image_stats,
            "class_validation_summary": {
                "status": class_validation_ok,
                "missing_expected_classes": missing_expected_classes,
                "unrecognized_discovered_classes": unrecognized_discovered_classes
            }
        }

        if validation_status and image_stats["corrupted_or_invalid_images"] == 0:
            logger.info("Image files and class discovery validated successfully with no issues.")
            if not image_stats["dimension_consistency"]:
                logger.warning("Images have inconsistent dimensions. This might require resizing during transformation.")
        else:
            logger.warning(f"Image file and class discovery validation completed with issues. "
                                f"Found {image_stats['corrupted_or_invalid_images']} corrupted/invalid images "
                                f"and {image_stats['zero_byte_files']} zero-byte files.")
            if missing_expected_classes:
                logger.error(f"Missing expected classes: {missing_expected_classes}")
            if unrecognized_discovered_classes:
                logger.error(f"Discovered unexpected classes: {unrecognized_discovered_classes}")

        return validation_status

    def attempt_repair_corrupted_images(self) -> dict:
        """
        Attempts to repair corrupted images by using alternative libraries or methods.
        Returns a dictionary with repair statistics.
        """
        if not hasattr(self, 'all_image_paths') or not self.all_image_paths:
            logger.warning("No image paths available for repair attempt.")
            return {"attempted": 0, "successful": 0, "failed": 0, "repaired_paths": []}
        
        repair_stats = {
            "attempted": 0,
            "successful": 0,
            "failed": 0,
            "repaired_paths": []
        }
        
        # Identify corrupted images by re-checking their integrity
        logger.info("Identifying corrupted images for repair attempt...")
        corrupted_images_for_repair = []
        for img_path in tqdm(self.all_image_paths, desc="Checking images for repair"):
            if img_path.stat().st_size == 0:
                # Already handled zero-byte files, not typically 'repairable' in this context
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
                    # If img.verify() passes, it's not corrupted for PIL.
                    # We only want to attempt repair on files that previously failed.
            except Exception:
                corrupted_images_for_repair.append(img_path)
        
        if not corrupted_images_for_repair:
            logger.info("No corrupted images identified for repair.")
            return repair_stats
        
        logger.info(f"Attempting to repair {len(corrupted_images_for_repair)} corrupted images...")
        
        # Check if OpenCV is available
        cv2_available = False
        try:
            import cv2
            cv2_available = True
        except ImportError:
            logger.warning("opencv-python not found. Skipping repair attempts using OpenCV.")
        except Exception as e:
            logger.warning(f"Error importing opencv-python: {e}. Skipping repair attempts using OpenCV.")

        # Store original PIL.ImageFile.LOAD_TRUNCATED_IMAGES state
        original_load_truncated_images = ImageFile.LOAD_TRUNCATED_IMAGES
        
        for img_path in tqdm(corrupted_images_for_repair, desc="Repairing images"):
            repair_stats["attempted"] += 1
            
            repaired_successfully = False
            
            # Method 1: Try with OpenCV (if available)
            if cv2_available:
                try:
                    img_np = cv2.imread(str(img_path))
                    if img_np is not None and img_np.size > 0: # Check if image was loaded and is not empty
                        # Save with a temporary name, or overwrite if desired (be cautious)
                        repair_path = img_path.with_name(f"{img_path.stem}_repaired{img_path.suffix}")
                        cv2.imwrite(str(repair_path), img_np)
                        repair_stats["successful"] += 1
                        repair_stats["repaired_paths"].append(str(repair_path))
                        repaired_successfully = True
                except Exception as e:
                    logger.debug(f"OpenCV repair failed for {img_path}: {e}")
            
            if repaired_successfully:
                continue # Move to the next image if repaired by OpenCV

            # Method 2: Try with PIL and LOAD_TRUNCATED_IMAGES
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True # Temporarily allow truncated images
                with Image.open(img_path) as img:
                    img.load() # This might succeed for truncated images now
                    # Save as a new file (or overwrite if that's the desired behavior)
                    repair_path = img_path.with_name(f"{img_path.stem}_repaired{img_path.suffix}")
                    img.save(repair_path)
                    repair_stats["successful"] += 1
                    repair_stats["repaired_paths"].append(str(repair_path))
                    repaired_successfully = True
            except Exception as e:
                logger.debug(f"PIL (truncated) repair failed for {img_path}: {e}")
            finally:
                # Reset PIL.ImageFile.LOAD_TRUNCATED_IMAGES
                ImageFile.LOAD_TRUNCATED_IMAGES = original_load_truncated_images
            
            if not repaired_successfully:
                repair_stats["failed"] += 1
                
        logger.info(f"Image repair attempt completed. "
                         f"Attempted: {repair_stats['attempted']}, "
                         f"Successful: {repair_stats['successful']}, "
                         f"Failed: {repair_stats['failed']}")
        
        # Add repair stats to validation report
        self.validation_report["repair_attempt_summary"] = repair_stats
        
        return repair_stats

    def save_validation_report(self) -> bool:
        """Save validation report to a JSON file."""
        try:
            report_file_path = Path(self.config.validation_report_file)
            report_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_file_path, 'w') as f:
                json.dump(self.validation_report, f, indent=4)

            logger.info(f"Validation report saved to {report_file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving validation report: {e}")
            return False

    def initiate_data_validation(self) -> bool:
        """Run all validation checks and return overall status."""
        logger.info("Starting data validation process...")

        overall_validation_status = True

        # Step 1: Validate initial directory structure
        initial_structure_ok = self.validate_initial_directory_structure()
        overall_validation_status = overall_validation_status and initial_structure_ok

        # Step 2: Validate image files and discover classes
        images_and_classes_ok = self.validate_image_files_and_discover_classes()
        overall_validation_status = overall_validation_status and images_and_classes_ok

        # Step 3: If validation failed due to corrupted images, attempt repair
        # Check if any images were corrupted and if there are actual images to repair
        if not images_and_classes_ok and \
           self.validation_report.get("image_and_class_validation", {}).get("image_stats", {}).get("corrupted_or_invalid_images", 0) > 0 and \
           self.all_image_paths: # Ensure all_image_paths is populated before attempting repair
            logger.info("Validation identified corrupted images. Attempting repair...")
            repair_stats = self.attempt_repair_corrupted_images()
            
            # If we successfully repaired some images, re-run image validation
            if repair_stats["successful"] > 0:
                logger.info(f"Successfully repaired {repair_stats['successful']} images. Re-running image and class validation...")
                # It's important to re-run only the image validation part, as structure is already confirmed
                # The re-run will pick up the newly repaired files if they were saved in place of original.
                # If they were saved with "_repaired" suffix, the next training step would need to know to use those.
                # For this class, it re-scans all_image_paths, which should now include newly validated files if they replaced the originals.
                # If repaired files are new, they need to be added to all_image_paths or a new list
                # For simplicity, if repair is successful, you might want to consider overwriting original files or adjusting paths.
                # Here, the repair saves to a new path. For the validation to reflect this,
                # you'd ideally want to validate the *repaired* paths.
                # A full re-run of validate_image_files_and_discover_classes on original paths
                # won't reflect the new "_repaired" files unless they overwrite the originals.
                # For this example, let's assume repair means fixing the original file or that
                # a later stage will use the repaired paths. For a true re-validation,
                # you'd need to re-scan for all images again, which is implicitly done by
                # calling validate_image_files_and_discover_classes again.
                images_and_classes_ok = self.validate_image_files_and_discover_classes()
                overall_validation_status = overall_validation_status and images_and_classes_ok
            else:
                logger.warning("Image repair attempt completed, but no images were successfully repaired or identified for repair.")


        # Aggregate overall status based on reported statuses
        final_overall_status = True
        if self.validation_report.get("initial_structure_validation", {}).get("status", False) is False:
             final_overall_status = False
        if self.validation_report.get("image_and_class_validation", {}).get("status", False) is False:
            final_overall_status = False

        self.validation_report["overall_validation_status"] = final_overall_status

        self.save_validation_report()

        if final_overall_status:
            logger.info("Data validation completed successfully.")
        else:
            logger.error("Data validation completed with failures. Check report for details.")

        return final_overall_status