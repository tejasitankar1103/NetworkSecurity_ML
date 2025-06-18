# ---------------------------------------------
# 📁 File: config_entity.py
# 🔍 Purpose:
# This file defines configuration classes used to manage file paths and settings 
# for each stage of the ML pipeline: ingestion, validation, transformation, and training.
# Instead of hardcoding paths, this file generates them dynamically using timestamps,
# making the pipeline organized, reproducible, and easy to manage.
#
# 🔄 Overall Workflow:
# 1. MongoDB → Raw CSVs (DataIngestionConfig)
# 2. Raw CSVs → Valid/Invalid + Drift Report (DataValidationConfig)
# 3. Valid Data → Transformed Data + Preprocessor (DataTransformationConfig)
# 4. Transformed Data → Trained Model (ModelTrainerConfig)
# ---------------------------------------------

from datetime import datetime
import os

# Import constant values like folder names, filenames, ratios, etc.
from networksecurity.constants import training_pipeline

# Just to check if constants are loading properly (for debugging)
print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)

# 🔧 Configuration for setting up the root artifact directory and final model saving
class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        # Create a unique timestamp for every run (e.g., 06_18_2025_12_30_22)
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        
        # Main pipeline name and root artifact folder name
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        
        # Full path: artifact/06_18_2025_12_30_22/
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        
        # Folder to save the final model (after training)
        self.model_dir = os.path.join("final_model")
        
        # Saving timestamp so other configs can access it
        self.timestamp: str = timestamp


# 📥 Configuration for the Data Ingestion stage (MongoDB → CSV files)
class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory to store all data ingestion artifacts
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )

        # Path where all raw data will be saved after extraction from MongoDB
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )

        # Train data file path
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Test data file path
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )

        # Ratio to split the data into training and testing sets (e.g., 80:20)
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION

        # MongoDB collection and database names to pull data from
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


# ✅ Configuration for the Data Validation stage (cleaning and checking data quality)
class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory to store all data validation outputs
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        # Folder and file paths for valid data
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )

        # Folder and file paths for invalid (bad or missing) data
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )

        # Path to save data drift report (shows if new data is different from training data)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )


# 🔄 Configuration for the Data Transformation stage (preprocessing)
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory to store all transformation outputs
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        # File paths for transformed training and testing data (as .npy files)
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy")
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy")
        )

        # File path to save the transformation/preprocessing object (like StandardScaler)
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )


# 🧠 Configuration for Model Training stage (train the ML model)
class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory to save all model training related artifacts
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME
        )

        # File path to save the final trained model (.pkl or .sav)
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_FILE_NAME
        )

        # Accuracy expected from model (threshold to decide success)
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE

        # Threshold to detect overfitting or underfitting
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
