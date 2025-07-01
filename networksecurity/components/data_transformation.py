# ==============================================================================
# DATA TRANSFORMATION MODULE FOR NETWORK SECURITY PROJECT
# ==============================================================================
# This module handles the transformation of raw data into a format suitable for 
# machine learning. It cleans the data, handles missing values, and prepares 
# features for model training.

# IMPORTING REQUIRED LIBRARIES
# ==============================================================================
import sys                    # For system-specific parameters and functions
import os                     # For operating system interface (file operations)
import numpy as np            # For numerical operations and array handling
import pandas as pd           # For data manipulation and analysis (DataFrames)
from sklearn.impute import KNNImputer      # For filling missing values using K-Nearest Neighbors
from sklearn.pipeline import Pipeline      # For chaining multiple data processing steps

# IMPORTING PROJECT-SPECIFIC MODULES
# ==============================================================================
from networksecurity.constants.training_pipeline import TARGET_COLUMN
from networksecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,    # Structure to store transformation results
    DataValidationArtifact         # Structure containing validated data paths
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

# ==============================================================================
# MAIN DATA TRANSFORMATION CLASS
# ==============================================================================
class DataTransformation:
    """
    This class is responsible for transforming raw data into a format suitable 
    for machine learning models. It handles missing values, feature scaling, 
    and data preprocessing.
    
    WORKFLOW:
    1. Initialize with validation artifacts and configuration
    2. Read the validated train and test data
    3. Separate features from target variable
    4. Create and fit a data transformer (KNN Imputer)
    5. Transform both train and test data
    6. Save the transformed data and transformer object
    """
    
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        Constructor: Sets up the data transformation process
        
        Args:
            data_validation_artifact: Contains paths to validated train/test files
            data_transformation_config: Contains configuration for transformation process
        """
        try:
            # Store the artifacts for later use
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            # If initialization fails, raise custom exception with error details
            raise NetworkSecurityException(e, sys)
    
    # ==============================================================================
    # UTILITY METHOD: READ DATA FROM FILE
    # ==============================================================================
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Static method to read CSV data from a given file path
        
        What is @staticmethod?
        - A static method belongs to the class but doesn't need an instance to be called
        - It doesn't access instance variables (self) or class variables (cls)
        - Can be called directly on the class: DataTransformation.read_data(path)
        
        Args:
            file_path: Path to the CSV file to be read
            
        Returns:
            pd.DataFrame: Pandas DataFrame containing the data
            
        What is pd.read_csv()?
        - Built-in pandas function to read CSV files
        - Automatically detects data types and creates a DataFrame
        - Handles various CSV formats and separators
        """
        try:
            return pd.read_csv(file_path)  # Read CSV and return as DataFrame
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    # ==============================================================================
    # DATA TRANSFORMER CREATION METHOD
    # ==============================================================================
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Creates and returns a data transformation pipeline
        
        What is KNNImputer?
        - A method to fill missing values in data
        - Uses K-Nearest Neighbors algorithm to predict missing values
        - Finds 'k' similar rows and uses their values to fill missing data
        - More intelligent than simple mean/median filling
        
        What is Pipeline?
        - A way to chain multiple data processing steps together
        - Ensures the same transformations are applied to both train and test data
        - Makes the process reproducible and organized
        
        Returns:
            Pipeline: A sklearn pipeline with KNN imputer for handling missing values
        """
        logging.info("Entered get_data_transformer_object method of Transformation class")
        
        try:
            # Create KNN Imputer with predefined parameters
            # **DATA_TRANSFORMATION_IMPUTER_PARAMS unpacks dictionary as keyword arguments
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            
            logging.info(f"Initialized KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            
            # Create a pipeline with the imputer as the first (and only) step
            # Pipeline format: [("step_name", transformer_object)]
            processor: Pipeline = Pipeline([("imputer", imputer)])
            
            return processor
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ==============================================================================
    # MAIN TRANSFORMATION METHOD
    # ==============================================================================
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Main method that orchestrates the entire data transformation process
        
        STEP-BY-STEP WORKFLOW:
        1. Read validated train and test data
        2. Separate input features from target variable
        3. Handle target variable encoding (-1 to 0)
        4. Create and fit data transformer
        5. Transform both datasets
        6. Combine features and target into arrays
        7. Save transformed data and transformer object
        8. Return transformation artifacts
        
        Returns:
            DataTransformationArtifact: Contains paths to all transformed files
        """
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        
        try:
            logging.info("Starting data transformation")
            
            # ==================================================================
            # STEP 1: READ THE VALIDATED DATA FILES
            # ==================================================================
            # Read train and test data that passed validation
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # ==================================================================
            # STEP 2: PREPARE TRAINING DATA
            # ==================================================================
            # Separate input features (X) from target variable (y) for training data
            
            # Drop the target column to get input features
            # axis=1 means drop column (axis=0 would drop rows)
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            
            # Extract the target column
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            # Replace -1 with 0 in target (binary classification: 0 and 1)
            # This is common in ML where -1/1 encoding is converted to 0/1
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            # ==================================================================
            # STEP 3: PREPARE TESTING DATA
            # ==================================================================
            # Same process for test data
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            # ==================================================================
            # STEP 4: CREATE AND FIT DATA TRANSFORMER
            # ==================================================================
            # Get the preprocessing pipeline (KNN Imputer)
            preprocessor = self.get_data_transformer_object()

            # Fit the preprocessor on training data
            # .fit() learns the parameters needed for transformation
            # (e.g., which values to use for imputing missing data)
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            
            # Transform both training and test input features
            # .transform() applies the learned transformations
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # ==================================================================
            # STEP 5: COMBINE FEATURES AND TARGET INTO FINAL ARRAYS
            # ==================================================================
            # What is np.c_[]?
            # - NumPy function to concatenate arrays column-wise
            # - Combines transformed features with target variable
            # - Creates final arrays ready for model training
            
            # Combine transformed features with target for training
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            
            # Combine transformed features with target for testing
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # ==================================================================
            # STEP 6: SAVE TRANSFORMED DATA AND OBJECTS
            # ==================================================================
            # Save transformed training data as numpy array
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, 
                array=train_arr
            )
            
            # Save transformed testing data as numpy array
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            
            # Save the fitted preprocessor object for future use
            # This is important for making predictions on new data
            save_object(
                self.data_transformation_config.transformed_object_file_path, 
                preprocessor_object
            )

            # Save preprocessor in final_model directory for deployment
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            # ==================================================================
            # STEP 7: CREATE AND RETURN TRANSFORMATION ARTIFACT
            # ==================================================================
            # Create artifact containing paths to all transformed files
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

# ==============================================================================
# SUMMARY OF THE ENTIRE WORKFLOW
# ==============================================================================
"""
HIGH-LEVEL WORKFLOW EXPLANATION:

1. INITIALIZATION:
   - Set up the transformation process with validated data paths and configuration

2. DATA READING:
   - Read the validated train and test CSV files into pandas DataFrames

3. DATA PREPARATION:
   - Separate input features (X) from target variable (y)
   - Convert target labels from -1/1 to 0/1 format
   - Do this for both train and test sets

4. TRANSFORMATION SETUP:
   - Create a KNN Imputer to handle missing values
   - Wrap it in a Pipeline for organized processing

5. DATA TRANSFORMATION:
   - Fit the imputer on training data (learn how to fill missing values)
   - Transform both train and test data using the fitted imputer
   - This fills any missing values intelligently

6. FINAL ARRAY CREATION:
   - Combine transformed features with target variables
   - Create numpy arrays ready for machine learning models

7. PERSISTENCE:
   - Save transformed train/test arrays as files
   - Save the fitted transformer object for future use

8. RETURN RESULTS:
   - Return artifact containing paths to all saved files

WHY THIS PROCESS IS IMPORTANT:
- Ensures data quality by handling missing values
- Maintains consistency between train and test data transformations
- Creates reproducible preprocessing pipeline
- Prepares data in the format expected by ML algorithms
- Enables the same transformations to be applied to new data in production
"""